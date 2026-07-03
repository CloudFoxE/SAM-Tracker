"""Streaming video tracking orchestrator.

This module contains the core business logic for streaming video processing,
separated from PyQt threading and I/O operations for testability.
"""

import enum
import logging
import shutil
import time
import traceback
from pathlib import Path
from typing import Optional, Callable, Dict
import numpy as np

from tracker.pipeline.video_io import VideoReader
from tracker.pipeline.memory import MemoryManager
from tracker.analysis.contours import ContourExtractor, ContourSelector
from tracker.io.exporters import ContourExporter
from tracker.io.overlay import FrameOverlay
from tracker.io.video_writer import VideoWriterManager
from tracker.config.settings import DEFAULT_OVERLAY_CONFIG, DEFAULT_VIDEO_PROCESSING_CONFIG
from tracker.tracking.base import TrackerBackend, ObjectPrompt, FramePrediction

logger = logging.getLogger(__name__)


class ResultStatus(enum.Enum):
    """Outcome of processing a single video."""
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrackingResult:
    """Result of tracking a single video."""

    def __init__(self,
                 status: ResultStatus = ResultStatus.SUCCESS,
                 frames_processed: int = 0,
                 frames_with_mask: int = 0,
                 elapsed_time: float = 0.0,
                 error_message: str = "",
                 # Keep a convenience property for backward-compat in simple checks
                 success: Optional[bool] = None):
        # If caller used the old success= kwarg, translate it
        if success is not None:
            self.status = ResultStatus.SUCCESS if success else ResultStatus.FAILED
        else:
            self.status = status
        self.frames_processed = frames_processed
        self.frames_with_mask = frames_with_mask
        self.elapsed_time = elapsed_time
        self.fps_achieved = frames_processed / elapsed_time if elapsed_time > 0 else 0
        self.error_message = error_message

    @property
    def success(self) -> bool:
        """Backward-compatible convenience: True only for SUCCESS."""
        return self.status == ResultStatus.SUCCESS


class StreamingOrchestrator:
    """Orchestrates streaming video processing with immediate export.

    This class contains the core business logic separated from threading and GUI.
    All progress callbacks are optional - this makes it testable without PyQt.
    """

    def __init__(self,
                 tracker: TrackerBackend,
                 video_processor,
                 memory_manager: Optional[MemoryManager] = None):
        """Initialize streaming orchestrator.

        Args:
            tracker: TrackerBackend instance
            video_processor: VideoProcessor instance
            memory_manager: MemoryManager instance (creates default if None)
        """
        self.tracker = tracker
        self.video_processor = video_processor
        self.memory_manager = memory_manager or MemoryManager()

        # Optional callbacks for progress reporting
        self.on_status_update: Optional[Callable[[str], None]] = None
        self.on_frame_progress: Optional[Callable[[int, int, str], None]] = None
        self.should_cancel: Optional[Callable[[], bool]] = None
        self.should_pause: Optional[Callable[[], bool]] = None

    def process_video(self,
                      video_path: Path,
                      annotation_file: Path,
                      tracked_videos_dir: Path,
                      contour_data_dir: Path,
                      export_videos: bool = True,
                      export_csv: bool = True) -> TrackingResult:
        """Process a single video with streaming export.

        Args:
            video_path: Path to input video
            annotation_file: Path to annotation JSON file
            tracked_videos_dir: Directory for tracked video output
            contour_data_dir: Directory for CSV contour data
            export_videos: Whether to export tracked video
            export_csv: Whether to export CSV contours

        Returns:
            TrackingResult with processing statistics
        """
        start_time = time.time()
        video_name = video_path.stem

        try:
            # Update status
            self._update_status(f"Loading → {video_name}")

            # Load annotation
            annotation = self.video_processor.load_annotation(str(annotation_file))
            if not annotation:
                return TrackingResult(
                    success=False,
                    error_message=f"No annotation found for {video_name}"
                )

            # Get video info
            video_info = self.video_processor.get_video_info(str(video_path))
            if not video_info:
                return TrackingResult(
                    success=False,
                    error_message=f"Failed to get video info for {video_name}"
                )

            width = video_info['width']
            height = video_info['height']
            fps = video_info['fps']
            total_frames = video_info['frame_count']

            # Initialize model for this video
            self._update_status(f"Initializing model → {video_name}")
            if not self.tracker.init_video(str(video_path),
                                           lambda c, t: self._update_frame_progress(c, t, "Extracting")):
                return TrackingResult(success=False, error_message=f"Failed to init model state for {video_name}")

            prompts = [ObjectPrompt(obj_id=1, points=[tuple(p) for p in annotation['points']],
                                    labels=list(annotation['labels']))]
            if not self.tracker.add_prompts(0, prompts):
                return TrackingResult(success=False, error_message=f"Failed to add prompts for {video_name}")

            # Process frames with streaming export
            result = self._process_frames_streaming(
                video_path=video_path,
                video_name=video_name,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                tracked_videos_dir=tracked_videos_dir,
                contour_data_dir=contour_data_dir,
                export_videos=export_videos,
                export_csv=export_csv
            )

            self._update_status(f"Completed → {video_name}")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_path.name}: {e}")
            traceback.print_exc()
            return TrackingResult(
                success=False,
                error_message=str(e)
            )

        finally:
            # reset() handles cleanup_after_video() on the shared MemoryManager
            self.tracker.reset()
            self.video_processor.clear_cache()

    def _process_frames_streaming(self,
                                   video_path: Path,
                                   video_name: str,
                                   width: int,
                                   height: int,
                                   fps: float,
                                   total_frames: int,
                                   tracked_videos_dir: Path,
                                   contour_data_dir: Path,
                                   export_videos: bool,
                                   export_csv: bool) -> TrackingResult:
        """Process frames with streaming export.

        Args:
            video_path: Path to input video
            video_name: Video name (stem)
            width: Video width
            height: Video height
            fps: Video FPS
            total_frames: Total frame count
            tracked_videos_dir: Output directory for tracked video
            contour_data_dir: Output directory for CSV data
            export_videos: Whether to export video
            export_csv: Whether to export CSV

        Returns:
            TrackingResult with statistics
        """
        mp4_writer = None
        frames_processed = 0
        frames_with_mask = 0
        start_time = time.time()  # Track processing time

        try:
            # Open video for reading
            with VideoReader(str(video_path)) as video_reader:
                # Create video writer if needed
                if export_videos:
                    output_mp4 = tracked_videos_dir / f"{video_name}_tracked.mp4"
                    mp4_writer = VideoWriterManager.create_writer(
                        str(output_mp4), fps, width, height
                    )
                    if not mp4_writer:
                        return TrackingResult(
                            success=False,
                            error_message=f"Failed to create video writer for {output_mp4}"
                        )

                # Create CSV output folder
                csv_folder = None
                if export_csv:
                    csv_folder = contour_data_dir / video_name
                    csv_folder.mkdir(parents=True, exist_ok=True)

                # Create overlay handler (reuse for all frames)
                overlay_handler = FrameOverlay(
                    color=DEFAULT_OVERLAY_CONFIG.color,
                    opacity=DEFAULT_OVERLAY_CONFIG.opacity,
                    contour_thickness=DEFAULT_OVERLAY_CONFIG.contour_thickness
                )

                # Create contour extractor/selector (reuse)
                contour_extractor = ContourExtractor()
                contour_selector = ContourSelector()

                # Update status
                self._update_status(f"Analyzing → {video_name}")

                # Stream process frame-by-frame
                for pred in self.tracker.propagate():
                    # Check cancellation
                    if self._check_should_cancel():
                        break

                    # Pause handling
                    while self._check_should_pause():
                        time.sleep(0.1)
                        if self._check_should_cancel():
                            break

                    frame_idx = pred.frame_idx
                    # Phase 0 preserves single-object export: take the one mask if present.
                    mask = next(iter(pred.masks.values())) if pred.masks else None

                    # Update progress
                    self._update_frame_progress(frame_idx + 1, total_frames, "Analyzing")

                    # Read next sequential frame (no seeking needed)
                    frame = video_reader.read_next_frame()
                    if frame is None:
                        logger.warning(f"Could not read frame {frame_idx}")
                        # Write blank frame if export enabled
                        if export_videos and mp4_writer:
                            frame = np.zeros((height, width, 3), dtype=np.uint8)
                            mp4_writer.write(frame)
                        continue

                    # OPTIMIZATION: Extract contours ONCE
                    contours = None
                    largest_contour = None

                    if mask is not None and np.any(mask):
                        contours = contour_extractor.find_contours(mask)
                        if contours:
                            largest_contour = contour_selector.select_largest(contours)
                            frames_with_mask += 1

                    # A. Write CSV immediately
                    if export_csv and csv_folder:
                        csv_path = csv_folder / f"frame_{frame_idx:04d}.csv"
                        if largest_contour is not None:
                            ContourExporter.export_to_csv(largest_contour, str(csv_path))
                        else:
                            csv_path.touch()

                    # B. Write MP4 frame immediately
                    if export_videos and mp4_writer:
                        if contours:
                            masked_frame = overlay_handler.apply_overlay(frame, mask, contours=contours)
                            mp4_writer.write(masked_frame)
                        else:
                            mp4_writer.write(frame)

                    # C. Discard immediately
                    mask = None
                    frame = None
                    contours = None
                    largest_contour = None
                    frames_processed += 1

                    # Periodic cleanup
                    if self.memory_manager.should_cleanup(frames_processed):
                        self.memory_manager.periodic_cleanup()
                        self._update_frame_progress(
                            frames_processed, total_frames,
                            "Exporting (memory cleanup)"
                        )

                    # Update status periodically
                    if frames_processed % DEFAULT_VIDEO_PROCESSING_CONFIG.status_update_interval == 0:
                        self._update_frame_progress(
                            frames_processed, total_frames,
                            "Exporting"
                        )

            # Calculate stats
            elapsed_time = time.time() - start_time
            cancelled = self._check_should_cancel()

            if cancelled:
                logger.info(
                    f"Cancelled after {frames_processed} frames in {elapsed_time:.1f}s"
                )
                return TrackingResult(
                    status=ResultStatus.CANCELLED,
                    frames_processed=frames_processed,
                    frames_with_mask=frames_with_mask,
                    elapsed_time=elapsed_time,
                    error_message="Cancelled by user",
                )

            logger.info(f"Processed {frames_processed} frames in {elapsed_time:.1f}s")
            logger.info(f"Frames with masks: {frames_with_mask}/{frames_processed}")

            return TrackingResult(
                status=ResultStatus.SUCCESS,
                frames_processed=frames_processed,
                frames_with_mask=frames_with_mask,
                elapsed_time=elapsed_time
            )

        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            traceback.print_exc()
            return TrackingResult(
                status=ResultStatus.FAILED,
                error_message=str(e)
            )

        finally:
            if mp4_writer:
                VideoWriterManager.release_writer(mp4_writer)

    def _update_status(self, message: str):
        """Update status message (calls callback if set)."""
        if self.on_status_update:
            self.on_status_update(message)

    def _update_frame_progress(self, current: int, total: int, status: str):
        """Update frame progress (calls callback if set)."""
        if self.on_frame_progress:
            self.on_frame_progress(current, total, status)

    def _check_should_cancel(self) -> bool:
        """Check if processing should be cancelled."""
        if self.should_cancel:
            return self.should_cancel()
        return False

    def _check_should_pause(self) -> bool:
        """Check if processing should be paused."""
        if self.should_pause:
            return self.should_pause()
        return False
