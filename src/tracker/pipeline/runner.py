"""Video processing utilities with streaming architecture for tracking.

This module handles video file operations and PyQt threading for streaming processing.
Core business logic has been moved to pipeline/orchestrator.py for testability.
"""
import logging
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from tracker.config.settings import SUPPORTED_VIDEO_EXTENSIONS, MAX_FRAME_DISPLAY_SIZE
from tracker.pipeline.memory import MemoryManager
from tracker.pipeline.orchestrator import StreamingOrchestrator, ResultStatus

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = frozenset(ext.lower() for ext in SUPPORTED_VIDEO_EXTENSIONS)


class VideoProcessor:
    """Handles video file processing operations."""

    def __init__(self):
        """Initialize video processor."""
        self._frame_cache: Dict[str, np.ndarray] = {}
        self._video_info_cache: Dict[str, Dict] = {}

    def get_video_files(self, folder_path: str) -> List[Path]:
        """Get list of video files in folder."""
        try:
            folder = Path(folder_path)
            if not folder.exists() or not folder.is_dir():
                logger.warning(f"Invalid folder path: {folder_path}")
                return []

            video_files = []
            for file in folder.iterdir():
                if file.is_file():
                    if file.suffix.lower() in _SUPPORTED_EXTENSIONS:
                        video_files.append(file)

            video_files.sort()
            logger.info(f"Found {len(video_files)} video files in {folder_path}")
            return video_files

        except Exception as e:
            logger.error(f"Error scanning folder: {e}")
            return []

    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """Get video metadata."""
        if video_path in self._video_info_cache:
            return self._video_info_cache[video_path]

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                logger.warning(f"Video reports fps={fps}, using fallback of 30.0: {video_path}")
                fps = 30.0

            info = {
                "path": video_path,
                "name": Path(video_path).name,
                "frame_count": frame_count,
                "fps": fps,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": int(frame_count / fps)
            }

            cap.release()
            self._video_info_cache[video_path] = info
            return info

        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None

    def extract_first_frame(self, video_path: str) -> Optional[np.ndarray]:
        """Extract first frame from video.

        Returns:
            First frame as RGB numpy array, or None if failed.
        """
        if video_path in self._frame_cache:
            return self._frame_cache[video_path].copy()

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None

            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.error(f"Cannot read first frame from: {video_path}")
                return None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frame_cache[video_path] = frame_rgb

            logger.info(f"Extracted first frame from: {Path(video_path).name}")
            return frame_rgb.copy()

        except Exception as e:
            logger.error(f"Error extracting first frame: {e}")
            return None

    def save_annotation(self, annotation_data: Dict[str, Any],
                        output_path: str,
                        mask: Optional[np.ndarray] = None) -> bool:
        """Save annotation data to JSON file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(annotation_data, f, indent=2)

            logger.info(f"Saved annotation to: {output_file}")

            if mask is not None:
                mask_path = output_file.with_suffix('.png')
                mask_uint8 = (mask * 255).astype(np.uint8)
                success = cv2.imwrite(str(mask_path), mask_uint8)

                if success:
                    logger.info(f"Saved mask to: {mask_path}")
                else:
                    logger.error(f"Failed to save mask to: {mask_path}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error saving annotation: {e}")
            return False

    def load_annotation(self, annotation_path: str) -> Optional[Dict[str, Any]]:
        """Load annotation data from JSON file."""
        try:
            annotation_file = Path(annotation_path)

            if not annotation_file.exists():
                logger.debug(f"Annotation file not found: {annotation_path}")
                return None

            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)

            required_fields = ['video_path', 'points', 'labels']
            for field in required_fields:
                if field not in annotation_data:
                    logger.warning(f"Missing required field '{field}' in annotation: {annotation_path}")
                    return None

            logger.info(f"Loaded annotation from: {annotation_file}")
            return annotation_data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in annotation file {annotation_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading annotation: {e}")
            return None

    def get_complete_videos(self, video_files: List[Path],
                            video_statuses: Dict[str, str]) -> List[Path]:
        """Get list of videos marked as complete."""
        complete_videos = []
        for video_path in video_files:
            if video_statuses.get(str(video_path)) == "Complete":
                complete_videos.append(video_path)
        return complete_videos

    def clear_cache(self):
        """Clear frame and info caches."""
        self._frame_cache.clear()
        self._video_info_cache.clear()
        logger.info("Cleared video processor cache")


class StreamedTrackingThread(QThread):
    """Thread for streaming video tracking with immediate export."""

    # Signals
    video_progress = pyqtSignal(int, int)  # current_video, total_videos
    frame_progress = pyqtSignal(int, int, str)  # current_frame, total_frames, status
    status_message = pyqtSignal(str)
    tracking_complete = pyqtSignal(int)  # number of videos processed (legacy)
    tracking_complete_detailed = pyqtSignal(list)  # list of (video_name, ResultStatus, error_msg)
    error_occurred = pyqtSignal(str)
    memory_usage = pyqtSignal(float, float)  # gpu_memory_mb, ram_mb

    def __init__(self, sam2_manager, video_processor):
        """Initialize streaming tracking thread."""
        super().__init__()
        self.sam2_manager = sam2_manager
        self.video_processor = video_processor
        self.complete_videos = []
        self.annotations_folder = None
        self.output_folder = None
        self.export_videos = True
        self.export_csv = True
        self._pause_event = threading.Event()    # set = paused
        self._cancel_event = threading.Event()   # set = cancel requested
        self._stop_after_current_event = threading.Event()  # set = stop after current
        self.videos_processed = 0

        # Single shared MemoryManager for all components
        self.memory_manager = MemoryManager()
        self.sam2_manager.memory_manager = self.memory_manager
        self.orchestrator = StreamingOrchestrator(
            tracker=sam2_manager,
            video_processor=video_processor,
            memory_manager=self.memory_manager
        )

        # Wire up orchestrator callbacks to emit Qt signals
        self.orchestrator.on_status_update = lambda msg: self.status_message.emit(msg)
        self.orchestrator.on_frame_progress = lambda c, t, s: self.frame_progress.emit(c, t, s)
        self.orchestrator.should_cancel = lambda: self._cancel_event.is_set()
        self.orchestrator.should_pause = lambda: self._pause_event.is_set()

    def setup(self, complete_videos: List[Path], annotations_folder: Path,
              output_folder: Path, export_videos: bool = True, export_csv: bool = True):
        """Set up tracking parameters."""
        self.complete_videos = complete_videos
        self.annotations_folder = annotations_folder
        self.output_folder = output_folder
        self.export_videos = export_videos
        self.export_csv = export_csv
        self.videos_processed = 0

    def pause(self):
        """Pause tracking."""
        self._pause_event.set()

    def resume(self):
        """Resume tracking."""
        self._pause_event.clear()

    def stop_after_current(self):
        """Stop after completing current video."""
        self._stop_after_current_event.set()

    def cancel(self):
        """Cancel tracking immediately."""
        self._cancel_event.set()

    def _report_memory_usage(self):
        """Report current memory usage."""
        try:
            gpu_mb, ram_mb = self.memory_manager.get_memory_usage()
            self.memory_usage.emit(gpu_mb, ram_mb)
        except Exception:
            pass

    def _cleanup_memory(self):
        """Aggressively clean up memory after each video."""
        try:
            # Use memory manager for cleanup
            self.memory_manager.cleanup_after_video()

            # Clear video processor caches
            self.video_processor.clear_cache()

            # Report memory after cleanup
            self._report_memory_usage()

        except Exception as e:
            logger.warning(f"Memory cleanup warning: {e}")

    def run(self):
        """Run streaming tracking process using orchestrator."""
        try:
            from tracker.io.video_writer import VideoWriterManager

            # Create export directories once
            export_dirs = VideoWriterManager.create_export_directories(self.output_folder)

            total_videos = len(self.complete_videos)
            video_results = []  # (video_name, ResultStatus, error_msg)

            for video_idx, video_path in enumerate(self.complete_videos):
                # Check if should stop
                if self._cancel_event.is_set():
                    self.status_message.emit("Tracking cancelled")
                    break

                if self._stop_after_current_event.is_set() and video_idx > 0:
                    self.status_message.emit(f"Stopped after video {video_idx}")
                    break

                # Wait if paused
                while self._pause_event.is_set() and not self._cancel_event.is_set():
                    self.msleep(100)

                # Update progress
                self.video_progress.emit(video_idx + 1, total_videos)
                self.status_message.emit(f"Processing video {video_idx + 1}/{total_videos} (streaming mode)")

                # Delegate to orchestrator for processing
                video_name = video_path.stem
                annotation_file = self.annotations_folder / f"{video_name}.json"

                result = self.orchestrator.process_video(
                    video_path=video_path,
                    annotation_file=annotation_file,
                    tracked_videos_dir=export_dirs['tracked_videos'],
                    contour_data_dir=export_dirs['contour_data'],
                    export_videos=self.export_videos,
                    export_csv=self.export_csv
                )

                video_results.append((video_path.name, result.status, result.error_message))

                if result.success:
                    self.videos_processed += 1
                    logger.info(
                        f"Successfully processed video {video_idx + 1}: {video_path.name} "
                        f"({result.fps_achieved:.1f} fps)"
                    )
                elif result.status == ResultStatus.CANCELLED:
                    logger.info(f"Cancelled during video {video_idx + 1}: {video_path.name}")
                    break  # Stop processing further videos
                else:
                    logger.error(
                        f"Failed to process video {video_idx + 1}: {video_path.name} - "
                        f"{result.error_message}"
                    )

                # Aggressive memory cleanup after EACH video
                self._cleanup_memory()

                # Brief pause to ensure cleanup
                self.msleep(100)

            # Emit detailed results (always) then legacy signal
            self.tracking_complete_detailed.emit(video_results)
            self.tracking_complete.emit(self.videos_processed)
            self.status_message.emit(f"Streaming processing complete: {self.videos_processed} videos")

        except Exception as e:
            logger.error(f"Streaming tracking error: {e}")
            self.error_occurred.emit(str(e))
