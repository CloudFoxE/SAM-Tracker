"""Batch processing controller for streaming video tracking.

Manages the streaming tracking thread lifecycle — start, pause, resume,
stop — and relays progress signals to the UI.
"""

import logging
from pathlib import Path
from typing import Optional, List, Callable

from tracker.tracking.sam2 import Sam2Tracker as SAM2Manager
from tracker.pipeline.runner import VideoProcessor, StreamedTrackingThread
from tracker.io.video_writer import VideoWriterManager

logger = logging.getLogger(__name__)


class BatchProcessingController:
    """Controls the streaming batch processing lifecycle.

    Manages StreamedTrackingThread creation and signal wiring.
    Communicates with the UI through callbacks.
    """

    def __init__(self,
                 sam2_manager: SAM2Manager,
                 video_processor: VideoProcessor,
                 on_status: Callable[[str], None]):
        """Initialize batch processing controller.

        Args:
            sam2_manager: SAM2 model manager
            video_processor: Video I/O processor
            on_status: Callback for status bar messages
        """
        self.sam2_manager = sam2_manager
        self.video_processor = video_processor
        self._on_status = on_status

        self._thread: Optional[StreamedTrackingThread] = None

        # Callbacks set by main window for signal relay
        self.on_video_progress: Optional[Callable[[int, int], None]] = None
        self.on_frame_progress: Optional[Callable[[int, int, str], None]] = None
        self.on_complete: Optional[Callable[[int], None]] = None
        self.on_complete_detailed: Optional[Callable[[list], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_memory_usage: Optional[Callable[[float, float], None]] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    @staticmethod
    def estimate_total_size(videos: List[Path], include_csv: bool) -> float:
        """Estimate total export size in MB."""
        return sum(
            VideoWriterManager.estimate_export_size(str(v), include_csv)
            for v in videos
        )

    @staticmethod
    def check_disk_space(output_folder: Path, required_mb: int) -> bool:
        """Check if sufficient disk space is available."""
        return VideoWriterManager.check_disk_space(output_folder, required_mb)

    def start(self,
              complete_videos: List[Path],
              annotations_folder: Path,
              output_folder: Path,
              export_videos: bool,
              export_csv: bool):
        """Start streaming processing.

        Args:
            complete_videos: List of annotated video paths
            annotations_folder: Path to annotations directory
            output_folder: Path to output directory
            export_videos: Whether to export tracked videos
            export_csv: Whether to export CSV contours
        """
        self._thread = StreamedTrackingThread(
            self.sam2_manager,
            self.video_processor
        )
        self._thread.setup(
            complete_videos,
            annotations_folder,
            output_folder,
            export_videos,
            export_csv,
        )

        # Wire signals to callbacks
        if self.on_video_progress:
            self._thread.video_progress.connect(self.on_video_progress)
        if self.on_frame_progress:
            self._thread.frame_progress.connect(self.on_frame_progress)
        if self.on_complete:
            self._thread.tracking_complete.connect(self.on_complete)
        if self.on_complete_detailed:
            self._thread.tracking_complete_detailed.connect(self.on_complete_detailed)
        if self.on_error:
            self._thread.error_occurred.connect(self.on_error)
        if self.on_memory_usage:
            self._thread.memory_usage.connect(self.on_memory_usage)
        self._thread.status_message.connect(self._on_status)

        self._thread.start()
        logger.info(f"Started streaming processing for {len(complete_videos)} videos")

    def pause(self):
        """Pause streaming processing."""
        if self._thread:
            self._thread.pause()
            self._on_status("Streaming paused")

    def resume(self):
        """Resume streaming processing."""
        if self._thread:
            self._thread.resume()
            self._on_status("Streaming resumed")

    def stop_after_current(self):
        """Stop after the current video completes."""
        if self._thread:
            self._thread.stop_after_current()
            self._on_status("Will stop after current video completes")

    def cancel_and_wait(self):
        """Cancel processing and wait for thread to finish."""
        if self._thread and self._thread.isRunning():
            self._thread.cancel()
            self._thread.wait()
