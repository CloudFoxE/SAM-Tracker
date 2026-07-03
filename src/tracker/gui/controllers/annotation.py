"""Annotation controller for first-frame mask annotation workflow.

Manages point annotation, mask generation, saving, and video navigation
during the annotation phase.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Callable

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from tracker.tracking.sam2 import Sam2Tracker as SAM2Manager
from tracker.pipeline.runner import VideoProcessor
from tracker.gui.widgets.image_viewer import ImageViewer

logger = logging.getLogger(__name__)


class MaskGenerateThread(QThread):
    """Thread for generating masks."""

    finished = pyqtSignal(object)

    def __init__(self, sam2_manager: SAM2Manager):
        super().__init__()
        self.sam2_manager = sam2_manager
        self.image = None
        self.points = []
        self.labels = []

    def setup(self, image, points, labels):
        """Set up generation parameters."""
        self.image = image
        self.points = points
        self.labels = labels

    def run(self):
        """Generate the mask."""
        mask = self.sam2_manager.generate_mask(self.image, self.points, self.labels)
        self.finished.emit(mask)


class AnnotationController:
    """Controls the first-frame annotation workflow.

    Manages video navigation, point annotation, mask generation/saving,
    and video status tracking. Communicates with the UI through callbacks.
    """

    def __init__(self,
                 sam2_manager: SAM2Manager,
                 video_processor: VideoProcessor,
                 image_viewer: ImageViewer,
                 on_status: Callable[[str], None],
                 on_navigation_changed: Callable[[], None],
                 on_error: Optional[Callable[[str, str], None]] = None):
        """Initialize annotation controller.

        Args:
            sam2_manager: SAM2 model manager
            video_processor: Video I/O processor
            image_viewer: Image viewer widget for annotation display
            on_status: Callback for status bar messages
            on_navigation_changed: Callback when video index/statuses change
            on_error: Callback for error dialogs (title, message)
        """
        self.sam2_manager = sam2_manager
        self.video_processor = video_processor
        self.image_viewer = image_viewer
        self._on_status = on_status
        self._on_navigation_changed = on_navigation_changed
        self._on_error = on_error

        # State
        self.current_image: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.current_video_index: int = 0
        self.video_files: List[Path] = []
        self.video_statuses: Dict[str, str] = {}
        self.annotations_folder: Optional[Path] = None

        # Thread reference (prevent GC)
        self._generate_thread: Optional[MaskGenerateThread] = None

        # Callback for mask generation result (set by main window)
        self.on_mask_generated: Optional[Callable[[object], None]] = None

    @property
    def complete_count(self) -> int:
        return sum(1 for s in self.video_statuses.values() if s == "Complete")

    @property
    def skip_count(self) -> int:
        return sum(1 for s in self.video_statuses.values() if s == "Skip")

    def load_folder(self, folder_path: str):
        """Load videos from a folder."""
        self.video_files = self.video_processor.get_video_files(folder_path)
        self.video_statuses.clear()
        self.current_video_index = 0

        # Restore statuses from persisted annotations
        self._restore_statuses_from_annotations()

        self._on_navigation_changed()

        if self.video_files:
            self.load_video_at_index(0)

    def _restore_statuses_from_annotations(self):
        """Rebuild video_statuses from saved annotation files on disk.

        Videos with a valid annotation JSON (points, labels, mask_generated=True)
        are marked "Complete" so batch export works after reopening a project.
        """
        if not self.annotations_folder or not self.annotations_folder.exists():
            return

        for video_path in self.video_files:
            annotation_file = self.annotations_folder / f"{video_path.stem}.json"
            if not annotation_file.exists():
                continue
            try:
                annotation = self.video_processor.load_annotation(str(annotation_file))
                if (annotation
                        and annotation.get('mask_generated', False)
                        and annotation.get('points')):
                    self.video_statuses[str(video_path)] = "Complete"
                    logger.info(f"Restored 'Complete' status for {video_path.name}")
            except Exception as e:
                logger.warning(f"Could not restore annotation for {video_path.name}: {e}")

    def load_video_at_index(self, index: int):
        """Load and display a video at the given index."""
        if not (0 <= index < len(self.video_files)):
            return

        self.current_video_index = index
        video_path = self.video_files[index]

        self._on_status(f"Loading video: {video_path.name}")

        frame = self.video_processor.extract_first_frame(str(video_path))
        if frame is not None:
            self.image_viewer.clear_points()
            self.current_mask = None
            self.current_image = frame
            self.image_viewer.set_image(frame)

            info = self.video_processor.get_video_info(str(video_path))
            if info:
                self._on_status(
                    f"Video: {info['name']} | "
                    f"Size: {info['width']}x{info['height']} | "
                    f"Frames: {info['frame_count']} | "
                    f"FPS: {info['fps']:.1f}"
                )

            self._load_existing_annotation(video_path)
        else:
            self._on_status("Failed to load video")
            if self._on_error:
                self._on_error("Error", "Failed to load video first frame")

        self._on_navigation_changed()

    def _load_existing_annotation(self, video_path: Path):
        """Load existing annotation if available."""
        if not self.annotations_folder:
            return
        try:
            annotation_file = self.annotations_folder / f"{video_path.stem}.json"
            if not annotation_file.exists():
                return

            annotation = self.video_processor.load_annotation(str(annotation_file))
            if not annotation:
                return

            points = annotation.get('points', [])
            labels = annotation.get('labels', [])
            for (x, y), label in zip(points, labels):
                self.image_viewer.add_point(x, y, label)

            if annotation.get('mask_generated', False) and points:
                self.generate_mask()

            logger.info(f"Loaded existing annotation for {video_path.name}")
        except Exception as e:
            logger.error(f"Error loading annotation: {e}")

    def generate_mask(self):
        """Start mask generation from current points."""
        if self.current_image is None:
            return False

        points = self.image_viewer.get_points()
        if not points:
            return False

        point_coords = [(p[0], p[1]) for p in points]
        point_labels = [p[2] for p in points]

        self._generate_thread = MaskGenerateThread(self.sam2_manager)
        self._generate_thread.setup(self.current_image, point_coords, point_labels)
        self._generate_thread.finished.connect(self._handle_mask_result)
        self._generate_thread.start()
        return True

    def _handle_mask_result(self, mask):
        """Handle mask generation completion."""
        if mask is not None:
            self.current_mask = mask
            self.image_viewer.set_mask(mask)

        if self.on_mask_generated:
            self.on_mask_generated(mask)

    def clear_points(self):
        """Clear all annotation points."""
        self.image_viewer.clear_points()

    def clear_mask(self):
        """Clear the current mask."""
        self.image_viewer.set_mask(None)
        self.current_mask = None

    def save_mask(self) -> bool:
        """Save current mask and annotation. Returns True on success."""
        if self.current_mask is None or not self.video_files:
            return False

        try:
            current_video = self.video_files[self.current_video_index]
            points = self.image_viewer.get_points()

            annotation_data = {
                "video_path": str(current_video),
                "points": [[p[0], p[1]] for p in points],
                "labels": [p[2] for p in points],
                "frame_idx": 0,
                "timestamp": datetime.now().isoformat(),
                "mask_generated": True,
            }

            if not self.annotations_folder:
                return False

            annotation_file = self.annotations_folder / f"{current_video.stem}.json"
            success = self.video_processor.save_annotation(
                annotation_data, str(annotation_file), self.current_mask
            )

            if success:
                self.video_statuses[str(current_video)] = "Complete"
                self._on_status(f"Saved annotation for {current_video.name}")
                self._on_navigation_changed()
            return success

        except Exception as e:
            logger.error(f"Error saving mask: {e}")
            return False

    def navigate_previous(self):
        """Navigate to the previous video."""
        if self.current_video_index > 0:
            self.load_video_at_index(self.current_video_index - 1)

    def navigate_next(self):
        """Navigate to the next video."""
        if self.current_video_index < len(self.video_files) - 1:
            self.load_video_at_index(self.current_video_index + 1)

    def skip_current(self):
        """Skip the current video."""
        if not self.video_files:
            return
        current_path = str(self.video_files[self.current_video_index])
        self.video_statuses[current_path] = "Skip"
        self._on_navigation_changed()

    def advance_to_next_or_complete(self) -> bool:
        """Advance to next video. Returns False if this was the last video."""
        if self.current_video_index < len(self.video_files) - 1:
            self.load_video_at_index(self.current_video_index + 1)
            return True
        return False

    def get_complete_videos(self) -> List[Path]:
        """Get list of videos marked as complete."""
        return self.video_processor.get_complete_videos(
            self.video_files, self.video_statuses
        )

    def get_display_files(self):
        """Get video files with status for display."""
        return [
            (path, self.video_statuses.get(str(path), ""))
            for path in self.video_files
        ]

    def can_navigate_prev(self) -> bool:
        return self.current_video_index > 0

    def can_navigate_next(self) -> bool:
        return self.current_video_index < len(self.video_files) - 1

    def has_mask(self) -> bool:
        return self.current_mask is not None
