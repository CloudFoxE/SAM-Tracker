"""Video I/O utilities for reading video files.

This module handles pure video file input operations with context manager support.
Separated from business logic for single responsibility and testability.
"""

import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Context manager for reading video files.

    Provides clean interface for video file operations with automatic resource cleanup.
    """

    def __init__(self, video_path: str):
        """Initialize video reader.

        Args:
            video_path: Path to video file
        """
        self.video_path = str(video_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False

    def __enter__(self):
        """Open video file."""
        self.cap = cv2.VideoCapture(self.video_path)
        if self.cap.isOpened():
            self._is_opened = True
            logger.debug(f"Opened video: {Path(self.video_path).name}")
        else:
            logger.error(f"Failed to open video: {self.video_path}")
            raise IOError(f"Cannot open video file: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close video file."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self._is_opened = False
            logger.debug(f"Closed video: {Path(self.video_path).name}")
        return False

    def read_next_frame(self) -> Optional[np.ndarray]:
        """Read the next sequential frame without seeking.

        More efficient than read_frame() when reading frames in order,
        as it avoids a seek operation per frame.

        Returns:
            Frame as BGR numpy array, or None if read fails.
        """
        if not self._is_opened or self.cap is None:
            logger.error("Video not opened")
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read a specific frame from video by seeking to the given index.

        Args:
            frame_idx: Frame index to read (0-based)

        Returns:
            Frame as BGR numpy array, or None if read fails.
            Note: Returns BGR (OpenCV default), not RGB.
        """
        if not self._is_opened or self.cap is None:
            logger.error("Video not opened")
            return None

        try:
            # Set position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # Read frame
            ret, frame = self.cap.read()

            if not ret:
                logger.warning(f"Could not read frame {frame_idx}")
                return None

            return frame

        except Exception as e:
            logger.error(f"Error reading frame {frame_idx}: {e}")
            return None

    def get_info(self) -> dict:
        """Get video metadata.

        Returns:
            Dictionary with video properties
        """
        if not self._is_opened or self.cap is None:
            return {}

        try:
            info = {
                "path": self.video_path,
                "name": Path(self.video_path).name,
                "frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            return info

        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}

    @property
    def is_opened(self) -> bool:
        """Check if video is currently opened."""
        return self._is_opened and self.cap is not None and self.cap.isOpened()
