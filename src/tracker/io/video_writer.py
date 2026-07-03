"""Video writer management utilities.

This module handles video file output operations.
Separated from overlay and export for single responsibility.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import cv2

logger = logging.getLogger(__name__)


class VideoWriterManager:
    """Manages video writer lifecycle and operations.

    Provides a clean interface for creating and managing video output.
    """

    @staticmethod
    def create_export_directories(output_folder: Path) -> Dict[str, Path]:
        """Create export directory structure.

        Args:
            output_folder: Base output folder path

        Returns:
            Dictionary with paths to created directories
        """
        exports_dir = output_folder / "exports"
        tracked_videos_dir = exports_dir / "tracked_videos"
        contour_data_dir = exports_dir / "contour_data"

        tracked_videos_dir.mkdir(parents=True, exist_ok=True)
        contour_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created export directories in {exports_dir}")

        return {
            "exports": exports_dir,
            "tracked_videos": tracked_videos_dir,
            "contour_data": contour_data_dir,
        }

    @staticmethod
    def create_writer(output_path: str,
                      fps: float,
                      width: int,
                      height: int,
                      codec: str = 'mp4v') -> Optional[cv2.VideoWriter]:
        """Create a video writer for streaming output.

        Args:
            output_path: Path for output video file
            fps: Frames per second
            width: Video width
            height: Video height
            codec: FourCC codec code

        Returns:
            VideoWriter object or None if failed
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                logger.error(f"Failed to open video writer for {output_path}")
                return None

            logger.info(f"Created video writer: {output_path} ({width}x{height} @ {fps} fps)")
            return writer

        except Exception as e:
            logger.error(f"Error creating video writer: {e}")
            return None

    @staticmethod
    def release_writer(writer: Optional[cv2.VideoWriter]) -> bool:
        """Safely release a video writer.

        Args:
            writer: VideoWriter object or None

        Returns:
            True if released successfully
        """
        try:
            if writer is not None and writer.isOpened():
                writer.release()
                logger.debug("Video writer released")
                return True
            return False

        except Exception as e:
            logger.error(f"Error releasing video writer: {e}")
            return False

    @staticmethod
    def check_disk_space(output_path: Path, required_mb: int = 1000) -> bool:
        """Check if sufficient disk space is available.

        Args:
            output_path: Path to check disk space
            required_mb: Required space in megabytes

        Returns:
            True if sufficient space available
        """
        try:
            import shutil
            stat = shutil.disk_usage(output_path)
            available_mb = stat.free / (1024 * 1024)

            if available_mb < required_mb:
                logger.warning(
                    f"Low disk space: {available_mb:.1f} MB available, "
                    f"{required_mb} MB required"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
            return True  # Assume space is available if check fails

    @staticmethod
    def estimate_export_size(video_path: str, include_csv: bool = True) -> float:
        """Estimate export size in MB (output folder only, excludes temp frames).

        Args:
            video_path: Path to video file
            include_csv: Whether to include CSV export in estimate

        Returns:
            Estimated size in MB
        """
        try:
            # Get video file size
            video_size_mb = Path(video_path).stat().st_size / (1024 * 1024)

            # Estimate tracked video size (usually similar to original)
            tracked_video_size = video_size_mb * 1.1

            # Estimate CSV size
            csv_size = 0
            if include_csv:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # Assume ~1KB per frame CSV
                    csv_size = frame_count * 0.001
                    cap.release()

            return tracked_video_size + csv_size

        except Exception as e:
            logger.error(f"Failed to estimate export size: {e}")
            return 100.0  # Default estimate

    @staticmethod
    def estimate_temp_frames_size(video_path: str) -> float:
        """Estimate disk space needed for extracted JPEG frames in temp dir.

        SAM2 requires all frames extracted to disk as JPEGs before processing.
        A reasonable estimate is ~50-150 KB per frame depending on resolution.

        Args:
            video_path: Path to video file

        Returns:
            Estimated temp size in MB
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 100.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # JPEG size heuristic: ~0.1 bytes per pixel (quality ~95)
            bytes_per_frame = width * height * 0.1
            total_bytes = bytes_per_frame * frame_count
            return total_bytes / (1024 * 1024)
        except Exception as e:
            logger.error(f"Failed to estimate temp frames size: {e}")
            return 100.0
