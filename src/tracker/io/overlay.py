"""Frame overlay utilities for mask visualization.

This module handles visual overlay operations for displaying masks on video frames.
Separated from CSV export and video I/O for single responsibility.
"""

import logging
from typing import Optional, Tuple, List
import numpy as np
import cv2
from tracker.config.settings import DEFAULT_OVERLAY_CONFIG

logger = logging.getLogger(__name__)


class FrameOverlay:
    """Applies mask overlays and contours to video frames.

    Focuses solely on visualization operations.
    Does not perform contour extraction - expects contours to be provided.
    """

    def __init__(self,
                 color: Tuple[int, int, int] = DEFAULT_OVERLAY_CONFIG.color,
                 opacity: float = DEFAULT_OVERLAY_CONFIG.opacity,
                 contour_thickness: int = DEFAULT_OVERLAY_CONFIG.contour_thickness):
        """Initialize frame overlay.

        Args:
            color: BGR color tuple for overlay
            opacity: Overlay opacity (0.0 to 1.0)
            contour_thickness: Thickness of contour lines in pixels
        """
        self.color = color
        self.opacity = opacity
        self.contour_thickness = contour_thickness

    def apply_overlay(self,
                      frame: np.ndarray,
                      mask: Optional[np.ndarray],
                      contours: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Apply mask overlay and contours to a frame.

        Args:
            frame: Input frame (BGR format from OpenCV)
            mask: Binary mask or None
            contours: Pre-extracted contours (optional, will extract if None and mask provided)

        Returns:
            Frame with overlay applied (BGR format)
        """
        try:
            # Make a copy to avoid modifying original
            output_frame = frame.copy()

            # If no mask, return original frame
            if mask is None or not np.any(mask):
                return output_frame

            # Ensure mask is the right size
            height, width = frame.shape[:2]
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height))
                mask = mask.astype(bool)

            # Create colored overlay
            overlay = np.zeros_like(frame)
            overlay[mask] = self.color

            # Blend with original frame
            output_frame = cv2.addWeighted(
                frame,
                1 - self.opacity,
                overlay,
                self.opacity,
                0
            )

            # Draw contours
            if contours is None:
                # Extract contours if not provided
                from tracker.analysis.contours import ContourExtractor
                extractor = ContourExtractor()
                contours = extractor.find_contours(mask.astype(np.uint8))

            if contours:
                cv2.drawContours(
                    output_frame,
                    contours,
                    -1,
                    self.color,
                    self.contour_thickness
                )

            return output_frame

        except Exception as e:
            logger.error(f"Error applying overlay: {e}")
            return frame  # Return original on error

    def apply_overlay_simple(self,
                             frame: np.ndarray,
                             mask: Optional[np.ndarray]) -> np.ndarray:
        """Apply mask overlay without contour extraction (faster).

        Args:
            frame: Input frame (BGR format)
            mask: Binary mask or None

        Returns:
            Frame with colored overlay (no contour lines)
        """
        try:
            output_frame = frame.copy()

            if mask is None or not np.any(mask):
                return output_frame

            # Ensure mask is the right size
            height, width = frame.shape[:2]
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height))
                mask = mask.astype(bool)

            # Create colored overlay
            overlay = np.zeros_like(frame)
            overlay[mask] = self.color

            # Blend
            output_frame = cv2.addWeighted(
                frame,
                1 - self.opacity,
                overlay,
                self.opacity,
                0
            )

            return output_frame

        except Exception as e:
            logger.error(f"Error applying simple overlay: {e}")
            return frame

    def draw_contours_only(self,
                           frame: np.ndarray,
                           contours: List[np.ndarray]) -> np.ndarray:
        """Draw contours on frame without overlay.

        Args:
            frame: Input frame (BGR format)
            contours: List of contour arrays

        Returns:
            Frame with contours drawn
        """
        try:
            output_frame = frame.copy()

            if contours:
                cv2.drawContours(
                    output_frame,
                    contours,
                    -1,
                    self.color,
                    self.contour_thickness
                )

            return output_frame

        except Exception as e:
            logger.error(f"Error drawing contours: {e}")
            return frame
