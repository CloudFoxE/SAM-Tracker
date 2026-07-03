"""Contour processing utilities for mask analysis.

This module provides clean separation of contour-related operations:
- Finding contours in binary masks
- Simplifying contours using various algorithms
- Selecting contours based on criteria
- Validating contour quality

All classes are designed as pure functions with no I/O side effects.
"""

import logging
from typing import List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ContourExtractor:
    """Extracts contours from binary masks.

    Provides a single source of truth for contour finding operations,
    eliminating duplicate cv2.findContours calls across the codebase.

    Always extracts full detailed contours with no approximation.
    """

    def __init__(self, mode: int = cv2.RETR_EXTERNAL):
        """Initialize contour extractor.

        Args:
            mode: Contour retrieval mode (cv2.RETR_*)
        """
        self.mode = mode

    def find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find contours in a binary mask.

        Args:
            mask: Binary mask array (H, W) with values 0 or 255 (uint8),
                  or boolean array

        Returns:
            List of contours as numpy arrays. Empty list if no contours found.
            Each contour has shape (N, 1, 2) following OpenCV convention.
        """
        if mask is None:
            logger.debug("Mask is None, returning empty contour list")
            return []

        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            mask_uint8 = mask.astype(np.uint8)
            if mask_uint8.max() == 1:
                mask_uint8 = mask_uint8 * 255
        else:
            mask_uint8 = mask.copy()

        # Check if mask has any non-zero values
        if not np.any(mask_uint8):
            logger.debug("Mask is empty (all zeros), returning empty contour list")
            return []

        try:
            # Find contours - always use CHAIN_APPROX_NONE for full detailed contours
            contours, _ = cv2.findContours(
                mask_uint8,
                self.mode,
                cv2.CHAIN_APPROX_NONE  # No approximation - all contour points
            )

            logger.debug(f"Found {len(contours)} contours with full detail")
            return list(contours)

        except Exception as e:
            logger.error(f"Error finding contours: {e}")
            return []


# ContourSimplifier class removed - simplification should be done in data processing, not here


class ContourSelector:
    """Selects contours from a list based on various criteria."""

    @staticmethod
    def select_largest(contours: List[np.ndarray]) -> Optional[np.ndarray]:
        """Select the largest contour by area.

        Args:
            contours: List of contour arrays

        Returns:
            Largest contour or None if list is empty
        """
        if not contours:
            logger.debug("No contours to select from")
            return None

        try:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            logger.debug(f"Selected largest contour with area {area:.1f} pixels")
            return largest

        except Exception as e:
            logger.error(f"Error selecting largest contour: {e}")
            return None

    @staticmethod
    def select_by_area_threshold(contours: List[np.ndarray],
                                 min_area: float = 10.0) -> List[np.ndarray]:
        """Select all contours above a minimum area threshold.

        Args:
            contours: List of contour arrays
            min_area: Minimum area in pixels

        Returns:
            List of contours meeting the area threshold
        """
        if not contours:
            return []

        try:
            filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
            logger.debug(f"Filtered {len(contours)} contours to {len(filtered)} above {min_area} pixels")
            return filtered

        except Exception as e:
            logger.error(f"Error filtering contours by area: {e}")
            return contours
