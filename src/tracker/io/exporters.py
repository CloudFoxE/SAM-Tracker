"""Contour CSV export utilities.

This module handles exporting contour coordinates to CSV files.
Separated from visualization and video I/O for single responsibility.
"""

import logging
import csv
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class ContourExporter:
    """Exports contour data to CSV files.

    Focuses solely on CSV writing operations.
    Expects contours to already be extracted and processed.
    """

    @staticmethod
    def export_to_csv(contour: Optional[np.ndarray], output_path: str) -> bool:
        """Export a single contour to CSV file.

        Args:
            contour: Contour array (N, 1, 2) or None
            output_path: Path for output CSV file

        Returns:
            True if export successful
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # If no contour, create empty file
            if contour is None or len(contour) == 0:
                Path(output_path).touch()
                logger.debug(f"Created empty CSV (no contour): {output_path}")
                return True

            # Write contour points to CSV
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write coordinates (no header)
                for point in contour:
                    x, y = point[0]
                    writer.writerow([x, y])

            logger.debug(f"Exported {len(contour)} contour points to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export contour CSV to {output_path}: {e}")
            return False

    @staticmethod
    def export_mask_to_csv(mask: np.ndarray,
                           output_path: str,
                           contour_extractor,
                           contour_selector) -> bool:
        """Export contour from mask to CSV (convenience method).

        Args:
            mask: Binary mask array
            output_path: Path for output CSV file
            contour_extractor: ContourExtractor instance
            contour_selector: ContourSelector instance

        Returns:
            True if export successful
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Check if mask is None or empty
            if mask is None or not np.any(mask):
                Path(output_path).touch()
                return True

            # Ensure mask is uint8
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
                if mask.max() == 1:
                    mask = mask * 255

            # Extract contours
            contours = contour_extractor.find_contours(mask)

            # If no contours, create empty file
            if not contours:
                Path(output_path).touch()
                return True

            # Get largest contour
            largest_contour = contour_selector.select_largest(contours)

            if largest_contour is None:
                Path(output_path).touch()
                return True

            # Export to CSV
            return ContourExporter.export_to_csv(largest_contour, output_path)

        except Exception as e:
            logger.error(f"Failed to export mask contour CSV: {e}")
            return False
