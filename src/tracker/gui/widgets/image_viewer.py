"""Interactive image viewer widget for point selection and mask display.

This module provides an interactive image display widget that allows users
to click points for mask generation and displays the resulting masks.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QColor, QPen,
                         QBrush, QWheelEvent, QMouseEvent)
from PyQt6.QtWidgets import QLabel, QSizePolicy

from tracker.config.settings import (POINT_RADIUS, FOREGROUND_COLOR, BACKGROUND_COLOR,
                          MASK_OPACITY, ZOOM_FACTOR)

logger = logging.getLogger(__name__)


class ImageViewer(QLabel):
    """Interactive image viewer with point selection capability."""

    # Signals
    point_added = pyqtSignal(int, int, int)  # x, y, label (1=foreground, 0=background)
    points_cleared = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize image viewer.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Set properties
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Ignored size policy + an explicit tiny minimum so the displayed pixmap never
        # drives the label's size request. A QLabel holding a pixmap with
        # scaledContents=False otherwise reports minimumSizeHint == pixmap size; when
        # zoomed in (pixmap ≈ widget width) that pins the right pane's minimum and forces
        # the QSplitter to squeeze the left control pane, while the re-centered pixmap
        # appears to "shift" on each redraw. Ignored still lets the viewer expand to fill
        # its pane. Pair with the resizeEvent below so zoom/pan stay correct on resize.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(1, 1)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; border: 1px solid #555; }")

        # Enable mouse tracking
        self.setMouseTracking(True)

        # State variables
        self._original_image: Optional[np.ndarray] = None
        self._display_pixmap: Optional[QPixmap] = None
        self._mask: Optional[np.ndarray] = None
        self._points: List[Tuple[int, int, int]] = []  # (x, y, label)
        self._zoom_level: float = 1.0
        self._pan_offset: QPoint = QPoint(0, 0)
        self._last_mouse_pos: Optional[QPoint] = None
        self._is_panning: bool = False

        # Display cache: avoids numpy copy + mask blend on every pan/zoom
        self._cached_base_pixmap: Optional[QPixmap] = None
        self._cache_dirty: bool = True

        # Set initial text
        self.setText("No image loaded\nSelect a video to display its first frame")
        self.setStyleSheet(self.styleSheet() + "color: #888;")

    def set_image(self, image: np.ndarray):
        """Set the image to display.

        Args:
            image: Image as numpy array (RGB)
        """
        try:
            self._original_image = image.copy()
            self._zoom_level = self._compute_fit_zoom()
            self._pan_offset = QPoint(0, 0)
            self.clear_points()
            self._mask = None
            self._cache_dirty = True
            self._update_display()

            # Remove text styling
            self.setStyleSheet("QLabel { background-color: #2b2b2b; border: 1px solid #555; }")

            logger.info(f"Set image with shape: {image.shape}")

        except Exception as e:
            logger.error(f"Error setting image: {e}")

    def set_mask(self, mask: np.ndarray):
        """Set the mask overlay; normalize to image size and boolean dtype."""
        if self._original_image is None:
            logger.warning("Cannot set mask without image")
            return

        # Handle None mask (for clearing)
        if mask is None:
            self._mask = None
            self._cache_dirty = True
            self._update_display()
            return

        try:
            m = mask

            # 1) If it's logits/probabilities elsewhere, binarize BEFORE passing here.
            #    Here we just coerce: >0 means foreground.
            if m.ndim == 3:  # e.g., HxWx1 or multiple channels
                m = m[..., 0]
            m = (m > 0)  # boolean

            H, W = self._original_image.shape[:2]
            Hm, Wm = m.shape[:2]

            # 2) Some pipelines output (W, H); if that exactly matches swapped dims, transpose once.
            if (Hm, Wm) == (W, H):
                m = m.T
                Hm, Wm = m.shape[:2]

            # 3) Resize mask to match the image size when needed.
            if (Hm, Wm) != (H, W):
                logger.debug(f"Resizing mask from {(Hm, Wm)} to {(H, W)}")
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

            self._mask = m
            self._cache_dirty = True
            self._update_display()
            logger.info("Mask overlay set")

        except Exception as e:
            logger.error(f"Error setting mask: {e}")

    def add_point(self, x: int, y: int, label: int):
        """Add a point to the display.

        Args:
            x: X coordinate
            y: Y coordinate
            label: Point label (1=foreground, 0=background)
        """
        self._points.append((x, y, label))
        self._update_display()

    def clear_points(self):
        """Clear all points."""
        self._points.clear()
        self._update_display()
        self.points_cleared.emit()

    def get_points(self) -> List[Tuple[int, int, int]]:
        """Get all points.

        Returns:
            List of (x, y, label) tuples
        """
        return self._points.copy()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events."""
        if self._original_image is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Add foreground point
            img_pos = self._widget_to_image_coords(event.pos())
            if img_pos:
                self.add_point(img_pos.x(), img_pos.y(), 1)
                self.point_added.emit(img_pos.x(), img_pos.y(), 1)

        elif event.button() == Qt.MouseButton.RightButton:
            # Add background point
            img_pos = self._widget_to_image_coords(event.pos())
            if img_pos:
                self.add_point(img_pos.x(), img_pos.y(), 0)
                self.point_added.emit(img_pos.x(), img_pos.y(), 0)

        elif event.button() == Qt.MouseButton.MiddleButton:
            # Start panning
            self._is_panning = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events."""
        if self._is_panning and self._last_mouse_pos:
            # Pan the image
            delta = event.pos() - self._last_mouse_pos
            self._pan_offset += delta

            # Limit pan offset to keep image visible
            if self._display_pixmap and self._original_image is not None:
                # Calculate scaled image size
                scaled_w = int(self._original_image.shape[1] * self._zoom_level)
                scaled_h = int(self._original_image.shape[0] * self._zoom_level)

                # Widget size
                widget_w = self.width()
                widget_h = self.height()

                # Calculate pan limits
                # When zoomed in, allow panning to see all edges of the image
                if scaled_w > widget_w:
                    # Can pan left until right edge of image aligns with right edge of widget
                    min_pan_x = widget_w - scaled_w
                    # Can pan right until left edge of image aligns with left edge of widget
                    max_pan_x = 0
                else:
                    # Image smaller than widget - center it
                    center_x = (widget_w - scaled_w) // 2
                    min_pan_x = max_pan_x = center_x

                if scaled_h > widget_h:
                    # Can pan up until bottom edge of image aligns with bottom edge of widget
                    min_pan_y = widget_h - scaled_h
                    # Can pan down until top edge of image aligns with top edge of widget
                    max_pan_y = 0
                else:
                    # Image smaller than widget - center it
                    center_y = (widget_h - scaled_h) // 2
                    min_pan_y = max_pan_y = center_y

                # Apply limits
                self._pan_offset.setX(max(min_pan_x, min(max_pan_x, self._pan_offset.x())))
                self._pan_offset.setY(max(min_pan_y, min(max_pan_y, self._pan_offset.y())))

            self._last_mouse_pos = event.pos()
            self._update_display()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming."""
        if self._original_image is None:
            return

        # Get scroll direction
        delta = event.angleDelta().y()

        # Calculate new zoom level
        if delta > 0:
            new_zoom = self._zoom_level * ZOOM_FACTOR
        else:
            new_zoom = self._zoom_level / ZOOM_FACTOR

        # Limit zoom range: fit-to-widget as floor, 5x as ceiling
        min_zoom = self._compute_fit_zoom()
        new_zoom = max(min_zoom, min(5.0, new_zoom))

        if new_zoom != self._zoom_level:
            # Reset pan if zooming back to fit level
            if new_zoom <= min_zoom:
                self._pan_offset = QPoint(0, 0)

            self._zoom_level = new_zoom
            self._update_display()

    def resizeEvent(self, event):
        """Regenerate the display when the widget is resized.

        Keeps fit-zoom as the floor (so the frame refits when the pane grows) and
        rebuilds the pixmap for the new size. Without this the pixmap is not
        regenerated on splitter/window resize, leaving a stale, mis-centered image.
        """
        super().resizeEvent(event)
        if self._original_image is not None:
            fit = self._compute_fit_zoom()
            if self._zoom_level < fit:
                self._zoom_level = fit
                self._pan_offset = QPoint(0, 0)
            self._update_display()

    def _compute_fit_zoom(self) -> float:
        """Compute zoom level that fits the entire image within the widget."""
        if self._original_image is None:
            return 1.0
        h, w = self._original_image.shape[:2]
        widget_w = max(self.width(), 1)
        widget_h = max(self.height(), 1)
        # Don't upscale small images beyond native resolution
        return min(widget_w / w, widget_h / h, 1.0)

    def _widget_to_image_coords(self, widget_pos: QPoint) -> Optional[QPoint]:
        """Convert widget coordinates to image coordinates.

        Args:
            widget_pos: Position in widget coordinates

        Returns:
            Position in image coordinates or None if outside image
        """
        if not self._display_pixmap:
            return None

        # Get display rect
        pixmap_rect = self._get_pixmap_rect()

        # Check if point is inside pixmap
        if not pixmap_rect.contains(widget_pos):
            return None

        # Convert to pixmap coordinates
        pixmap_x = widget_pos.x() - pixmap_rect.x()
        pixmap_y = widget_pos.y() - pixmap_rect.y()

        # Account for pan offset
        img_x = int((pixmap_x - self._pan_offset.x()) / self._zoom_level)
        img_y = int((pixmap_y - self._pan_offset.y()) / self._zoom_level)

        # Check bounds
        h, w = self._original_image.shape[:2]
        if 0 <= img_x < w and 0 <= img_y < h:
            return QPoint(img_x, img_y)

        return None

    def _get_pixmap_rect(self) -> QRect:
        """Get the rectangle where pixmap is displayed."""
        if not self._display_pixmap:
            return QRect()

        # Calculate position to center pixmap
        widget_size = self.size()
        pixmap_size = self._display_pixmap.size()

        x = (widget_size.width() - pixmap_size.width()) // 2
        y = (widget_size.height() - pixmap_size.height()) // 2

        return QRect(x, y, pixmap_size.width(), pixmap_size.height())

    def _rebuild_base_pixmap(self):
        """Stage A: Rebuild the cached base pixmap from numpy image + mask.

        Only called when image or mask changes (_cache_dirty is True).
        """
        display_img = self._original_image.copy()

        if self._mask is not None:
            mask_overlay = np.zeros_like(display_img)
            mask_overlay[self._mask > 0] = [255, 255, 0]  # Yellow
            alpha = MASK_OPACITY
            display_img = cv2.addWeighted(display_img, 1 - alpha,
                                          mask_overlay, alpha, 0)

        h, w, c = display_img.shape
        bytes_per_line = c * w
        q_image = QImage(display_img.data, w, h, bytes_per_line,
                         QImage.Format.Format_RGB888)
        self._cached_base_pixmap = QPixmap.fromImage(q_image)
        self._cache_dirty = False

    def _update_display(self):
        """Update the displayed image with zoom, pan, and points.

        Uses a two-stage approach:
        - Stage A (only when dirty): numpy blend -> cached QPixmap
        - Stage B (always): scale, pan, draw points (Qt-native ops only)
        """
        if self._original_image is None:
            return

        try:
            # Stage A: rebuild base pixmap only when image/mask changed
            if self._cache_dirty or self._cached_base_pixmap is None:
                self._rebuild_base_pixmap()

            # Stage B: scale, pan, and draw points using cached pixmap
            scaled_size = self._cached_base_pixmap.size() * self._zoom_level
            scaled_pixmap = self._cached_base_pixmap.scaled(
                int(scaled_size.width()), int(scaled_size.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            widget_size = self.size()
            final_width = min(scaled_pixmap.width(), widget_size.width())
            final_height = min(scaled_pixmap.height(), widget_size.height())

            self._display_pixmap = QPixmap(final_width, final_height)
            self._display_pixmap.fill(Qt.GlobalColor.transparent)

            painter = QPainter(self._display_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            source_rect = QRect(
                max(0, -self._pan_offset.x()),
                max(0, -self._pan_offset.y()),
                final_width,
                final_height
            )
            target_rect = QRect(
                max(0, self._pan_offset.x()),
                max(0, self._pan_offset.y()),
                final_width,
                final_height
            )

            painter.drawPixmap(target_rect, scaled_pixmap, source_rect)

            # Draw points
            if self._points:
                for x, y, label in self._points:
                    display_x = int(x * self._zoom_level + self._pan_offset.x())
                    display_y = int(y * self._zoom_level + self._pan_offset.y())

                    if (0 <= display_x <= final_width and
                            0 <= display_y <= final_height):
                        if label == 1:
                            color = QColor(*FOREGROUND_COLOR)
                        else:
                            color = QColor(*BACKGROUND_COLOR)

                        painter.setPen(QPen(color, 2))
                        painter.setBrush(QBrush(color))
                        painter.drawEllipse(QPoint(display_x, display_y),
                                            int(POINT_RADIUS * self._zoom_level),
                                            int(POINT_RADIUS * self._zoom_level))

            painter.end()
            self.setPixmap(self._display_pixmap)

        except Exception as e:
            logger.error(f"Error updating display: {e}")
