"""File browser widget for video selection with status indicators.

This module provides a widget for browsing folders and selecting video files
for processing, with visual status indicators for annotation progress.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLineEdit, QListWidget, QListWidgetItem, QLabel,
                             QFileDialog, QGroupBox)
from PyQt6.QtGui import QColor

from tracker.config.settings import VIDEO_LIST_WIDTH

logger = logging.getLogger(__name__)


class FileBrowser(QWidget):
    """Widget for browsing and selecting video files."""

    # Signals
    folder_changed = pyqtSignal(str)  # Emitted when folder is selected
    video_selected = pyqtSignal(str)  # Emitted when video is selected

    def __init__(self, parent=None):
        """Initialize file browser.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._current_folder: Optional[str] = None
        self._video_files: List[Path] = []
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Folder selection group
        folder_group = QGroupBox("Video Folder")
        folder_layout = QHBoxLayout()

        # Folder path display
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText("No folder selected")
        folder_layout.addWidget(self.folder_edit)

        # Browse button
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._on_browse_clicked)
        self.browse_btn.setMaximumWidth(100)
        folder_layout.addWidget(self.browse_btn)

        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        # Video list group
        list_group = QGroupBox("Video Files")
        list_layout = QVBoxLayout()

        # Video list widget
        self.video_list = QListWidget()
        self.video_list.setMaximumWidth(VIDEO_LIST_WIDTH)
        self.video_list.itemClicked.connect(self._on_video_clicked)
        self.video_list.setAlternatingRowColors(True)
        list_layout.addWidget(self.video_list)

        # Info label
        self.info_label = QLabel("0 videos")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #888;")
        list_layout.addWidget(self.info_label)

        list_group.setLayout(list_layout)
        layout.addWidget(list_group)

        # Set fixed width for the browser
        self.setMaximumWidth(VIDEO_LIST_WIDTH + 50)

    def _on_browse_clicked(self):
        """Handle browse button click."""
        # Get initial directory
        initial_dir = self._current_folder or str(Path.home())

        # Open folder dialog
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Video Folder",
            initial_dir,
            QFileDialog.Option.ShowDirsOnly
        )

        if folder:
            self.set_folder(folder)

    def _on_video_clicked(self, item: QListWidgetItem):
        """Handle video list item click.

        Args:
            item: Clicked list item
        """
        video_path = item.data(Qt.ItemDataRole.UserRole)
        if video_path:
            logger.info(f"Video selected: {Path(video_path).name}")
            self.video_selected.emit(video_path)

    def set_folder(self, folder_path: str):
        """Set the current folder and update video list.

        Args:
            folder_path: Path to folder
        """
        try:
            folder = Path(folder_path)
            if not folder.exists() or not folder.is_dir():
                logger.error(f"Invalid folder: {folder_path}")
                return

            self._current_folder = folder_path
            self.folder_edit.setText(folder_path)

            # Emit signal
            self.folder_changed.emit(folder_path)

            # Update video list (will be called by main window)
            logger.info(f"Folder set: {folder_path}")

        except Exception as e:
            logger.error(f"Error setting folder: {e}")

    def update_video_list(self, video_files: List[Path]):
        """Update the displayed video list.

        Args:
            video_files: List of video file paths
        """
        try:
            self.video_list.clear()
            self._video_files = video_files

            for video_path in video_files:
                # Create list item
                item = QListWidgetItem(video_path.name)
                item.setData(Qt.ItemDataRole.UserRole, str(video_path))

                # Add file size info
                try:
                    size_mb = video_path.stat().st_size / (1024 * 1024)
                    item.setToolTip(f"Size: {size_mb:.1f} MB\nPath: {video_path}")
                except Exception:
                    pass

                self.video_list.addItem(item)

            # Update info label
            count = len(video_files)
            self.info_label.setText(f"{count} video{'s' if count != 1 else ''}")

            logger.info(f"Updated video list with {count} files")

        except Exception as e:
            logger.error(f"Error updating video list: {e}")

    def update_video_list_with_status(self, video_files_with_status: List[Tuple[Path, str]]):
        """Update the displayed video list with status indicators.

        Args:
            video_files_with_status: List of (video_path, status) tuples
                where status is "Complete", "Skip", or empty string
        """
        try:
            # Save current selection
            current_row = self.video_list.currentRow()

            self.video_list.clear()
            self._video_files = [video for video, _ in video_files_with_status]

            for video_path, status in video_files_with_status:
                # Create display text with status
                display_text = video_path.name
                if status:
                    display_text += f" [{status}]"

                # Create list item
                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, str(video_path))

                # Set color based on status
                if status == "Complete":
                    item.setForeground(QColor(0, 128, 0))  # Green
                elif status == "Skip":
                    item.setForeground(QColor(192, 0, 0))  # Red

                # Add file size and status info to tooltip
                try:
                    size_mb = video_path.stat().st_size / (1024 * 1024)
                    tooltip = f"Size: {size_mb:.1f} MB\nPath: {video_path}"
                    if status:
                        tooltip += f"\nStatus: {status}"
                    item.setToolTip(tooltip)
                except Exception:
                    pass

                self.video_list.addItem(item)

            # Restore selection
            if 0 <= current_row < self.video_list.count():
                self.video_list.setCurrentRow(current_row)

            # Update info label with status counts
            total_count = len(video_files_with_status)
            complete_count = sum(1 for _, status in video_files_with_status if status == "Complete")
            skip_count = sum(1 for _, status in video_files_with_status if status == "Skip")

            info_text = f"{total_count} video{'s' if total_count != 1 else ''}"
            if complete_count > 0 or skip_count > 0:
                info_text += f" ({complete_count} complete, {skip_count} skipped)"

            self.info_label.setText(info_text)

            logger.info(f"Updated video list with {total_count} files and status indicators")

        except Exception as e:
            logger.error(f"Error updating video list with status: {e}")

    def get_selected_video(self) -> Optional[str]:
        """Get currently selected video path.

        Returns:
            Path to selected video or None
        """
        current_item = self.video_list.currentItem()
        if current_item:
            return current_item.data(Qt.ItemDataRole.UserRole)
        return None

    def clear(self):
        """Clear the browser."""
        self._current_folder = None
        self._video_files.clear()
        self.folder_edit.clear()
        self.video_list.clear()
        self.info_label.setText("0 videos")
