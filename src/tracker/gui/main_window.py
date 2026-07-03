"""Main application window for Tracker with streaming architecture.

This module provides the main window as a thin UI shell. All business logic
is delegated to AnnotationController and BatchProcessingController.
"""

import logging
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QStatusBar, QMessageBox, QSplitter,
                             QGroupBox, QLineEdit, QFileDialog, QLabel, QFrame,
                             QCheckBox, QProgressBar, QSizePolicy)
from PyQt6.QtGui import QAction

from tracker.gui.widgets.image_viewer import ImageViewer
from tracker.gui.widgets.file_browser import FileBrowser
from tracker.gui.controllers.annotation import AnnotationController
from tracker.gui.controllers.batch import BatchProcessingController
from tracker.tracking.factory import make_tracker
from tracker.pipeline.runner import VideoProcessor
from tracker.pipeline.orchestrator import ResultStatus
from tracker.config.settings import (WINDOW_WIDTH, WINDOW_HEIGHT, MIN_WINDOW_WIDTH,
                          MIN_WINDOW_HEIGHT, APP_NAME, APP_VERSION,
                          MSG_NO_VIDEO_SELECTED, MSG_LOADING_MODEL,
                          MSG_MODEL_LOADED, MSG_MODEL_LOAD_ERROR,
                          MSG_PROCESSING_MASK, MSG_MASK_COMPLETE,
                          MSG_NO_POINTS)

logger = logging.getLogger(__name__)


class ModelLoadThread(QThread):
    """Thread for loading the tracking model."""

    finished = pyqtSignal(bool)

    def __init__(self, sam2_manager):
        super().__init__()
        self.sam2_manager = sam2_manager

    def run(self):
        """Load the model."""
        success = self.sam2_manager.load_model()
        self.finished.emit(success)


class MainWindow(QMainWindow):
    """Main application window with streaming processing.

    Thin UI shell that delegates business logic to controllers:
    - AnnotationController: point annotation, mask generation, video navigation
    - BatchProcessingController: streaming tracking thread lifecycle
    """

    def __init__(self):
        """Initialize main window."""
        super().__init__()

        # Core components
        self.sam2_manager = make_tracker()
        self.video_processor = VideoProcessor()
        self.output_folder: Optional[str] = None

        # Set up UI first (creates widgets needed by controllers)
        self._setup_ui()
        self._setup_menu()

        # Create controllers
        self.annotation_ctrl = AnnotationController(
            sam2_manager=self.sam2_manager,
            video_processor=self.video_processor,
            image_viewer=self.image_viewer,
            on_status=self.status_bar.showMessage,
            on_navigation_changed=self._on_navigation_changed,
            on_error=self._show_error,
        )
        self.annotation_ctrl.on_mask_generated = self._on_mask_generated

        self.batch_ctrl = BatchProcessingController(
            sam2_manager=self.sam2_manager,
            video_processor=self.video_processor,
            on_status=self.status_bar.showMessage,
        )
        self.batch_ctrl.on_video_progress = self._on_streaming_video_progress
        self.batch_ctrl.on_frame_progress = self._on_streaming_frame_progress
        self.batch_ctrl.on_complete = self._on_streaming_complete
        self.batch_ctrl.on_complete_detailed = self._on_streaming_complete_detailed
        self.batch_ctrl.on_error = self._on_streaming_error
        self.batch_ctrl.on_memory_usage = self._on_memory_usage_update

        self._connect_signals()

        # Load the tracking model in background
        self._load_model()

    # ------------------------------------------------------------------ #
    #  UI Setup
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} - Streaming Mode")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - File browser with output folder
        left_widget = QWidget()
        # Maximum horizontal policy: the pane never grows wider than its natural content
        # (≈ the file panels' width), so on maximize all surplus width goes to the image
        # viewer instead of ballooning the controls. No gap, no dragging it wide.
        left_widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.file_browser = FileBrowser()
        left_layout.addWidget(self.file_browser)

        # Output folder selection
        output_group = QGroupBox("Output Folder")
        output_layout = QHBoxLayout()

        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setReadOnly(True)
        self.output_folder_edit.setPlaceholderText("Default: {video_folder}/sam2_project/")
        output_layout.addWidget(self.output_folder_edit)

        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self._on_output_browse_clicked)
        self.output_browse_btn.setMaximumWidth(100)
        output_layout.addWidget(self.output_browse_btn)

        self.output_default_btn = QPushButton("Default")
        self.output_default_btn.clicked.connect(self._on_output_default_clicked)
        self.output_default_btn.setMaximumWidth(60)
        output_layout.addWidget(self.output_default_btn)

        output_group.setLayout(output_layout)
        left_layout.addWidget(output_group)

        # Progress info
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.video_progress_label = QLabel("No videos loaded")
        self.video_progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.video_progress_label)

        self.annotation_progress_label = QLabel("")
        self.annotation_progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.annotation_progress_label)

        progress_group.setLayout(progress_layout)
        left_layout.addWidget(progress_group)

        # Streaming Processing section
        tracking_group = QGroupBox("Streaming Processing")
        tracking_layout = QVBoxLayout()

        self.export_videos_check = QCheckBox("Export tracked videos with overlays")
        self.export_videos_check.setChecked(True)
        tracking_layout.addWidget(self.export_videos_check)

        self.export_csv_check = QCheckBox("Generate contour coordinate CSV files")
        self.export_csv_check.setChecked(True)
        tracking_layout.addWidget(self.export_csv_check)

        self.start_streaming_btn = QPushButton("Start Streamed Analyze & Export")
        self.start_streaming_btn.clicked.connect(self._on_start_streaming_clicked)
        self.start_streaming_btn.setEnabled(False)
        self.start_streaming_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        tracking_layout.addWidget(self.start_streaming_btn)

        self.streaming_video_progress = QLabel("")
        tracking_layout.addWidget(self.streaming_video_progress)

        self.frame_progress_bar = QProgressBar()
        self.frame_progress_bar.setTextVisible(True)
        tracking_layout.addWidget(self.frame_progress_bar)

        self.streaming_frame_progress = QLabel("")
        tracking_layout.addWidget(self.streaming_frame_progress)

        self.memory_usage_label = QLabel("")
        self.memory_usage_label.setStyleSheet("color: #666; font-size: 10pt;")
        tracking_layout.addWidget(self.memory_usage_label)

        # Pause / Resume share a row; "Stop After Current Video" gets its own full-width
        # row below. The old 3-in-a-row layout forced the whole left column to ~490px
        # wide (Stop's label alone wants ~300px), which is why it couldn't shrink to the
        # file-panel width. Stacking lets the column collapse to ~350px cleanly.
        tracking_controls_layout = QHBoxLayout()

        self.pause_streaming_btn = QPushButton("Pause")
        self.pause_streaming_btn.clicked.connect(self._on_pause_streaming_clicked)
        self.pause_streaming_btn.setEnabled(False)
        tracking_controls_layout.addWidget(self.pause_streaming_btn)

        self.resume_streaming_btn = QPushButton("Resume")
        self.resume_streaming_btn.clicked.connect(self._on_resume_streaming_clicked)
        self.resume_streaming_btn.setEnabled(False)
        tracking_controls_layout.addWidget(self.resume_streaming_btn)

        tracking_layout.addLayout(tracking_controls_layout)

        self.stop_after_current_btn = QPushButton("Stop After Current Video")
        self.stop_after_current_btn.clicked.connect(self._on_stop_after_current_clicked)
        self.stop_after_current_btn.setEnabled(False)
        tracking_layout.addWidget(self.stop_after_current_btn)

        self.streaming_status_label = QLabel("")
        self.streaming_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.streaming_status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        tracking_layout.addWidget(self.streaming_status_label)

        tracking_group.setLayout(tracking_layout)
        left_layout.addWidget(tracking_group)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # Right side - Viewer and controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        viewer_group = QGroupBox("Image Viewer - First Frame Annotation")
        viewer_layout = QVBoxLayout()

        self.image_viewer = ImageViewer()
        viewer_layout.addWidget(self.image_viewer)

        button_layout = QHBoxLayout()

        self.generate_mask_btn = QPushButton("Generate Mask")
        self.generate_mask_btn.clicked.connect(self._on_generate_mask)
        self.generate_mask_btn.setEnabled(False)
        button_layout.addWidget(self.generate_mask_btn)

        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self._on_clear_points)
        button_layout.addWidget(self.clear_points_btn)

        self.clear_mask_btn = QPushButton("Clear Mask")
        self.clear_mask_btn.clicked.connect(self._on_clear_mask)
        self.clear_mask_btn.setEnabled(False)
        button_layout.addWidget(self.clear_mask_btn)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        button_layout.addWidget(separator)

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self._on_previous_clicked)
        self.prev_btn.setEnabled(False)
        button_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self._on_next_clicked)
        self.next_btn.setEnabled(False)
        button_layout.addWidget(self.next_btn)

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self._on_skip_clicked)
        self.skip_btn.setEnabled(False)
        button_layout.addWidget(self.skip_btn)

        self.save_mask_btn = QPushButton("Save Mask")
        self.save_mask_btn.clicked.connect(self._on_save_mask_clicked)
        self.save_mask_btn.setEnabled(False)
        button_layout.addWidget(self.save_mask_btn)

        button_layout.addStretch()

        viewer_layout.addLayout(button_layout)
        viewer_group.setLayout(viewer_layout)
        right_layout.addWidget(viewer_group)

        splitter.addWidget(right_widget)
        # Keep the left control column compact (≈ the Video Folder / Video Files width)
        # and give ALL surplus width to the image viewer. Stretch factor 0 vs 1 means the
        # left pane holds its width while the right pane absorbs window growth/maximize —
        # previously both had stretch 0, so on maximize they grew proportionally and the
        # left column ballooned far past the ~350px it needs.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 850])

        main_layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Streaming Mode")

    def _setup_menu(self):
        """Set up the menu bar."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        open_folder_action = QAction("Open Folder...", self)
        open_folder_action.setShortcut("Ctrl+O")
        open_folder_action.triggered.connect(self.file_browser._on_browse_clicked)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        instructions_action = QAction("Instructions", self)
        instructions_action.triggered.connect(self._show_instructions)
        help_menu.addAction(instructions_action)

    def _connect_signals(self):
        """Connect signals and slots."""
        self.file_browser.folder_changed.connect(self._on_folder_changed)
        self.file_browser.video_selected.connect(self._on_video_selected_manual)
        self.image_viewer.point_added.connect(self._on_point_added)
        self.image_viewer.points_cleared.connect(self._on_points_cleared)

    # ------------------------------------------------------------------ #
    #  Model Loading
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """Load the tracking model in background."""
        self.status_bar.showMessage(MSG_LOADING_MODEL)
        self.load_thread = ModelLoadThread(self.sam2_manager)
        self.load_thread.finished.connect(self._on_model_loaded)
        self.load_thread.start()

    def _on_model_loaded(self, success: bool):
        """Handle model load completion."""
        if success:
            self.status_bar.showMessage(MSG_MODEL_LOADED)
            self.generate_mask_btn.setEnabled(True)
        else:
            self.status_bar.showMessage(MSG_MODEL_LOAD_ERROR)
            QMessageBox.warning(
                self,
                "Model Load Error",
                "Failed to load the tracking model.\n\n"
                "Please check that the model files are available and the paths are correct."
            )

    # ------------------------------------------------------------------ #
    #  Folder / Video Selection (delegates to AnnotationController)
    # ------------------------------------------------------------------ #

    def _on_folder_changed(self, folder_path: str):
        """Handle folder change.

        Resets output/annotation context before loading videos so that
        a previously selected output folder does not bleed into the new
        project. The default output folder is always recomputed.
        """
        # Reset output context before loading the new folder
        self.output_folder = None
        self.output_folder_edit.clear()
        self.annotation_ctrl.annotations_folder = None
        self.annotation_ctrl.current_image = None
        self.annotation_ctrl.current_mask = None
        self.image_viewer.clear_points()
        self.image_viewer.set_mask(None)

        # Set up the default output folder for the new source folder
        self._on_output_default_clicked()

        # Now load videos (will also restore statuses from annotations)
        self.annotation_ctrl.load_folder(folder_path)
        self.status_bar.showMessage(
            f"Found {len(self.annotation_ctrl.video_files)} videos in folder"
        )

    def _on_video_selected_manual(self, video_path: str):
        """Handle manual video selection from list."""
        for i, path in enumerate(self.annotation_ctrl.video_files):
            if str(path) == video_path:
                self.annotation_ctrl.load_video_at_index(i)
                break

    # ------------------------------------------------------------------ #
    #  Navigation Changed Callback (called by AnnotationController)
    # ------------------------------------------------------------------ #

    def _on_navigation_changed(self):
        """Update UI when video list, status, or index changes."""
        self._update_video_list_display()
        self._update_navigation_buttons()
        self._update_progress_display()
        # Highlight current video in list
        idx = self.annotation_ctrl.current_video_index
        if 0 <= idx < self.file_browser.video_list.count():
            self.file_browser.video_list.setCurrentRow(idx)

    def _update_video_list_display(self):
        """Update the video list with status indicators."""
        self.file_browser.update_video_list_with_status(
            self.annotation_ctrl.get_display_files()
        )

    def _update_progress_display(self):
        """Update progress labels."""
        ctrl = self.annotation_ctrl
        if not ctrl.video_files:
            self.video_progress_label.setText("No videos loaded")
            self.annotation_progress_label.setText("")
            return

        self.video_progress_label.setText(
            f"Video {ctrl.current_video_index + 1} of {len(ctrl.video_files)}"
        )

        total_processed = ctrl.complete_count + ctrl.skip_count
        self.annotation_progress_label.setText(
            f"First frames annotated: {total_processed}/{len(ctrl.video_files)}"
        )

        if ctrl.complete_count > 0:
            self.start_streaming_btn.setEnabled(True)

    def _update_navigation_buttons(self):
        """Update navigation button states."""
        ctrl = self.annotation_ctrl
        if not ctrl.video_files:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.skip_btn.setEnabled(False)
            self.save_mask_btn.setEnabled(False)
            return

        self.prev_btn.setEnabled(ctrl.can_navigate_prev())
        self.next_btn.setEnabled(ctrl.can_navigate_next())
        self.skip_btn.setEnabled(True)
        self.save_mask_btn.setEnabled(ctrl.has_mask())

    # ------------------------------------------------------------------ #
    #  Annotation Handlers (delegate to AnnotationController)
    # ------------------------------------------------------------------ #

    def _on_point_added(self, x: int, y: int, label: int):
        """Handle point addition."""
        point_type = "foreground" if label == 1 else "background"
        self.status_bar.showMessage(f"Added {point_type} point at ({x}, {y})")

    def _on_points_cleared(self):
        """Handle points cleared."""
        self.status_bar.showMessage("Points cleared")

    def _on_previous_clicked(self):
        """Navigate to previous video."""
        self.annotation_ctrl.navigate_previous()

    def _on_next_clicked(self):
        """Navigate to next video."""
        self.annotation_ctrl.navigate_next()

    def _on_skip_clicked(self):
        """Skip current video and move to next."""
        self.annotation_ctrl.skip_current()
        if not self.annotation_ctrl.advance_to_next_or_complete():
            self._show_completion_message()

    def _on_generate_mask(self):
        """Generate mask from current points."""
        if self.annotation_ctrl.current_image is None:
            QMessageBox.warning(self, "No Image", MSG_NO_VIDEO_SELECTED)
            return

        points = self.image_viewer.get_points()
        if not points:
            QMessageBox.warning(self, "No Points", MSG_NO_POINTS)
            return

        self.status_bar.showMessage(MSG_PROCESSING_MASK)
        self.generate_mask_btn.setEnabled(False)
        self.annotation_ctrl.generate_mask()

    def _on_mask_generated(self, mask):
        """Handle mask generation completion."""
        self.generate_mask_btn.setEnabled(True)

        if mask is not None:
            self.clear_mask_btn.setEnabled(True)
            self.save_mask_btn.setEnabled(True)
            self.status_bar.showMessage(MSG_MASK_COMPLETE)
        else:
            self.status_bar.showMessage("Failed to generate mask")
            QMessageBox.warning(self, "Error", "Failed to generate mask")

    def _on_save_mask_clicked(self):
        """Save current mask and annotation."""
        success = self.annotation_ctrl.save_mask()
        if success:
            if not self.annotation_ctrl.advance_to_next_or_complete():
                self._show_completion_message()
        elif self.annotation_ctrl.has_mask():
            QMessageBox.warning(self, "Save Error", "Failed to save annotation")

    def _on_clear_points(self):
        """Clear all points."""
        self.annotation_ctrl.clear_points()

    def _on_clear_mask(self):
        """Clear the mask overlay."""
        self.annotation_ctrl.clear_mask()
        self.clear_mask_btn.setEnabled(False)
        self.save_mask_btn.setEnabled(False)
        self.status_bar.showMessage("Mask cleared")

    def _show_completion_message(self):
        """Show completion message when all videos are processed."""
        ctrl = self.annotation_ctrl
        QMessageBox.information(
            self,
            "Annotation Complete",
            f"All videos have been processed!\n\n"
            f"Complete: {ctrl.complete_count}\n"
            f"Skipped: {ctrl.skip_count}\n\n"
            f"Annotations saved to:\n{ctrl.annotations_folder}\n\n"
            f"You can now start streaming analysis and export."
        )

    # ------------------------------------------------------------------ #
    #  Output Folder
    # ------------------------------------------------------------------ #

    def _on_output_browse_clicked(self):
        """Handle output folder browse button click."""
        initial_dir = self.output_folder or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", initial_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self._set_output_folder(folder)

    def _on_output_default_clicked(self):
        """Set output folder to default location."""
        if self.file_browser._current_folder:
            default_folder = Path(self.file_browser._current_folder) / "sam2_project"
            self._set_output_folder(str(default_folder))

    def _set_output_folder(self, folder_path: str):
        """Set the output folder and sync to annotation controller."""
        try:
            output_path = Path(folder_path)
            output_path.mkdir(parents=True, exist_ok=True)
            annotations_path = output_path / "annotations"
            annotations_path.mkdir(exist_ok=True)

            self.output_folder = str(output_path)
            self.annotation_ctrl.annotations_folder = annotations_path
            self.output_folder_edit.setText(str(output_path))

            logger.info(f"Output folder set to: {output_path}")
            self.status_bar.showMessage(f"Output folder: {output_path}")

        except PermissionError:
            QMessageBox.warning(
                self, "Permission Error",
                f"Cannot create folder at: {folder_path}\n\n"
                "Please select a different location."
            )
        except Exception as e:
            logger.error(f"Error setting output folder: {e}")
            QMessageBox.warning(self, "Error", f"Failed to set output folder: {str(e)}")

    # ------------------------------------------------------------------ #
    #  Processing Lock (prevents annotation/navigation during batch run)
    # ------------------------------------------------------------------ #

    def _set_processing_lock(self, locked: bool):
        """Enable or disable UI controls that conflict with batch processing.

        When locked=True, annotation, navigation, folder browsing, and output
        changes are all disabled so the shared tracker backend is not accessed
        concurrently from the UI thread and the worker thread.
        """
        interactive = not locked

        # Annotation buttons
        self.generate_mask_btn.setEnabled(interactive and self.sam2_manager.is_loaded)
        self.clear_points_btn.setEnabled(interactive)
        self.clear_mask_btn.setEnabled(interactive and self.annotation_ctrl.has_mask())
        self.save_mask_btn.setEnabled(interactive and self.annotation_ctrl.has_mask())

        # Navigation
        self.prev_btn.setEnabled(interactive and self.annotation_ctrl.can_navigate_prev())
        self.next_btn.setEnabled(interactive and self.annotation_ctrl.can_navigate_next())
        self.skip_btn.setEnabled(interactive and bool(self.annotation_ctrl.video_files))

        # Folder / output browsing
        self.file_browser.setEnabled(interactive)
        self.output_browse_btn.setEnabled(interactive)
        self.output_default_btn.setEnabled(interactive)

        # Image viewer click interaction
        self.image_viewer.setEnabled(interactive)

    # ------------------------------------------------------------------ #
    #  Streaming Processing (delegates to BatchProcessingController)
    # ------------------------------------------------------------------ #

    def _on_start_streaming_clicked(self):
        """Start streaming processing with immediate export."""
        complete_videos = self.annotation_ctrl.get_complete_videos()

        if not complete_videos:
            QMessageBox.warning(
                self, "No Complete Videos",
                "No videos have been marked as complete.\n\n"
                "Please annotate at least one video before starting streaming processing."
            )
            return

        if not self.export_videos_check.isChecked() and not self.export_csv_check.isChecked():
            QMessageBox.warning(
                self, "No Export Options",
                "Please select at least one export option."
            )
            return

        # Check disk space on output drive
        from tracker.io.video_writer import VideoWriterManager
        import tempfile

        total_size_mb = BatchProcessingController.estimate_total_size(
            complete_videos, self.export_csv_check.isChecked()
        )

        if not BatchProcessingController.check_disk_space(
            Path(self.output_folder), int(total_size_mb * 1.5)
        ):
            reply = QMessageBox.warning(
                self, "Low Disk Space (Output)",
                f"Export requires approximately {total_size_mb:.1f} MB on the output drive.\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Check disk space on temp drive (frame extraction goes to system temp)
        # Only the largest single video matters since frames are cleaned up per-video
        max_temp_mb = max(
            VideoWriterManager.estimate_temp_frames_size(str(v))
            for v in complete_videos
        )
        temp_dir = Path(tempfile.gettempdir())
        if not VideoWriterManager.check_disk_space(temp_dir, int(max_temp_mb * 1.5)):
            reply = QMessageBox.warning(
                self, "Low Disk Space (Temp)",
                f"Frame extraction requires approximately {max_temp_mb:.0f} MB "
                f"on the temp drive ({temp_dir.drive or temp_dir}).\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Confirm start
        reply = QMessageBox.question(
            self, "Start Streaming Processing",
            f"Start streaming analysis and export for {len(complete_videos)} videos?\n\n"
            "This will process and export each video frame-by-frame to minimize memory usage.\n"
            "Processing time depends on video length.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Lock annotation/navigation UI and enable streaming controls
        self._set_processing_lock(True)
        self.start_streaming_btn.setEnabled(False)
        self.pause_streaming_btn.setEnabled(True)
        self.stop_after_current_btn.setEnabled(True)

        # Start via controller
        self.batch_ctrl.start(
            complete_videos,
            self.annotation_ctrl.annotations_folder,
            Path(self.output_folder),
            self.export_videos_check.isChecked(),
            self.export_csv_check.isChecked(),
        )
        self.streaming_status_label.setText("Streaming: Active")
        self.streaming_status_label.setStyleSheet(
            "QLabel { background-color: #90EE90; padding: 5px; }"
        )

    def _on_pause_streaming_clicked(self):
        """Pause streaming processing."""
        self.batch_ctrl.pause()
        self.pause_streaming_btn.setEnabled(False)
        self.resume_streaming_btn.setEnabled(True)
        self.streaming_status_label.setText("Streaming: Paused")
        self.streaming_status_label.setStyleSheet(
            "QLabel { background-color: #FFFFE0; padding: 5px; }"
        )

    def _on_resume_streaming_clicked(self):
        """Resume streaming processing."""
        self.batch_ctrl.resume()
        self.pause_streaming_btn.setEnabled(True)
        self.resume_streaming_btn.setEnabled(False)
        self.streaming_status_label.setText("Streaming: Active")
        self.streaming_status_label.setStyleSheet(
            "QLabel { background-color: #90EE90; padding: 5px; }"
        )

    def _on_stop_after_current_clicked(self):
        """Stop after current video completes."""
        self.batch_ctrl.stop_after_current()
        self.stop_after_current_btn.setEnabled(False)
        self.streaming_status_label.setText("Streaming: Stopping...")
        self.streaming_status_label.setStyleSheet(
            "QLabel { background-color: #FFB6C1; padding: 5px; }"
        )

    def _on_streaming_video_progress(self, current: int, total: int):
        """Update video streaming progress."""
        self.streaming_video_progress.setText(
            f"Processing video {current} of {total} (streaming mode)"
        )

    def _on_streaming_frame_progress(self, current: int, total: int, status: str):
        """Update frame streaming progress."""
        if total:
            self.frame_progress_bar.setMaximum(total)
            self.frame_progress_bar.setValue(current)
            self.frame_progress_bar.setFormat(f"Frame {current}/{total} - {status}")
            self.streaming_frame_progress.setText(f"Frame {current}/{total} → {status}")
        else:
            self.streaming_frame_progress.setText(f"Frame {current} → {status}")

    def _on_memory_usage_update(self, gpu_mb: float, ram_mb: float):
        """Update memory usage display."""
        if gpu_mb > 0:
            self.memory_usage_label.setText(f"Memory: GPU {gpu_mb:.0f}MB | RAM {ram_mb:.0f}MB")
        else:
            self.memory_usage_label.setText(f"Memory: RAM {ram_mb:.0f}MB")

    def _on_streaming_complete(self, videos_processed: int):
        """Handle streaming completion (legacy signal, minimal)."""
        # Detailed handler (_on_streaming_complete_detailed) shows the summary.
        # This is kept for backward-compat but only resets UI if the detailed
        # signal was not connected (shouldn't happen in normal flow).
        pass

    def _on_streaming_complete_detailed(self, video_results: list):
        """Handle streaming completion with per-video results.

        Args:
            video_results: list of (video_name, ResultStatus, error_msg)
        """
        self._reset_streaming_ui()
        export_folder = Path(self.output_folder) / "exports"

        succeeded = [r for r in video_results if r[1] == ResultStatus.SUCCESS]
        failed = [r for r in video_results if r[1] == ResultStatus.FAILED]
        cancelled = [r for r in video_results if r[1] == ResultStatus.CANCELLED]

        lines = []
        lines.append(f"Succeeded: {len(succeeded)}")
        if failed:
            lines.append(f"Failed: {len(failed)}")
        if cancelled:
            lines.append(f"Cancelled: {len(cancelled)}")

        if failed:
            lines.append("")
            lines.append("Failed videos:")
            for name, _, err in failed:
                lines.append(f"  - {name}: {err}")

        if cancelled:
            lines.append("")
            lines.append("Cancelled during:")
            for name, _, _ in cancelled:
                lines.append(f"  - {name}")

        lines.append(f"\nResults saved to:\n{export_folder}")

        title = "Streaming Complete"
        if failed or cancelled:
            title = "Streaming Complete (with issues)"
            QMessageBox.warning(self, title, "\n".join(lines))
        else:
            QMessageBox.information(self, title, "\n".join(lines))

    def _on_streaming_error(self, error_msg: str):
        """Handle streaming error."""
        self._reset_streaming_ui()
        QMessageBox.critical(
            self, "Streaming Error",
            f"An error occurred during streaming:\n\n{error_msg}"
        )

    def _reset_streaming_ui(self):
        """Reset streaming UI elements and unlock annotation/navigation."""
        self._set_processing_lock(False)
        self.start_streaming_btn.setEnabled(True)
        self.pause_streaming_btn.setEnabled(False)
        self.resume_streaming_btn.setEnabled(False)
        self.stop_after_current_btn.setEnabled(False)
        self.streaming_video_progress.setText("")
        self.streaming_frame_progress.setText("")
        self.frame_progress_bar.setValue(0)
        self.memory_usage_label.setText("")
        self.streaming_status_label.setText("")
        self.streaming_status_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 5px; }"
        )

    # ------------------------------------------------------------------ #
    #  Dialogs
    # ------------------------------------------------------------------ #

    def _show_error(self, title: str, message: str):
        """Show error dialog (callback for controllers)."""
        QMessageBox.warning(self, title, message)

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"{APP_NAME} v{APP_VERSION}\n\n"
            "Streaming Mode Edition\n\n"
            "A desktop application for object tracking in videos.\n"
            "Uses frame-by-frame streaming to prevent memory overload.\n\n"
            "Built with PyQt6."
        )

    def _show_instructions(self):
        """Show instructions dialog."""
        instructions = """
        <h3>Tracker - Streaming Mode</h3>

        <p><b>Key Features:</b></p>
        <ul>
        <li><b>Streaming Processing:</b> Processes videos frame-by-frame with immediate export</li>
        <li><b>Memory Efficient:</b> Clears memory after each video to prevent overload</li>
        <li><b>Real-time Export:</b> MP4 and CSV files are written as frames are processed</li>
        </ul>

        <p><b>Workflow:</b></p>
        <ol>
        <li><b>Select folder:</b> Browse to folder with video files</li>
        <li><b>Annotate first frames:</b>
            <ul>
            <li>Left-click: Add foreground points (green)</li>
            <li>Right-click: Add background points (red)</li>
            <li>Generate Mask → Save Mask</li>
            </ul>
        </li>
        <li><b>Start streaming:</b> Click "Start Streamed Analyze & Export"</li>
        <li><b>Monitor progress:</b> Watch frame-by-frame processing</li>
        </ol>

        <p><b>Controls:</b></p>
        <ul>
        <li><b>Pause/Resume:</b> Temporarily halt processing</li>
        <li><b>Stop After Current:</b> Complete current video then stop</li>
        <li><b>Memory display:</b> Shows GPU and RAM usage</li>
        </ul>

        <p><b>Output:</b></p>
        <ul>
        <li>MP4 videos with mask overlays</li>
        <li>CSV files with contour coordinates per frame</li>
        <li>All saved to exports folder</li>
        </ul>
        """

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Instructions - Streaming Mode")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(instructions)
        msg_box.exec()

    # ------------------------------------------------------------------ #
    #  Window Close
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        """Handle window close event."""
        self.batch_ctrl.cancel_and_wait()
        self.sam2_manager.cleanup()
        self.video_processor.clear_cache()
        event.accept()
