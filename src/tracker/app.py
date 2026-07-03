#!/usr/bin/env python3
"""Tracker Video Application

A PyQt6 desktop application for interactive object tracking in videos.

Usage:
    python -m tracker.app
    python -m tracker.app --debug  # For debug logging

Configuration:
    - All contours are exported with full detail (no simplification)
    - Simplification can be done in downstream data processing if needed
    - See tracker/config/settings.py for configuration options
"""

import torch  # noqa: F401  # must precede PyQt6 to avoid Windows WinError 1114 DLL conflict

import sys
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from tracker.gui.main_window import MainWindow
from tracker.config.settings import APP_NAME, LOG_FORMAT, LOG_DATE_FORMAT


def setup_logging():
    """Configure logging for the application."""
    # Set debug level for troubleshooting
    log_level = logging.DEBUG if "--debug" in sys.argv else logging.INFO

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            # Optionally add file handler
            # logging.FileHandler('tracker.log')
        ]
    )

    # Set specific logger levels
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # Enable debug for our modules when debugging
    if log_level == logging.DEBUG:
        logging.getLogger('tracker.tracking.sam2').setLevel(logging.DEBUG)


def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Starting {APP_NAME}")

    # Create Qt application
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("SAM2VideoTracker")

    # Set application style (optional)
    app.setStyle("Fusion")  # Modern cross-platform style

    try:
        # Create and show main window
        window = MainWindow()
        window.show()

        logger.info("Application window created")

        # Run application
        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
