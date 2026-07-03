"""Custom exceptions for the tracking application.

This module defines a hierarchy of exceptions for different error scenarios,
enabling better error handling and recovery strategies.
"""


class TrackingError(Exception):
    """Base exception for all tracking errors.

    All custom exceptions in this application inherit from this base class.
    """
    pass


# Video-related errors
class VideoError(TrackingError):
    """Base exception for video-related errors."""
    pass


class VideoReadError(VideoError):
    """Video file cannot be opened or read.

    This may occur when:
    - Video file is corrupted
    - Unsupported video codec
    - File does not exist
    - Insufficient permissions
    """
    pass


class VideoWriteError(VideoError):
    """Video output cannot be written.

    This may occur when:
    - Disk is full
    - Insufficient permissions
    - Invalid output path
    - Codec not available
    """
    pass


class FrameExtractionError(VideoError):
    """Frame extraction from video failed.

    This may occur when:
    - Video is corrupted at specific frame
    - Memory exhausted during extraction
    - Temporary directory not writable
    """
    pass


# Model errors
class ModelError(TrackingError):
    """Base exception for tracking model errors."""
    pass


class ModelNotLoadedError(ModelError):
    """Tracking model has not been loaded.

    Raised when attempting to use the model before calling load_model().
    """
    pass


class ModelLoadError(ModelError):
    """Tracking model failed to load.

    This may occur when:
    - Config file not found
    - Checkpoint file not found
    - Insufficient GPU memory
    - CUDA not available (if required)
    """
    pass


class InferenceError(ModelError):
    """Model inference failed.

    This may occur when:
    - Invalid input dimensions
    - Out of memory during inference
    - Model internal error
    """
    pass


# Annotation errors
class AnnotationError(TrackingError):
    """Base exception for annotation-related errors."""
    pass


class AnnotationNotFoundError(AnnotationError):
    """Annotation file not found for video.

    Raised when attempting to process a video without annotation.
    """
    pass


class InvalidAnnotationError(AnnotationError):
    """Annotation file is malformed or invalid.

    This may occur when:
    - JSON is malformed
    - Required fields are missing
    - Point coordinates are invalid
    """
    pass


# Resource errors
class ResourceError(TrackingError):
    """Base exception for resource-related errors."""
    pass


class InsufficientDiskSpaceError(ResourceError):
    """Insufficient disk space for export.

    This is a recoverable error - user can free up space or select different output location.
    """
    def __init__(self, required_mb: float, available_mb: float):
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            f"Insufficient disk space: {available_mb:.1f} MB available, "
            f"{required_mb:.1f} MB required"
        )


class InsufficientMemoryError(ResourceError):
    """Insufficient memory for operation.

    This may occur when:
    - Video is too large for available RAM
    - GPU memory exhausted
    - System is under memory pressure
    """
    pass


# Export errors
class ExportError(TrackingError):
    """Base exception for export-related errors."""
    pass


class CSVExportError(ExportError):
    """CSV contour export failed."""
    pass


class VideoExportError(ExportError):
    """Video export failed."""
    pass


# Processing errors
class ProcessingError(TrackingError):
    """Base exception for processing errors."""
    pass


class ProcessingCancelledError(ProcessingError):
    """Processing was cancelled by user."""
    pass


class ProcessingTimeoutError(ProcessingError):
    """Processing exceeded timeout limit."""
    pass


# Configuration errors
class ConfigurationError(TrackingError):
    """Invalid configuration detected."""
    pass


def is_recoverable_error(error: Exception) -> bool:
    """Check if an error is potentially recoverable.

    Args:
        error: Exception to check

    Returns:
        True if error might be recoverable with user action
    """
    recoverable_types = (
        InsufficientDiskSpaceError,
        VideoReadError,  # User can provide different video
        AnnotationNotFoundError,  # User can create annotation
        ProcessingCancelledError,  # Not really an error
    )
    return isinstance(error, recoverable_types)


def get_user_friendly_message(error: Exception) -> str:
    """Get a user-friendly error message.

    Args:
        error: Exception to format

    Returns:
        Human-readable error message
    """
    if isinstance(error, InsufficientDiskSpaceError):
        return (
            f"Not enough disk space. Please free up at least "
            f"{error.required_mb:.1f} MB and try again."
        )
    elif isinstance(error, VideoReadError):
        return (
            "Cannot open video file. Please check that the file exists "
            "and is a valid video format."
        )
    elif isinstance(error, ModelNotLoadedError):
        return "Tracking model is not loaded. Please wait for model initialization."
    elif isinstance(error, ModelLoadError):
        return (
            "Failed to load tracking model. Please check that the model files "
            "are in the correct location."
        )
    elif isinstance(error, AnnotationNotFoundError):
        return "No annotation found for this video. Please annotate the video first."
    elif isinstance(error, ProcessingCancelledError):
        return "Processing was cancelled."
    else:
        # Generic message for unknown errors
        return f"An error occurred: {str(error)}"
