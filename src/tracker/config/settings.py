"""Configuration constants for SAM2 Tracking Application.

This module contains all configuration parameters used throughout the application,
including paths, model settings, UI constants, and batch annotation settings.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import cv2
import os

# Application Info
APP_NAME = "Tracker"
APP_VERSION = "3.0.0"

# Window Settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600

# File Settings
SUPPORTED_VIDEO_EXTENSIONS = [".avi", ".mp4", ".mov", ".mkv"]
VIDEO_LIST_WIDTH = 300  # Increased to accommodate status text

# Image Viewer Settings
POINT_RADIUS = 5
FOREGROUND_COLOR = (0, 255, 0)  # Green
BACKGROUND_COLOR = (255, 0, 0)   # Red
MASK_OPACITY = 0.5
ZOOM_FACTOR = 1.1

# SAM2 Model Settings (temporary A/B backend)
# Paths resolve dynamically so they work on any machine (override via env vars):
#   - checkpoint: <repo>/checkpoints/sam2.1_hiera_large.pt (Meta public download)
#   - config: the sam2.1_hiera_l.yaml shipped inside the installed `sam2` package
#     (this SAM2 build accepts an absolute config path, which also satisfies the
#     load-time existence check; falls back to the Hydra config name if sam2 is absent)
import importlib.util as _ilu

_repo_root = Path(__file__).resolve().parents[3]  # src/tracker/config/settings.py -> repo root
_default_checkpoint = str(_repo_root / "checkpoints" / "sam2.1_hiera_large.pt")


def _resolve_sam2_config() -> str:
    spec = _ilu.find_spec("sam2")
    if spec and spec.submodule_search_locations:
        cand = Path(list(spec.submodule_search_locations)[0]) / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
        if cand.exists():
            return str(cand)
    return "configs/sam2.1/sam2.1_hiera_l.yaml"  # Hydra-name fallback


_default_config = _resolve_sam2_config()

SAM2_CONFIG_PATH = os.environ.get('SAM2_CONFIG_PATH', _default_config)
SAM2_CHECKPOINT_PATH = os.environ.get('SAM2_CHECKPOINT_PATH', _default_checkpoint)

_config_logger = logging.getLogger(__name__)
if not Path(SAM2_CONFIG_PATH).exists():
    _config_logger.debug(f"SAM2 config not found: {SAM2_CONFIG_PATH}")
if not Path(SAM2_CHECKPOINT_PATH).exists():
    _config_logger.debug(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT_PATH}")

# Default model parameters
DEFAULT_SAM2_PARAMS: Dict[str, Any] = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "crop_n_layers": 0,
    "crop_n_points_downscale_factor": 1,
}

# SAM2 compute precision. Default float32 (validated). Set bfloat16/float16 to run the
# image + video predictors under torch.autocast — cuts activation memory for very long
# sessions. Only applied on CUDA; CPU always uses float32. (SAM2 also keeps its internal
# memory-bank features in bf16 regardless — that's baked into the predictor.)
SAM2_COMPUTE_DTYPE = os.environ.get("SAM2_COMPUTE_DTYPE", "float32")  # float32 | bfloat16 | float16

# SAM3 model settings (used from Phase 1 onward; HF gated weights)
TRACKER_MODEL_ID = os.environ.get("TRACKER_MODEL_ID", "facebook/sam3")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
TRACKER_BACKEND = os.environ.get("TRACKER_BACKEND", "sam3")  # "sam3" (default) | "sam2" (validated)

# SAM3 compute precision for the VIDEO tracking path (model weights + inference
# session). Default float32 — validated, and VRAM stays bounded because frames are
# CPU-offloaded and streamed to the GPU one at a time. Switch to bfloat16 (or float16)
# to roughly halve per-frame VRAM and the CPU-stored preprocessed clip for very long
# sessions. Only applied on CUDA; CPU always uses float32 (bf16 CPU kernels are slow).
# Model dtype and session dtype must match — Sam3Tracker keeps them in lockstep.
SAM3_COMPUTE_DTYPE = os.environ.get("SAM3_COMPUTE_DTYPE", "float32")  # float32 | bfloat16 | float16

# Video Processing Settings
MAX_FRAME_DISPLAY_SIZE: Tuple[int, int] = (800, 600)
FRAME_CACHE_SIZE = 10  # Number of frames to cache

# Batch Annotation Settings
DEFAULT_OUTPUT_FOLDER_NAME = "tracker_project"
ANNOTATIONS_SUBFOLDER_NAME = "annotations"
ANNOTATION_FILE_EXTENSION = ".json"
MASK_FILE_EXTENSION = ".png"

# Export Settings
STATUS_COLOR_COMPLETE = (0, 128, 0)  # Green
STATUS_COLOR_SKIP = (192, 0, 0)      # Red
STATUS_COLOR_DEFAULT = (0, 0, 0)     # Black

# Contour Export Settings
# Note: All contours are now exported with full detail (CHAIN_APPROX_NONE)
# Simplification should be done in downstream data processing if needed

# UI Messages
MSG_NO_VIDEO_SELECTED = "Please select a video from the list"
MSG_LOADING_MODEL = "Loading model..."
MSG_MODEL_LOADED = "Model loaded successfully"
MSG_MODEL_LOAD_ERROR = "Failed to load model"
MSG_PROCESSING_MASK = "Generating mask..."
MSG_MASK_COMPLETE = "Mask generated successfully"
MSG_NO_POINTS = "Please click on the image to add points"
MSG_ANNOTATION_SAVED = "Annotation saved successfully"
MSG_ANNOTATION_LOADED = "Loaded existing annotation"
MSG_ALL_VIDEOS_COMPLETE = "All videos have been processed!"

# Logging Settings
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# Configuration Dataclasses
# ============================================================================
# These provide structured, type-safe configuration objects to replace
# scattered magic numbers and improve code clarity.


@dataclass(frozen=True)
class ContourConfig:
    """Configuration for contour extraction and processing.

    Attributes:
        min_area_threshold: Minimum contour area in pixels
        retrieval_mode: OpenCV contour retrieval mode

    Note: All contours are extracted with full detail (CHAIN_APPROX_NONE).
    """
    min_area_threshold: int = 10
    retrieval_mode: int = cv2.RETR_EXTERNAL


@dataclass(frozen=True)
class OverlayConfig:
    """Configuration for mask overlay visualization.

    Attributes:
        color: BGR color tuple for mask overlay
        opacity: Overlay opacity/transparency (0.0 to 1.0)
        contour_thickness: Thickness of contour lines in pixels
    """
    color: Tuple[int, int, int] = (0, 255, 0)  # Green
    opacity: float = 0.3
    contour_thickness: int = 2


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for memory management during video processing.

    Attributes:
        cleanup_interval_frames: Run periodic cleanup every N frames
        gc_passes: Number of garbage collection passes per cleanup
        enable_gpu_cleanup: Whether to clear GPU cache during cleanup
    """
    cleanup_interval_frames: int = 500
    gc_passes: int = 2
    enable_gpu_cleanup: bool = True


@dataclass(frozen=True)
class VideoProcessingConfig:
    """Configuration for video processing parameters.

    Attributes:
        status_update_interval: Update progress every N frames
        frame_read_retry_count: Number of retries for failed frame reads
        temp_frame_format: Image format for temporary frames (jpg/png)
    """
    status_update_interval: int = 10
    frame_read_retry_count: int = 3
    temp_frame_format: str = "jpg"


# Default configuration instances
DEFAULT_CONTOUR_CONFIG = ContourConfig()
DEFAULT_OVERLAY_CONFIG = OverlayConfig()
DEFAULT_MEMORY_CONFIG = MemoryConfig()
DEFAULT_VIDEO_PROCESSING_CONFIG = VideoProcessingConfig()
