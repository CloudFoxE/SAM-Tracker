"""SAM2 model management with CPU offloading for memory efficiency.

This module provides an interface to the SAM2 model with CPU offloading
to prevent GPU memory overload when processing multiple videos.
"""

import inspect
import logging
import tempfile
import shutil
import traceback
import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Generator
import numpy as np
import torch
import cv2

try:
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    logging.warning("SAM2 not installed. Please install from https://github.com/facebookresearch/sam2")
    build_sam2_video_predictor = None
    build_sam2 = None
    SAM2ImagePredictor = None

from tracker.pipeline.memory import MemoryManager
from tracker.config.settings import SAM2_CONFIG_PATH, SAM2_CHECKPOINT_PATH, SAM2_COMPUTE_DTYPE
from tracker.tracking.base import ObjectPrompt, FramePrediction
from tracker.tracking.precision import resolve_compute_dtype

logger = logging.getLogger(__name__)


class Sam2Tracker:
    """Manages SAM2 model with CPU offloading for video processing."""

    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """Initialize SAM2 manager.

        Args:
            memory_manager: Shared MemoryManager instance (creates one if None)
        """
        self.model = None
        self.predictor = None
        self.video_predictor = None
        self.device = None
        self.is_loaded = False
        # Compute dtype for autocast; resolved once the device is known (load_model).
        # float32 by default (no autocast); bf16/fp16 wrap inference to save memory.
        self.compute_dtype = torch.float32
        self.current_inference_state = None
        self._temp_frames_dir: Optional[Path] = None
        self.memory_manager = memory_manager or MemoryManager()

    def load_model(self, config_path: Optional[str] = None,
                   checkpoint_path: Optional[str] = None) -> bool:
        """Load SAM2 model for both image and video prediction.

        Args:
            config_path: Path to SAM2 config file (uses default if None)
            checkpoint_path: Path to SAM2 checkpoint (uses default if None)

        Returns:
            True if model loaded successfully, False otherwise
        """
        if build_sam2 is None or build_sam2_video_predictor is None:
            logger.error("SAM2 is not installed")
            return False

        try:
            # Use provided paths or defaults
            config = config_path or SAM2_CONFIG_PATH
            checkpoint = checkpoint_path or SAM2_CHECKPOINT_PATH

            # Check if files exist
            if not Path(config).exists():
                logger.error(f"Config file not found: {config}")
                return False

            if not Path(checkpoint).exists():
                logger.error(f"Checkpoint file not found: {checkpoint}")
                return False

            # Detect device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.compute_dtype = resolve_compute_dtype(self.device, SAM2_COMPUTE_DTYPE)
            logger.info(f"Using device: {self.device} (compute dtype={self.compute_dtype})")

            # Load model for image prediction
            logger.info("Loading SAM2 model...")
            self.model = build_sam2(config, checkpoint, device=self.device)

            # Create image predictor
            self.predictor = SAM2ImagePredictor(self.model)

            # Create video predictor
            logger.info("Loading SAM2 video predictor...")
            self.video_predictor = build_sam2_video_predictor(config, checkpoint, device=self.device)

            self.is_loaded = True
            logger.info("SAM2 model and video predictor loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            self.is_loaded = False
            return False

    def _autocast(self):
        """Autocast context for the configured compute dtype.

        Returns a no-op context for float32 or non-CUDA, so the default path is
        byte-for-byte unchanged; on CUDA with bf16/fp16 it wraps SAM2's forward passes
        in ``torch.autocast`` (the official SAM2 pattern) to cut activation memory.
        """
        if self.device == "cuda" and self.compute_dtype != torch.float32:
            return torch.autocast(device_type="cuda", dtype=self.compute_dtype)
        return nullcontext()

    def generate_mask(self, image: np.ndarray,
                      points: List[Tuple[int, int]],
                      labels: List[int]) -> Optional[np.ndarray]:
        """Generate mask from points on a single image.

        Args:
            image: Input image as numpy array (H, W, C)
            points: List of (x, y) coordinates
            labels: List of point labels (1 for foreground, 0 for background)

        Returns:
            Binary mask as numpy array, None if failed
        """
        if not self.is_loaded:
            logger.error("Model not loaded")
            return None

        if len(points) == 0:
            logger.warning("No points provided")
            return None

        if len(points) != len(labels):
            logger.error("Points and labels must have same length")
            return None

        try:
            with self._autocast():
                # Set image
                self.predictor.set_image(image)

                # Convert points to numpy array
                point_coords = np.array(points, dtype=np.float32)
                point_labels = np.array(labels, dtype=np.int32)

                # Generate mask
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

            # Select best mask (highest score)
            best_idx = np.argmax(scores)
            mask = masks[best_idx]

            logger.info(f"Generated mask with score: {scores[best_idx]:.3f}")
            return mask

        except Exception as e:
            logger.error(f"Failed to generate mask: {e}")
            return None

    def extract_frames_to_temp(self, video_path: str,
                               progress_callback: Optional[callable] = None) -> Optional[Path]:
        """Extract video frames to temporary directory.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(current_frame, total_frames)

        Returns:
            Path to temporary directory with frames, None if failed
        """
        cap = None
        try:
            # Create temporary directory
            temp_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
            self._temp_frames_dir = temp_dir

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                shutil.rmtree(temp_dir)
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0

            logger.info(f"Extracting {total_frames} frames to {temp_dir}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame as JPEG
                frame_path = temp_dir / f"{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)

                # Progress callback
                if progress_callback:
                    progress_callback(frame_idx + 1, total_frames)

                frame_idx += 1

            logger.info(f"Extracted {frame_idx} frames successfully")
            return temp_dir

        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            return None

        finally:
            # Ensure video capture is released
            if cap is not None:
                cap.release()

    def init_video_state(self, frames_dir: Path, video_height: int, video_width: int) -> bool:
        """Initialize video inference state with CPU offloading.

        This method now uses CPU offloading to prevent GPU memory explosion
        by keeping frames in system RAM and only loading to GPU as needed.

        Args:
            frames_dir: Directory containing extracted frames
            video_height: Height of video frames
            video_width: Width of video frames

        Returns:
            True if initialized successfully
        """
        if not self.is_loaded or self.video_predictor is None:
            logger.error("Video predictor not loaded")
            return False

        try:
            # Get sorted frame paths
            frame_paths = sorted(frames_dir.glob("*.jpg"))
            if not frame_paths:
                logger.error("No frames found in directory")
                return False

            # Build init_state kwargs based on what the installed SAM2 version supports
            init_kwargs = {"video_path": str(frames_dir)}

            supported_params = set(inspect.signature(
                self.video_predictor.init_state
            ).parameters.keys())

            # CPU offloading params — critical for keeping frames in RAM instead of GPU VRAM
            offload_params = {
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": True,
                "async_loading": True,
            }
            applied = []
            for param, value in offload_params.items():
                if param in supported_params:
                    init_kwargs[param] = value
                    applied.append(param)

            if applied:
                logger.info(f"SAM2 init_state supports: {', '.join(applied)}")
            else:
                logger.warning(
                    "SAM2 version does not support CPU offloading — "
                    "this may consume large amounts of GPU memory"
                )

            with self._autocast():
                self.current_inference_state = self.video_predictor.init_state(**init_kwargs)
            logger.info("Video inference state initialized")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize video state: {e}")
            return False

    def add_annotation_to_video(self, points: List[Tuple[int, int]],
                                labels: List[int],
                                frame_idx: int = 0,
                                obj_id: int = 1) -> bool:
        """Add annotation points to video tracking.

        Args:
            points: List of (x, y) coordinates
            labels: List of point labels (1 for foreground, 0 for background)
            frame_idx: Frame index to add annotation (default: 0)
            obj_id: Object id to associate with these points (default: 1)

        Returns:
            True if added successfully
        """
        if self.current_inference_state is None:
            logger.error("No video inference state initialized")
            return False

        try:
            # Convert to numpy arrays
            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)

            # Detect whether add_new_points uses keyword or positional args
            sig = inspect.signature(self.video_predictor.add_new_points)
            with self._autocast():
                if "inference_state" in sig.parameters:
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points(
                        inference_state=self.current_inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=point_coords,
                        labels=point_labels,
                    )
                else:
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points(
                        self.current_inference_state,
                        frame_idx,
                        obj_id,
                        point_coords,
                        point_labels,
                    )

            logger.info(f"Added {len(points)} annotation points to frame {frame_idx}")
            logger.debug(f"Output object IDs: {out_obj_ids}")
            return True

        except Exception as e:
            logger.error(f"Failed to add annotation to video: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def _tensor_to_2d_mask(tensor) -> np.ndarray:
        """Convert a mask tensor/array of any dimensionality to a 2D boolean mask."""
        mask = (tensor > 0).cpu().numpy() if hasattr(tensor, 'cpu') else (np.asarray(tensor) > 0)
        # Collapse leading dimensions: (B, C, H, W) → (H, W)
        while mask.ndim > 2:
            mask = mask[0]
        return mask

    def _extract_mask(self, video_res, frame_idx: int) -> Optional[np.ndarray]:
        """Extract a 2D boolean mask from SAM2 propagation output.

        SAM2 returns either:
        - A dict keyed by obj_id (standard API)
        - A raw tensor (some versions)

        Args:
            video_res: Raw output from SAM2 propagate_in_video
            frame_idx: Current frame index (for logging)

        Returns:
            2D boolean numpy array, or None if extraction failed
        """
        try:
            if isinstance(video_res, dict):
                # Standard SAM2 format: {obj_id: mask_tensor}
                mask_data = video_res.get(1)  # obj_id=1
                if mask_data is None and video_res:
                    # Grab first available object if obj_id 1 not found
                    mask_data = next(iter(video_res.values()))
                if mask_data is not None:
                    return self._tensor_to_2d_mask(mask_data)
            elif hasattr(video_res, 'cpu') or isinstance(video_res, np.ndarray):
                # Raw tensor or numpy array
                return self._tensor_to_2d_mask(video_res)
        except Exception as e:
            logger.warning(f"Mask extraction failed for frame {frame_idx}: {e}")

        logger.warning(f"Could not extract mask for frame {frame_idx}")
        return None

    def propagate_in_video(self, progress_callback: Optional[callable] = None) -> Generator[Dict, None, None]:
        """Propagate masks through video frames.

        Args:
            progress_callback: Optional callback(current_frame, total_frames)

        Yields:
            Dictionary with frame_idx and mask data
        """
        if self.current_inference_state is None:
            logger.error("No video inference state initialized")
            return

        try:
            # Propagate through video (autocast wraps the per-frame forward passes;
            # no-op under the default float32).
            with self._autocast():
                for frame_idx, obj_ids, video_res in self.video_predictor.propagate_in_video(
                        self.current_inference_state
                ):
                    # Extract mask from video_res (dict keyed by obj_id, or raw tensor)
                    mask = self._extract_mask(video_res, frame_idx)

                    # Progress callback
                    if progress_callback:
                        progress_callback(frame_idx + 1, None)

                    yield {
                        'frame_idx': frame_idx,
                        'mask': mask,
                        'obj_ids': obj_ids
                    }

        except Exception as e:
            logger.error(f"Error during video propagation: {e}")
            traceback.print_exc()

    def reset_video_state(self):
        """Reset video inference state and clean up temporary files.

        Uses MemoryManager for cleanup operations.
        """
        try:
            # Reset inference state
            if self.video_predictor and self.current_inference_state:
                try:
                    self.video_predictor.reset_state(self.current_inference_state)
                except Exception as e:
                    logger.warning(f"Error resetting video state: {e}")

            # Clear the reference
            self.current_inference_state = None

            # Use memory manager for cleanup (including temp frames deletion)
            self.memory_manager.cleanup_after_video(self._temp_frames_dir)
            self._temp_frames_dir = None

        except Exception as e:
            logger.error(f"Error resetting video state: {e}")

    def cleanup(self):
        """Clean up all resources with enhanced memory management."""
        try:
            # Reset video state
            self.reset_video_state()

            # Clear image predictor state safely
            if self.predictor is not None:
                try:
                    # Try various methods to clear predictor state
                    if hasattr(self.predictor, 'reset'):
                        self.predictor.reset()
                    elif hasattr(self.predictor, 'features'):
                        self.predictor.features = None
                    elif hasattr(self.predictor, 'is_image_set'):
                        self.predictor.is_image_set = False

                    # Clear any cached data
                    if hasattr(self.predictor, 'original_size'):
                        self.predictor.original_size = None
                    if hasattr(self.predictor, 'input_size'):
                        self.predictor.input_size = None

                except Exception as e:
                    logger.warning(f"Could not fully reset predictor: {e}")

            # Use memory manager for comprehensive cleanup
            self.memory_manager.full_cleanup(reset_stats=True)

            logger.info("SAM2 manager cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    # ------------------------------------------------------------------
    # TrackerBackend interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        if not self.load_model():
            from tracker.errors import ModelLoadError
            raise ModelLoadError("Failed to load SAM2 model")

    def predict_image_masks(self, image, prompts):
        out = {}
        for p in prompts:                                   # Phase 0: one prompt
            m = self.generate_mask(image, p.points, p.labels)
            if m is not None:
                out[p.obj_id] = m
        return out

    def init_video(self, video_path, progress_callback=None) -> bool:
        temp_dir = self.extract_frames_to_temp(video_path, progress_callback)
        if not temp_dir:
            return False
        # height/width are read by init_video_state from the frames themselves in the port;
        # if the upstream signature needs them, read from the first extracted frame here.
        first = sorted(temp_dir.glob("*.jpg"))[0]
        img = cv2.imread(str(first))
        h, w = img.shape[:2]
        return self.init_video_state(temp_dir, h, w)

    def add_prompts(self, frame_idx, prompts) -> bool:
        ok = True
        for p in prompts:                                   # obj_id now parameterized (see Step 2)
            ok = self.add_annotation_to_video(p.points, p.labels, frame_idx, obj_id=p.obj_id) and ok
        return ok

    def propagate(self, progress_callback=None):
        for item in self.propagate_in_video(progress_callback):
            mask = item.get('mask')
            masks = {1: mask} if mask is not None else {}
            yield FramePrediction(frame_idx=item['frame_idx'], masks=masks)

    def reset(self) -> None:
        self.reset_video_state()
