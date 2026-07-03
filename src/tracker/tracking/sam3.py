"""SAM3 tracker backend (Hugging Face transformers Sam3TrackerVideo* / Sam3Tracker*).

API pinned by .superpowers/sdd/phase1-sam3-api-spike.md against transformers 5.12.1.
Coordinates are pixel (points xy, boxes xyxy) — identical to ObjectPrompt, no conversion.
"""
from __future__ import annotations
import logging
from typing import Iterator, Optional, Callable
import numpy as np
import cv2
import torch

from tracker.tracking.base import ObjectPrompt, FramePrediction
from tracker.tracking.precision import resolve_compute_dtype
from tracker.config.settings import TRACKER_MODEL_ID, SAM3_COMPUTE_DTYPE
from tracker.errors import ModelLoadError

logger = logging.getLogger(__name__)

try:
    from transformers import (
        Sam3TrackerVideoModel, Sam3TrackerVideoProcessor,
        Sam3TrackerModel, Sam3TrackerProcessor,
    )
except ImportError:  # transformers without SAM3
    Sam3TrackerVideoModel = Sam3TrackerVideoProcessor = None
    Sam3TrackerModel = Sam3TrackerProcessor = None


def _decode_video_rgb(video_path: str) -> list[np.ndarray]:
    """Decode a video file into a list of RGB uint8 HWC frames (SAM3 wants decoded RGB frames)."""
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def _masks_from_output(object_ids, processed_masks) -> dict[int, np.ndarray]:
    """Map a post-processed (num_objects,1,H,W) bool tensor to {obj_id: 2D bool ndarray}.

    object_ids[i] corresponds to processed_masks[i] (insertion order, per the spike).
    """
    out: dict[int, np.ndarray] = {}
    for i, obj_id in enumerate(object_ids):
        out[int(obj_id)] = processed_masks[i, 0].cpu().numpy().astype(bool)
    return out


class Sam3Tracker:
    """TrackerBackend implementation backed by HF SAM3 video + image tracker models."""

    def __init__(self, model_id: str = TRACKER_MODEL_ID, device: str = "cuda",
                 memory_manager=None):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        # Video-path compute dtype (float32 by default; SAM3_COMPUTE_DTYPE-configurable).
        # Kept identical between the model weights (load) and the inference session
        # (init_video); a mismatch raises a conv dtype error in the model's image encoder.
        self.compute_dtype = resolve_compute_dtype(self.device, SAM3_COMPUTE_DTYPE)
        self.model = None
        self.processor = None
        self.image_model = None
        self.image_processor = None
        self.session = None
        self._num_frames = 0
        self._prompt_frame_idx: Optional[int] = None
        self.memory_manager = memory_manager

    def load(self) -> None:
        if Sam3TrackerVideoModel is None:
            raise ModelLoadError("transformers build lacks SAM3 (Sam3TrackerVideoModel)")
        try:
            self.model = Sam3TrackerVideoModel.from_pretrained(
                self.model_id, dtype=self.compute_dtype).to(self.device).eval()
            self.processor = Sam3TrackerVideoProcessor.from_pretrained(self.model_id)
            # Frame-0 preview stays float32 (single-image, cheap, max fidelity).
            self.image_model = Sam3TrackerModel.from_pretrained(self.model_id).to(self.device).eval()
            self.image_processor = Sam3TrackerProcessor.from_pretrained(self.model_id)
            logger.info("Loaded SAM3 tracker '%s' on %s (video dtype=%s)",
                        self.model_id, self.device, self.compute_dtype)
        except Exception as e:  # gated access / config mismatch surfaces here
            raise ModelLoadError(f"Failed to load SAM3 '{self.model_id}': {e}") from e

    # ------------------------------------------------------------------
    # GUI-facing wrappers (SAM2-flavored surface expected by the GUI layer)
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self, config_path=None, checkpoint_path=None) -> bool:
        """GUI-facing load returning bool (SAM3 ignores the SAM2 path args)."""
        try:
            self.load()
            return True
        except Exception as e:
            logger.error("SAM3 load_model failed: %s", e)
            return False

    def generate_mask(self, image, points, labels):
        """Single-object frame-0 preview mask (GUI annotation path)."""
        if not points:
            return None
        masks = self.predict_image_masks(
            image, [ObjectPrompt(obj_id=1, points=[tuple(p) for p in points], labels=list(labels))])
        return masks.get(1)

    def cleanup(self) -> None:
        self.reset()

    def predict_image_masks(self, image: np.ndarray,
                            prompts: list[ObjectPrompt]) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        for p in prompts:
            inputs = self.image_processor(
                images=image,
                input_points=[[[list(pt) for pt in p.points]]] if p.points else None,
                input_labels=[[list(p.labels)]] if p.labels else None,
                input_boxes=[[list(p.box)]] if p.box else None,
                return_tensors="pt",
            ).to(self.image_model.device)
            with torch.inference_mode():
                outputs = self.image_model(**inputs, multimask_output=False)
            best = self.image_processor.post_process_masks(
                outputs.pred_masks, inputs["original_sizes"], binarize=True)[0]
            result[p.obj_id] = best[0, 0].cpu().numpy().astype(bool)
        return result

    def init_video(self, video_path: str,
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> bool:
        frames = _decode_video_rgb(video_path)
        if not frames:
            logger.error("No frames decoded from %s", video_path)
            return False
        # Preprocess the clip on the CPU. If processing_device is left unset it defaults
        # to inference_device (cuda), and init_video_session builds the ENTIRE clip as one
        # (num_frames, 3, 1008, 1008) float32 tensor ON THE GPU
        # (processing_sam3_tracker_video.py:546/551) *before* video_storage_device="cpu"
        # ever moves it off — ~11.6 MiB/frame, i.e. tens of GiB in a single CUDA allocation
        # → OOM at init on long clips. With processing/storage/state on the CPU, only one
        # frame at a time is streamed to the GPU during propagation
        # (modeling_sam3_tracker_video.py:322), so VRAM stays bounded.
        self.session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            inference_state_device="cpu",
            dtype=self.compute_dtype,
        )
        self._num_frames = len(frames)
        return True

    def add_prompts(self, frame_idx: int, prompts: list[ObjectPrompt]) -> bool:
        if not prompts:
            return True
        # All objects for a frame MUST be added in ONE call: the processor sets
        # `inference_session.obj_with_new_inputs = obj_ids` (overwrite, not append),
        # so separate per-object calls leave only the last object registered as
        # having new inputs — the rest get no conditioning output and error with
        # empty maskmem at propagation. Batch by object (image->object->... nesting).
        obj_ids = [p.obj_id for p in prompts]
        has_pts = any(p.points for p in prompts)
        all_box = all(p.box for p in prompts)  # boxes must be all-or-none (one per object)
        if any(p.box for p in prompts) and not all_box:
            logger.warning(
                "add_prompts: mixed box/no-box across objects in one frame — boxes are "
                "dropped (every object in a single call must have a box). Add box-prompted "
                "objects in their own add_prompts call/frame. (Phase-2 edge case.)")
        input_points = [[[list(pt) for pt in p.points] for p in prompts]] if has_pts else None
        input_labels = [[list(p.labels) for p in prompts]] if has_pts else None
        input_boxes = [[list(p.box) for p in prompts]] if all_box else None
        self.processor.add_inputs_to_inference_session(
            inference_session=self.session, frame_idx=frame_idx, obj_ids=obj_ids,
            input_points=input_points, input_labels=input_labels, input_boxes=input_boxes,
            clear_old_inputs=True)
        # Track the earliest prompted frame — propagate() runs forward on it first.
        self._prompt_frame_idx = (frame_idx if self._prompt_frame_idx is None
                                  else min(self._prompt_frame_idx, frame_idx))
        return True

    def propagate(self, progress_callback: Optional[Callable[[int, int], None]] = None
                  ) -> Iterator[FramePrediction]:
        # SAM3 needs the prompted frame's conditioning + memory features materialized
        # before propagation. Run forward on it once (this also runs the memory
        # encoder); the iterator then auto-starts there, re-yields it from cache, and
        # propagates the remaining frames using that memory.
        start = self._prompt_frame_idx if self._prompt_frame_idx is not None else 0
        self.model(self.session, frame_idx=start)
        for out in self.model.propagate_in_video_iterator(
                self.session, show_progress_bar=False):
            processed = self.processor.post_process_masks(
                masks=[out.pred_masks],
                original_sizes=[[self.session.video_height, self.session.video_width]],
                binarize=True)[0]
            masks = _masks_from_output(out.object_ids, processed)
            if progress_callback:
                progress_callback(out.frame_idx, self._num_frames)
            yield FramePrediction(frame_idx=out.frame_idx, masks=masks)

    def reset(self) -> None:
        if self.session is not None:
            self.session.reset_inference_session()
            self.session = None
        self._prompt_frame_idx = None
