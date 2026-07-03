"""In-memory TrackerBackend for tests (no GPU/model)."""
from __future__ import annotations
from typing import Iterator, Optional, Callable
import numpy as np
from tracker.tracking.base import FramePrediction, ObjectPrompt

class FakeTracker:
    def __init__(self, masks_per_frame: list[dict[int, np.ndarray]]):
        self._masks_per_frame = masks_per_frame
        self.loaded = False
        self.added_prompts: list[ObjectPrompt] = []

    def load(self) -> None:
        self.loaded = True

    def predict_image_masks(self, image, prompts: list[ObjectPrompt]) -> dict[int, np.ndarray]:
        return self._masks_per_frame[0] if self._masks_per_frame else {}

    def init_video(self, video_path: str, progress_callback: Optional[Callable] = None) -> bool:
        return True

    def add_prompts(self, frame_idx: int, prompts: list[ObjectPrompt]) -> bool:
        self.added_prompts.extend(prompts)
        return True

    def propagate(self, progress_callback: Optional[Callable] = None) -> Iterator[FramePrediction]:
        for i, masks in enumerate(self._masks_per_frame):
            if progress_callback:
                progress_callback(i + 1, len(self._masks_per_frame))
            yield FramePrediction(frame_idx=i, masks=masks)

    def reset(self) -> None:
        self.added_prompts.clear()
