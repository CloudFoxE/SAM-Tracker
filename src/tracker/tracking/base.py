"""Model-agnostic tracking interface and data types."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol, runtime_checkable, Callable
import numpy as np

@dataclass
class ObjectPrompt:
    obj_id: int
    points: list[tuple[float, float]] = field(default_factory=list)   # (x, y) pixel coords
    labels: list[int] = field(default_factory=list)                   # 1 = foreground, 0 = background
    box: Optional[tuple[float, float, float, float]] = None           # (x1, y1, x2, y2), optional

@dataclass
class FramePrediction:
    frame_idx: int
    masks: dict[int, np.ndarray]        # obj_id -> 2D bool mask

@runtime_checkable
class TrackerBackend(Protocol):
    def load(self) -> None: ...
    def predict_image_masks(self, image: np.ndarray,
                            prompts: list[ObjectPrompt]) -> dict[int, np.ndarray]: ...
    def init_video(self, video_path: str,
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> bool: ...
    def add_prompts(self, frame_idx: int, prompts: list[ObjectPrompt]) -> bool: ...
    def propagate(self, progress_callback: Optional[Callable[[int, int], None]] = None
                  ) -> Iterator[FramePrediction]: ...
    def reset(self) -> None: ...
