"""Compute-precision resolution shared by the tracker backends.

Both SAM2 and SAM3 expose a ``*_COMPUTE_DTYPE`` config knob (default ``float32``,
which is validated and, thanks to CPU offloading, keeps VRAM bounded). Switching to
``bfloat16``/``float16`` roughly halves activation/storage memory for very long
sessions. On non-CUDA devices we always fall back to float32 because bf16/fp16 CPU
kernels are slow or incomplete.
"""
from __future__ import annotations
import torch

_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_compute_dtype(device: str, name: str) -> torch.dtype:
    """Map a ``*_COMPUTE_DTYPE`` config string to a torch dtype for ``device``.

    Non-CUDA devices always resolve to float32. Unknown names also fall back to
    float32 (the safe default) rather than raising.
    """
    if device != "cuda":
        return torch.float32
    return _DTYPE_BY_NAME.get((name or "").lower(), torch.float32)
