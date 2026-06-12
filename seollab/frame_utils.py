"""Shared Atari frame normalization for V2/V3 visualization."""

from __future__ import annotations

import numpy as np
from PIL import Image


def normalize_v2_frame(img: np.ndarray) -> np.ndarray:
    """Undo DreamerV2 Atari env rotation so frames match DreamerV3 orientation."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    # dreamerv2/common/envs.py Atari._render uses buf.transpose(1, 0, 2) → 90° CCW vs V3.
    return np.rot90(arr, k=-1).copy()


def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def resize_frame(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    rgb = to_rgb_uint8(img)
    if rgb.shape[:2] == size:
        return rgb
    return np.array(Image.fromarray(rgb).resize(size, Image.BILINEAR))
