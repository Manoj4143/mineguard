"""Turbo-style colormap for depth visualisation.

Builds a 256-entry BGR lookup table once at import time and maps normalised
depth values onto it. Using a precomputed LUT keeps per-frame colourisation a
cheap array index rather than a per-pixel computation.
"""

from __future__ import annotations

import numpy as np


def _build_turbo_lut() -> np.ndarray:
    """Construct a 256x3 BGR lookup table approximating the 'turbo' colormap."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            r, g, b = 0, int(t * 4 * 255), 255
        elif t < 0.5:
            r, g, b = 0, 255, int((1 - (t - 0.25) * 4) * 255)
        elif t < 0.75:
            r, g, b = int((t - 0.5) * 4 * 255), 255, 0
        else:
            r, g, b = 255, int((1 - (t - 0.75) * 4) * 255), 0
        lut[i] = [b, g, r]  # stored as BGR for OpenCV
    return lut


TURBO_LUT: np.ndarray = _build_turbo_lut()


def depth_to_color(depth_norm: np.ndarray) -> np.ndarray:
    """Map normalised depth in ``[0, 1]`` to BGR colours via the turbo LUT."""
    idx = (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)
    return TURBO_LUT[idx]
