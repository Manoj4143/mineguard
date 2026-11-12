"""Monocular depth estimation via Depth Anything V2.

Wraps a Hugging Face depth-estimation pipeline, normalises the output to
``[0, 1]``, and applies exponential temporal smoothing to reduce flicker
between frames.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import ModelConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class DepthEstimator:
    """Estimates a normalised, temporally-smoothed depth map per frame."""

    def __init__(self, cfg: ModelConfig, device: str, smoothing: float = 0.6):
        self.cfg = cfg
        self.device = device
        self._smoothing = smoothing
        self._prev: np.ndarray | None = None
        self._pipe = None

    def load(self) -> None:
        from transformers import pipeline as hf_pipeline

        logger.info("Loading depth model: %s", self.cfg.depth_model)
        hf_device = 0 if self.device == "cuda" else -1
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=self.cfg.depth_model,
            device=hf_device,
        )
        logger.info("Depth model ready")

    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        """Return a normalised depth map (``[0, 1]``) for a BGR frame."""
        from PIL import Image

        if self._pipe is None:
            raise RuntimeError("DepthEstimator.load() must be called before estimate()")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._pipe(Image.fromarray(rgb))
        depth = np.array(result["depth"], dtype=np.float32)
        depth = cv2.resize(
            depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR
        )

        lo, hi = float(depth.min()), float(depth.max())
        if hi - lo > 1e-5:
            depth = (depth - lo) / (hi - lo)

        if self._prev is not None and self._prev.shape == depth.shape:
            a = self._smoothing
            depth = a * self._prev + (1.0 - a) * depth
        self._prev = depth.copy()
        return depth
