"""YOLO-based vehicle detector wrapper.

Thin adapter around Ultralytics YOLO that resolves the device, restricts
detections to the configured vehicle classes, and returns results as a
``supervision.Detections`` object for downstream tracking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ModelConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class VehicleDetector:
    """Detects vehicles in a frame using a YOLO model."""

    def __init__(self, cfg: ModelConfig, device: str, weights_path: str | None = None):
        self.cfg = cfg
        self.device = device
        self.weights_path = weights_path or cfg.detector_weights
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO

        logger.info("Loading detector weights: %s", self.weights_path)
        self._model = YOLO(self.weights_path)
        self._model.to(self.device)
        logger.info("Detector ready on %s", self.device)

    def detect(self, frame: np.ndarray):
        """Run detection on a BGR frame and return ``supervision.Detections``."""
        import supervision as sv

        if self._model is None:
            raise RuntimeError("VehicleDetector.load() must be called before detect()")

        results = self._model(
            frame,
            conf=self.cfg.confidence,
            classes=list(self.cfg.vehicle_classes.keys()),
            verbose=False,
            device=self.device,
        )
        return sv.Detections.from_ultralytics(results[0])

    @staticmethod
    def resolve_weights(weights: str, asset_dir: Path) -> str:
        """Resolve a weights filename to an absolute path if it lives in assets."""
        candidate = Path(weights)
        if candidate.is_file():
            return str(candidate)
        bundled = asset_dir / weights
        if bundled.is_file():
            return str(bundled)
        # Fall back to the bare name so Ultralytics can auto-download it.
        return weights
