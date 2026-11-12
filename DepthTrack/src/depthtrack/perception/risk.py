"""Collision-risk classification.

Maps an object's estimated distance and lateral position to a discrete risk
level. ``DANGER`` is reserved for objects that are both close *and* roughly in
the ego lane, so a nearby parked car off to the side is not flagged as a
collision threat.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..config import RiskConfig


class RiskLevel(str, Enum):
    """Ordered collision-risk levels (lowest to highest severity)."""

    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    DANGER = "DANGER"


def classify_risk(
    distance_m: float,
    box: np.ndarray,
    frame_w: int,
    cfg: RiskConfig,
) -> RiskLevel:
    """Classify collision risk for a single tracked object.

    Args:
        distance_m: Estimated distance to the object in metres.
        box: Bounding box ``[x1, y1, x2, y2]`` in pixel coordinates.
        frame_w: Frame width, used to compute lateral offset from centre.
        cfg: Risk thresholds.

    Returns:
        The :class:`RiskLevel` for the object.
    """
    center_x = (box[0] + box[2]) / 2.0
    half_w = frame_w / 2.0
    lateral_offset = abs(center_x - half_w) / half_w if half_w else 1.0

    if distance_m < cfg.danger_m and lateral_offset < cfg.danger_lateral_offset:
        return RiskLevel.DANGER
    if distance_m < cfg.warning_m:
        return RiskLevel.WARNING
    if distance_m < cfg.caution_m:
        return RiskLevel.CAUTION
    return RiskLevel.SAFE
