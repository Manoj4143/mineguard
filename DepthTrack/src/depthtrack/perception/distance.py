"""Monocular distance estimation.

Fuses two cues into a single per-object distance estimate:

1. **Depth cue** - the median normalised depth inside the bounding box is mapped
   onto a metric near/far range.
2. **Projection cue** - a pinhole-camera estimate from the apparent pixel width
   of the object and the class-specific real-world vehicle width.

The two are blended (controlled by ``DistanceConfig.depth_weight``) and clamped
to a sane range.
"""

from __future__ import annotations

import numpy as np

from ..config import DistanceConfig

_FALLBACK_DISTANCE_M = 50.0


def estimate_distance(
    depth_map: np.ndarray,
    box: np.ndarray,
    frame_w: int,
    frame_h: int,
    cfg: DistanceConfig,
    class_id: int = 2,
) -> float:
    """Estimate distance (metres) to the object in ``box``.

    Args:
        depth_map: Normalised depth in ``[0, 1]`` with shape ``(H, W)``.
        box: Bounding box ``[x1, y1, x2, y2]`` in pixel coordinates.
        frame_w: Width of the frame the box is expressed in.
        frame_h: Height of the frame the box is expressed in.
        cfg: Distance estimator parameters.
        class_id: COCO class id of the detected object. Used to look up the
            correct real-world width so trucks/buses are not confused with cars.

    Returns:
        Distance in metres, rounded to one decimal place.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w - 1, x2), min(frame_h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return _FALLBACK_DISTANCE_M

    patch = depth_map[y1:y2, x1:x2]
    median_depth = float(np.median(patch))

    # Depth Anything V2 outputs INVERSE depth: high normalised value = NEAR,
    # low value = FAR. So a high median maps to a SHORT distance.
    depth_span = cfg.depth_far_m - cfg.depth_near_m
    dist_from_depth = cfg.depth_near_m + (1.0 - median_depth) * depth_span

    real_width = cfg.width_for_class(class_id)
    pixel_width = max(x2 - x1, 1)
    dist_from_projection = (cfg.focal_length_px * real_width) / pixel_width

    w = cfg.depth_weight
    fused = w * dist_from_depth + (1.0 - w) * dist_from_projection
    clamped = float(np.clip(fused, cfg.min_distance_m, cfg.max_distance_m))
    return round(clamped, 1)