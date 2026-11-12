"""Multi-object tracking with trajectory and velocity bookkeeping.

Wraps ByteTrack (via the ``supervision`` library) and additionally records, per
track id, a bounded history of centre points (for drawing trails) and a simple
finite-difference velocity estimate (for motion arrows).
"""

from __future__ import annotations

import collections


class ObjectTracker:
    """Associates detections across frames and tracks their motion history."""

    def __init__(self, max_history: int = 30):
        import supervision as sv

        self.tracker = sv.ByteTrack()
        self.trajectories: dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=max_history)
        )
        self.velocities: dict[int, tuple[float, float]] = collections.defaultdict(
            lambda: (0.0, 0.0)
        )
        self._prev_centers: dict[int, tuple[int, int]] = {}

    def update(self, detections):
        """Update tracks with the latest detections; return tracked detections."""
        tracked = self.tracker.update_with_detections(detections)
        if tracked.tracker_id is None:
            return tracked

        for i, tid in enumerate(tracked.tracker_id):
            tid = int(tid)
            box = tracked.xyxy[i]
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            self.trajectories[tid].append((cx, cy))
            if tid in self._prev_centers:
                px, py = self._prev_centers[tid]
                self.velocities[tid] = (cx - px, cy - py)
            self._prev_centers[tid] = (cx, cy)
        return tracked
