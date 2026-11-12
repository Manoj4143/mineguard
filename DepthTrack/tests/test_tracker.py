"""Regression tests for ObjectTracker."""

import numpy as np
import supervision as sv
from depthtrack.models.tracker import ObjectTracker


def test_object_tracker_initialization():
    tracker = ObjectTracker(max_history=15)
    assert tracker.tracker is not None
    assert tracker.trajectories.default_factory is not None


def test_object_tracker_update_and_history():
    tracker = ObjectTracker(max_history=5)
    
    # Frame 1
    xyxy = np.array([[100, 100, 200, 200]], dtype=float)
    conf = np.array([0.9], dtype=float)
    cls = np.array([2], dtype=int)
    dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
    
    res1 = tracker.update(dets)
    assert res1.tracker_id is not None
    tid = int(res1.tracker_id[0])
    assert len(tracker.trajectories[tid]) == 1
    assert tracker.velocities[tid] == (0.0, 0.0)
    
    # Frame 2 (moving +10 in x, +5 in y)
    xyxy2 = np.array([[110, 105, 210, 205]], dtype=float)
    dets2 = sv.Detections(xyxy=xyxy2, confidence=conf, class_id=cls)
    res2 = tracker.update(dets2)
    
    assert int(res2.tracker_id[0]) == tid
    assert len(tracker.trajectories[tid]) == 2
    assert tracker.velocities[tid] == (10.0, 5.0)

