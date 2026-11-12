"""Integration tests for the UnifiedMineGuardPerception engine."""

import numpy as np
import pytest
from mineguard_depthtrack import UnifiedMineGuardPerception


@pytest.fixture(scope="module")
def unified_engine():
    engine = UnifiedMineGuardPerception()
    return engine


def test_unified_engine_initialization(unified_engine):
    assert unified_engine.system_status == "ONLINE"
    assert unified_engine.detector is not None
    assert unified_engine.depth_estimator is not None
    assert unified_engine.tracker is not None


def test_unified_engine_process_frame(unified_engine):
    # Create synthetic test frame 640x480
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background road
    frame[240:, :] = (70, 70, 70)
    # Simulated vehicle box
    frame[260:360, 200:320] = (0, 0, 220)

    res = unified_engine.process_frame(frame, timestamp=1.0)
    assert "telemetry" in res
    assert "objects" in res
    assert "composite_frame" in res
    assert "depth_map" in res

    tel = res["telemetry"]
    assert tel["system_status"] == "ONLINE"
    assert tel["highest_risk"] in ("SAFE", "CAUTION", "CRITICAL")
    assert tel["road_status"] in ("ROAD CLEAR", "ROAD BLOCKED / HAZARD", "PROXIMITY WARNING")
    assert tel["fps"] > 0
    assert res["composite_frame"].shape[0] == 540
    assert res["composite_frame"].shape[1] == 1924


def test_unified_engine_consecutive_tracking_and_ttc(unified_engine):
    # Simulate an approaching vehicle across 5 frames
    for i in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[240:, :] = (70, 70, 70)
        # Getting larger and closer
        y1 = int(260 + i * 5)
        y2 = int(360 + i * 10)
        x1 = int(200 - i * 5)
        x2 = int(320 + i * 5)
        frame[y1:y2, x1:x2] = (0, 0, 220)
        
        res = unified_engine.process_frame(frame, timestamp=1.0 + i * 0.1)
        assert res["telemetry"]["frame_id"] > 0
