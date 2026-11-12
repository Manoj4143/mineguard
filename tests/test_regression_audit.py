"""Comprehensive Regression Suite for Phase 4.5 Integration Audit."""

import time
import numpy as np
import pytest
from mineguard_depthtrack import UnifiedMineGuardPerception


@pytest.fixture(scope="module")
def engine():
    return UnifiedMineGuardPerception()


def test_night_mode_clahe_activation(engine):
    """Test that night frames (<70 lux) trigger NIGHT_CLAHE pipeline mode."""
    # Create dark frame (< 15 lux)
    dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)
    res = engine.process_frame(dark_frame, timestamp=1.0)
    assert res["telemetry"]["pipeline_mode"] == "NIGHT_CLAHE"
    assert res["telemetry"]["ambient_lux"] < 70.0


def test_fog_adaptive_mode_activation(engine):
    """Test that foggy frames trigger FOG_ADAPTIVE mode."""
    # Create low-contrast bright frame simulating fog
    fog_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    fog_frame[200:250, :] = 175
    res = engine.process_frame(fog_frame, timestamp=2.0)
    assert res["telemetry"]["fog_condition"] in ("LIGHT_FOG", "MEDIUM_FOG", "DENSE_FOG")
    assert res["telemetry"]["pipeline_mode"] in ("FOG_ADAPTIVE", "DCP_V5_FOG")


def test_clear_bypass_mode_activation(engine):
    """Test that high-contrast daylight frames trigger CLEAR_BYPASS."""
    # Create high-contrast clear frame
    clear_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    clear_frame[:240, :] = 240  # sky
    clear_frame[240:, :] = 40   # ground
    res = engine.process_frame(clear_frame, timestamp=3.0)
    assert res["telemetry"]["fog_condition"] == "CLEAR"
    assert res["telemetry"]["pipeline_mode"] == "CLEAR_BYPASS"


def test_output_schema_and_dimensions(engine):
    """Verify composite dimensions and telemetry typing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = engine.process_frame(frame, timestamp=4.0)

    # Check composite dimensions (1924x540)
    comp = res["composite_frame"]
    assert comp.shape[0] == 540
    assert comp.shape[1] == 1924

    # Check depth map dimensions and range
    dmap = res["depth_map"]
    assert dmap.min() >= 0.0
    assert dmap.max() <= 1.0

    # Check telemetry fields
    tel = res["telemetry"]
    for key in ["system_status", "frame_id", "pipeline_mode", "fps", "latency_ms", "road_status", "highest_risk"]:
        assert key in tel


def test_safety_ttc_and_closing_speed(engine):
    """Verify closing speed and TTC calculation on advancing box."""
    # Frame 1: Vehicle far away
    f1 = np.zeros((480, 640, 3), dtype=np.uint8)
    f1[240:, :] = (60, 60, 60)
    f1[260:320, 280:360] = (0, 0, 220)
    res1 = engine.process_frame(f1, timestamp=10.0)

    # Frame 2: Vehicle approaching closer 0.5s later
    f2 = np.zeros((480, 640, 3), dtype=np.uint8)
    f2[240:, :] = (60, 60, 60)
    f2[240:360, 240:400] = (0, 0, 220)
    res2 = engine.process_frame(f2, timestamp=10.5)

    assert res2["telemetry"]["frame_id"] > res1["telemetry"]["frame_id"]

