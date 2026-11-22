"""Tests for Feature 1: Automatic Incident Recorder.

Validates rolling buffer pre/post event duration, automated triggers,
async background MP4 encoding worker, and SQLite metadata indexing.
"""

import time
import numpy as np
import pytest
from mineguard_depthtrack.incident_recorder import (
    RollingFrameBuffer,
    AutomaticIncidentRecorder,
    IncidentEventType
)


def test_rolling_frame_buffer():
    # 20 fps, 2 seconds max -> maxlen = 40 frames
    buf = RollingFrameBuffer(max_seconds=2.0, fps=20.0)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    now = time.time()
    for i in range(50):
        buf.append(now + (i * 0.05), dummy_frame)

    frames = buf.get_all_frames()
    assert len(frames) == 40  # Capped at 40 frames


def test_incident_trigger_and_metadata():
    recorder = AutomaticIncidentRecorder(
        db_path=":memory:",
        pre_event_seconds=1.0,
        post_event_seconds=1.0,
        fps=10.0
    )

    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(15):
        recorder.add_frame(dummy_frame)

    # Trigger critical risk incident
    inc_id = recorder.trigger_incident(
        event_type=IncidentEventType.CRITICAL_COLLISION_RISK,
        severity="CRITICAL",
        vehicle_id="MINE-TRUCK-001",
        telemetry={"speed_kmh": 35.0},
        description="Test critical collision risk event",
        ttc=1.2,
        distance_m=14.5
    )
    assert inc_id is not None
    assert inc_id.startswith("INC_")

    # Duplicate trigger within cooldown should return None
    inc_dup = recorder.trigger_incident(
        event_type=IncidentEventType.CRITICAL_COLLISION_RISK,
        severity="CRITICAL",
        vehicle_id="MINE-TRUCK-001",
        telemetry={"speed_kmh": 35.0},
        description="Duplicate trigger"
    )
    assert inc_dup is None

    # Simulate feeding frames past the post-event window (1.0 sec)
    end_time = time.time() + 1.2
    while time.time() < end_time:
        recorder.add_frame(dummy_frame)
        time.sleep(0.05)

    # Wait for async encoding worker to commit to DB
    time.sleep(1.0)

    summary = recorder.get_summary()
    assert summary["total_incidents"] >= 1
    assert summary["critical_incidents"] >= 1
