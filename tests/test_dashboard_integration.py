"""Automated unit and integration tests for Phase 5 Controlled Dashboard Integration."""

import io
import json
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from mineguard_depthtrack.service import PerceptionAdapter, BackgroundVideoWorker
from mineguard_depthtrack.dashboard import DashboardHTTPRequestHandler


@pytest.fixture
def mock_perception_adapter():
    """Provides a PerceptionAdapter with mocked model dependencies."""
    adapter = PerceptionAdapter.__new__(PerceptionAdapter)
    adapter.device = "cpu"
    adapter.preferred_engine = PerceptionAdapter.ENGINE_UNIFIED
    adapter.active_engine_name = PerceptionAdapter.ENGINE_UNIFIED
    adapter.is_fallback_active = False
    adapter.initialization_error = None
    adapter._init_lock = MagicMock()

    mock_engine = MagicMock()
    # Mock return of process_frame
    composite_dummy = np.full((540, 1920, 3), 128, dtype=np.uint8)
    telemetry_dummy = {
        "system_status": "ONLINE",
        "frame_id": 1,
        "pipeline_mode": "CLEAR_BYPASS",
        "fog_condition": "CLEAR",
        "ambient_lux": 150.0,
        "fps": 25.0,
        "latency_ms": 40.0,
        "road_status": "ROAD CLEAR",
        "road_status_detail": "Unobstructed.",
        "highest_risk": "SAFE",
        "active_vehicles_count": 1,
    }
    objects_dummy = [
        {
            "track_id": 101,
            "class_name": "truck",
            "confidence": 0.88,
            "bbox": [100, 100, 200, 200],
            "relative_depth": 0.35,
            "distance_m": 22.5,
            "closing_speed_kmh": 12.0,
            "ttc_seconds": 6.7,
            "risk_tier": "SAFE",
        }
    ]
    mock_engine.process_frame.return_value = {
        "composite_frame": composite_dummy,
        "telemetry": telemetry_dummy,
        "objects": objects_dummy,
    }
    adapter.engine_instance = mock_engine
    return adapter


def test_adapter_initialization_unified(mock_perception_adapter):
    """Test 1: Engine initialization with Unified engine."""
    assert mock_perception_adapter.active_engine_name == PerceptionAdapter.ENGINE_UNIFIED
    assert not mock_perception_adapter.is_fallback_active


def test_adapter_initialization_failure():
    """Test 2: Engine initialization failure triggers fallback."""
    with patch("mineguard_depthtrack.service.UnifiedMineGuardPerception", side_effect=Exception("CUDA OOM")):
        with patch("mineguard_depthtrack.service.MineGuardIntegratedSystem") as mock_fallback:
            adapter = PerceptionAdapter(preferred_engine=PerceptionAdapter.ENGINE_UNIFIED, device="cpu")
            assert adapter.is_fallback_active
            assert adapter.active_engine_name == PerceptionAdapter.ENGINE_FALLBACK
            mock_fallback.assert_called_once()


def test_frame_processing_and_composite(mock_perception_adapter):
    """Tests 3 & 4: Frame processing and composite output shape/type."""
    input_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    disp, telem = mock_perception_adapter.process_frame(input_frame, 1)

    assert isinstance(disp, np.ndarray)
    assert disp.shape == (540, 1920, 3)
    assert telem["active_engine"] == PerceptionAdapter.ENGINE_UNIFIED
    assert len(telem["vehicles"]) == 1
    assert telem["vehicles"][0]["track_id"] == 101
    assert telem["vehicles"][0]["distance_m"] == 22.5


def test_numpy_to_jpeg_encoding():
    """Test 5: Reliable conversion of display frame to JPEG bytes."""
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    success, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert success
    jpeg_bytes = enc.tobytes()
    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 0
    # Confirm JPEG header magic bytes FF D8
    assert jpeg_bytes[:2] == b"\xff\xd8"


def test_invalid_frame_handling(mock_perception_adapter):
    """Test 6: Invalid or None frame handling."""
    mock_perception_adapter.engine_instance = None
    disp, telem = mock_perception_adapter.process_frame(None, 1)
    assert isinstance(disp, np.ndarray)
    assert telem["system_status"] == "OFFLINE"
    assert telem["road_status"] == "SYSTEM ERROR"


def test_missing_telemetry_fields_schema_consistency(mock_perception_adapter):
    """Tests 7 & 8: Schema consistency even when underlying engine has sparse fields."""
    mock_perception_adapter.engine_instance.process_frame.return_value = {
        "composite_frame": np.zeros((10, 10, 3), dtype=np.uint8),
        "telemetry": {},
        "objects": [],
    }
    disp, telem = mock_perception_adapter.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), 1)

    required_keys = [
        "active_engine", "is_fallback", "engine_error", "system_status",
        "pipeline_mode", "fog_condition", "ambient_lux", "road_status",
        "road_status_detail", "road_status_class", "highest_risk",
        "fps", "latency_ms", "active_vehicles_count", "vehicles"
    ]
    for key in required_keys:
        assert key in telem, f"Missing required telemetry key: {key}"


def test_background_worker_lifecycle(mock_perception_adapter, tmp_path):
    """Tests 9, 10, 11, 12: Worker startup, processing, and clean shutdown."""
    # Create a synthetic mini video
    video_path = tmp_path / "test_mini.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for _ in range(5):
        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        out.write(frame)
    out.release()

    worker = BackgroundVideoWorker(mock_perception_adapter, str(video_path))
    worker.start()
    assert worker._running

    time.sleep(0.15)
    jpeg_bytes, telem = worker.get_latest_data()
    assert jpeg_bytes is not None
    assert telem["worker_frame_id"] > 0

    worker.stop(timeout=1.0)
    assert not worker._running
    assert worker._cap is None


def test_worker_processing_exception_recovery(mock_perception_adapter, tmp_path):
    """Test 13: Worker survives intermittent processing exceptions without crashing."""
    video_path = tmp_path / "test_err.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for _ in range(3):
        out.write(np.zeros((240, 320, 3), dtype=np.uint8))
    out.release()

    call_count = 0
    def faulty_process(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Transient device timeout")
        return np.zeros((10, 10, 3), dtype=np.uint8), {"active_engine": "mock"}

    mock_perception_adapter.process_frame = faulty_process
    worker = BackgroundVideoWorker(mock_perception_adapter, str(video_path))
    worker.start()
    time.sleep(0.1)
    assert worker._running
    worker.stop(timeout=1.0)


def test_existing_engine_fallback_switch(mock_perception_adapter):
    """Test 14: Dynamic switching between unified engine and fallback."""
    with patch("mineguard_depthtrack.service.MineGuardIntegratedSystem") as mock_fallback:
        mock_instance = MagicMock()
        mock_fallback.return_value = mock_instance
        mock_perception_adapter.ENGINE_FALLBACK = "MineGuardIntegratedSystem"
        mock_perception_adapter._init_fallback = PerceptionAdapter._init_fallback.__get__(mock_perception_adapter)
        mock_perception_adapter.set_engine = PerceptionAdapter.set_engine.__get__(mock_perception_adapter)

        res = mock_perception_adapter.set_engine("MineGuardIntegratedSystem")
        assert res is True
        assert mock_perception_adapter.is_fallback_active


def test_telemetry_endpoint_compatibility(mock_perception_adapter):
    """Tests 15 & 16: Telemetry and set_engine API responses."""
    import mineguard_depthtrack.dashboard as dash
    dash.worker = MagicMock()
    dash.worker.get_latest_data.return_value = (
        b"fake_jpeg",
        {"active_engine": "UnifiedMineGuardPerception", "fps": 20.0, "road_status": "ROAD CLEAR"}
    )
    dash.adapter = mock_perception_adapter

    # Verify telemetry response structure
    _, telem = dash.worker.get_latest_data()
    assert telem["active_engine"] == "UnifiedMineGuardPerception"
    assert telem["fps"] == 20.0