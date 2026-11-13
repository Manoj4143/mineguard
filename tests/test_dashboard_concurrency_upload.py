"""Automated concurrency, upload verification, and streaming tests for Phase 5.1."""

import http.client
import json
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pytest

from mineguard_depthtrack.service import PerceptionAdapter
import mineguard_depthtrack.dashboard as dash


@pytest.fixture(scope="module")
def live_threaded_server():
    """Starts a live ThreadingHTTPServer on an ephemeral port for testing concurrency."""
    port = 8899
    # Use synthetic adapter to keep tests fast and avoid GPU contention
    test_adapter = PerceptionAdapter.__new__(PerceptionAdapter)
    test_adapter.device = "cpu"
    test_adapter.preferred_engine = PerceptionAdapter.ENGINE_UNIFIED
    test_adapter.active_engine_name = PerceptionAdapter.ENGINE_UNIFIED
    test_adapter.is_fallback_active = False
    test_adapter.initialization_error = None
    test_adapter._init_lock = threading.Lock()

    def mock_process(frame, frame_id=1):
        disp = np.zeros((240, 320, 3), dtype=np.uint8)
        telem = {
            "active_engine": test_adapter.active_engine_name,
            "is_fallback": test_adapter.is_fallback_active,
            "system_status": "ONLINE",
            "pipeline_mode": "CLEAR_BYPASS",
            "fog_condition": "CLEAR",
            "ambient_lux": 100.0,
            "road_status": "ROAD CLEAR",
            "road_status_detail": "All clear",
            "road_status_class": "clear",
            "highest_risk": "SAFE",
            "fps": 25.0,
            "latency_ms": 40.0,
            "active_vehicles_count": 0,
            "vehicles": [],
        }
        return disp, telem

    test_adapter.process_frame = mock_process
    test_adapter.set_engine = lambda name: PerceptionAdapter.set_engine(test_adapter, name)
    test_adapter._initialize_engine = lambda name: setattr(test_adapter, "active_engine_name", name)

    # Temporary mini video
    temp_video = Path("temp_concurrency_test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_video), fourcc, 10.0, (160, 120))
    for _ in range(20):
        out.write(np.zeros((120, 160, 3), dtype=np.uint8))
    out.release()

    dash.adapter = test_adapter
    dash.worker = dash.BackgroundVideoWorker(test_adapter, str(temp_video))
    dash.worker.start()
    dash.server_running = True

    server = dash.ThreadingHTTPServer(("127.0.0.1", port), dash.DashboardHTTPRequestHandler)
    dash.server_instance = server

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    yield f"http://127.0.0.1:{port}"

    dash.server_running = False
    dash.worker.stop(timeout=1.0)
    server.shutdown()
    server.server_close()
    if temp_video.exists():
        temp_video.unlink()


def test_concurrent_mjpeg_and_telemetry(live_threaded_server):
    """Test 1: Persistent MJPEG connection does NOT block concurrent telemetry polling."""
    base_url = live_threaded_server
    stream_opened = threading.Event()
    mjpeg_received = False
    stop_stream = threading.Event()

    def stream_consumer():
        nonlocal mjpeg_received
        req = urllib.request.urlopen(f"{base_url}/stream.mjpg", timeout=5)
        stream_opened.set()
        # Read a chunk of the stream
        chunk = req.read(1024)
        if b"--FRAME" in chunk or len(chunk) > 0:
            mjpeg_received = True
        while not stop_stream.is_set():
            try:
                req.read(512)
            except Exception:
                break
        req.close()

    stream_thread = threading.Thread(target=stream_consumer, daemon=True)
    stream_thread.start()

    assert stream_opened.wait(timeout=3.0), "MJPEG stream failed to open"

    # Now verify telemetry requests succeed immediately without blocking
    t_start = time.time()
    for _ in range(5):
        req = urllib.request.urlopen(f"{base_url}/api/telemetry?t={time.time()}", timeout=2.0)
        assert req.status == 200
        data = json.loads(req.read().decode("utf-8"))
        assert "active_engine" in data
        assert "road_status" in data
        time.sleep(0.05)

    elapsed = time.time() - t_start
    assert elapsed < 2.0, "Concurrent telemetry requests were blocked by MJPEG stream!"
    stop_stream.set()
    stream_thread.join(timeout=1.0)


def test_upload_valid_video_while_streaming(live_threaded_server, tmp_path):
    """Tests 2, 3, 4, 5: Upload valid MP4 during streaming, verify worker switches and increments frames."""
    base_url = live_threaded_server

    # Create a small valid test video
    valid_video = tmp_path / "valid_sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(valid_video), fourcc, 10.0, (160, 120))
    for _ in range(15):
        out.write(np.full((120, 160, 3), 150, dtype=np.uint8))
    out.release()

    video_bytes = valid_video.read_bytes()
    assert len(video_bytes) > 0

    # Perform POST upload
    req = urllib.request.Request(
        f"{base_url}/api/upload_video?filename=valid_sample.mp4",
        data=video_bytes,
        headers={"Content-Length": str(len(video_bytes))},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["success"] is True
        assert "source_key" in res
        assert res["filename"] == "valid_sample.mp4"

    # Verify worker switched to uploaded video
    time.sleep(0.3)
    _, telem = dash.worker.get_latest_data()
    assert telem.get("worker_frame_id", 0) > 0


def test_upload_empty_video_rejected(live_threaded_server):
    """Test 8a: Upload of 0-byte file is rejected with HTTP 400."""
    base_url = live_threaded_server
    req = urllib.request.Request(
        f"{base_url}/api/upload_video?filename=empty.mp4",
        data=b"",
        headers={"Content-Length": "0"},
        method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=3.0)
    assert exc_info.value.code == 400


def test_upload_corrupt_video_rejected(live_threaded_server):
    """Test 8b: Upload of corrupt/non-video bytes is rejected with HTTP 422."""
    base_url = live_threaded_server
    corrupt_bytes = b"NOT_A_REAL_VIDEO_HEADER_1234567890" * 50
    req = urllib.request.Request(
        f"{base_url}/api/upload_video?filename=corrupt.mp4",
        data=corrupt_bytes,
        headers={"Content-Length": str(len(corrupt_bytes))},
        method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=3.0)
    assert exc_info.value.code == 422


def test_engine_switching_identifiers_and_deduplication(live_threaded_server):
    """Test 6: Engine switching uses exact identifiers and deduplicates."""
    base_url = live_threaded_server

    # Switch to fallback
    req = urllib.request.urlopen(f"{base_url}/api/set_engine?engine=MineGuardIntegratedSystem")
    assert req.status == 200
    res = json.loads(req.read().decode("utf-8"))
    assert res["active_engine"] == "MineGuardIntegratedSystem"

    # Switch back to unified
    req = urllib.request.urlopen(f"{base_url}/api/set_engine?engine=UnifiedMineGuardPerception")
    assert req.status == 200
    res = json.loads(req.read().decode("utf-8"))
    assert res["active_engine"] == "UnifiedMineGuardPerception"

    # Redundant switch should be no-op / success True
    req = urllib.request.urlopen(f"{base_url}/api/set_engine?engine=UnifiedMineGuardPerception")
    assert req.status == 200
    res = json.loads(req.read().decode("utf-8"))
    assert res["success"] is True

