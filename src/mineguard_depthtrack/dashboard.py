"""MineGuard AI - Interactive Web-Based Monitoring Dashboard.

Integrated with UnifiedMineGuardPerception perception engine,
ThreadingHTTPServer concurrency, validated streaming uploads,
and fallback engine support.
"""

from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import logging
from pathlib import Path
import re
import sys
import threading
import time
from typing import Optional, Dict, Any
import urllib.parse

import cv2
import numpy as np

# Ensure root paths are in sys.path
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
DEPTHTRACK_SRC = REPO_ROOT / "DepthTrack" / "src"
MINEGUARD_FALLBACK_DIR = Path("D:/MINI_GUARD_PROJECT/MineGuard-AI")

for p in [str(SRC_DIR), str(REPO_ROOT / "src"), str(DEPTHTRACK_SRC), str(MINEGUARD_FALLBACK_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mineguard_depthtrack.service import PerceptionAdapter, BackgroundVideoWorker
from mineguard_depthtrack.dataset import MineGuardDatasetManager

logger = logging.getLogger("MineGuardDashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

UPLOAD_DIR = REPO_ROOT / "DepthTrack" / "data" / "user_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Maximum upload size limit: 500 MB
MAX_UPLOAD_SIZE = 500 * 1024 * 1024

# Load Organized Datasets (Group A: Synthetic Testing & Group B: Real Mining Testing)
dataset_manager = MineGuardDatasetManager()
AVAILABLE_SCENARIOS = dataset_manager.get_available_scenarios()

AVAILABLE_VIDEOS = {}
for k, v in AVAILABLE_SCENARIOS.items():
    if Path(v["path"]).exists():
        AVAILABLE_VIDEOS[k] = v["path"]

# Fallback compatibility keys for legacy references
if "synthetic_clear" in AVAILABLE_VIDEOS:
    AVAILABLE_VIDEOS["mine_clear"] = AVAILABLE_VIDEOS["synthetic_clear"]
if "synthetic_light_fog" in AVAILABLE_VIDEOS:
    AVAILABLE_VIDEOS["mine_light_fog"] = AVAILABLE_VIDEOS["synthetic_light_fog"]
if "synthetic_dense_fog" in AVAILABLE_VIDEOS:
    AVAILABLE_VIDEOS["mine_dense_fog"] = AVAILABLE_VIDEOS["synthetic_dense_fog"]
if "synthetic_night" in AVAILABLE_VIDEOS:
    AVAILABLE_VIDEOS["mine_night"] = AVAILABLE_VIDEOS["synthetic_night"]
if "synthetic_sample" in AVAILABLE_VIDEOS:
    AVAILABLE_VIDEOS["sample_test"] = AVAILABLE_VIDEOS["synthetic_sample"]
if "real_mine_truck" in AVAILABLE_VIDEOS:
    AVAILABLE_VIDEOS["mine_truck"] = AVAILABLE_VIDEOS["real_mine_truck"]

# Global Controller State
DEFAULT_SOURCE = "real_mine_truck" if "real_mine_truck" in AVAILABLE_VIDEOS else "mine_clear"
current_source_key = DEFAULT_SOURCE
current_source_path = AVAILABLE_VIDEOS[DEFAULT_SOURCE]

adapter: Optional[PerceptionAdapter] = None
worker: Optional[BackgroundVideoWorker] = None
server_instance: Optional[ThreadingHTTPServer] = None
server_running = True

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MineGuard AI — Mine Vehicle Safety & Road Clearance Dashboard</title>
    <style>
        :root {
            --bg-base: #0c1017;
            --bg-card: rgba(22, 27, 34, 0.90);
            --border: #30363d;
            --text-main: #f0f6fc;
            --text-dim: #8b949e;
            --accent-blue: #58a6ff;
            --safe-green: #3fb950;
            --caution-yellow: #d29922;
            --hazard-red: #f85149;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg-base); color: var(--text-main); padding: 18px; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 16px; }
        .logo-group { display: flex; align-items: center; gap: 12px; }
        .logo-badge { background: linear-gradient(135deg, #1f6feb, #238636); padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 15px; }
        h1 { font-size: 20px; font-weight: 600; }
        .status-pill { display: flex; align-items: center; gap: 8px; background: var(--bg-card); border: 1px solid var(--border); padding: 6px 14px; border-radius: 20px; font-size: 13px; }
        .pulse-dot { width: 9px; height: 9px; background: var(--safe-green); border-radius: 50%; box-shadow: 0 0 8px var(--safe-green); }

        /* Road Clearance & Alert Banner */
        #road-clearance-bar {
            padding: 14px 20px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-weight: 700;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.3s ease;
        }
        .road-clear { background: rgba(63, 185, 80, 0.18); border: 2px solid var(--safe-green); color: var(--safe-green); }
        .road-caution { background: rgba(210, 153, 34, 0.22); border: 2px solid var(--caution-yellow); color: var(--caution-yellow); }
        .road-blocked { background: rgba(248, 81, 73, 0.28); border: 2px solid var(--hazard-red); color: var(--hazard-red); animation: blink-border 0.9s infinite alternate; }
        @keyframes blink-border { from { box-shadow: 0 0 4px var(--hazard-red); } to { box-shadow: 0 0 20px var(--hazard-red); } }

        /* Controls bar */
        .controls-bar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 16px; background: var(--bg-card); padding: 12px 16px; border-radius: 8px; border: 1px solid var(--border); }
        select, button { background: #161b22; color: #fff; border: 1px solid var(--border); padding: 8px 14px; border-radius: 6px; font-size: 13px; cursor: pointer; }
        button:hover { background: #1f242c; border-color: var(--accent-blue); }
        .upload-btn { background: #238636; border-color: #2ea043; font-weight: bold; }
        .engine-badge { font-weight: bold; padding: 4px 10px; border-radius: 4px; font-size: 12px; border: 1px solid var(--border); }
        .engine-badge-unified { background: rgba(31, 111, 235, 0.2); color: var(--accent-blue); border-color: var(--accent-blue); }
        .engine-badge-fallback { background: rgba(210, 153, 34, 0.2); color: var(--caution-yellow); border-color: var(--caution-yellow); }

        /* Grid */
        .dashboard-grid { display: grid; grid-template-columns: 2.2fr 1fr; gap: 18px; }
        .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
        .card-title { font-size: 14px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 12px; }

        /* Video Container */
        .video-container { position: relative; border-radius: 8px; overflow: hidden; background: #000; border: 1px solid var(--border); min-height: 420px; display: flex; align-items: center; justify-content: center; }
        .video-container img { width: 100%; height: auto; display: block; object-fit: contain; }

        /* Metrics */
        .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 16px; }
        .stat-box { background: rgba(13, 17, 23, 0.7); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
        .stat-label { font-size: 12px; color: var(--text-dim); margin-bottom: 4px; }
        .stat-val { font-size: 18px; font-weight: 700; color: var(--accent-blue); }

        table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
        th { text-align: left; color: var(--text-dim); padding: 8px 6px; border-bottom: 1px solid var(--border); font-weight: 500; }
        td { padding: 8px 6px; border-bottom: 1px solid rgba(48, 54, 61, 0.5); }
        .tier-badge { padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .badge-SAFE { background: rgba(63, 185, 80, 0.2); color: var(--safe-green); }
        .badge-CAUTION { background: rgba(210, 153, 34, 0.25); color: var(--caution-yellow); }
        .badge-CRITICAL { background: rgba(248, 81, 73, 0.3); color: var(--hazard-red); }
    </style>
</head>
<body>
    <header>
        <div class="logo-group">
            <div class="logo-badge">MINEGUARD AI</div>
            <h1>Mine Vehicle Safety & Real-Time Road Clearance Center</h1>
        </div>
        <div class="status-pill">
            <div class="pulse-dot" id="header-pulse"></div>
            <span id="pipeline-status">Perception Engine Active (GPU RTX 3050)</span>
        </div>
    </header>

    <!-- Source Selector & Engine Selector -->
    <div class="controls-bar">
        <label style="font-size: 13px; color: var(--text-dim);">Video Source:</label>
        <select id="source-select" onchange="changeVideoSource()">
            <optgroup label="Real Mining Testing (Group B)">
                <option value="real_mine_truck" selected>Real Mining: Main Haul Truck Road</option>
                <option value="real_clear_clean">Real Mining: Clear Haul Road (GT Baseline)</option>
                <option value="real_light_fog">Real Mining: Light Fog Road (Koschmieder beta=0.35)</option>
                <option value="real_medium_fog">Real Mining: Medium Fog Road (Koschmieder beta=0.75)</option>
                <option value="real_dense_fog">Real Mining: Dense Fog Road (Koschmieder beta=1.30)</option>
                <option value="real_quarry">Real Mining: Dusty Quarry Haul Truck</option>
                <option value="real_night">Real Mining: Night Haul Road (Headlights Active)</option>
            </optgroup>
            <optgroup label="Synthetic Testing (Group A)">
                <option value="synthetic_clear">Synthetic: Clear Ground Road</option>
                <option value="synthetic_light_fog">Synthetic: Light Fog Simulation</option>
                <option value="synthetic_dense_fog">Synthetic: Dense Fog Simulation</option>
                <option value="synthetic_night">Synthetic: Night Pit Road</option>
                <option value="synthetic_sample">Synthetic: Sample Pit Test Pattern</option>
            </optgroup>
        </select>
        <button onclick="document.getElementById('video-upload').click()" class="upload-btn">+ Upload Custom Video (.mp4)</button>
        <input type="file" id="video-upload" accept="video/mp4,video/*" style="display: none;" onchange="uploadCustomVideo(this)">

        <label style="font-size: 13px; color: var(--text-dim); margin-left: 12px;">Active Engine:</label>
        <select id="engine-select" onchange="changeEngine()">
            <option value="UnifiedMineGuardPerception" selected>Unified DepthTrack (Depth+BEV+Safety)</option>
            <option value="MineGuardIntegratedSystem">Original MineGuard Baseline (Fallback)</option>
        </select>
        <span id="engine-status-badge" class="engine-badge engine-badge-unified">Unified DepthTrack</span>

        <span id="upload-status" style="font-size: 12px; color: var(--accent-blue); margin-left: auto;">Ready</span>
    </div>

    <!-- Prominent Road Clearance Banner -->
    <div id="road-clearance-bar" class="road-clear">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span id="road-status-tag" style="padding: 4px 10px; border-radius: 4px; background: rgba(0,0,0,0.35);">ROAD CLEAR</span>
            <span id="road-banner-text">Path is unobstructed. No vehicles or hazards detected in roadway.</span>
        </div>
        <span id="banner-fps" style="font-size: 14px; opacity: 0.95;">FPS: --</span>
    </div>

    <div class="dashboard-grid">
        <!-- Live AI Feed -->
        <div class="card">
            <div class="card-title">Live AI Vision & Safety Stream</div>
            <div class="video-container">
                <img id="stream-img" src="/stream.mjpg" alt="MineGuard Live AI Feed">
            </div>
        </div>

        <!-- Telemetry & Road Status -->
        <div class="card">
            <div class="card-title">Atmospheric & Road Intelligence</div>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Road Clearance</div>
                    <div class="stat-val" id="metric-road-status" style="color: var(--safe-green);">Checking...</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Atmosphere / Fog</div>
                    <div class="stat-val" id="metric-fog">--</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Pipeline Mode</div>
                    <div class="stat-val" id="metric-mode" style="font-size: 15px;">--</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Pipeline Latency</div>
                    <div class="stat-val" id="metric-latency">-- ms</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Worker Frame ID</div>
                    <div class="stat-val" id="metric-frame-id" style="font-size: 15px;">--</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Active Obstacles</div>
                    <div class="stat-val" id="metric-obstacles-count" style="font-size: 15px;">0</div>
                </div>
            </div>

            <div class="card-title" style="margin-top: 14px;">Active Tracked Obstacles & Collision Proximity</div>
            <table>
                <thead>
                    <tr>
                        <th>Object / ID</th>
                        <th>Distance (rel)</th>
                        <th>Speed</th>
                        <th>TTC</th>
                        <th>Risk State</th>
                    </tr>
                </thead>
                <tbody id="vehicle-tbody">
                    <tr><td colspan="5" style="text-align: center; color: var(--text-dim);">Connecting to telemetry feed...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        async function updateTelemetry() {
            try {
                const res = await fetch('/api/telemetry?t=' + Date.now());
                if (!res.ok) return;
                const data = await res.json();

                // Road Clearance Banner
                const roadBar = document.getElementById('road-clearance-bar');
                const roadTag = document.getElementById('road-status-tag');
                const roadText = document.getElementById('road-banner-text');

                roadBar.className = 'road-' + (data.road_status_class || 'clear');
                roadTag.innerText = data.road_status || 'ROAD CLEAR';
                roadText.innerText = data.road_status_detail || '';

                // Stats Cards
                const statusBox = document.getElementById('metric-road-status');
                statusBox.innerText = data.road_status || '--';
                if (data.road_status_class === 'blocked') statusBox.style.color = 'var(--hazard-red)';
                else if (data.road_status_class === 'caution') statusBox.style.color = 'var(--caution-yellow)';
                else statusBox.style.color = 'var(--safe-green)';

                document.getElementById('metric-fog').innerText = (data.fog_condition !== undefined && data.fog_condition !== null) ? data.fog_condition : '--';
                document.getElementById('metric-mode').innerText = (data.pipeline_mode !== undefined && data.pipeline_mode !== null) ? data.pipeline_mode : '--';
                document.getElementById('metric-latency').innerText = (data.latency_ms !== undefined && data.latency_ms !== null) ? data.latency_ms.toFixed(1) + ' ms' : '--';
                document.getElementById('banner-fps').innerText = 'FPS: ' + ((data.fps !== undefined && data.fps !== null) ? data.fps.toFixed(1) : '--');
                document.getElementById('metric-frame-id').innerText = (data.worker_frame_id !== undefined && data.worker_frame_id !== null) ? '#' + data.worker_frame_id : '--';
                document.getElementById('metric-obstacles-count').innerText = (data.active_vehicles_count !== undefined && data.active_vehicles_count !== null) ? data.active_vehicles_count : '0';

                // Engine Badge Sync
                const badge = document.getElementById('engine-status-badge');
                if (data.active_engine === 'UnifiedMineGuardPerception') {
                    badge.className = 'engine-badge engine-badge-unified';
                    badge.innerText = 'Unified DepthTrack';
                } else if (data.active_engine === 'MineGuardIntegratedSystem') {
                    badge.className = 'engine-badge engine-badge-fallback';
                    badge.innerText = 'Fallback Baseline';
                } else {
                    badge.className = 'engine-badge';
                    badge.innerText = data.active_engine || 'UNKNOWN';
                }

                // Vehicle Obstacle Table
                const tbody = document.getElementById('vehicle-tbody');
                if (!data.vehicles || data.vehicles.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--safe-green);">Road is clear — No obstacles detected</td></tr>';
                } else {
                    let html = '';
                    for (const v of data.vehicles) {
                        const ttcText = (v.ttc_s !== null && v.ttc_s !== undefined) ? v.ttc_s + 's' : '--';
                        const distText = (v.distance_m !== null && v.distance_m !== undefined) ? v.distance_m + 'm' : '--';
                        const speedText = (v.speed_kmh !== null && v.speed_kmh !== undefined) ? v.speed_kmh + ' km/h' : '--';
                        const riskTier = v.risk_tier || 'UNKNOWN';
                        html += `<tr>
                            <td><b>${v.class_name ? v.class_name : 'VEHICLE'} #${v.track_id}</b></td>
                            <td>${distText}</td>
                            <td>${speedText}</td>
                            <td>${ttcText}</td>
                            <td><span class="tier-badge badge-${riskTier}">${riskTier}</span></td>
                        </tr>`;
                    }
                    tbody.innerHTML = html;
                }
            } catch (err) {
                console.error("Telemetry error:", err);
            }
        }

        async function changeVideoSource() {
            const select = document.getElementById('source-select');
            const val = select.value;
            await fetch('/api/set_source?source=' + encodeURIComponent(val));
            document.getElementById('upload-status').innerText = 'Switched to: ' + select.options[select.selectedIndex].text;
        }

        async function changeEngine() {
            const select = document.getElementById('engine-select');
            const val = select.value;
            const res = await fetch('/api/set_engine?engine=' + encodeURIComponent(val));
            const data = await res.json();
            if (data.success) {
                document.getElementById('upload-status').innerText = 'Active Engine: ' + data.active_engine;
            } else {
                document.getElementById('upload-status').innerText = 'Engine switch failed: ' + (data.error || 'Unknown');
            }
        }

        function uploadCustomVideo(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const status = document.getElementById('upload-status');
            status.innerText = "Uploading (" + (file.size / (1024*1024)).toFixed(1) + " MB)...";

            const xhr = new XMLHttpRequest();
            xhr.open("POST", "/api/upload_video?filename=" + encodeURIComponent(file.name), true);

            xhr.upload.onprogress = function(e) {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    status.innerText = "Uploading: " + percent + "%";
                }
            };

            xhr.onload = function() {
                try {
                    const json = JSON.parse(xhr.responseText);
                    if (xhr.status === 200 && json.success) {
                        status.innerText = "Success: Playing " + json.filename;
                        const select = document.getElementById('source-select');
                        const opt = document.createElement("option");
                        opt.value = json.source_key;
                        opt.innerText = "Uploaded: " + json.filename;
                        opt.selected = true;
                        select.appendChild(opt);
                    } else {
                        status.innerText = "Upload failed: " + (json.error || xhr.statusText);
                    }
                } catch (e) {
                    status.innerText = "Upload failed (Status " + xhr.status + ")";
                }
            };

            xhr.onerror = function() {
                status.innerText = "Upload network error.";
            };

            xhr.send(file);
        }

        // Poll telemetry every 300ms
        updateTelemetry();
        setInterval(updateTelemetry, 300);
    </script>
</body>
</html>
"""


class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles web requests, telemetry, engine selection, MJPEG stream, and uploads concurrently."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))

        elif parsed.path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if worker is not None:
                _, telemetry = worker.get_latest_data()
            else:
                telemetry = {"system_status": "OFFLINE", "active_engine": "NONE"}

            self.wfile.write(json.dumps(telemetry).encode("utf-8"))

        elif parsed.path == "/api/set_source":
            query = urllib.parse.parse_qs(parsed.query)
            src_key = query.get("source", [DEFAULT_SOURCE])[0]
            global current_source_key, current_source_path

            success = False
            if src_key in AVAILABLE_VIDEOS:
                current_source_key = src_key
                current_source_path = AVAILABLE_VIDEOS[src_key]
                if worker is not None:
                    worker.switch_source(current_source_path)
                success = True

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": success, "source": current_source_key}).encode("utf-8"))

        elif parsed.path == "/api/set_engine":
            query = urllib.parse.parse_qs(parsed.query)
            engine_req = query.get("engine", ["UnifiedMineGuardPerception"])[0]

            success = False
            err_msg = None
            if adapter is not None:
                success = adapter.set_engine(engine_req)
                if not success:
                    err_msg = adapter.initialization_error

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": success,
                "active_engine": adapter.active_engine_name if adapter else "NONE",
                "is_fallback": adapter.is_fallback_active if adapter else True,
                "error": err_msg
            }).encode("utf-8"))

        elif parsed.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            try:
                last_frame_bytes = None
                while server_running:
                    if worker is not None:
                        frame_bytes, _ = worker.get_latest_data()
                        if frame_bytes is not None and frame_bytes != last_frame_bytes:
                            self.wfile.write(b"--FRAME\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(frame_bytes)))
                            self.end_headers()
                            self.wfile.write(frame_bytes)
                            self.wfile.write(b"\r\n")
                            last_frame_bytes = frame_bytes

                    time.sleep(0.033)  # ~30 FPS polling
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/upload_video":
            content_length_str = self.headers.get("Content-Length")
            if not content_length_str:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Missing Content-Length"}).encode("utf-8"))
                return

            try:
                content_length = int(content_length_str)
            except ValueError:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Invalid Content-Length"}).encode("utf-8"))
                return

            if content_length <= 0:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Uploaded video file is empty (0 bytes)"}).encode("utf-8"))
                return

            if content_length > MAX_UPLOAD_SIZE:
                self.send_response(413)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": f"File exceeds maximum allowed limit of {MAX_UPLOAD_SIZE // (1024*1024)} MB"}).encode("utf-8"))
                return

            # Sanitize filename
            query = urllib.parse.parse_qs(parsed.query)
            raw_filename = query.get("filename", ["video.mp4"])[0]
            clean_name = re.sub(r"[^a-zA-Z0-9_\.-]", "_", Path(raw_filename).name)
            if not clean_name.lower().endswith(".mp4"):
                clean_name = f"{Path(clean_name).stem}.mp4"
            if not clean_name or clean_name == ".mp4":
                clean_name = "user_video.mp4"

            timestamp_str = int(time.time())
            save_name = f"{Path(clean_name).stem}_{timestamp_str}.mp4"
            save_path = UPLOAD_DIR / save_name

            # Stream-read payload safely
            chunk_size = 1024 * 1024
            bytes_left = content_length

            try:
                with open(save_path, "wb") as f_out:
                    while bytes_left > 0:
                        read_len = min(chunk_size, bytes_left)
                        chunk = self.rfile.read(read_len)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        bytes_left -= len(chunk)
            except Exception as e:
                if save_path.exists():
                    save_path.unlink()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": f"Write failed: {str(e)}"}).encode("utf-8"))
                return

            # Validate uploaded file
            if not save_path.exists() or save_path.stat().st_size == 0:
                if save_path.exists():
                    save_path.unlink()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Uploaded video file is empty"}).encode("utf-8"))
                return

            # OpenCV VideoCapture validation
            cap_test = cv2.VideoCapture(str(save_path))
            is_valid = False
            if cap_test.isOpened():
                ret_test, frame_test = cap_test.read()
                if ret_test and frame_test is not None and frame_test.size > 0:
                    is_valid = True
            cap_test.release()

            if not is_valid:
                save_path.unlink()
                self.send_response(422)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Uploaded file is not a valid or decodable MP4 video"}).encode("utf-8"))
                return

            global current_source_key, current_source_path
            source_key = f"upload_{timestamp_str}"
            AVAILABLE_VIDEOS[source_key] = str(save_path)
            current_source_key = source_key
            current_source_path = str(save_path)

            if worker is not None:
                worker.switch_source(current_source_path)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": True,
                "source_key": source_key,
                "filename": clean_name
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def start_dashboard_server(host="0.0.0.0", port=8080):
    """Initializes adapter, background worker, and ThreadingHTTPServer."""
    global adapter, worker, server_instance, server_running

    logger.info("Initializing PerceptionAdapter for MineGuard Dashboard...")
    adapter = PerceptionAdapter(preferred_engine=PerceptionAdapter.ENGINE_UNIFIED)

    logger.info("Starting BackgroundVideoWorker on %s...", current_source_path)
    worker = BackgroundVideoWorker(adapter, current_source_path)
    worker.start()

    server_instance = ThreadingHTTPServer((host, port), DashboardHTTPRequestHandler)
    print("\n" + "=" * 65)
    print("MINEGUARD AI — INTEGRATED ROAD CLEARANCE DASHBOARD ONLINE")
    print(f"Local Access: http://127.0.0.1:{port}")
    print(f"Active Engine: {adapter.active_engine_name}")
    print("=" * 65 + "\n")

    try:
        server_instance.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        server_running = False
        if worker is not None:
            worker.stop()
        server_instance.server_close()


if __name__ == "__main__":
    start_dashboard_server()
