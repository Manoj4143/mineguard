"""MineGuard AI - Interactive Web-Based Monitoring Dashboard.

Integrated with UnifiedMineGuardPerception perception engine,
ThreadingHTTPServer concurrency, validated streaming uploads,
and fallback engine support.

Features:
1. Mine Fleet Coordination (Feature 3)
2. Mine Road Condition Intelligence (Feature 2)
3. Automatic Incident Recorder (Feature 1)
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
from mineguard_depthtrack.fleet_models import TelemetryMessage

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
            --purple: #bc8cff;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg-base); color: var(--text-main); padding: 18px; line-height: 1.4; }
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
        select, button, input { background: #161b22; color: #fff; border: 1px solid var(--border); padding: 8px 14px; border-radius: 6px; font-size: 13px; cursor: pointer; }
        button:hover { background: #1f242c; border-color: var(--accent-blue); }
        .upload-btn { background: #238636; border-color: #2ea043; font-weight: bold; }
        .engine-badge { font-weight: bold; padding: 4px 10px; border-radius: 4px; font-size: 12px; border: 1px solid var(--border); }
        .engine-badge-unified { background: rgba(31, 111, 235, 0.2); color: var(--accent-blue); border-color: var(--accent-blue); }
        .engine-badge-fallback { background: rgba(210, 153, 34, 0.2); color: var(--caution-yellow); border-color: var(--caution-yellow); }

        /* Primary Live Grid */
        .dashboard-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 18px; margin-bottom: 24px; }
        .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
        .card-title { font-size: 14px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }

        /* Expanded Live Video Container */
        .video-container { position: relative; border-radius: 8px; overflow: hidden; background: #000; border: 1px solid var(--border); min-height: 480px; width: 100%; display: flex; align-items: center; justify-content: center; }
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
        .badge-SIM { background: rgba(188, 140, 255, 0.2); color: var(--purple); border: 1px solid rgba(188, 140, 255, 0.4); }

        /* Full-Width Stacked Section Layout */
        .feature-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 24px;
        }
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
            margin-bottom: 16px;
        }
        .section-title {
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: var(--accent-blue);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-subtitle {
            font-size: 12px;
            color: var(--text-dim);
            font-weight: normal;
        }

        .summary-card-row {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 14px;
            margin-bottom: 18px;
        }

        /* Evidence Preview Frame */
        .evidence-preview-container {
            display: flex;
            gap: 18px;
            align-items: flex-start;
        }
        .evidence-img-box {
            width: 320px;
            min-height: 180px;
            background: #000;
            border: 1px solid var(--border);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .evidence-img-box img {
            width: 100%;
            height: auto;
            display: block;
        }

        /* Video Modal Player */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.85);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal-content {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            max-width: 800px;
            width: 90%;
        }
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

    <!-- EXPANDED LIVE AI VISION STREAM & TELEMETRY GRID -->
    <div class="dashboard-grid">
        <div class="card">
            <div class="card-title">
                <span>Live AI Vision & Safety Stream (Expanded Field of View)</span>
                <span style="font-size: 11px; color: var(--safe-green); font-weight: normal;">MONOCULAR DEPTH FUSED</span>
            </div>
            <div class="video-container">
                <img id="stream-img" src="/stream.mjpg" alt="MineGuard Live AI Feed">
            </div>
        </div>

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

    <!-- FEATURE 2: MINE ROAD CONDITION INTELLIGENCE -->
    <section class="feature-section" id="section-road-intelligence">
        <div class="section-header">
            <div>
                <div class="section-title">
                    <span>MINE ROAD CONDITION INTELLIGENCE</span>
                    <span class="tier-badge" id="road-model-badge" style="background: rgba(88,166,255,0.2); color: var(--accent-blue);">CORRIDOR TEXTURE ANALYZER</span>
                </div>
                <div class="section-subtitle">Traversable Corridor Segmentation, Potholes, Rocks, Loose Debris, Water & Road Edge Analysis</div>
            </div>
            <div id="road-model-status-text" style="font-size: 12px; color: var(--text-dim);">Status: Honest readiness reporting</div>
        </div>

        <div class="summary-card-row">
            <div class="stat-box">
                <div class="stat-label">Active Hazards</div>
                <div class="stat-val" id="road-active-hazards" style="color: var(--caution-yellow);">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Critical / Blockages</div>
                <div class="stat-val" id="road-critical-hazards" style="color: var(--hazard-red);">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Resolved Hazards</div>
                <div class="stat-val" id="road-resolved-hazards" style="color: var(--safe-green);">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Logged</div>
                <div class="stat-val" id="road-total-hazards">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Corridor Status</div>
                <div class="stat-val" id="road-corridor-status" style="font-size: 15px; color: var(--safe-green);">CLEAR</div>
            </div>
        </div>

        <div class="evidence-preview-container">
            <div style="flex: 2;">
                <div class="card-title">Tracked Road Surface Hazards & Corridors</div>
                <table>
                    <thead>
                        <tr>
                            <th>Hazard ID</th>
                            <th>Category</th>
                            <th>Severity</th>
                            <th>Confidence</th>
                            <th>Status</th>
                            <th>Reported By</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="road-hazards-tbody">
                        <tr><td colspan="7" style="text-align: center; color: var(--safe-green);">Road traversable corridor is clear of surface hazards</td></tr>
                    </tbody>
                </table>
            </div>
            <div style="flex: 1;">
                <div class="card-title">Evidence Snapshot Viewer</div>
                <div class="evidence-img-box">
                    <img id="evidence-preview-img" src="" alt="Select hazard to view evidence" style="display: none;">
                    <span id="evidence-placeholder" style="font-size: 12px; color: var(--text-dim);">No hazard selected</span>
                </div>
                <div id="evidence-caption" style="font-size: 12px; color: var(--text-dim); margin-top: 6px;"></div>
            </div>
        </div>
    </section>

    <!-- FEATURE 1: AUTOMATIC INCIDENT RECORDER -->
    <section class="feature-section" id="section-incident-recorder">
        <div class="section-header">
            <div>
                <div class="section-title">
                    <span>AUTOMATIC INCIDENT RECORDER</span>
                    <span class="tier-badge badge-SAFE" style="font-size: 11px;">EVIDENCE SNAPSHOT VAULT ONLINE</span>
                </div>
                <div class="section-subtitle">Automated Triggering on Collision Risks, Hard Near-Misses, Instantaneous Evidence Snapshots & Audit Vault</div>
            </div>
            <div style="font-size: 12px; color: var(--text-dim);" id="recorder-disk-status">Storage: data/incidents</div>
        </div>

        <div class="summary-card-row">
            <div class="stat-box">
                <div class="stat-label">Total Incidents</div>
                <div class="stat-val" id="inc-total-count">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Critical Collision Risks</div>
                <div class="stat-val" id="inc-critical-count" style="color: var(--hazard-red);">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Near-Miss Events</div>
                <div class="stat-val" id="inc-near-miss-count" style="color: var(--caution-yellow);">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Road Obstructions</div>
                <div class="stat-val" id="inc-hazard-count">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Today's Incidents</div>
                <div class="stat-val" id="inc-today-count" style="color: var(--accent-blue);">0</div>
            </div>
        </div>

        <div class="card-title">Recorded Incident Event Log & Evidence Snapshots</div>
        <table>
            <thead>
                <tr>
                    <th>Incident ID</th>
                    <th>Event Type</th>
                    <th>Severity</th>
                    <th>Time</th>
                    <th>Vehicle</th>
                    <th>TTC / Dist</th>
                    <th>Description</th>
                    <th>Incident Snapshot</th>
                </tr>
            </thead>
            <tbody id="incidents-tbody">
                <tr><td colspan="8" style="text-align: center; color: var(--safe-green);">No incident events recorded</td></tr>
            </tbody>
        </table>
    </section>

    <!-- Incident Snapshot Modal -->
    <div class="modal-overlay" id="snapshot-modal">
        <div class="modal-content" style="max-width: 820px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 id="snapshot-modal-title" style="font-size: 16px;">Incident Snapshot Review</h3>
                <button onclick="closeSnapshotModal()" style="padding: 4px 10px;">Close ✕</button>
            </div>
            <div style="background: #000; border-radius: 6px; overflow: hidden; display: flex; justify-content: center; align-items: center; min-height: 380px;">
                <img id="snapshot-modal-img" src="" alt="Incident Snapshot" style="max-width: 100%; max-height: 520px; object-fit: contain;">
            </div>
            <div id="snapshot-modal-meta" style="font-size: 12px; color: var(--text-dim); margin-top: 10px; line-height: 1.5;"></div>
        </div>
    </div>

    <script>
        // Telemetry Update Loop
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

        // Feature 2: Road Condition Poller
        async function updateRoadHazards() {
            try {
                const [summaryRes, hazardsRes] = await Promise.all([
                    fetch('/api/road/summary?t=' + Date.now()),
                    fetch('/api/road/hazards?t=' + Date.now())
                ]);

                if (summaryRes.ok) {
                    const s = await summaryRes.json();
                    document.getElementById('road-active-hazards').innerText = s.active_hazards;
                    document.getElementById('road-critical-hazards').innerText = s.critical_hazards;
                    document.getElementById('road-resolved-hazards').innerText = s.resolved_hazards;
                    document.getElementById('road-total-hazards').innerText = s.total_hazards_logged;
                    document.getElementById('road-model-status-text').innerText = s.model_status;
                }

                if (hazardsRes.ok) {
                    const hazards = await hazardsRes.json();
                    const tbody = document.getElementById('road-hazards-tbody');
                    if (!hazards || hazards.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--safe-green);">Road traversable corridor is clear of surface hazards</td></tr>';
                    } else {
                        let html = '';
                        for (const h of hazards) {
                            const sevClass = h.severity === 'CRITICAL' ? 'badge-CRITICAL' : 'badge-CAUTION';
                            const evParam = encodeURIComponent(h.evidence_image_path || '');
                            html += `<tr>
                                <td><b>${h.hazard_id}</b></td>
                                <td>${h.category}</td>
                                <td><span class="tier-badge ${sevClass}">${h.severity}</span></td>
                                <td>${(h.confidence * 100).toFixed(0)}%</td>
                                <td>${h.status}</td>
                                <td>${h.source_vehicle_id}</td>
                                <td><button onclick="previewEvidence('${h.hazard_id}', '${evParam}', '${h.category}')" style="padding: 4px 8px; font-size: 11px;">View Snapshot</button></td>
                            </tr>`;
                        }
                        tbody.innerHTML = html;
                    }
                }
            } catch (e) {
                console.error("Road intelligence error:", e);
            }
        }

        function previewEvidence(id, encodedPath, category) {
            const path = decodeURIComponent(encodedPath);
            const img = document.getElementById('evidence-preview-img');
            const placeholder = document.getElementById('evidence-placeholder');
            const caption = document.getElementById('evidence-caption');

            if (!path || path === 'None' || path === 'null') {
                img.style.display = 'none';
                placeholder.style.display = 'block';
                placeholder.innerText = 'No evidence image stored for ' + id;
                caption.innerText = '';
            } else {
                img.src = '/api/road/evidence?path=' + encodeURIComponent(path);
                img.style.display = 'block';
                placeholder.style.display = 'none';
                caption.innerText = `${id} — ${category}`;
            }
        }

        // Feature 1: Automatic Incident Recorder Poller
        async function updateIncidents() {
            try {
                const [sumRes, listRes] = await Promise.all([
                    fetch('/api/incidents/summary?t=' + Date.now()),
                    fetch('/api/incidents/list?t=' + Date.now())
                ]);

                if (sumRes.ok) {
                    const s = await sumRes.json();
                    document.getElementById('inc-total-count').innerText = s.total_incidents;
                    document.getElementById('inc-critical-count').innerText = s.critical_incidents;
                    document.getElementById('inc-near-miss-count').innerText = s.near_misses;
                    document.getElementById('inc-hazard-count').innerText = s.road_hazard_incidents;
                    document.getElementById('inc-today-count').innerText = s.today_incidents;
                }

                if (listRes.ok) {
                    const list = await listRes.json();
                    const tbody = document.getElementById('incidents-tbody');
                    if (!list || list.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--safe-green);">No incident events recorded</td></tr>';
                    } else {
                        let html = '';
                        for (const inc of list) {
                            const sevClass = inc.severity === 'CRITICAL' ? 'badge-CRITICAL' : 'badge-CAUTION';
                            const timeStr = new Date(inc.timestamp * 1000).toLocaleTimeString();
                            const ttcDist = `${inc.ttc_seconds ? inc.ttc_seconds.toFixed(1) + 's' : '--'} / ${inc.distance_m ? inc.distance_m.toFixed(1) + 'm' : '--'}`;
                            html += `<tr>
                                <td><b>${inc.incident_id}</b></td>
                                <td>${inc.event_type}</td>
                                <td><span class="tier-badge ${sevClass}">${inc.severity}</span></td>
                                <td>${timeStr}</td>
                                <td>${inc.vehicle_id}</td>
                                <td>${ttcDist}</td>
                                <td>${inc.description || '--'}</td>
                                <td><button onclick="previewIncidentSnapshot('${inc.incident_id}', '${encodeURIComponent(inc.video_path || '')}', '${inc.event_type}', '${inc.description || ''}')" style="padding: 4px 10px; font-size: 11px; background: #238636; border-color: #2ea043; font-weight: bold;">📷 View Snapshot</button></td>
                            </tr>`;
                        }
                        tbody.innerHTML = html;
                    }
                }
            } catch (e) {
                console.error("Incident recorder error:", e);
            }
        }

        function previewIncidentSnapshot(incId, encodedPath, eventType, description) {
            const path = decodeURIComponent(encodedPath);
            const modal = document.getElementById('snapshot-modal');
            const img = document.getElementById('snapshot-modal-img');
            const title = document.getElementById('snapshot-modal-title');
            const meta = document.getElementById('snapshot-modal-meta');

            title.innerText = "Incident Evidence Snapshot — " + incId;
            img.src = "/api/incidents/snapshot?id=" + encodeURIComponent(incId) + "&path=" + encodeURIComponent(path);
            meta.innerHTML = `<b>Event:</b> ${eventType} | <b>Incident Vault ID:</b> ${incId}<br><b>Details:</b> ${description || 'Automated safety trigger snapshot captured.'}`;
            modal.style.display = 'flex';
        }

        function closeSnapshotModal() {
            const img = document.getElementById('snapshot-modal-img');
            img.src = "";
            document.getElementById('snapshot-modal').style.display = 'none';
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

        // Pollers: Live stream 400ms, Road 2500ms, Incidents 3500ms
        updateTelemetry();
        setInterval(updateTelemetry, 400);

        updateRoadHazards();
        setInterval(updateRoadHazards, 2500);

        updateIncidents();
        setInterval(updateIncidents, 3500);
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

        # --- FEATURE 3: FLEET COORDINATION ENDPOINTS ---
        elif parsed.path == "/api/fleet/summary":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = adapter.fleet_manager.get_fleet_summary() if adapter and hasattr(adapter, "fleet_manager") else {}
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif parsed.path == "/api/fleet/vehicles":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = adapter.fleet_manager.get_all_vehicles() if adapter and hasattr(adapter, "fleet_manager") else []
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif parsed.path == "/api/fleet/alerts":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = adapter.fleet_manager.get_active_alerts() if adapter and hasattr(adapter, "fleet_manager") else []
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif parsed.path == "/api/fleet/hazards":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = adapter.fleet_manager.get_shared_hazards() if adapter and hasattr(adapter, "fleet_manager") else []
            self.wfile.write(json.dumps(data).encode("utf-8"))

        # --- FEATURE 2: ROAD CONDITION INTELLIGENCE ENDPOINTS ---
        elif parsed.path == "/api/road/summary":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = adapter.road_intelligence.get_summary() if adapter and hasattr(adapter, "road_intelligence") else {}
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif parsed.path == "/api/road/hazards":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = adapter.road_intelligence.get_all_hazards() if adapter and hasattr(adapter, "road_intelligence") else []
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif parsed.path == "/api/road/evidence":
            query = urllib.parse.parse_qs(parsed.query)
            ev_path = query.get("path", [""])[0]
            target_p = Path(ev_path)
            if target_p.exists() and target_p.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(target_p, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()

        # --- FEATURE 1: AUTOMATIC INCIDENT RECORDER ENDPOINTS ---
        elif parsed.path == "/api/incidents/summary":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = adapter.incident_recorder.get_summary() if adapter and hasattr(adapter, "incident_recorder") else {}
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif parsed.path == "/api/incidents/list":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = adapter.incident_recorder.get_all_incidents() if adapter and hasattr(adapter, "incident_recorder") else []
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif parsed.path == "/api/incidents/snapshot":
            query = urllib.parse.parse_qs(parsed.query)
            inc_id = query.get("id", [""])[0]
            req_path = query.get("path", [""])[0]
            target_file = None

            if req_path and Path(req_path).exists() and Path(req_path).is_file():
                p = Path(req_path)
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    target_file = p

            if target_file is None and adapter and hasattr(adapter, "incident_recorder"):
                incidents = adapter.incident_recorder.get_all_incidents()
                for inc in incidents:
                    if inc.get("incident_id") == inc_id:
                        p = Path(inc.get("video_path", ""))
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.exists():
                            target_file = p
                            break
                        # Check companion snapshot next to mp4
                        snap_candidate = p.parent / f"{inc_id}_snap.jpg"
                        if snap_candidate.exists():
                            target_file = snap_candidate
                            break

            if target_file and target_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(target_file.stat().st_size))
                self.end_headers()
                with open(target_file, "rb") as sf:
                    self.wfile.write(sf.read())
            else:
                self.send_response(404)
                self.end_headers()

        elif parsed.path == "/api/incidents/video":
            query = urllib.parse.parse_qs(parsed.query)
            inc_id = query.get("id", [""])[0]
            video_file = None
            if adapter and hasattr(adapter, "incident_recorder"):
                incidents = adapter.incident_recorder.get_all_incidents()
                for inc in incidents:
                    if inc.get("incident_id") == inc_id:
                        p = Path(inc.get("video_path", ""))
                        if p.exists() and p.is_file():
                            video_file = p
                            break

            if video_file:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(video_file.stat().st_size))
                self.end_headers()
                with open(video_file, "rb") as vf:
                    self.wfile.write(vf.read())
            else:
                self.send_response(404)
                self.end_headers()

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

                    time.sleep(0.030)  # Smooth 30+ FPS transmission
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/fleet/register":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
                v_id = payload.get("vehicle_id")
                v_name = payload.get("vehicle_name", v_id)
                v_type = payload.get("vehicle_type", "HAUL_TRUCK")
                is_sim = payload.get("is_simulated", False)
                if adapter and hasattr(adapter, "fleet_manager"):
                    success, msg = adapter.fleet_manager.register_vehicle(v_id, v_name, v_type, is_sim)
                else:
                    success, msg = False, "Fleet manager unavailable"
                self.send_response(200 if success else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": success, "message": msg}).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))

        elif parsed.path == "/api/fleet/telemetry":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
                telem = TelemetryMessage(
                    vehicle_id=payload.get("vehicle_id", ""),
                    timestamp=payload.get("timestamp", time.time()),
                    latitude=payload.get("latitude"),
                    longitude=payload.get("longitude"),
                    speed_kmh=payload.get("speed_kmh"),
                    heading_deg=payload.get("heading_deg"),
                    camera_status=payload.get("camera_status", "ONLINE"),
                    perception_status=payload.get("perception_status", "ACTIVE"),
                    is_simulated=payload.get("is_simulated", False)
                )
                if adapter and hasattr(adapter, "fleet_manager"):
                    success, msg = adapter.fleet_manager.ingest_telemetry(telem)
                else:
                    success, msg = False, "Fleet manager unavailable"
                self.send_response(200 if success else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": success, "message": msg}).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))

        elif parsed.path == "/api/upload_video":
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
    print("MINEGUARD AI — ADVANCED MINING SAFETY PLATFORM ONLINE")
    print(f"Local Access: http://127.0.0.1:{port}")
    print(f"Active Engine: {adapter.active_engine_name}")
    print("Feature 3: Mine Fleet Coordination [ACTIVE]")
    print("Feature 2: Mine Road Condition Intelligence [ACTIVE]")
    print("Feature 1: Automatic Incident Recorder [ACTIVE]")
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
