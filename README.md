# 🛡️ MineGuard AI (Mineguard_project_2)
### Intelligent Mining Vehicle Safety, Monocular Depth Perception & Real-Time Proximity Alert System

MineGuard AI is a multi-modal perception and collision-avoidance system designed for heavy vehicles operating in open-pit mines, quarries, and hazardous haul roads. It combines **YOLOv8** object detection, **ByteTrack** multi-object tracking, and **Depth Anything V2** monocular depth estimation with adaptive environmental filters (CLAHE dehazing and low-light enhancement).

---

## 📂 Directory Structure Overview

```text
D:\Mineguard_project_2\
│
├── src/                                  # Core Python application modules
│   └── mineguard_depthtrack/
│       ├── __init__.py
│       ├── dashboard.py                  # Threaded HTTP & MJPEG live streaming dashboard server
│       ├── service.py                    # PerceptionAdapter & BackgroundVideoWorker streaming pipeline
│       ├── unified_engine.py             # UnifiedMineGuardPerception engine (detection + depth + safety rules)
│       ├── dataset.py                    # MineGuardDatasetManager cataloging scenarios & validating annotations
│       └── evaluation.py                 # Ground-truth detection & tracking evaluation engine
│
├── DepthTrack/                           # DepthTrack perception library & models
│   ├── assets/                           # Model weights (yolov8n.pt, yolo11n.pt, etc.)
│   ├── config/                           # Default YAML configuration (default.yaml)
│   ├── src/depthtrack/                   # Depth, tracking, and renderer modules
│   │   ├── config.py                     # Pipeline dataclass configurations (confidence, classes, etc.)
│   │   ├── models/                       # Detector, DepthEstimator, ObjectTracker wrappers
│   │   ├── perception/                   # Pinhole & monocular distance estimation
│   │   └── visualization/                # 3D Bird's-Eye View (BEV) and composite rendering
│   └── data/
│       ├── scenarios/                    # Baseline test videos (mine_clear, mine_dense_fog, etc.)
│       └── user_uploads/                 # Uploaded custom video MP4 files
│
├── data/                                 # Mining scenarios and ground-truth validation datasets
│   ├── scenarios/
│   │   ├── real_mining/                  # Authentic haul truck videos (Clear, Light Fog, Medium Fog, Dense Fog, Quarry, Night)
│   │   └── synthetic/                    # Geometric and simulated weather test sequences
│   └── ground_truth/                     # 100 CVAT-annotated keyframes across 4 visibility conditions
│       ├── annotation_manifest.csv       # Verification manifest mapping frames to conditions
│       └── labels/train/                 # YOLO bounding box label files
│
├── reports/                              # Engineering audits, benchmark reports & performance evidence
│   ├── PHASE_5_3_IMPLEMENTATION_REPORT.md# Comprehensive Phase 5.3 dataset & integration report
│   ├── PHASE_5_INTEGRATION_REPORT.md     # Concurrency, adapter, and telemetry integration audit
│   ├── phase5_3/                         # Detection & tracking evaluation breakdowns and screenshots
│   │   ├── detection_evaluation_report.md# Precision, recall, and IoU metrics
│   │   ├── tracking_evaluation_report.md # ID stability and continuity analysis
│   │   └── evidence/                     # Captured dashboard composite frames & telemetry JSONs
│   └── ...                               # Historic benchmark reports (Phase 1 through Phase 4.5)
│
├── tests/                                # Automated pytest regression and integration test suites
│   ├── test_dataset_evaluation.py        # Dataset manager & ground-truth validation tests
│   ├── test_dashboard_concurrency_upload.py # Concurrency, streaming, and upload repair tests
│   ├── test_dashboard_integration.py     # Dashboard adapter & telemetry schema unit tests
│   └── test_regression_audit.py          # Regression tests for models and pipeline
│
├── tools/                                # Automation scripts, benchmarks & inspection utilities
│   ├── test_rules.py                     # Script verifying the 4 directional lane safety rules
│   ├── run_phase5_live_validation.py     # End-to-end dashboard automation validator
│   ├── benchmark_detection.py            # YOLO detector benchmark runner
│   └── benchmark_pipeline.py             # Full pipeline FPS and latency benchmark
│
├── backups/                              # Pre-modification backups for safety and rollback
├── .gitignore                            # Excludes caches, large MP4s, and temporary files
└── README.md                             # This documentation file
```

---

## 🚀 Quick Start: How to Run the Dashboard

1. **Open PowerShell in the project directory:**
   ```powershell
   cd D:\Mineguard_project_2
   ```

2. **Start the Dashboard Server:**
   ```powershell
   .\DepthTrack\.venv\Scripts\python.exe -u .\src\mineguard_depthtrack\dashboard.py
   ```

3. **Open the Web Browser:**
   Navigate to:
   👉 **[http://127.0.0.1:8080](http://127.0.0.1:8080)** (or `http://localhost:8080`)

4. **Stop the Server:**
   Press `Ctrl + C` in the terminal.

---

## 🎯 Key Features & Capabilities

### 1. Dual-View Dashboard
* **Left Screen (Live AI Vision):** Annotated video displaying bounding boxes, vehicle class, confidence, distance, relative velocity, and dynamic hazard states.
* **Right Screen (3D Bird's-Eye View):** Real-time spatial radar grid mapping forward distance and relative depth from a single 2D camera.

### 2. Visibility & Environmental Adaptation
* **CLEAR_BYPASS:** Standard daylight processing.
* **FOG_ADAPTIVE (CLAHE):** Contrast-limited adaptive histogram equalization to cut through mine dust and fog.
* **NIGHT_CLAHE:** Low-light boost for night haul road operations.

### 3. Obstacle Failsafe & Unusual Object Detection
* Automatically analyzes high-frequency gradient anomalies in the driving corridor.
* Flags faint, distant, or unusual road obstacles as **`UNUSUAL OBJECT / OBSTACLE`** even when standard YOLO feature maps cannot classify them.

### 4. 4-Tier Directional & Lane-Aware Safety Rules
* **Rule 1 (Moving Forward):** Green box with `[MOVING FORWARD (SAFE)]` and `"ROAD CLEAR"`.
* **Rule 2 (Adjacent Lane Approach):** Amber box with `[WARNING: ADJACENT APPROACH]` and `"PROXIMITY WARNING"`.
* **Rule 3 (Same Lane Head-On Approach):** Red box with `[DANGER: HEAD-ON APPROACH]` and `"ROAD BLOCKED / DANGER"`.
* **Rule 4 (Vehicle Stopped Ahead):** Red/Amber box with `[VEHICLE STOPPED IN LANE]` and `"ROAD BLOCKED / HAZARD"`.

---

## 🧪 Running Tests

To run the automated test suite with the project's virtual environment:

```powershell
$env:PYTHONPATH="src;DepthTrack\src"
.\DepthTrack\.venv\Scripts\python.exe -m pytest tests/test_dataset_evaluation.py tests/test_dashboard_integration.py -v
```

---

## 📊 Git Repository Information
* **Remote Repository:** [https://github.com/Manoj4143/mineguard.git](https://github.com/Manoj4143/mineguard.git)
* **Working Branch:** `phase5-dashboard-integration`
