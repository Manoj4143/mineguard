# PHASE 1 CODEBASE AUDIT REPORT: DEPTH TRACK

**Date:** 2026-11-12  
**Auditor:** Senior Computer Vision & AI/ML Engineer  
**System:** Windows 11 | NVIDIA RTX 3050 6GB Laptop GPU | Python 3.13.1 | PyTorch 2.11.0+cu128  

---

## 1. Executive Summary

DepthTrack (`depthtrack==1.2.0`) is a modular perception library designed to process vehicle dashcam video streams. It performs multi-class vehicle detection, bounding-box tracking, monocular relative depth estimation, inverse-perspective Bird's-Eye-View (BEV) mapping, and collision-risk tiering.

The system is organized into decoupled layers:
- Configuration (`depthtrack.config`)
- Pipeline orchestration (`depthtrack.pipeline`)
- AI Model abstraction (`depthtrack.models`)
- Perception & physics estimation (`depthtrack.perception`)
- UI & multi-panel visualization (`depthtrack.visualization`)

---

## 2. Architecture & Component Decomposition

### 2.1 Configuration Layer (`depthtrack.config`)
- Implemented using Python `dataclasses` with YAML deserialization (`default.yaml`).
- Config sub-trees:
  - `ModelConfig`: weights path, model IDs, confidence/IoU thresholds, tracking history, device selection (`cpu`/`cuda`).
  - `DistanceConfig`: near/far depth planes (default 2.0m to 80.0m), pinhole focal length (default 700.0 px), blend weight (`depth_weight: 0.6`), class-specific vehicle width lookup (Cars: 1.8m, Trucks: 2.5m, Buses: 2.5m, Motorcycles: 0.8m).
  - `RiskConfig`: distance thresholds (Danger: < 10m, Warning: < 20m, Caution: < 35m) and lateral lane threshold (`danger_lateral_offset: 0.35`).
  - `BEVConfig`: grid width/height, range (50m forward, 15m lateral), ego-vehicle rendering parameters.
  - `VideoConfig`: work resolution, depth model input dimensions, codec/bitrate settings.

### 2.2 Model Layer (`depthtrack.models`)
- **Detector (`VehicleDetector`)**:
  - Wraps Ultralytics YOLO (`YOLO('yolov8n.pt')`).
  - Filters COCO vehicle classes: `{2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}`.
  - Converts outputs to `supervision.Detections`.
- **Depth Estimator (`DepthEstimator`)**:
  - Integrates Hugging Face `transformers.AutoModelForDepthEstimation` using checkpoint `depth-anything/Depth-Anything-V2-Small-hf`.
  - Normalizes output inverse depth maps to $[0, 1]$ float32.
- **Tracker (`ObjectTracker`)**:
  - Encapsulates `supervision.ByteTrack`.
  - Maintains a bounded trajectory deque (`max_history=30`) and computes finite-difference 2D pixel velocities for motion arrows.

### 2.3 Perception Layer (`depthtrack.perception`)
- **Distance Estimation (`estimate_distance`)**:
  - Combines two monocular cues:
    1. *Relative Depth Cue*: extracts the median depth value within the bounding box and maps $[0, 1]$ linearly to $[\text{depth\_near\_m}, \text{depth\_far\_m}]$.
    2. *Pinhole Projection Cue*: calculates $Z = \frac{f \cdot W_{\text{real}}}{W_{\text{pixel}}}$.
  - Fuses the two using weight $\alpha = 0.6$: $Z_{\text{fused}} = \alpha Z_{\text{depth}} + (1 - \alpha) Z_{\text{proj}}$.
- **Risk Assessment (`classify_risk`)**:
  - Evaluates both longitudinal distance and lateral lane offset from camera center.
  - Produces categorical risk tiers: `SAFE`, `CAUTION`, `WARNING`, `DANGER`.

### 2.4 Visualization Layer (`depthtrack.visualization`)
- Renders a multi-panel synchronized dashboard:
  - **Left Panel (960×540)**: Camera view with annotated bounding boxes, class labels, tracked IDs, distances, risk color codes, and trajectory trails.
  - **Right Panel (960×540)**:
    - Top: Depth map colorized using Inferno/Plasma colormaps.
    - Bottom: Top-down Bird's-Eye-View (BEV) radar map with ego vehicle, vehicle positions, speed vectors, and distance arc rings.
  - **HUD Overlay**: Real-time FPS, frame counters, and device telemetry.

---

## 3. Dependency & Compatibility Audit

| Package | Installed Version | Compatibility Assessment | Risk Level |
|---|---|---|---|
| **Python** | 3.13.1 | Core libraries compatible. PyTorch 2.11+cu128 functional. | Low |
| **PyTorch** | 2.11.0+cu128 | CUDA 12.8 acceleration validated on RTX 3050 Laptop GPU. | Minimal |
| **Ultralytics** | 8.4.211 | Both YOLOv8 and YOLO11 architectures supported natively. | Minimal |
| **Supervision** | 0.30.9 | `sv.ByteTrack()` emits `FutureWarning: deprecated since v0.28.0, removed in v0.31.0`. | Medium |
| **Transformers**| 5.22.2 | `AutoModelForDepthEstimation` loads Depth Anything V2 correctly. | Low |
| **OpenCV** | 5.0.0.93 | Codec handling functional; mp4v fallback operational without ffmpeg. | Low |

---

## 4. Key Limitations & Architectural Warnings

1. **Monocular Depth Ambiguity**:
   - Depth Anything V2 produces *relative* (affine-invariant) inverse depth, not calibrated metric distance.
   - The linear scaling $[0, 1] \to [2\text{m}, 80\text{m}]$ is an approximation and is sensitive to camera pitch, horizon shifts, and scene topography.
2. **ByteTrack Deprecation in Supervision**:
   - Supervision v0.30.9 deprecates top-level `sv.ByteTrack()`. A compatibility abstraction is needed to guarantee future-proof operation without breakages.
3. **Sequential Pipeline Latency**:
   - Running detector, tracker, depth model, and panel compositing synchronously in Python creates an end-to-end latency bottleneck dominated by depth inference (~39–46ms on GPU).

