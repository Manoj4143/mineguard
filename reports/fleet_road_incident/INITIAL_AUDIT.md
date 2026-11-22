# MineGuard AI — Feature Integration Initial Audit Report
## Pre-Integration Architecture & Readiness Assessment

**Date:** November 23, 2026  
**Auditor / Architect:** Senior AI/ML & Mining Systems Integration Engineer  
**Workspace:** `D:\Mineguard_project_2`  
**Current Git Branch:** `phase5-dashboard-integration` (Local only — Strictly zero pushes/commits)  
**Fallback Project Status:** `D:\MINI_GUARD_PROJECT\MineGuard-AI` strictly untouched (Read-Only)  
**Pre-modification Backup:** `D:\Mineguard_project_2\backups\pre_fleet_road_incident_integration\` verified with:
  - `dashboard.py.bak`
  - `service.py.bak`
  - `unified_engine.py.bak`

---

### 1. Existing System Baseline & Capabilities
- **Perception Engine:** `UnifiedMineGuardPerception` coordinating:
  - Object Detection: YOLOv8n (`DepthTrack/assets/yolov8n.pt`) with vehicle classes (car, truck, bus, motorcycle, bicycle) and corridor anomaly failsafe.
  - Tracking: ByteTrack multi-object tracking.
  - Depth Estimation: Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) providing monocular depth for distance fusion.
  - Environmental Adaptation: CLAHE and DCP-v5 dehazing with NightVisionProcessor.
- **Service & Concurrency Layer:** `ThreadingHTTPServer` running non-blocking MJPEG streaming (`/stream.mjpg`) and REST API telemetry polling (`/api/telemetry`).
- **Dashboard UI:** Side-by-side display (Left: Live AI Vision, Right: 3D BEV / Depth point cloud).

---

### 2. Required Modifications & Roadmap
1. **Frontend Layout Evolution:**
   - Remove 3D BEV / Bird's-Eye View visualization from `composite_frame` and dashboard layout.
   - Expand the Live AI Detection View to full card width while strictly preserving aspect ratio.
   - Retain backend depth estimation (Depth Anything V2) for monocular distance fusion and safety rules.
   - Vertically stack the three new modules beneath the existing Atmospheric & Road Intelligence panel:
     1. **FEATURE 3 — MINE FLEET COORDINATION**
     2. **FEATURE 2 — MINE ROAD CONDITION INTELLIGENCE**
     3. **FEATURE 1 — AUTOMATIC INCIDENT RECORDER**
2. **Feature 3 Architecture (`fleet_models.py`, `fleet_manager.py`, `fleet_simulator.py`):**
   - SQLite backed registry (`data/fleet/fleet.db`).
   - REST endpoints: `/api/fleet/register`, `/api/fleet/telemetry`, `/api/fleet/vehicles`, `/api/fleet/alerts`, `/api/fleet/hazards`.
   - V2V proximity, convoy spacing, intersection conflict detection, communication timeout detection.
   - Real vs SIMULATED data separation with 3-vehicle simulator.
3. **Feature 2 Architecture (`road_intelligence.py`):**
   - Road hazard taxonomy: POTHOLE, ROCK, LOOSE_DEBRIS, ROAD_OBSTRUCTION, MUDDY_SURFACE, WATER_ACCUMULATION, ROAD_EDGE_DAMAGE.
   - Geometric road ROI extraction, gradient & textural anomaly segmentation, persistent hazard tracking with evidence snapshots.
   - Explicit reporting: if a trained CNN/YOLO hazard model is not loaded, clearly report `ROAD HAZARD MODEL NOT CONFIGURED` rather than fabricating detections.
4. **Feature 1 Architecture (`incident_recorder.py`):**
   - Rolling ring buffer (15s pre-event, 15s post-event) storing timestamped frames with bounded RAM.
   - Asynchronous worker saving MP4s and metadata to `data/incidents/`.
   - Triggers: `CRITICAL_COLLISION_RISK`, `NEAR_MISS`, `ROAD_OBSTRUCTION`, `ROAD_HAZARD`, `VEHICLE_PROXIMITY`, `PERCEPTION_FAILURE`.
   - Restful endpoints for incident playback, JSON metadata, and summary statistics.
