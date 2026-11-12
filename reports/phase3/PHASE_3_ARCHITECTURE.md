# PHASE 3: DEPTH TRACK & MINEGUARD ARCHITECTURE MAPPING

**Target:** Modular Perception Integration  
**Original System Location:** `D:\MINI_GUARD_PROJECT\MineGuard-AI` (Preserved intact as immutable fallback)  
**Integrated Engine Location:** `D:\Mineguard_project_2\src\mineguard_depthtrack`  

---

## 1. Subsystem Responsibility Matrix

| Responsibility | Original MineGuard (`MineGuard-AI`) | DepthTrack (`depthtrack`) | Integrated Engine Architecture |
|---|---|---|---|
| **Environmental Visibility** | `assess_atmospheric_visibility` (dark mean + RMS contrast) | None | **MineGuard Atmospheric Assessor** (reused) |
| **Day/Night Assessor** | `NightVisionProcessor` (lux + hysteresis) | None | **MineGuard NightVisionProcessor** (reused) |
| **Enhancement Dispatch** | Dynamic CLAHE, DCP V5, Night Vision | None | **Decoupled Enhancement Engine** (CLAHE at night; bypass default for daytime; DCP V5 user toggle) |
| **Vehicle Detector** | YOLO11n (`yolo11n.pt`) | YOLOv8n (`assets/yolov8n.pt`) | **Configurable Detector Plugin** (Default: YOLOv8n based on Phase 2 empirical superiority) |
| **Multi-Object Tracker** | Ultralytics built-in ByteTrack | `ObjectTracker` (Supervision ByteTrack) | **DepthTrack ObjectTracker** (with trajectory buffer & velocity vector) |
| **Monocular Depth** | Heuristic bounding box + ground ratio | `Depth Anything V2 Small` (DPT DinoV2) | **Depth Anything V2 Small** (fed original natural frame) |
| **Distance Estimation** | Projection + ground heuristic | Fusion: Relative depth median + pinhole projection | **Fused Monocular Range Estimator** |
| **Closing Velocity & TTC** | `VehicleRiskAnalyzer` ($\Delta d / \Delta t$) | Finite difference pixel velocity | **Temporal Velocity & TTC Engine** (from `VehicleRiskAnalyzer`) |
| **Risk Classification** | 3-tier: `SAFE`, `CAUTION`, `CRITICAL` | 4-tier: `SAFE`, `CAUTION`, `WARNING`, `DANGER` | **Unified 3-Tier Safety Layer** (`SAFE`, `CAUTION`, `CRITICAL` mapping) |
| **BEV Radar** | None | Inverse Perspective Grid (`bev.py`) | **Integrated BEV Radar Display** |
| **Dashboard & Web API** | Native HTTP server (`dashboard.py`), live MJPEG stream, REST telemetry | CLI video writer | **MineGuard Live Streaming Dashboard + REST Telemetry API** |
| **Road Clearance** | `evaluate_road_clearance` | None | **MineGuard Road Clearance Module** (reused) |

---

## 2. Unified Perception Data Schema

```json
{
  "frame_id": 142,
  "timestamp": 5.917,
  "telemetry": {
    "system_status": "ONLINE",
    "pipeline_mode": "CLEAR_BYPASS",
    "fog_condition": "CLEAR",
    "ambient_lux": 92.4,
    "fps": 16.4,
    "latency_ms": 61.0,
    "road_status": "ROAD CLEAR",
    "highest_risk": "SAFE"
  },
  "objects": [
    {
      "track_id": 1,
      "class_id": 7,
      "class_name": "truck",
      "confidence": 0.88,
      "bbox": [320, 180, 560, 420],
      "relative_depth": 0.68,
      "distance_m": 24.5,
      "closing_speed_kmh": 4.2,
      "ttc_seconds": 21.0,
      "risk_tier": "SAFE",
      "color_bgr": [0, 255, 0]
    }
  ]
}
```

