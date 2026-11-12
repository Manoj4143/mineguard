# AUDIT E: DASHBOARD AND API INTEGRATION CONTRACT

**Audit Date:** 2026-11-12  
**Target File:** `reports/phase4_5/DASHBOARD_INTEGRATION_READINESS.md`  

---

## 1. Integration Contract Specification

This document provides the integration contract for connecting `mineguard_depthtrack.UnifiedMineGuardPerception` into the existing `dashboard.py` server when live dashboard integration is authorized.

### 1.1 Class Initialization
```python
from mineguard_depthtrack import UnifiedMineGuardPerception

# Instantiate engine once at server startup
engine = UnifiedMineGuardPerception(
    detector_weights="DepthTrack/assets/yolov8n.pt",
    device="cuda",  # Auto-falls back to "cpu" if CUDA unavailable
    enable_heavy_fog_dcp=False  # Keep False by default to maintain 16 FPS
)
```

### 1.2 Per-Frame Invocation Interface
```python
# Feed incoming OpenCV BGR frame (numpy.ndarray, shape [H, W, 3])
output = engine.process_frame(frame, timestamp=time.time())
```

### 1.3 Expected Output Schema
The engine returns a dictionary with four structured elements:
```python
{
    # 1. High-level dashboard telemetry & system state
    "telemetry": {
        "system_status": "ONLINE",
        "frame_id": 142,
        "pipeline_mode": "CLEAR_BYPASS",     # "CLEAR_BYPASS", "FOG_ADAPTIVE", "NIGHT_CLAHE", "DCP_V5_FOG"
        "fog_condition": "CLEAR",            # "CLEAR", "LIGHT_FOG", "MEDIUM_FOG", "DENSE_FOG"
        "ambient_lux": 92.4,
        "fps": 16.4,
        "latency_ms": 61.0,
        "road_status": "ROAD CLEAR",         # "ROAD CLEAR", "PROXIMITY WARNING", "ROAD BLOCKED / HAZARD"
        "road_status_detail": "Vehicles at safe distance.",
        "highest_risk": "SAFE",              # "SAFE", "CAUTION", "CRITICAL"
        "active_vehicles_count": 1
    },

    # 2. Per-object telemetry list for live tables & API consumers
    "objects": [
        {
            "track_id": 1,
            "class_id": 7,
            "class_name": "truck",
            "confidence": 0.88,
            "bbox": [140, 302, 493, 495],
            "relative_depth": 0.652,
            "distance_m": 24.2,
            "closing_speed_kmh": 4.2,
            "ttc_seconds": 20.7,
            "risk_tier": "CAUTION",
            "color_bgr": [0, 215, 255]
        }
    ],

    # 3. Synchronized 1924x540 3-panel composite frame for live MJPEG streaming
    # (Left: Camera + ADAS BBoxes + Trails | Divider: 4px | Right-Top: Depth Map | Right-Bottom: 3D BEV Radar)
    "composite_frame": np.ndarray(shape=(540, 1924, 3), dtype=np.uint8),

    # 4. Raw normalized depth map [0, 1] float32 for downstream analytical queries
    "depth_map": np.ndarray(shape=(540, 960), dtype=np.float32)
}
```

---

## 2. Streaming Compatibility with `dashboard.py`

In `dashboard.py`, the streaming worker loop currently encodes `processed_frame`:
```python
_, buffer = cv2.imencode('.jpg', current_frame_jpeg)
```
To display the unified perception view, `dashboard.py` merely needs to update:
```python
current_frame_jpeg = output["composite_frame"]
current_telemetry = output["telemetry"]
current_telemetry["vehicles"] = output["objects"]
```
All existing dashboard telemetry endpoints (`/telemetry`, `/video_feed`, `/upload`) remain **100% backward-compatible**.

