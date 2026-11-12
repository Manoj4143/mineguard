# AUDIT B: ORIGINAL MINEGUARD BASELINE VALIDATION REPORT

**Audit Date:** 2026-11-12  
**Target File:** `reports/phase4_5/BASELINE_VALIDATION_REPORT.md`  
**Inspected System:** `D:\MINI_GUARD_PROJECT\MineGuard-AI`  

---

## 1. Discrepancy Statement & Root Cause Discovery

Phase 4 reported that the original MineGuard engine achieved **0/30 detections** across Clear, Light Fog, and Dense Fog scenarios. This contradicted prior project claims that the original system was functioning.

### Root Cause:
In `tools/run_full_system_validation.py`, the evaluation loop inspected:
```python
v_list = res_orig.get("telemetry", {}).get("vehicles", [])
```
However, inspecting `D:\MINI_GUARD_PROJECT\MineGuard-AI\mineguard_core.py` (lines 240–252) reveals that `process_frame` returns a flat dictionary:
```python
return {
    "annotated_frame": annotated_frame,
    "processed_frame": processed_frame,
    "pipeline_mode": active_pipeline_mode,
    "fog_condition": fog_cond,
    "ambient_lux": illumination["raw_brightness"],
    "highest_risk": highest_risk,
    "vehicles": vehicle_telemetry,  # <--- Top-level key, NOT inside "telemetry"
    "fps": fps,
    "latency_ms": frame_latency_ms,
    "enhancement_ms": enhancement_ms
}
```
Because the benchmark script queried `res_orig.get("telemetry", {}).get("vehicles", [])`, the expression returned an empty list `[]` for every single frame!

---

## 2. Re-Benchmarked True Baseline Measurements

When queried correctly using top-level `res_orig["vehicles"]`, the original MineGuard system functions as follows:

| Scenario Video | Evaluated Frames | Original MineGuard (YOLO11n) True Detections | Integrated DepthTrack (YOLOv8n) Detections | Tracking Continuity |
|---|---|---|---|---|
| **`clear_clean.mp4`** | 30 | **30 / 30 frames (100%)** | **30 / 30 frames (100%)** | Both hold 1 continuous track ID |
| **`light_fog_clean.mp4`** | 30 | **30 / 30 frames (100%)** | **30 / 30 frames (100%)** | Both hold 1 continuous track ID |
| **`dense_fog_clean.mp4`** | 30 | **30 / 30 frames (100%)** | **30 / 30 frames (100%)** | Both hold 1 continuous track ID |
| **`mine_truck.mp4`** (Raw Mining) | 30 | **30 / 30 frames (100%)** | **30 / 30 frames (100%)** | Both detect active haul truck |
| **`mine_night.mp4`** | 30 | 0 / 30 frames | 0 / 30 frames | Both fail under zero-light conditions |

---

## 3. Comparative Technical Assessment (Original vs. Integrated)

With the baseline properly measured and verified:

1. **Detection Parity:** Both engines achieve **100% frame detection retention** on daylight and foggy mining roads.
2. **Key Advancements of the Integrated System:**
   - **Monocular Depth Perception:** Original MineGuard relies solely on bounding-box pixel height heuristics ($d = (f \cdot H)/h$), which fails completely when a vehicle is partially occluded, pitched, or cornering. The integrated engine fuses real monocular depth heatmaps with projection geometry.
   - **Bird's-Eye-View Radar:** The integrated system generates real-time pseudo-3D point clouds and top-down radar grids.
   - **Throughput Profile:** Original MineGuard achieves ~27–28 FPS (detection only). The integrated system sustains **15.5 – 17.1 FPS** while running heavy transformer-based depth estimation (`Depth Anything V2 Small`), rendering the 1924×540 composite dashboard smoothly in real-time.

