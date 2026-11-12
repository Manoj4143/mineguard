# AUDIT A: NIGHT DETECTION DISCREPANCY INVESTIGATION

**Audit Date:** 2026-11-12  
**Target File:** `reports/phase4_5/NIGHT_DETECTION_INVESTIGATION.md`  
**Hardware Evaluated:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)  

---

## 1. Discrepancy Statement & Root Cause Diagnosis

In Phase 2.5, CLAHE was reported to produce a **250% recall improvement** under night conditions (from 2 up to 7 detections on sample frames). However, Phase 4 reported **0/30 detections** for both Original MineGuard and Integrated DepthTrack.

### Root Causes Discovered:
1. **Automated Condition Detection Trigger:**
   - In `mineguard_depthtrack/unified_engine.py`:
     ```python
     illum = self.night_processor.assess_scene_illumination(work_frame)
     if illum["is_night"]:
         pipeline_mode = "NIGHT_CLAHE"
         detector_input, _ = enhance_clahe(work_frame)
     ```
   - On `mine_night.mp4`, the raw scene illumination measured **7.7 to 9.0 lux**. The threshold for night is `night_threshold = 70.0`. Hence `illum["is_night"]` was indeed evaluated as **True** and `NIGHT_CLAHE` was actively dispatched.
2. **COCO Class Filtering Mismatch on Degraded Night Targets:**
   - On the synthetic night scenario `mine_night.mp4`, vehicles are located outside the headlight beam or partially illuminated.
   - When inspecting the raw un-filtered YOLOv8n detector logits across the 30 frames:
     - YOLOv8n detected objects in multiple frames (Frames 16, 24, 25, 26, 27, 28) with confidence scores between 0.17 and 0.50.
     - However, the predicted COCO class IDs were **Class 32 (`sports ball`)** and **Class 64 (`mouse`)**, triggered by the circular high-contrast headlight beams.
     - Because `VehicleDetector` strictly filters `{1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}`, these candidates were rightly discarded as non-vehicles.
3. **Contrast Limit & Headlight Glare:**
   - In real mining conditions, optical camera sensors at night without active headlights produce pure sensor noise. Headlights create blooming highlights that overpower peripheral vehicle contours on generic pretrained weights.

---

## 2. Experimental Measurement Matrix on Night Footage

Evaluated on 30 consecutive frames of `mine_night.mp4` on GPU:

| Configuration | Enhancement Latency | Candidate Boxes (Raw) | Target Vehicle Classes (2, 3, 5, 7) | Confirmed Vehicles |
|---|---|---|---|---|
| **Raw Input** | 0.0 ms | 7 | 0 | 0 |
| **CLAHE Enhanced** | 5.4 ms | 15 (+114%) | 0 | 0 |
| **Night Vision Processor** | 7.6 ms | 1 | 0 | 0 |

### Verdict & Hardware Scope Restriction:
- **Conclusion:** CLAHE genuinely doubles feature candidate extraction under extreme dark conditions, but pretrained standard COCO weights misclassify headlights as non-vehicle classes in total darkness.
- **Operational Requirement:** For nighttime mining operations, the system must either:
  1. Rely on high-lumen vehicle headlamps / working lights illuminating the front 30–50m road surface, or
  2. Integrate Long-Wave Infrared (LWIR) / Thermal imaging sensors for night hazard detection.

