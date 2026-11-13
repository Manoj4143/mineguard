# MINEGUARD AI — PHASE 5.3 DATASET RECOMMENDATIONS & TRANSITION PLAN

## Objective
Provide a practical, structured transition plan to replace synthetic scenario clips in `DepthTrack/data/scenarios/` with authentic real-world mining clips.

---

## 1. Target Directory Structure for Phase 5.3 Implementation

Upon user authorization, organize `DepthTrack/data/` as follows:

```
DepthTrack/data/
├── scenarios/
│   ├── mine_clear.mp4          # Trimmed 100 frames from clear_clean.mp4 (Real Haul Truck)
│   ├── mine_light_fog.mp4      # Trimmed 100 frames from light_fog_clean.mp4 (Koschmieder beta=0.35)
│   ├── mine_dense_fog.mp4      # Trimmed 100 frames from dense_fog_clean.mp4 (Koschmieder beta=1.30)
│   ├── mine_night.mp4          # Trimmed 100 frames from night_vision_demo.mp4 (Real Night Road)
│   └── quarry_dust.mp4         # Trimmed 100 frames from mixkit_yellow_truck.mp4 (Dusty Road)
├── sample_test.mp4             # Trimmed clip from real mine_truck.mp4 (replaces 29.7 KB texture)
├── ground_truth/
│   ├── data.yaml               # YOLO class specification (truck, car, etc.)
│   ├── labels/                 # Normalized bounding box annotations (.txt)
│   └── annotation_manifest.csv # Frame-to-image mapping
└── user_uploads/               # Preserved for runtime browser video uploads
```

---

## 2. Technical Benefits of Real-World Replacement

1. **Dashboard Demonstration Authenticity:**
   - Instead of 0 detections on primitive drawn rectangles, the live dashboard will display real-time bounding boxes around real haul trucks.
   - ByteTrack will maintain consistent tracking IDs (`ID #1`).
   - Relative depth maps will show actual vehicle geometry against haul road terrain.
   - The road clearance banner will dynamically transition from `ROAD CLEAR` to `PROXIMITY WARNING` or `ROAD BLOCKED` as trucks approach.
2. **Standardization of Benchmark Suites:**
   - Regression and performance benchmarks will measure genuine computer vision accuracy (Precision, Recall, mAP) against verified ground truth rather than flat artificial images.
3. **Storage Efficiency:**
   - Instead of copying 100 MB full-length files, trimming clips to 100–120 frames at 720p keeps each scenario video under 5–8 MB, ensuring rapid git operations and fast test runs.

---

## 3. Transition Safeguards

- **Strict Non-Destructive Ingestion:** Never overwrite files in `D:\MINI_GUARD_PROJECT\MineGuard-AI`. Only copy and trim selected frames into `D:\Mineguard_project_2\DepthTrack\data\`.
- **Zero Perception Architecture Changes:** `UnifiedMineGuardPerception`, `BackgroundVideoWorker`, and the dashboard API routes remain identical.
- **Rollback Archival:** Existing synthetic scenario videos will be archived in `D:\Mineguard_project_2\backups\synthetic_scenarios\` before replacement.

---

## 4. Next Step Checklist (Pending User Approval)

- [ ] User approves transition to real-world dataset.
- [ ] Run automated clip extraction script to generate trimmed 100-frame real scenario videos.
- [ ] Unpack and link `ground_truth` annotations.
- [ ] Run full 57-test suite to confirm zero regressions.
- [ ] Run live demo acceptance test to capture real-world detection evidence across all 6 scenarios.

