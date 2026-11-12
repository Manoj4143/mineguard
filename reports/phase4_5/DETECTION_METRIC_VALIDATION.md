# AUDIT C: DETECTION METRIC VERIFICATION & TRACKING AUDIT

**Audit Date:** 2026-11-12  
**Target File:** `reports/phase4_5/DETECTION_METRIC_VALIDATION.md`  

---

## 1. Clarification of "30/30 Detections"

In the previous report, the terminology **"30/30 detections"** was defined as:
> **"30 out of 30 evaluated sequential video frames contained at least one tracked vehicle instance whose class matched the target set `{car, motorcycle, bus, truck}`."**

### Important Scientific Distinctions:
- It **does NOT** denote 100% mAP or zero false negatives across arbitrary multi-object mine yards.
- Because formalized bounding-box coordinate annotations (e.g., Pascal VOC XML or COCO JSON ground truth) were not packaged with `cvat_clean_videos/*.mp4`, formal **mAP@50** and **mAP@50-95** cannot be claimed without inventing fictitious annotations.
- The 30/30 metric is an empirical measurement of **Frame-Level Track Retention** across consecutive frames of video.

---

## 2. Tracking Continuity & ID Switches

Across 30 consecutive frames on the real haul truck footage (`videos/cvat_clean_videos/clear_clean.mp4` and `videos/mine_truck.mp4`):
- **Initial Assignment:** Vehicle assigned `track_id = 1` upon initial entry.
- **Track Persistence:** Held `track_id = 1` continuously for all 30 frames.
- **ID Switches:** **0 ID switches observed.**
- **Track Fragmentation:** **0 fragmentation events.**
- **Trajectory History:** Bounded queue maintained 25–30 historical centroids, generating motion trails and stable velocity vectors without jitter.

