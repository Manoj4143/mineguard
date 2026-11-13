"""Ground-truth detection and tracking evaluation engine for MineGuard AI Phase 5.3."""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("MineGuardEvaluation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def yolo_to_xyxy(bbox_norm: List[float], width: int, height: int) -> List[float]:
    """Converts normalized [xc, yc, w, h] to pixel coordinates [x1, y1, x2, y2]."""
    xc, yc, w, h = bbox_norm
    x1 = (xc - w / 2.0) * width
    y1 = (yc - h / 2.0) * height
    x2 = (xc + w / 2.0) * width
    y2 = (yc + h / 2.0) * height
    return [float(x1), float(y1), float(x2), float(y2)]


class DetectionEvaluator:
    """Evaluates object detector predictions against ground-truth bounding boxes.

    Enforces strict one-to-one matching, class equality, and configurable IoU thresholds.
    """

    def __init__(self, iou_threshold: float = 0.50, target_class_name: str = "truck"):
        self.iou_threshold = iou_threshold
        self.target_class_name = target_class_name.lower()

    def evaluate_video(
        self,
        video_path: str,
        ground_truth: Dict[int, List[Dict[str, Any]]],
        detector_engine: Any,
        conf_threshold: float = 0.25,
        target_classes_eval: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluates detection performance on a video using exact annotated frame indices.

        Args:
            video_path: Path to MP4 video
            ground_truth: Dictionary of {frame_idx: [gt_box_dict, ...]}
            detector_engine: UnifiedMineGuardPerception or detector object with process_frame
            conf_threshold: Minimum confidence for positive predictions
            target_classes_eval: List of allowed class names (default: ['truck'])
        """
        target_classes = [c.lower() for c in (target_classes_eval or [self.target_class_name])]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for evaluation: {video_path}")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tp = 0
        fp = 0
        fn = 0
        class_mismatches = 0  # High IoU but wrong class (e.g. truck predicted as bus)
        ious_collected: List[float] = []
        frame_eval_records = []

        total_annotated_frames = len(ground_truth)
        evaluated_frames = 0

        # Sort frame indices to preserve temporal order
        sorted_frame_indices = sorted(ground_truth.keys())

        for f_idx in sorted_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            evaluated_frames += 1
            gt_boxes_raw = ground_truth[f_idx]
            gt_boxes_xyxy = [
                {
                    "class_name": b["class_name"].lower(),
                    "xyxy": yolo_to_xyxy(b["bbox_norm"], frame_w, frame_h)
                }
                for b in gt_boxes_raw
            ]

            # Run inference
            out = detector_engine.process_frame(frame)
            pred_objs = out.get("objects", [])

            # Compute work_w and work_h using engine cfg
            cfg_work_w = getattr(getattr(detector_engine, "cfg", None), "video", None)
            raw_work_w = getattr(cfg_work_w, "work_width", None) if cfg_work_w else 640
            max_work_w = raw_work_w if isinstance(raw_work_w, (int, float)) else 640
            work_w = min(frame_w, max_work_w)
            work_h = int(frame_h * (work_w / max(float(frame_w), 1.0)))

            # Map predictions from work_frame coordinates back to original native frame resolution
            scale_x = frame_w / max(float(work_w), 1.0)
            scale_y = frame_h / max(float(work_h), 1.0)

            # Filter valid predictions above confidence threshold and rescale to native frame
            preds = [
                {
                    "class_name": p.get("class_name", "unknown").lower(),
                    "confidence": float(p.get("confidence", 0.0)),
                    "xyxy": [
                        float(p.get("bbox", [0, 0, 0, 0])[0]) * scale_x,
                        float(p.get("bbox", [0, 0, 0, 0])[1]) * scale_y,
                        float(p.get("bbox", [0, 0, 0, 0])[2]) * scale_x,
                        float(p.get("bbox", [0, 0, 0, 0])[3]) * scale_y,
                    ]
                }
                for p in pred_objs
                if float(p.get("confidence", 0.0)) >= conf_threshold
            ]

            # One-to-one matching via greedy IoU assignment
            matched_gt = set()
            matched_pred = set()

            for p_idx, pred in enumerate(preds):
                best_iou = 0.0
                best_gt_idx = -1

                for g_idx, gt in enumerate(gt_boxes_xyxy):
                    if g_idx in matched_gt:
                        continue
                    iou = compute_iou(pred["xyxy"], gt["xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= self.iou_threshold:
                    gt_class = gt_boxes_xyxy[best_gt_idx]["class_name"]
                    pred_class = pred["class_name"]

                    # Strict class check
                    if pred_class in target_classes and gt_class in target_classes:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                        matched_pred.add(p_idx)
                        ious_collected.append(best_iou)
                    else:
                        # High IoU but mismatched category
                        class_mismatches += 1
                        fp += 1
                else:
                    fp += 1

            # Any unmatched ground truth box is a False Negative
            unmatched_gts = len(gt_boxes_xyxy) - len(matched_gt)
            fn += unmatched_gts

            frame_eval_records.append({
                "frame_idx": f_idx,
                "gt_count": len(gt_boxes_xyxy),
                "pred_count": len(preds),
                "tp": tp,
                "fp": fp,
                "fn": fn
            })

        cap.release()

        precision = tp / max(tp + fp, 1e-6)
        recall = tp / max(tp + fn, 1e-6)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
        mean_iou = float(np.mean(ious_collected)) if ious_collected else 0.0

        return {
            "video_path": str(video_path),
            "iou_threshold": self.iou_threshold,
            "conf_threshold": conf_threshold,
            "target_classes": target_classes,
            "total_annotated_frames": total_annotated_frames,
            "evaluated_frames": evaluated_frames,
            "total_gt_objects": sum(len(b) for b in ground_truth.values()),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "class_mismatches": class_mismatches,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mean_bounding_box_iou": round(mean_iou, 4)
        }


class TrackingEvaluator:
    """Evaluates multi-object tracking metrics across consecutive frames."""

    @staticmethod
    def evaluate_sequence(
        video_path: str,
        detector_engine: Any,
        start_frame: int = 0,
        max_frames: int = 60
    ) -> Dict[str, Any]:
        """Measures track ID retention, fragmentation, and continuity on an authentic sequence."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Cannot open {video_path}"}

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        active_track_ids_history = []
        frames_evaluated = 0

        while frames_evaluated < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames_evaluated += 1

            out = detector_engine.process_frame(frame)
            objs = out.get("objects", [])
            tids = [int(o["track_id"]) for o in objs if "track_id" in o]
            active_track_ids_history.append(tids)

        cap.release()

        all_unique_ids = set(tid for f_tids in active_track_ids_history for tid in f_tids)
        frames_with_detections = sum(1 for f_tids in active_track_ids_history if len(f_tids) > 0)

        # Track ID switches / fragmentations
        # A primary track ID is the one with highest frequency
        id_frequencies: Dict[int, int] = {}
        for f_tids in active_track_ids_history:
            for tid in f_tids:
                id_frequencies[tid] = id_frequencies.get(tid, 0) + 1

        primary_track_id = max(id_frequencies, key=id_frequencies.get) if id_frequencies else None
        primary_retention_ratio = (id_frequencies[primary_track_id] / max(frames_with_detections, 1e-6)) if primary_track_id else 0.0

        return {
            "video_path": str(video_path),
            "frames_evaluated": frames_evaluated,
            "frames_with_detections": frames_with_detections,
            "unique_track_ids_count": len(all_unique_ids),
            "primary_track_id": primary_track_id,
            "primary_track_retention": round(primary_retention_ratio, 4),
            "track_id_continuity": "STABLE" if len(all_unique_ids) <= 2 else "FRAGMENTED",
            "unsupported_metrics_note": "MOTA/MOTP cannot be computed because ground-truth lacks temporal ID associations (annotations are independent frame detection bboxes)."
        }
