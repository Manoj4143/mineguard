#!/usr/bin/env python3
"""benchmark_tracking.py - Compare tracking stability with YOLOv8n vs YOLO11n."""

import csv
import json
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from supervision.tracker.byte_tracker.core import ByteTrack
import supervision as sv

def benchmark_tracking():
    print("=" * 60)
    print("PHASE 2: TRACKING STABILITY BENCHMARK (ByteTrack)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v8 = YOLO("DepthTrack/assets/yolov8n.pt").to(device)
    v11 = YOLO("DepthTrack/assets/yolo11n.pt").to(device)
    
    rep_dir = Path("reports/phase2")
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = ["clear", "light_fog", "dense_fog", "night"]
    records = []
    
    for sc in scenarios:
        vid_path = Path(f"DepthTrack/data/scenarios/mine_{sc}.mp4")
        if not vid_path.exists():
            continue
            
        def eval_tracker_for_model(model):
            tracker = ByteTrack()
            cap = cv2.VideoCapture(str(vid_path))
            unique_ids = set()
            total_tracked_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                res = model(frame, conf=0.25, verbose=False, device=device)[0]
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                
                dets = sv.Detections(xyxy=boxes, confidence=confs, class_id=classes)
                tracked = tracker.update_with_detections(dets)
                
                if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
                    for tid in tracked.tracker_id:
                        unique_ids.add(int(tid))
                    total_tracked_frames += 1
            cap.release()
            return len(unique_ids), total_tracked_frames

        u_v8, f_v8 = eval_tracker_for_model(v8)
        u_v11, f_v11 = eval_tracker_for_model(v11)
        
        records.append({
            "scenario": sc,
            "yolov8n_unique_track_ids": u_v8,
            "yolov8n_frames_with_tracks": f_v8,
            "yolo11n_unique_track_ids": u_v11,
            "yolo11n_frames_with_tracks": f_v11,
        })
        print(f"Scenario '{sc}': v8 -> {u_v8} IDs ({f_v8} frames), v11 -> {u_v11} IDs ({f_v11} frames)")

    with open(rep_dir / "tracking_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Saved tracking benchmark to {rep_dir / 'tracking_benchmark.csv'}")

if __name__ == "__main__":
    benchmark_tracking()

