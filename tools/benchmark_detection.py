#!/usr/bin/env python3
"""benchmark_detection.py - Rigorous benchmark comparing YOLOv8n vs YOLO11n on CUDA."""

import os
import time
import json
import csv
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def benchmark_detectors():
    print("=" * 60)
    print("PHASE 2: YOLOv8n vs YOLO11n DETECTOR BENCHMARK")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    v8_path = Path("DepthTrack/assets/yolov8n.pt")
    v11_path = Path("DepthTrack/assets/yolo11n.pt")
    
    evidence_dir = Path("reports/phase2/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path("reports/phase2")
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\nLoading models...")
    t0 = time.time()
    model_v8 = YOLO(str(v8_path)).to(device)
    t_v8_load = time.time() - t0
    
    t0 = time.time()
    model_v11 = YOLO(str(v11_path)).to(device)
    t_v11_load = time.time() - t0
    
    print(f"YOLOv8n: {v8_path.stat().st_size / (1024**2):.2f} MB, Load Time: {t_v8_load:.3f}s")
    print(f"YOLO11n: {v11_path.stat().st_size / (1024**2):.2f} MB, Load Time: {t_v11_load:.3f}s")
    
    # 1. Measure pure inference speed on standardized frame sizes (640x640)
    print("\n--- [1] Inference Latency & Speed Benchmark (Batch=1, Resolution=640x640) ---")
    test_tensor = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        _ = model_v8(test_tensor, verbose=False, device=device)
        _ = model_v11(test_tensor, verbose=False, device=device)
        
    def profile_model(model, name):
        torch.cuda.reset_peak_memory_stats()
        times = []
        torch.cuda.synchronize()
        for _ in range(50):
            t_s = time.perf_counter()
            _ = model(test_tensor, verbose=False, device=device)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t_s) * 1000)
            
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        return {
            "model": name,
            "mean_ms": round(float(np.mean(times)), 2),
            "median_ms": round(float(np.median(times)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "fps": round(1000.0 / float(np.mean(times)), 2),
            "peak_vram_mb": round(peak_vram, 1),
        }
        
    perf_v8 = profile_model(model_v8, "YOLOv8n")
    perf_v11 = profile_model(model_v11, "YOLO11n")
    
    print(f"YOLOv8n: Mean={perf_v8['mean_ms']}ms, P95={perf_v8['p95_ms']}ms, FPS={perf_v8['fps']}, VRAM={perf_v8['peak_vram_mb']}MB")
    print(f"YOLO11n: Mean={perf_v11['mean_ms']}ms, P95={perf_v11['p95_ms']}ms, FPS={perf_v11['fps']}, VRAM={perf_v11['peak_vram_mb']}MB")
    
    with open(rep_dir / "performance_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(perf_v8.keys()))
        writer.writeheader()
        writer.writerows([perf_v8, perf_v11])

    # 2. Evaluate across scenario video frames (Clear, Light Fog, Dense Fog, Night)
    print("\n--- [2] Detection Evaluation Across Scenarios ---")
    scenarios = ["clear", "light_fog", "dense_fog", "night"]
    det_records = []
    
    # Ground truth vehicles per frame in synthesized scenarios = 2 (1 Haul Truck center, 1 Pickup right)
    GT_VEHICLES = 2
    
    for sc in scenarios:
        vid_path = Path(f"DepthTrack/data/scenarios/mine_{sc}.mp4")
        if not vid_path.exists():
            continue
            
        cap = cv2.VideoCapture(str(vid_path))
        frame_idx = 0
        v8_dets_total = 0
        v11_dets_total = 0
        v8_conf_sum = 0.0
        v11_conf_sum = 0.0
        
        saved_evidence = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run v8
            res_v8 = model_v8(frame, conf=0.30, verbose=False, device=device)[0]
            boxes_v8 = res_v8.boxes
            v8_count = len(boxes_v8)
            v8_dets_total += v8_count
            if v8_count > 0:
                v8_conf_sum += float(boxes_v8.conf.mean())
                
            # Run v11
            res_v11 = model_v11(frame, conf=0.30, verbose=False, device=device)[0]
            boxes_v11 = res_v11.boxes
            v11_count = len(boxes_v11)
            v11_dets_total += v11_count
            if v11_count > 0:
                v11_conf_sum += float(boxes_v11.conf.mean())
                
            # Save visual side-by-side comparison on frame 20
            if frame_idx == 20 and not saved_evidence:
                vis_v8 = res_v8.plot()
                vis_v11 = res_v11.plot()
                
                # Annotate titles
                cv2.putText(vis_v8, f"YOLOv8n [{sc.upper()}] Dets:{v8_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_v11, f"YOLO11n [{sc.upper()}] Dets:{v11_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                comparison = np.hstack([vis_v8, vis_v11])
                cv2.imwrite(str(evidence_dir / f"compare_{sc}_frame20.png"), comparison)
                saved_evidence = True
                
            frame_idx += 1
        cap.release()
        
        total_gt = frame_idx * GT_VEHICLES
        det_records.append({
            "scenario": sc,
            "total_frames": frame_idx,
            "gt_instances": total_gt,
            "yolov8n_total_detections": v8_dets_total,
            "yolov8n_mean_conf": round(v8_conf_sum / max(frame_idx, 1), 3),
            "yolo11n_total_detections": v11_dets_total,
            "yolo11n_mean_conf": round(v11_conf_sum / max(frame_idx, 1), 3),
        })
        print(f"Scenario '{sc}': v8 detections={v8_dets_total}, v11 detections={v11_dets_total}")
        
    with open(rep_dir / "detection_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(det_records[0].keys()))
        writer.writeheader()
        writer.writerows(det_records)
        
    # Save complete summary JSON
    summary = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vram_gb": 6.0,
            "torch": torch.__version__,
        },
        "performance": [perf_v8, perf_v11],
        "scenario_detection": det_records,
    }
    with open(rep_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nPhase 2 detection and performance benchmarks completed!")

if __name__ == "__main__":
    benchmark_detectors()

