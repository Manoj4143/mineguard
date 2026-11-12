#!/usr/bin/env python3
"""run_enhancement_compatibility.py - Phase 2.5 Enhancement Compatibility Suite."""

import os
import sys
import time
import json
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Add MineGuard-AI to sys.path to import original algorithms directly without modifying
mineguard_path = Path("D:/MINI_GUARD_PROJECT/MineGuard-AI")
if str(mineguard_path) not in sys.path:
    sys.path.insert(0, str(mineguard_path))

from dehaze_lib import enhance_clahe, dehaze_dcp_v3, dehaze_dcp_v4, dehaze_dcp_v5
from night_vision import NightVisionProcessor

def run_phase_2_5():
    print("=" * 70)
    print("PHASE 2.5: MINEGUARD ENHANCEMENT COMPATIBILITY BENCHMARK")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load YOLOv8n detector
    detector_path = Path("DepthTrack/assets/yolov8n.pt")
    detector = YOLO(str(detector_path)).to(device)

    # Load Depth Anything V2 Small
    depth_model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(depth_model_id)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id).to(device)
    depth_model.eval()

    night_proc = NightVisionProcessor()

    evidence_dir = Path("reports/phase2_5/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path("reports/phase2_5")
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Scenarios to test: Real video frames from cvat_clean_videos or scenarios
    # 1. Clear, 2. Light Fog, 3. Dense Fog, 4. Night
    video_sources = {
        "clear": "D:/MINI_GUARD_PROJECT/MineGuard-AI/videos/cvat_clean_videos/clear_clean.mp4",
        "light_fog": "D:/MINI_GUARD_PROJECT/MineGuard-AI/videos/cvat_clean_videos/light_fog_clean.mp4",
        "dense_fog": "D:/MINI_GUARD_PROJECT/MineGuard-AI/videos/cvat_clean_videos/dense_fog_clean.mp4",
        "night": "DepthTrack/data/scenarios/mine_night.mp4",
    }

    # Enhancement algorithms to test
    enhancements = ["ORIGINAL", "CLAHE", "DCP_V3", "DCP_V5", "NIGHT_VISION"]

    def apply_enhancement(frame, mode):
        t0 = time.perf_counter()
        if mode == "ORIGINAL":
            return frame.copy(), 0.0
        elif mode == "CLAHE":
            res, _ = enhance_clahe(frame)
            return res, (time.perf_counter() - t0) * 1000
        elif mode == "DCP_V3":
            res, _ = dehaze_dcp_v3(frame)
            return res, (time.perf_counter() - t0) * 1000
        elif mode == "DCP_V5":
            res, _ = dehaze_dcp_v5(frame)
            return res, (time.perf_counter() - t0) * 1000
        elif mode == "NIGHT_VISION":
            night_proc.assess_scene_illumination(frame)
            res, _ = night_proc.enhance_night_frame(frame)
            return res, (time.perf_counter() - t0) * 1000
        return frame.copy(), 0.0

    enhancement_results = []
    detection_metrics = []
    depth_comparison = []
    latency_records = []

    print("\n--- Running Tests Across Scenarios ---")

    for sc_name, v_path in video_sources.items():
        if not os.path.exists(v_path):
            print(f"Warning: {v_path} not found. Skipping {sc_name}.")
            continue

        cap = cv2.VideoCapture(v_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Select 20 evenly-spaced representative frames per scenario
        frame_indices = np.linspace(5, max(5, total_frames - 5), 20, dtype=int)
        
        frames = {}
        for idx in range(total_frames):
            ret, fr = cap.read()
            if not ret:
                break
            if idx in frame_indices:
                frames[idx] = fr
        cap.release()

        print(f"\nScenario: {sc_name.upper()} (Evaluated on {len(frames)} frames)")

        for mode in enhancements:
            enh_times = []
            det_times = []
            total_detections = 0
            conf_list = []
            sharpness_list = []
            depth_diffs = []

            saved_comparison_frame = False

            for idx, raw_frame in frames.items():
                # 1. Apply Enhancement
                enhanced_frame, enh_ms = apply_enhancement(raw_frame, mode)
                enh_times.append(enh_ms)

                # Compute image sharpness (Laplacian variance)
                gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                sharpness_list.append(sharpness)

                # 2. Run Detector (Enhanced Frame -> YOLOv8n)
                torch.cuda.synchronize()
                t_det = time.perf_counter()
                res = detector(enhanced_frame, conf=0.30, verbose=False, device=device)[0]
                torch.cuda.synchronize()
                det_ms = (time.perf_counter() - t_det) * 1000
                det_times.append(det_ms)

                boxes = res.boxes
                total_detections += len(boxes)
                if len(boxes) > 0:
                    conf_list.extend(boxes.conf.cpu().numpy().tolist())

                # 3. Depth Impact Experiment
                # Compare: Depth(Original) vs Depth(Enhanced)
                inputs_orig = processor(images=raw_frame, return_tensors="pt").to(device)
                inputs_enh = processor(images=enhanced_frame, return_tensors="pt").to(device)
                with torch.no_grad():
                    d_orig = depth_model(**inputs_orig).predicted_depth.squeeze().cpu().numpy()
                    d_enh = depth_model(**inputs_enh).predicted_depth.squeeze().cpu().numpy()
                
                # Normalize both to [0, 1]
                d_orig_n = (d_orig - d_orig.min()) / (d_orig.max() - d_orig.min() + 1e-8)
                d_enh_n = (d_enh - d_enh.min()) / (d_enh.max() - d_enh.min() + 1e-8)
                mae_depth = float(np.mean(np.abs(d_orig_n - d_enh_n)))
                depth_diffs.append(mae_depth)

                # Save evidence visual comparison on the 10th evaluated frame
                if not saved_comparison_frame:
                    vis_det = res.plot()
                    cv2.putText(vis_det, f"{mode} | Dets:{len(boxes)} | Enh:{enh_ms:.1f}ms", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imwrite(str(evidence_dir / f"{sc_name}_{mode}_det.png"), vis_det)
                    saved_comparison_frame = True

            mean_enh_ms = float(np.mean(enh_times))
            mean_det_ms = float(np.mean(det_times))
            total_lat = mean_enh_ms + mean_det_ms
            effective_fps = 1000.0 / max(total_lat, 1e-6)
            mean_conf = float(np.mean(conf_list)) if conf_list else 0.0
            mean_sharpness = float(np.mean(sharpness_list))
            mean_depth_mae = float(np.mean(depth_diffs))

            rec = {
                "scenario": sc_name,
                "enhancement": mode,
                "mean_enhancement_ms": round(mean_enh_ms, 2),
                "mean_detection_ms": round(mean_det_ms, 2),
                "total_latency_ms": round(total_lat, 2),
                "effective_fps": round(effective_fps, 2),
                "total_detections": total_detections,
                "mean_confidence": round(mean_conf, 3),
                "mean_sharpness": round(mean_sharpness, 1),
                "depth_divergence_mae": round(mean_depth_mae, 4),
            }
            enhancement_results.append(rec)
            print(f"  [{mode:12s}] Enh:{mean_enh_ms:5.1f}ms | Det:{mean_det_ms:5.1f}ms | Dets:{total_detections:2d} | Conf:{mean_conf:.2f} | Sharp:{mean_sharpness:5.1f} | DepthMAE:{mean_depth_mae:.3f}")

    # Save CSV outputs
    with open(rep_dir / "enhancement_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(enhancement_results[0].keys()))
        writer.writeheader()
        writer.writerows(enhancement_results)

    with open(rep_dir / "benchmark_summary.json", "w") as f:
        json.dump(enhancement_results, f, indent=2)

    print(f"\nPhase 2.5 suite successfully completed! Results saved to {rep_dir}")

if __name__ == "__main__":
    run_phase_2_5()

