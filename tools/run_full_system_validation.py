#!/usr/bin/env python3
"""run_full_system_validation.py - Phase 4: Comparative Validation & Regression Suite."""

import os
import sys
import time
import json
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

# Paths
ROOT = Path("D:/Mineguard_project_2")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "DepthTrack/src"))

MINEGUARD_DIR = Path("D:/MINI_GUARD_PROJECT/MineGuard-AI")
sys.path.insert(0, str(MINEGUARD_DIR))

# Import Original MineGuard Engine
from mineguard_core import MineGuardIntegratedSystem as OriginalMineGuard

# Import Integrated MineGuard-DepthTrack Engine
from mineguard_depthtrack import UnifiedMineGuardPerception as IntegratedMineGuard

def run_phase_4_validation():
    print("=" * 70)
    print("PHASE 4: FULL-SYSTEM COMPARATIVE VALIDATION & REGRESSION TESTING")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Platform: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    rep_dir = Path("reports/phase4")
    rep_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = rep_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Initialize both systems
    print("\n[1] Initializing Original MineGuard Engine (YOLO11n baseline)...")
    orig_engine = OriginalMineGuard(
        model_path="D:/MINI_GUARD_PROJECT/yolo11n.pt",
        device=0 if device == "cuda" else "cpu",
    )

    print("\n[2] Initializing Integrated MineGuard Engine (DepthTrack perception)...")
    integ_engine = IntegratedMineGuard(
        detector_weights="DepthTrack/assets/yolov8n.pt",
        device=device,
    )

    test_videos = {
        "clear_mine_road": "D:/MINI_GUARD_PROJECT/MineGuard-AI/videos/cvat_clean_videos/clear_clean.mp4",
        "light_fog_road": "D:/MINI_GUARD_PROJECT/MineGuard-AI/videos/cvat_clean_videos/light_fog_clean.mp4",
        "dense_fog_road": "D:/MINI_GUARD_PROJECT/MineGuard-AI/videos/cvat_clean_videos/dense_fog_clean.mp4",
        "night_haul_road": "DepthTrack/data/scenarios/mine_night.mp4",
    }

    perf_comparison = []
    detection_comparison = []
    tracking_comparison = []
    alert_comparison = []

    print("\n[3] Executing Comparative Runs Across Real & Synthesized Scenarios...")

    for sc_name, v_path in test_videos.items():
        if not os.path.exists(v_path):
            print(f"Skipping {sc_name} (path not found: {v_path})")
            continue

        print(f"\n--- Scenario: {sc_name.upper()} ---")
        cap = cv2.VideoCapture(v_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        eval_count = min(30, total_frames)

        frames = []
        for _ in range(eval_count):
            ret, fr = cap.read()
            if not ret:
                break
            frames.append(fr)
        cap.release()

        # Benchmark Original MineGuard
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        orig_dets = 0
        orig_tracks = set()
        orig_alerts = {"SAFE": 0, "CAUTION": 0, "CRITICAL": 0}

        for idx, fr in enumerate(frames):
            ts = idx * 0.1
            res_orig = orig_engine.process_frame(fr, frame_id=idx)
            v_list = res_orig.get("telemetry", {}).get("vehicles", [])
            orig_dets += len(v_list)
            for v in v_list:
                orig_tracks.add(v.get("id"))
                tier = v.get("risk_tier", "SAFE")
                if tier in orig_alerts:
                    orig_alerts[tier] += 1

        t_orig_total = time.perf_counter() - t0
        orig_fps = len(frames) / max(t_orig_total, 1e-6)
        orig_vram = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

        # Benchmark Integrated MineGuard
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        integ_dets = 0
        integ_tracks = set()
        integ_alerts = {"SAFE": 0, "CAUTION": 0, "CRITICAL": 0}
        saved_comp = False

        for idx, fr in enumerate(frames):
            ts = idx * 0.1
            res_integ = integ_engine.process_frame(fr, timestamp=ts)
            objs = res_integ.get("objects", [])
            integ_dets += len(objs)
            for o in objs:
                integ_tracks.add(o.get("track_id"))
                tier = o.get("risk_tier", "SAFE")
                if tier in integ_alerts:
                    integ_alerts[tier] += 1

            if idx == 15 and not saved_comp:
                # Save visual comparison frame
                comp_img = res_integ["composite_frame"]
                cv2.imwrite(str(evidence_dir / f"final_{sc_name}_frame15.png"), comp_img)
                saved_comp = True

        t_integ_total = time.perf_counter() - t0
        integ_fps = len(frames) / max(t_integ_total, 1e-6)
        integ_vram = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

        print(f"  [Original MineGuard]   FPS: {orig_fps:5.1f} | Detections: {orig_dets:2d} | Unique Tracks: {len(orig_tracks)} | VRAM: {orig_vram:5.1f}MB")
        print(f"  [Integrated DepthTrack]FPS: {integ_fps:5.1f} | Detections: {integ_dets:2d} | Unique Tracks: {len(integ_tracks)} | VRAM: {integ_vram:5.1f}MB")

        perf_comparison.append({
            "scenario": sc_name,
            "original_fps": round(orig_fps, 1),
            "integrated_fps": round(integ_fps, 1),
            "original_vram_mb": round(orig_vram, 1),
            "integrated_vram_mb": round(integ_vram, 1),
        })

        detection_comparison.append({
            "scenario": sc_name,
            "frames_evaluated": len(frames),
            "original_total_detections": orig_dets,
            "integrated_total_detections": integ_dets,
            "detection_gain_pct": round(((integ_dets - orig_dets) / max(orig_dets, 1)) * 100, 1),
        })

        tracking_comparison.append({
            "scenario": sc_name,
            "original_unique_tracks": len(orig_tracks),
            "integrated_unique_tracks": len(integ_tracks),
        })

        alert_comparison.append({
            "scenario": sc_name,
            "original_safe": orig_alerts["SAFE"],
            "original_caution": orig_alerts["CAUTION"],
            "original_critical": orig_alerts["CRITICAL"],
            "integrated_safe": integ_alerts["SAFE"],
            "integrated_caution": integ_alerts["CAUTION"],
            "integrated_critical": integ_alerts["CRITICAL"],
        })

    # Save CSV outputs
    def save_csv(filename, data):
        with open(rep_dir / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    save_csv("PERFORMANCE_EVALUATION.csv", perf_comparison)
    save_csv("DETECTION_EVALUATION.csv", detection_comparison)
    save_csv("TRACKING_EVALUATION.csv", tracking_comparison)
    save_csv("ALERT_EVALUATION.csv", alert_comparison)

    summary = {
        "hardware": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        },
        "performance": perf_comparison,
        "detection": detection_comparison,
        "tracking": tracking_comparison,
        "alerts": alert_comparison,
    }
    with open(rep_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPhase 4 Validation completed! Output files written to {rep_dir}")

if __name__ == "__main__":
    run_phase_4_validation()

