#!/usr/bin/env python3
"""benchmark_pipeline.py - Validate end-to-end DepthTrack pipeline across scenarios on CUDA."""

import os
import time
import json
import csv
from pathlib import Path
import torch

from depthtrack.config import AppConfig
from depthtrack.pipeline import PerceptionPipeline

def run_pipeline_benchmarks():
    print("=" * 60)
    print("END-TO-END PIPELINE VALIDATION ON CUDA")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_yaml = Path("DepthTrack/config/default.yaml")
    if default_yaml.exists():
        cfg = AppConfig.from_yaml(default_yaml)
    else:
        cfg = AppConfig()
    cfg.model.device = device
    
    scenarios_dir = Path("DepthTrack/data/scenarios")
    videos = sorted(list(scenarios_dir.glob("*.mp4")))
    if not videos:
        videos = [Path("DepthTrack/data/sample_test.mp4")]
        
    results = []
    
    out_dir = Path("DepthTrack/output/scenarios")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for vid in videos:
        print(f"\nProcessing scenario: {vid.name} on {device}...")
        cfg.video.output_dir = str(out_dir)
        cfg.video.output_name = f"result_{vid.stem}.mp4"
        
        pipeline = PerceptionPipeline(cfg, asset_dir=Path("DepthTrack/assets"))
        
        # Reset peak memory
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            
        t0 = time.time()
        final_video = pipeline.run(str(vid))
        total_time = time.time() - t0
        
        max_vram = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
        
        # Open video to get frame count
        import cv2
        cap = cv2.VideoCapture(str(vid))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        in_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        effective_fps = n_frames / max(total_time, 1e-6)
        
        rec = {
            "scenario": vid.stem,
            "device": device,
            "resolution": f"{w}x{h}",
            "frames": n_frames,
            "input_fps": in_fps,
            "duration_s": round(total_time, 2),
            "effective_fps": round(effective_fps, 2),
            "max_vram_mb": round(max_vram, 1),
            "output_video": str(final_video),
            "output_size_mb": round(final_video.stat().st_size / (1024**2), 2),
        }
        results.append(rec)
        print(f"  -> Finished {n_frames} frames in {total_time:.2f}s ({effective_fps:.2f} FPS), VRAM={max_vram:.1f}MB")

    # Save summary
    rep_dir = Path("reports/phase1")
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    with open(rep_dir / "pipeline_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open(rep_dir / "pipeline_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nSaved pipeline benchmark to {rep_dir / 'pipeline_benchmark.csv'}")

if __name__ == "__main__":
    run_pipeline_benchmarks()
