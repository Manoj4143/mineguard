"""Live validation and performance benchmark script for Phase 5 Dashboard Integration."""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DEPTHTRACK_DIR = ROOT / "DepthTrack" / "src"
MINEGUARD_DIR = Path("D:/MINI_GUARD_PROJECT/MineGuard-AI")

for p in [str(SRC_DIR), str(DEPTHTRACK_DIR), str(MINEGUARD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mineguard_depthtrack.service import PerceptionAdapter, BackgroundVideoWorker

SCENARIOS = {
    "mine_clear": str(ROOT / "DepthTrack" / "data" / "scenarios" / "mine_clear.mp4"),
    "mine_light_fog": str(ROOT / "DepthTrack" / "data" / "scenarios" / "mine_light_fog.mp4"),
    "mine_dense_fog": str(ROOT / "DepthTrack" / "data" / "scenarios" / "mine_dense_fog.mp4"),
    "mine_night": str(ROOT / "DepthTrack" / "data" / "scenarios" / "mine_night.mp4"),
}

EVIDENCE_DIR = ROOT / "reports" / "phase5" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def run_scenario_benchmark(adapter: PerceptionAdapter, name: str, video_path: str, num_frames: int = 50):
    print(f"\n--- Benchmarking scenario: {name} ({num_frames} frames) ---")
    worker = BackgroundVideoWorker(adapter, video_path)
    worker.start()

    latencies = []
    frames_collected = 0
    t_start = time.time()
    last_frame_id = -1
    evidence_saved = False

    while frames_collected < num_frames and (time.time() - t_start) < 60.0:
        jpeg_bytes, telem = worker.get_latest_data()
        fid = telem.get("worker_frame_id", -1)
        if fid > last_frame_id and jpeg_bytes is not None:
            last_frame_id = fid
            frames_collected += 1
            lat = telem.get("latency_ms", 0.0)
            if lat > 0:
                latencies.append(lat)

            if frames_collected == 20 and not evidence_saved:
                # Save snapshot evidence
                np_arr = np.frombuffer(jpeg_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                evidence_path = EVIDENCE_DIR / f"dashboard_{name}_frame20.jpg"
                cv2.imwrite(str(evidence_path), img)
                evidence_saved = True
                print(f"Saved evidence screenshot: {evidence_path.name}")

        time.sleep(0.01)

    worker.stop(timeout=2.0)

    elapsed = time.time() - t_start
    actual_fps = frames_collected / max(elapsed, 1e-6)
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    median_lat = float(np.median(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    # Read GPU memory if CUDA is active
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)

    metrics = {
        "scenario": name,
        "frames_collected": frames_collected,
        "duration_sec": round(elapsed, 2),
        "actual_worker_fps": round(actual_fps, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "gpu_vram_mb": vram_mb,
        "road_status": telem.get("road_status", "UNKNOWN"),
        "fog_condition": telem.get("fog_condition", "UNKNOWN"),
        "active_vehicles": telem.get("active_vehicles_count", 0),
    }
    print(f"Result {name}: FPS={actual_fps:.2f}, Latency={mean_lat:.1f}ms, Road={metrics['road_status']}")
    return metrics


def main():
    print("=" * 65)
    print("MINEGUARD AI — PHASE 5 DASHBOARD INTEGRATION VALIDATION BENCHMARK")
    print("=" * 65)

    adapter = PerceptionAdapter(preferred_engine=PerceptionAdapter.ENGINE_UNIFIED)
    results = {}

    for name, path in SCENARIOS.items():
        if not Path(path).exists():
            print(f"Warning: Scenario video {path} not found.")
            continue
        metrics = run_scenario_benchmark(adapter, name, path, num_frames=35)
        results[name] = metrics

    # Also test fallback engine compatibility
    print("\n--- Testing Fallback Baseline (MineGuardIntegratedSystem) ---")
    adapter.set_engine(PerceptionAdapter.ENGINE_FALLBACK)
    fallback_metrics = run_scenario_benchmark(adapter, "fallback_baseline", SCENARIOS["mine_clear"], num_frames=25)
    results["fallback_baseline"] = fallback_metrics

    # Save summary
    out_json = ROOT / "reports" / "phase5" / "live_validation_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll benchmark metrics saved to: {out_json}")


if __name__ == "__main__":
    main()

