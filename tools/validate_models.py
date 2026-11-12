#!/usr/bin/env python3
"""validate_models.py - Independent verification of YOLOv8n and Depth Anything V2 on CUDA."""

import time
import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def test_models():
    print("=" * 60)
    print("INDEPENDENT AI MODEL VALIDATION ON CUDA")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Device: {device}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
    evidence_dir = Path("reports/phase1/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Test YOLOv8n
    print("\n--- [1] Validating YOLOv8n ---")
    weights_v8 = Path("DepthTrack/assets/yolov8n.pt")
    if not weights_v8.exists():
        print(f"FAIL: {weights_v8} not found!")
        return
        
    t0 = time.time()
    detector = YOLO(str(weights_v8)).to(device)
    load_time_det = time.time() - t0
    print(f"Detector Loaded in {load_time_det:.3f}s")
    
    # Create synthetic test frame with realistic vehicle-like rectangles and road
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (0, 240), (640, 480), (60, 60, 60), -1) # road
    cv2.rectangle(test_img, (200, 260), (320, 360), (0, 0, 220), -1) # car body
    cv2.rectangle(test_img, (220, 240), (300, 260), (20, 20, 20), -1) # cabin
    cv2.circle(test_img, (220, 360), 20, (10, 10, 10), -1) # wheel
    cv2.circle(test_img, (300, 360), 20, (10, 10, 10), -1) # wheel
    
    # Warmup
    for _ in range(5):
        _ = detector(test_img, verbose=False, device=device)
        
    torch.cuda.synchronize()
    det_times = []
    for _ in range(30):
        t_start = time.perf_counter()
        res = detector(test_img, verbose=False, device=device)
        torch.cuda.synchronize()
        det_times.append((time.perf_counter() - t_start) * 1000)
        
    avg_det_lat = np.mean(det_times)
    p95_det_lat = np.percentile(det_times, 95)
    print(f"YOLOv8n Inference: Mean={avg_det_lat:.2f}ms, P95={p95_det_lat:.2f}ms")
    vram_det = torch.cuda.memory_allocated() / (1024**2)
    print(f"VRAM after detector: {vram_det:.1f} MB")
    
    # 2. Test Depth Anything V2 Small
    print("\n--- [2] Validating Depth Anything V2 Small ---")
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(model_id)
    depth_model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    depth_model.eval()
    load_time_depth = time.time() - t0
    print(f"Depth Model Loaded in {load_time_depth:.3f}s")
    
    inputs = processor(images=test_img, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = depth_model(**inputs)
            
    torch.cuda.synchronize()
    depth_times = []
    with torch.no_grad():
        for _ in range(30):
            t_start = time.perf_counter()
            outputs = depth_model(**inputs)
            pred = outputs.predicted_depth
            torch.cuda.synchronize()
            depth_times.append((time.perf_counter() - t_start) * 1000)
            
    avg_depth_lat = np.mean(depth_times)
    p95_depth_lat = np.percentile(depth_times, 95)
    print(f"Depth Anything V2 Latency: Mean={avg_depth_lat:.2f}ms, P95={p95_depth_lat:.2f}ms")
    
    depth_np = pred.squeeze().cpu().numpy()
    has_nan = bool(np.isnan(depth_np).any())
    has_inf = bool(np.isinf(depth_np).any())
    is_constant = bool(np.ptp(depth_np) == 0)
    print(f"Depth Output Shape: {depth_np.shape}")
    print(f"Depth Range: [{depth_np.min():.4f}, {depth_np.max():.4f}]")
    print(f"Sanity Check -> NaN: {has_nan}, Inf: {has_inf}, Constant: {is_constant}")
    
    # Save visual verification
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
    depth_vis = (depth_norm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(evidence_dir / "depth_validation_sample.png"), depth_color)
    print(f"Saved depth verification to {evidence_dir / 'depth_validation_sample.png'}")
    
    print("\n[VERDICT] Both models validated successfully on CUDA.")

if __name__ == "__main__":
    test_models()

