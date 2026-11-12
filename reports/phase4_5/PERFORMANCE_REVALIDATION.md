# AUDIT F: PERFORMANCE & RESOURCE REVALIDATION REPORT

**Audit Date:** 2026-11-12  
**Target File:** `reports/phase4_5/PERFORMANCE_REVALIDATION.md`  
**Hardware Evaluated:** NVIDIA GeForce RTX 3050 6GB Laptop GPU | Windows 11  

---

## 1. VRAM Measurement Scope Clarification

In earlier sections, detector VRAM was reported at both ~24.4 MB and ~99.4 MB. The audit investigated the methodology:
- **24.4 MB:** Measured strictly as `torch.cuda.memory_allocated()` immediately after loading the raw PyTorch model tensor parameters.
- **99.4 MB:** Measured as `torch.cuda.max_memory_allocated()` during high-resolution batch forward pass tensor allocation (including intermediate activation maps and NMS tensors).
- **217.3 – 227.4 MB:** Measured during the **complete integrated pipeline execution** combining YOLOv8n, Depth Anything V2 Small transformer backbone, ByteTrack, and OpenCV rendering buffers.

---

## 2. Extended 100-Frame Stability Benchmark Results

Tested on real mining haul truck footage (`mine_truck.mp4`):

```
Metric                             Measured Result
---------------------------------  -----------------------------------
Frames Processed                   100 consecutive frames
Mean Frame Latency                 64.57 ms
Median Frame Latency               56.65 ms
P95 Frame Latency                  69.76 ms
Maximum Latency (Initialization)   728.13 ms (first frame cache warming)
Sustained Real-Time FPS            15.49 FPS
Active PyTorch VRAM Allocated      152.8 MB
Peak PyTorch VRAM Reserved         264.0 MB (out of 6144 MB available)
Process Host RAM (Start -> End)    835.7 MB -> 1547.8 MB (stable PyTorch CUDA context)
GPU Memory Leaks                   None observed (VRAM stabilized at 152.8 MB)
```

### Throughput Conclusion
The integrated system delivers consistent, stable throughput at **~15.5 FPS** on the RTX 3050 Laptop GPU, with predictable latency (P95 < 70 ms) and a GPU memory footprint under 300 MB.

