# MINEGUARD AI — PHASE 5 PERFORMANCE REPORT
## Live Integrated Dashboard Performance Benchmark

**Date:** November 12, 2026  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA 12.8)  
**Host:** Windows 11, Intel 13th Gen, Python 3.13.1  
**Methodology:** Live background worker processing synthetic scenarios with frame-by-frame telemetry capture and JPEG compression.

---

## 1. Scenario Benchmark Metrics

| Metric | mine_clear | mine_light_fog | mine_dense_fog | mine_night | Fallback Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Active Engine** | Unified DepthTrack | Unified DepthTrack | Unified DepthTrack | Unified DepthTrack | MineGuardIntegratedSystem |
| **Frames Sampled** | 35 frames | 35 frames | 35 frames | 35 frames | 25 frames |
| **Worker Real-time FPS** | **10.63 FPS** | **13.41 FPS** | **13.55 FPS** | **13.68 FPS** | **24.79 FPS** |
| **Mean Latency (ms)** | 142.67 ms* | 66.30 ms | 54.58 ms | 53.71 ms | 30.96 ms |
| **Median Latency (ms)** | 97.10 ms | 69.10 ms | 55.10 ms | 54.10 ms | 20.01 ms |
| **P95 Latency (ms)** | 338.44 ms* | 76.28 ms | 55.70 ms | 54.80 ms | 67.24 ms |
| **GPU VRAM Allocated** | 152.79 MB | 152.79 MB | 152.79 MB | 152.79 MB | 114.50 MB |
| **Road Clearance Status**| ROAD CLEAR | ROAD CLEAR | ROAD CLEAR | ROAD CLEAR | ROAD CLEAR |
| **Atmospheric Assessment**| CLEAR | LIGHT_FOG | DENSE_FOG | CLEAR | CLEAR |

*\*First run on `mine_clear` includes initial CUDA runtime kernel instantiation and graph caching. All subsequent scenarios demonstrated tight latency clusters between 53 ms and 69 ms.*

---

## 2. Resource Utilization & Stability Observations

1. **VRAM Footprint:**
   - Both YOLOv8n and Depth Anything V2 Small share the CUDA device with an allocated memory footprint of only **152.79 MB** (out of 6,144 MB available).
   - No memory leakage was observed across 140+ continuous live frames.
2. **Worker Queue Resilience:**
   - Under real-time MJPEG consumption, the latest-frame buffer prevents memory expansion. If a slow client drops connection, the worker maintains steady throughput without stuttering or memory spikes.
3. **Comparison Against Phase 4.5 Baseline:**
   - Phase 4.5 Standalone Benchmark: 15.49 FPS.
   - Phase 5 Dashboard Integrated: 13.4–13.7 FPS (difference accounted for by composite frame JPEG compression, HTTP socket serialization, and UI telemetry aggregation).

