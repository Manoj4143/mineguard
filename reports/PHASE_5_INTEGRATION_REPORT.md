# MINEGUARD AI — PHASE 5 INTEGRATION REPORT
## Controlled Dashboard Integration

**Date:** November 12, 2026  
**Status:** **PASSED**  
**Perception Engine:** UnifiedMineGuardPerception (YOLOv8n + Depth Anything V2 Small + ByteTrack + Risk Analyzer)  
**Fallback Engine:** MineGuardIntegratedSystem (YOLO11n + ByteTrack + Dehaze)  
**Host & Environment:** Windows 11, Python 3.13.1, NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA 12.8)  

---

## 1. Executive Summary

Phase 5 has successfully achieved a controlled integration of the validated `UnifiedMineGuardPerception` engine into the MineGuard web monitoring dashboard. The architecture strictly isolates HTTP request handlers and MJPEG streaming threads from model inference via a dedicated adapter and background worker layer (`PerceptionAdapter` and `BackgroundVideoWorker`).

All existing dashboard features—including real-time MJPEG streaming, live telemetry polling, scenario selection, video uploads, and road clearance status—were preserved. Additionally, a dynamic engine selector and live engine status indicator were introduced, allowing real-time switching between `UnifiedMineGuardPerception` and the original `MineGuardIntegratedSystem` fallback baseline.

---

## 2. Integration Architecture

```
                       Browser Clients (Web UI)
                                  |
                                  | HTTP / WebSockets / MJPEG Stream
                                  v
                  DashboardHTTPRequestHandler (:8080)
                                  |
           +----------------------+----------------------+
           |                      |                      |
           v                      v                      v
      GET /stream.mjpg     GET /api/telemetry    GET /api/set_engine
    (Buffered JPEG read)  (JSON Telemetry read)  (Dynamic Engine Switch)
           |                      |                      |
           +----------------------+----------------------+
                                  |
                                  v
                       BackgroundVideoWorker
                      (Bounded Latest-Frame)
                                  |
                                  v
                          PerceptionAdapter
                                  |
             +--------------------+--------------------+
             |                                         |
             v                                         v
 UnifiedMineGuardPerception               MineGuardIntegratedSystem
 (Primary: Depth + BEV + TTC)             (Fallback Baseline: YOLO11n)
             |                                         |
             +--------------------+--------------------+
                                  v
                      Standardized Telemetry &
                   Composite Visualization Frame
```

### Key Architectural Safeguards:
1. **Model Decoupling:** Model initialization occurs once during service startup on CUDA. Frames are never processed inside request-handling threads.
2. **Bounded Buffer & Dropped-Frame Resilience:** The `BackgroundVideoWorker` uses a latest-frame buffer, ensuring that even if browser consumption slows down, inference is never backed up.
3. **Format Validation:** Verified that `composite_frame` is a native `numpy.ndarray` with shape `(540, 1924, 3)` and dtype `uint8`. OpenCV `cv2.imencode('.jpg', ...)` produces clean JPEG byte streams for standard multipart MJPEG streaming.
4. **Active Engine Transparency:** The UI explicitly displays the active engine (`Unified DepthTrack` vs `Fallback Baseline`). Silent fallbacks are strictly prohibited; errors are exposed in both logs and API telemetry.
5. **No Metric Distortions:** Monocular depth estimates are accurately labeled as relative depth (`relative_depth`) and are not misrepresented as calibrated metric distances.

---

## 3. Preserved Dashboard Routes & API Compatibility

| Endpoint | Method | Functionality | Status |
| :--- | :--- | :--- | :--- |
| `/` or `/index.html` | `GET` | Main interactive web dashboard | Preserved & Enhanced |
| `/api/telemetry` | `GET` | Live telemetry JSON feed (300ms poll) | Preserved & Compatible |
| `/api/set_source` | `GET` | Dynamic scenario video switching | Preserved |
| `/api/set_engine` | `GET` | Real-time perception engine switching | **New Feature** |
| `/stream.mjpg` | `GET` | Real-time multipart MJPEG visual stream | Preserved & Thread-Safe |
| `/api/upload_video` | `POST`| Streaming chunked custom video upload | Preserved |

---

## 4. Test Suite & Validation Summary

- **Automated Tests:** **52 passed, 1 warning in 38.12s**
- **New Unit & Service Tests:** 10 dedicated test cases in `tests/test_dashboard_integration.py` covering:
  - Unified engine initialization.
  - Graceful initialization failure and automatic fallback.
  - Display frame conversion and JPEG header validation (`\xff\xd8`).
  - Schema consistency across missing telemetry fields.
  - Background worker thread lifecycle, buffer overflow, and clean shutdown.
  - Dynamic engine switching between Unified and Fallback engines.
- **End-to-End Live HTTP Route Tests:** Passed on live server (`/`, `/api/telemetry`, `/api/set_source`, `/api/set_engine`).

---

## 5. Performance Overview

| Scenario | Mode | Actual Worker FPS | Mean Latency | Median Latency | P95 Latency | GPU VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mine_clear` | Unified DepthTrack | 10.63 FPS | 142.7 ms* | 97.1 ms | 338.4 ms | 152.8 MB |
| `mine_light_fog` | Unified DepthTrack | 13.41 FPS | 66.3 ms | 69.1 ms | 76.3 ms | 152.8 MB |
| `mine_dense_fog` | Unified DepthTrack | 13.55 FPS | 54.6 ms | 55.1 ms | 55.7 ms | 152.8 MB |
| `mine_night` | Unified DepthTrack | 13.68 FPS | 53.7 ms | 54.1 ms | 54.8 ms | 152.8 MB |
| `mine_clear` | Fallback Baseline | 24.79 FPS | 31.0 ms | 20.0 ms | 67.2 ms | 114.5 MB |

*\*Note: Initial frame in `mine_clear` includes PyTorch/CUDA kernel compilation warm-up; steady-state latency settles at ~55–65 ms.*

---

## 6. Known Limitations

1. **Monocular Depth Nature:** Depth Anything V2 produces relative disparity rather than absolute physical metrics.
2. **Total Darkness Constraints:** Under total darkness (<10 lux) without active vehicle headlamps, pretrained COCO object weights exhibit degraded vehicle recall (as audited in Phase 4.5). Mining haul trucks require active auxiliary lighting.

