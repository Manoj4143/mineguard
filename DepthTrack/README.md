# DepthTrack

DepthTrack is a monocular dashcam perception pipeline combining **YOLO** vehicle detection,
**Depth Anything V2** depth estimation, and **ByteTrack** multi-object tracking
into an annotated, ADAS-style analysis video — no LiDAR, stereo rig, or radar
required.

From a single RGB dashcam stream it detects and tracks surrounding vehicles,
estimates depth and per-object distance, scores collision risk, and renders a
synchronized **bird's-eye-view (BEV)** of the scene. The output is a
side-by-side video: annotated dashcam on the left, 3D depth/BEV reconstruction
on the right.

---

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Development](#development)
- [Docker](#docker)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Features

- **Vehicle detection** — YOLOv8 restricted to vehicle classes (car, truck,
  bus, motorcycle, bicycle).
- **Monocular depth estimation** — Depth Anything V2 produces a dense depth map
  from a single camera, with temporal smoothing to reduce frame-to-frame
  flicker.
- **Multi-object tracking** — ByteTrack assigns stable IDs across frames, with
  per-track trajectory trails and velocity arrows.
- **Distance estimation** — fuses a depth cue with a pinhole-projection
  estimate for each tracked vehicle.
- **Collision-risk scoring** — classifies each object as `SAFE`, `CAUTION`,
  `WARNING`, or `DANGER` based on distance and lateral lane position.
- **Bird's-eye-view rendering** — projects the depth map into a pseudo-3D BEV
  point cloud with a perspective ground grid and 3D object boxes.
- **Configurable & reproducible** — every parameter lives in a typed config
  with YAML and CLI overrides; dependencies are pinned.
- **CPU and CUDA** — runs on CPU out of the box; uses the GPU when available.

---

## How it works

Each frame flows through the following pipeline:

```
                 +-----------------+
   input frame   |   YOLOv8        |  detections
  ------------->-|   detector      |------------+
       |         +-----------------+            v
       |                                 +---------------+   tracked
       |                                 |  ByteTrack    |   objects
       |                                 |  tracker      |-----------+
       |         +-----------------+     +---------------+           |
       +-------->-| Depth Anything |  depth map                      |
                 | V2 (smoothed)  |---------------+                  |
                 +-----------------+               v                  v
                                          +-------------------------------+
                                          | distance + collision-risk     |
                                          | estimation (per object)       |
                                          +-------------------------------+
                                                       |
                          +----------------------------+----------------------------+
                          v                                                         v
                 +-----------------+                                      +-----------------+
                 | Left panel:     |                                      | Right panel:    |
                 | dashcam + boxes |                                      | BEV point cloud |
                 | + trails + risk |                                      | + 3D risk boxes |
                 +-----------------+                                      +-----------------+
                          |                                                         |
                          +-----------------------+---------------------------------+
                                                  v
                                    composite + HUD -> ffmpeg encode -> output.mp4
```

The distance estimate for each bounding box blends two independent cues:

1. **Depth cue** — the median normalised depth inside the box, mapped onto a
   metric near/far range.
2. **Projection cue** — a pinhole estimate
   `distance = (focal_length × real_width) / pixel_width`.

Risk is flagged as `DANGER` only when an object is both close **and** roughly
within the ego lane, so a nearby parked car off to the side is not flagged as
an imminent collision.

---

## Project structure

```
.
├── src/depthtrack/
│   ├── cli.py                  # command-line interface
│   ├── config.py               # typed, validated configuration
│   ├── logging_utils.py        # logging setup
│   ├── pipeline.py             # orchestration + video I/O + ffmpeg
│   ├── models/
│   │   ├── detector.py         # YOLOv8 wrapper
│   │   ├── depth.py            # Depth Anything V2 wrapper
│   │   └── tracker.py          # ByteTrack + trajectory/velocity history
│   ├── perception/
│   │   ├── distance.py         # distance estimation
│   │   └── risk.py             # collision-risk classification
│   └── visualization/
│       ├── colormap.py         # turbo depth colormap
│       ├── bev.py              # bird's-eye-view projection
│       └── panels.py           # panel rendering, compositing, HUD
├── config/default.yaml         # documented default configuration
├── assets/yolov8n.pt           # bundled detector weights
├── tests/                      # unit tests (28 passing)
├── pyproject.toml              # packaging + tooling
├── requirements.txt            # pinned runtime dependencies
├── Dockerfile
└── Makefile
```

---

## Installation

**Requirements:** Python 3.9+, and [ffmpeg](https://ffmpeg.org/download.html) on
your `PATH` for the final video compression step (optional — DepthTrack writes
an uncompressed file if ffmpeg is absent).

```bash
git clone https://github.com/your-username/DepthTrack
cd DepthTrack

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# CPU (default)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pip install -e .
```

Or with Make:

```bash
make install
```

> **GPU users:** install the CUDA build of `torch` / `torchvision` for your
> CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)
> *before* the rest, then pass `--device cuda`.

---

## Usage

```bash
depthtrack path/to/dashcam.mp4
```

Module form (no install needed if `src/` is on `PYTHONPATH`):

```bash
python -m depthtrack path/to/dashcam.mp4
```

### Common options

```bash
# Supply a custom config file
depthtrack input.mp4 --config config/default.yaml

# Run on GPU
depthtrack input.mp4 --device cuda

# Tighten the confidence threshold and redirect output
depthtrack input.mp4 --conf 0.5 --output-dir results/

# Verbose logging
depthtrack input.mp4 --log-level DEBUG
```

| Option | Description | Default |
|---|---|---|
| `input` | Path to input dashcam video | — |
| `-c, --config` | YAML config file | none |
| `--device` | `cpu` or `cuda` | `cpu` |
| `--conf` | YOLO confidence threshold | `0.35` |
| `-o, --output-dir` | Output directory | `output` |
| `--assets-dir` | Directory containing model weights | `assets` |
| `--log-level` | Logging verbosity | `INFO` |

---

## Configuration

All tunables are documented in [`config/default.yaml`](config/default.yaml).
Copy it, edit what you need, and pass it with `--config`. Omitted keys fall
back to built-in defaults; invalid values are rejected at startup.

| Section | Controls |
|---|---|
| `model` | Detector weights, depth model, confidence, device, vehicle classes |
| `distance` | Focal length, assumed vehicle width, depth range, blend weight |
| `risk` | Distance thresholds for each risk level and in-lane lateral offset |
| `video` | Output resolution, working resolution, ffmpeg encode settings |

---

## Output

Results land in the output directory (`output/` by default):

- **`final_output.mp4`** — the composited video, H.264-encoded via ffmpeg
  (`+faststart` for web playback) if available, otherwise raw `mp4v`.

The **left panel** shows the dashcam feed annotated with colour-coded detection
boxes (green → cyan → orange → red by risk), stable-ID trajectory trails, and
velocity arrows. The **right panel** shows the bird's-eye-view depth point cloud
with 3D risk boxes. A HUD strip reports live FPS, frame progress, and device.

---

## Development

```bash
make dev          # pip install -e ".[dev]"
make test         # pytest  (28 tests, no GPU required)
make lint         # ruff
make typecheck    # mypy
```

The test suite covers config validation, distance estimation, risk
classification, the depth colormap, and BEV projection. It runs without the
heavy deep-learning stack and is fast enough for CI on every push.

---

## Docker

```bash
docker build -t depthtrack .

docker run --rm \
  -v "$PWD/data:/data" \
  -v "$PWD/output:/app/output" \
  depthtrack /data/input.mp4
```

---

## Limitations

DepthTrack is a perception **demonstrator**, not a calibrated safety system:

- Distance and BEV positions are heuristic estimates from a single uncalibrated
  camera and should not be used for vehicle control.
- The pinhole cue assumes a fixed real-world vehicle width of 2 m.
- Depth is relative (per-frame normalised), not absolute metric depth.
- Real-time throughput generally requires a GPU.

Do not use this software for real driving decisions.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — vehicle detection
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — monocular depth estimation
- [Supervision / ByteTrack](https://github.com/roboflow/supervision) — multi-object tracking

---

## License

MIT © 2026 Harsh Patil — see [LICENSE](LICENSE).
