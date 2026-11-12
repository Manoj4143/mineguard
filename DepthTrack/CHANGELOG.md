# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.2] - 2026-06-06

### Fixed
- **Inverted depth cue (major accuracy bug).** Depth Anything V2 outputs
  *inverse* depth (high value = near), but the distance estimator treated a
  high value as far. This inverted every estimate: vehicles directly ahead
  were reported as distant/`SAFE` while far-off vehicles read as close. The
  depth term in `estimate_distance()` is now correctly inverted so high depth
  maps to a short distance.
- BEV rendering was inverted to match: the point cloud and 3D object boxes now
  place near objects at the bottom of the view (where the perspective grid is
  widest) and far objects toward the horizon, with near objects drawn larger.

## [1.2.1] - 2026-06-06

### Fixed
- Panel titles displayed `??` instead of `·` because OpenCV's built-in fonts
  are ASCII-only. Replaced the Unicode middle-dot separator with `|`.
- Distance estimation was biased by vehicle class: all objects used a fixed
  2.0 m real-world width, causing trucks (actual ~2.5 m) to be reported ~25%
  closer than reality and motorcycles/bicycles to be wildly overestimated.
  Added a `vehicle_widths_m` lookup in `DistanceConfig` with per-class widths
  (car=1.8 m, truck/bus=2.5 m, motorcycle=0.8 m, bicycle=0.6 m). The
  `estimate_distance()` function now accepts a `class_id` argument and uses
  the correct width for each object. Widths are configurable in
  `config/default.yaml`.

## [1.2.0] - 2026-06-06

### Changed
- Refactored the monolithic `app.py` into an installable `depthtrack`
  package with clear `models`, `perception`, and `visualization` modules.
- Replaced runtime dependency auto-installation with `requirements.txt` and
  `pyproject.toml`.
- Replaced `print` logging with the standard `logging` module.
- Moved all magic numbers into a typed, validated configuration system
  (`config.py`) with YAML support and CLI overrides.

### Added
- Unit test suite (`pytest`) covering config, distance, risk, and rendering.
- `RiskLevel` enum for collision-risk levels.
- Dockerfile, Makefile, CI workflow, and contributor docs.
- A real CLI with `--config`, `--device`, `--conf`, `--output-dir`, and
  `--log-level` options plus `--version`.

### Fixed
- Corrected a misleading dependency comment that referenced a non-existent
  torch version.
- CUDA requests now cleanly fall back to CPU when unavailable.