"""Typed, validated configuration for the perception pipeline.

All tunable parameters live here as dataclasses so that nothing in the codebase
relies on scattered "magic numbers". A configuration can be built from defaults,
overridden from a YAML file, or patched from command-line arguments.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# COCO class IDs emitted by the YOLO model that we treat as "vehicles".
DEFAULT_VEHICLE_CLASSES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class ModelConfig:
    """Detection / depth / tracking model settings."""

    detector_weights: str = "yolov8n.pt"
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    confidence: float = 0.35
    device: str = "cpu"  # "cpu" or "cuda"
    track_history: int = 30
    vehicle_classes: dict[int, str] = field(
        default_factory=lambda: dict(DEFAULT_VEHICLE_CLASSES)
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(f"confidence must be in (0, 1], got {self.confidence}")
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {self.device!r}")
        if self.track_history < 1:
            raise ValueError("track_history must be >= 1")
        # YAML maps come back with string keys; normalise to int.
        self.vehicle_classes = {int(k): v for k, v in self.vehicle_classes.items()}


@dataclass
class DistanceConfig:
    """Parameters for the depth + pinhole-projection distance estimator."""

    focal_length_px: float = 700.0
    real_vehicle_width_m: float = 2.0  # fallback for unknown classes
    # Per-class real-world widths (metres). Keys are COCO class IDs.
    # A truck/bus is ~2.5m wide; a car ~1.8m; motorcycle/bicycle much narrower.
    vehicle_widths_m: dict[int, float] = field(default_factory=lambda: {
        1: 0.6,   # bicycle
        2: 1.8,   # car
        3: 0.8,   # motorcycle
        5: 2.5,   # bus
        7: 2.5,   # truck
    })
    depth_near_m: float = 1.0
    depth_far_m: float = 80.0
    depth_weight: float = 0.5  # blend between depth-based and projection-based estimate
    min_distance_m: float = 1.0
    max_distance_m: float = 100.0

    def width_for_class(self, class_id: int) -> float:
        """Return the real-world width (m) for a given COCO class id."""
        return self.vehicle_widths_m.get(class_id, self.real_vehicle_width_m)

    def __post_init__(self) -> None:
        if not 0.0 <= self.depth_weight <= 1.0:
            raise ValueError("depth_weight must be in [0, 1]")
        if self.min_distance_m >= self.max_distance_m:
            raise ValueError("min_distance_m must be < max_distance_m")
        # Normalise YAML string keys to int
        self.vehicle_widths_m = {int(k): float(v) for k, v in self.vehicle_widths_m.items()}


@dataclass
class RiskConfig:
    """Distance thresholds (metres) for collision-risk classification."""

    danger_m: float = 8.0
    warning_m: float = 15.0
    caution_m: float = 25.0
    danger_lateral_offset: float = 0.4  # fraction of half-width; danger only if in-lane

    def __post_init__(self) -> None:
        if not self.danger_m < self.warning_m < self.caution_m:
            raise ValueError("risk thresholds must satisfy danger < warning < caution")


@dataclass
class VideoConfig:
    """Input/output video and rendering settings."""

    output_dir: str = "output"
    output_name: str = "final_output.mp4"
    output_width: int = 1920
    output_height: int = 540
    divider_width: int = 4
    work_width: int = 640  # frames are downscaled to this width for inference
    depth_input_width: int = 320
    depth_input_height: int = 192
    default_fps: float = 25.0
    # ffmpeg re-encode settings
    encode_codec: str = "libx264"
    encode_preset: str = "fast"
    encode_crf: int = 23

    @property
    def panel_width(self) -> int:
        return self.output_width // 2

    @property
    def panel_height(self) -> int:
        return self.output_height


@dataclass
class AppConfig:
    """Top-level configuration aggregating every sub-config."""

    model: ModelConfig = field(default_factory=ModelConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    # ---- construction helpers -------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AppConfig:
        """Build a config from a (possibly partial) nested mapping."""
        kwargs: dict[str, Any] = {}
        section_types = {f.name: f.type for f in fields(cls)}
        type_map = {
            "model": ModelConfig,
            "distance": DistanceConfig,
            "risk": RiskConfig,
            "video": VideoConfig,
        }
        for name, section_cls in type_map.items():
            section = data.get(name, {}) or {}
            if not isinstance(section, Mapping):
                raise TypeError(f"config section '{name}' must be a mapping")
            known = {f.name for f in fields(section_cls)}
            unknown = set(section) - known
            if unknown:
                raise ValueError(
                    f"unknown keys in '{name}' section: {sorted(unknown)}"
                )
            kwargs[name] = section_cls(**section)
        _ = section_types  # silence linters about unused var
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Load a config from a YAML file, falling back to defaults for omissions."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


def _asdict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    return obj