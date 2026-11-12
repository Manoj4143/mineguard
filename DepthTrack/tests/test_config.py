"""Tests for configuration loading, validation, and overrides."""

import textwrap

import pytest

from depthtrack.config import AppConfig, ModelConfig, RiskConfig


def test_defaults_are_valid():
    cfg = AppConfig()
    assert cfg.model.confidence == 0.35
    assert cfg.model.device == "cpu"
    assert cfg.video.panel_width == cfg.video.output_width // 2
    assert 2 in cfg.model.vehicle_classes  # car


def test_invalid_confidence_rejected():
    with pytest.raises(ValueError):
        ModelConfig(confidence=0.0)
    with pytest.raises(ValueError):
        ModelConfig(confidence=1.5)


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        ModelConfig(device="tpu")


def test_risk_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        RiskConfig(danger_m=20, warning_m=15, caution_m=25)


def test_from_dict_partial_override():
    cfg = AppConfig.from_dict({"model": {"confidence": 0.5}})
    assert cfg.model.confidence == 0.5
    # untouched sections keep defaults
    assert cfg.risk.danger_m == 8.0


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        AppConfig.from_dict({"model": {"nonexistent": 1}})


def test_yaml_roundtrip(tmp_path):
    yaml_text = textwrap.dedent(
        """
        model:
          confidence: 0.42
          device: cpu
        risk:
          danger_m: 5.0
          warning_m: 12.0
          caution_m: 20.0
        """
    )
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    cfg = AppConfig.from_yaml(p)
    assert cfg.model.confidence == 0.42
    assert cfg.risk.danger_m == 5.0


def test_vehicle_classes_keys_coerced_to_int():
    cfg = AppConfig.from_dict({"model": {"vehicle_classes": {"2": "car"}}})
    assert cfg.model.vehicle_classes[2] == "car"


def test_to_dict_is_nested():
    d = AppConfig().to_dict()
    assert "model" in d and "confidence" in d["model"]
