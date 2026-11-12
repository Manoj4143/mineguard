"""Tests for collision-risk classification."""

import numpy as np
import pytest

from depthtrack.config import RiskConfig
from depthtrack.perception.risk import RiskLevel, classify_risk


@pytest.fixture
def cfg():
    return RiskConfig()


def _centered_box(frame_w=1000):
    cx = frame_w // 2
    return np.array([cx - 20, 100, cx + 20, 200])


def test_close_and_in_lane_is_danger(cfg):
    assert classify_risk(5.0, _centered_box(), 1000, cfg) == RiskLevel.DANGER


def test_close_but_off_to_side_is_not_danger(cfg):
    # Far to the right edge -> high lateral offset -> downgraded to WARNING.
    box = np.array([950, 100, 990, 200])
    assert classify_risk(5.0, box, 1000, cfg) == RiskLevel.WARNING


def test_warning_band(cfg):
    assert classify_risk(12.0, _centered_box(), 1000, cfg) == RiskLevel.WARNING


def test_caution_band(cfg):
    assert classify_risk(20.0, _centered_box(), 1000, cfg) == RiskLevel.CAUTION


def test_safe_when_far(cfg):
    assert classify_risk(40.0, _centered_box(), 1000, cfg) == RiskLevel.SAFE


def test_thresholds_are_configurable():
    cfg = RiskConfig(danger_m=3.0, warning_m=6.0, caution_m=10.0)
    assert classify_risk(5.0, _centered_box(), 1000, cfg) == RiskLevel.WARNING


def test_risk_level_is_str_enum():
    assert RiskLevel.DANGER.value == "DANGER"
    assert str(RiskLevel.SAFE.value) == "SAFE"
