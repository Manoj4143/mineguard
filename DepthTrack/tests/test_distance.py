"""Tests for the monocular distance estimator."""

import numpy as np
import pytest

from depthtrack.config import DistanceConfig
from depthtrack.perception.distance import estimate_distance


@pytest.fixture
def cfg():
    return DistanceConfig()


def test_degenerate_box_returns_fallback(cfg):
    depth = np.zeros((100, 100), dtype=np.float32)
    # x2 <= x1
    assert estimate_distance(depth, np.array([50, 50, 50, 60]), 100, 100, cfg) == 50.0


def test_result_is_clamped_to_range(cfg):
    depth = np.ones((100, 100), dtype=np.float32)  # far depth
    box = np.array([10, 10, 12, 12])  # tiny box -> large projection distance
    d = estimate_distance(depth, box, 100, 100, cfg)
    assert cfg.min_distance_m <= d <= cfg.max_distance_m


def test_closer_object_has_smaller_distance(cfg):
    # Lower normalised depth => nearer; wider box => nearer by projection too.
    depth = np.zeros((200, 200), dtype=np.float32)
    near_box = np.array([10, 10, 150, 150])   # large/near
    far_box = np.array([90, 90, 110, 110])    # small/far
    d_near = estimate_distance(depth, near_box, 200, 200, cfg)
    d_far = estimate_distance(depth, far_box, 200, 200, cfg)
    assert d_near < d_far


def test_box_clipped_to_frame_bounds(cfg):
    depth = np.full((100, 100), 0.5, dtype=np.float32)
    # box extends beyond frame; should not raise and stay in range
    d = estimate_distance(depth, np.array([-20, -20, 200, 200]), 100, 100, cfg)
    assert cfg.min_distance_m <= d <= cfg.max_distance_m


def test_depth_weight_one_low_depth_is_far():
    cfg = DistanceConfig(depth_weight=1.0, depth_near_m=1.0, depth_far_m=80.0)
    depth = np.zeros((100, 100), dtype=np.float32)  # depth 0 => far => depth_far_m
    d = estimate_distance(depth, np.array([10, 10, 90, 90]), 100, 100, cfg, class_id=2)
    assert d == pytest.approx(80.0, abs=0.1)


def test_returns_one_decimal_place(cfg):
    depth = np.full((100, 100), 0.37, dtype=np.float32)
    d = estimate_distance(depth, np.array([10, 10, 60, 60]), 100, 100, cfg)
    assert d == round(d, 1)

def test_truck_farther_than_car_same_pixels(cfg):
    """Truck should report greater distance than car at equal pixel width,
    because truck_width_m (2.5) > car_width_m (1.8)."""
    depth = np.zeros((200, 200), dtype=np.float32)
    box = np.array([50, 50, 150, 150])  # 100px wide for both
    d_car   = estimate_distance(depth, box, 200, 200, cfg, class_id=2)  # car
    d_truck = estimate_distance(depth, box, 200, 200, cfg, class_id=7)  # truck
    assert d_truck > d_car, f"truck {d_truck}m should be > car {d_car}m at same pixel width"


def test_width_for_class_fallback(cfg):
    assert cfg.width_for_class(2) == 1.8   # car
    assert cfg.width_for_class(7) == 2.5   # truck
    assert cfg.width_for_class(99) == cfg.real_vehicle_width_m  # unknown -> fallback


def test_inverse_depth_near_object_is_close(cfg):
    """Depth Anything outputs inverse depth (high = near). A high-depth patch
    must produce a SHORT distance, not a long one."""
    near_depth = np.full((100, 100), 0.95, dtype=np.float32)  # high => near
    far_depth = np.full((100, 100), 0.05, dtype=np.float32)   # low  => far
    box = np.array([20, 20, 80, 80])
    d_near = estimate_distance(near_depth, box, 100, 100, cfg, class_id=2)
    d_far = estimate_distance(far_depth, box, 100, 100, cfg, class_id=2)
    assert d_near < d_far, f"high-depth (near) {d_near}m should be < low-depth (far) {d_far}m"


def test_depth_weight_one_high_depth_is_near():
    """With full depth weight, a near (high) patch maps near depth_near_m."""
    cfg = DistanceConfig(depth_weight=1.0, depth_near_m=1.0, depth_far_m=80.0)
    depth = np.ones((100, 100), dtype=np.float32)  # depth=1 => nearest
    d = estimate_distance(depth, np.array([10, 10, 90, 90]), 100, 100, cfg, class_id=2)
    assert d == pytest.approx(1.0, abs=0.1)