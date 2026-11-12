"""Tests for the turbo colormap and BEV rendering helpers."""

import numpy as np

from depthtrack.visualization import bev
from depthtrack.visualization.colormap import TURBO_LUT, depth_to_color


def test_lut_shape_and_dtype():
    assert TURBO_LUT.shape == (256, 3)
    assert TURBO_LUT.dtype == np.uint8


def test_depth_to_color_maps_range():
    depth = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    colors = depth_to_color(depth)
    assert colors.shape == (3, 3)
    assert colors.dtype == np.uint8


def test_depth_to_color_clips_out_of_range():
    depth = np.array([-1.0, 2.0], dtype=np.float32)
    colors = depth_to_color(depth)
    # Should equal the clamped endpoints.
    assert np.array_equal(colors[0], TURBO_LUT[0])
    assert np.array_equal(colors[1], TURBO_LUT[255])


def test_project_pointcloud_writes_within_bounds():
    depth = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    bev.project_pointcloud(depth, canvas)
    # Some pixels should have been coloured.
    assert canvas.sum() > 0


def test_draw_grid_runs_without_error():
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    bev.draw_grid(canvas, 300, 300)
    assert canvas.sum() > 0


def test_draw_bev_box_does_not_raise():
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    bev.draw_bev_box(canvas, 0.5, 10.0, "car 10m WARNING", (0, 140, 255), 300, 300)
    assert canvas.shape == (300, 300, 3)
