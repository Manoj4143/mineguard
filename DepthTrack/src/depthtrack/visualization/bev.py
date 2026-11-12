"""Bird's-eye-view (BEV) rendering helpers.

Projects the lower half of a normalised depth map into a pseudo-3D bird's-eye
point cloud and draws perspective grid lines and 3D-ish object boxes onto a BEV
canvas. These are purely cosmetic/illustrative projections, not a metric
calibration.
"""

from __future__ import annotations

import cv2
import numpy as np

from .colormap import depth_to_color

# Far plane (metres) used to normalise distance into BEV depth.
BEV_Z_FAR_M = 60.0

# Layout constants for the perspective projection (fractions of canvas size).
_HORIZON_FRAC = 0.28
_DEPTH_FRAC = 0.70


def draw_grid(canvas: np.ndarray, width: int, height: int) -> None:
    """Draw a perspective ground grid (horizontal arcs + radial lines)."""
    for frac in np.linspace(0.3, 1.0, 10):
        y = int(frac * height)
        xl = int(width * 0.5 - width * 0.4 * frac)
        xr = int(width * 0.5 + width * 0.4 * frac)
        alpha = int(80 + 120 * frac)
        cv2.line(canvas, (xl, y), (xr, y), (alpha, 0, alpha // 2), 1, cv2.LINE_AA)

    vp_x, vp_y = width // 2, int(height * _HORIZON_FRAC)
    for k in np.linspace(-0.42, 0.42, 16):
        bx = int(width * 0.5 + k * width)
        alpha = int(60 + 80 * (1 - abs(k) / 0.45))
        cv2.line(canvas, (vp_x, vp_y), (bx, height), (0, alpha, alpha // 2), 1, cv2.LINE_AA)


def project_pointcloud(
    depth_norm: np.ndarray, canvas: np.ndarray, step: int = 3
) -> None:
    """Splat a sparse, colourised point cloud from depth onto the BEV canvas."""
    h, w = depth_norm.shape
    ch, cw = canvas.shape[:2]
    ys = np.arange(h // 2, h, step)
    xs = np.arange(0, w, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    d = depth_norm[yy, xx].astype(np.float32)
    u = xx.astype(np.float32) / w
    # d is inverse depth (high = near). The grid widens toward the bottom, so
    # near points belong at the bottom (large y) and far points near the horizon.
    bev_y = (ch * _HORIZON_FRAC + d * ch * _DEPTH_FRAC).astype(np.int32)
    spread = 0.38 + 0.52 * d
    bev_x = (cw * 0.5 + (u - 0.5) * cw * spread).astype(np.int32)

    mask = (bev_x >= 0) & (bev_x < cw) & (bev_y >= 0) & (bev_y < ch)
    canvas[bev_y[mask].ravel(), bev_x[mask].ravel()] = depth_to_color(d[mask].ravel())


def draw_bev_box(
    canvas: np.ndarray,
    cx_norm: float,
    dist_m: float,
    label: str,
    color: tuple[int, int, int],
    width: int,
    height: int,
) -> None:
    """Draw a glowing pseudo-3D box for one object at its BEV position."""
    # Proximity: 1.0 = at the camera, 0.0 = at/beyond the far plane.
    prox = float(np.clip(1.0 - dist_m / BEV_Z_FAR_M, 0.0, 1.0))
    # Near objects sit at the bottom (large y); far objects near the horizon.
    bev_y = int(height * _HORIZON_FRAC + prox * height * _DEPTH_FRAC)
    spread = 0.38 + 0.52 * prox
    bev_x = int(width * 0.5 + (cx_norm - 0.5) * width * spread)
    scale = 0.5 + 1.5 * prox

    bw, bh, bd = int(55 * scale), int(35 * scale), int(20 * scale)
    fl, fr = bev_x - bw // 2, bev_x + bw // 2
    ft, fb = bev_y - bh // 2, bev_y + bh // 2
    bl, br = fl + bd, fr + bd
    bt, bb = ft - bd // 2, fb - bd // 2

    def glow_line(p1: tuple[int, int], p2: tuple[int, int]) -> None:
        for t, a in [(3, 0.2), (2, 0.5), (1, 1.0)]:
            cv2.line(canvas, p1, p2, tuple(int(v * a) for v in color), t, cv2.LINE_AA)

    # front face
    for p1, p2 in [((fl, ft), (fr, ft)), ((fr, ft), (fr, fb)),
                   ((fr, fb), (fl, fb)), ((fl, fb), (fl, ft))]:
        glow_line(p1, p2)
    # back face
    for p1, p2 in [((bl, bt), (br, bt)), ((br, bt), (br, bb)),
                   ((br, bb), (bl, bb)), ((bl, bb), (bl, bt))]:
        glow_line(p1, p2)
    # connecting edges
    for p1, p2 in [((fl, ft), (bl, bt)), ((fr, ft), (br, bt)),
                   ((fr, fb), (br, bb)), ((fl, fb), (bl, bb))]:
        glow_line(p1, p2)

    fs = 0.38 * (0.6 + 0.8 * scale)
    (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
    tx, ty = bev_x - tw // 2, ft - 6
    if 0 < ty < height:
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, fs, color, 1, cv2.LINE_AA)