"""Frame rendering: the left ADAS panel, right BEV panel, compositing and HUD.

The renderer is configured once and exposes methods that take per-frame data
(the working frame, tracked detections, depth map) and return drawn panels.
Keeping rendering in a class lets it hold onto config and risk-colour lookups
without re-deriving them every frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import AppConfig
from ..models.tracker import ObjectTracker
from ..perception import RiskLevel, classify_risk, estimate_distance
from . import bev

# BGR colours for each risk level.
RISK_COLORS: dict[RiskLevel, tuple[int, int, int]] = {
    RiskLevel.SAFE: (0, 255, 80),
    RiskLevel.CAUTION: (0, 220, 255),
    RiskLevel.WARNING: (0, 140, 255),
    RiskLevel.DANGER: (0, 0, 255),
}

_FONT = cv2.FONT_HERSHEY_DUPLEX


class Renderer:
    """Renders annotated ADAS + BEV panels for each frame."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.panel_w = cfg.video.panel_width
        self.panel_h = cfg.video.panel_height

    # ---- left panel: dashcam + detections ------------------------------------

    def draw_left_panel(
        self,
        frame: np.ndarray,
        tracked,
        depth_map: np.ndarray,
        tracker: ObjectTracker,
    ) -> np.ndarray:
        out = frame.copy()
        self._draw_lane(out)
        h, w = out.shape[:2]

        if tracked is not None and len(tracked) > 0:
            for i in range(len(tracked.xyxy)):
                box = tracked.xyxy[i].astype(int)
                cls = int(tracked.class_id[i]) if tracked.class_id is not None else 2
                tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else 0
                label = self.cfg.model.vehicle_classes.get(cls, "vehicle")
                dist = estimate_distance(depth_map, box, w, h, self.cfg.distance, cls)
                risk = classify_risk(dist, box, w, self.cfg.risk)
                color = RISK_COLORS[risk]

                self._glow_box(out, box[0], box[1], box[2], box[3], color)

                text = f"{label} {dist:.0f}m {risk.value}"
                fs = 0.48
                (tw, th), _ = cv2.getTextSize(text, _FONT, fs, 1)
                lx, ly = box[0], max(box[1] - 6, th + 4)
                cv2.rectangle(out, (lx - 2, ly - th - 4), (lx + tw + 4, ly + 2), (0, 0, 0), -1)
                cv2.putText(out, text, (lx, ly), _FONT, fs, color, 1, cv2.LINE_AA)

                traj = list(tracker.trajectories[tid])
                for k in range(1, len(traj)):
                    cv2.line(out, traj[k - 1], traj[k], color, 1, cv2.LINE_AA)

                vx, vy = tracker.velocities.get(tid, (0, 0))
                cx_b = (box[0] + box[2]) // 2
                cy_b = (box[1] + box[3]) // 2
                cv2.arrowedLine(
                    out, (cx_b, cy_b), (cx_b + int(vx * 8), cy_b + int(vy * 8)),
                    (255, 255, 0), 1, cv2.LINE_AA, tipLength=0.4,
                )

        cv2.putText(out, "DASHCAM | ADAS PERCEPTION", (10, 22),
                    _FONT, 0.55, (0, 255, 80), 1, cv2.LINE_AA)
        return out

    # ---- right panel: bird's-eye view ----------------------------------------

    def draw_right_panel(
        self, depth_norm: np.ndarray, tracked, frame_w: int, frame_h: int
    ) -> np.ndarray:
        canvas = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
        d_small = cv2.resize(depth_norm, (self.panel_w, self.panel_h))
        bev.project_pointcloud(d_small, canvas)

        horizon = int(self.panel_h * 0.3)
        grad = np.linspace(200, 0, horizon, dtype=np.uint8)
        canvas[:horizon, :, 0] = np.minimum(
            canvas[:horizon, :, 0].astype(np.int16) + grad[:, None] // 4, 255
        ).astype(np.uint8)

        bev.draw_grid(canvas, self.panel_w, self.panel_h)

        if tracked is not None and len(tracked) > 0 and tracked.tracker_id is not None:
            d_full = cv2.resize(depth_norm, (frame_w, frame_h))
            for i in range(len(tracked.xyxy)):
                box = tracked.xyxy[i]
                cls = int(tracked.class_id[i]) if tracked.class_id is not None else 2
                name = self.cfg.model.vehicle_classes.get(cls, "vehicle")
                dist = estimate_distance(d_full, box, frame_w, frame_h, self.cfg.distance, cls)
                risk = classify_risk(dist, box, frame_w, self.cfg.risk)
                color = RISK_COLORS[risk]
                cx_norm = ((box[0] + box[2]) / 2) / frame_w
                bev.draw_bev_box(
                    canvas, cx_norm, dist, f"{name} {dist:.0f}m {risk.value}",
                    color, self.panel_w, self.panel_h,
                )

        cv2.putText(canvas, "3D BEV | DEPTH PERCEPTION", (10, 22),
                    _FONT, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
        return canvas

    # ---- compositing & HUD ---------------------------------------------------

    def compose(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_r = cv2.resize(left, (self.panel_w, self.panel_h))
        right_r = cv2.resize(right, (self.panel_w, self.panel_h))
        div = np.full((self.panel_h, self.cfg.video.divider_width, 3), (0, 255, 80), dtype=np.uint8)
        return np.hstack([left_r, div, right_r])

    def add_hud(
        self, frame: np.ndarray, fps: float, frame_no: int, total: int, device_label: str
    ) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 28), (w, h), (5, 5, 5), -1)
        pct = frame_no / max(total, 1) * 100
        text = (f"  FPS: {fps:5.1f}  |  Frame: {frame_no}/{total} ({pct:.1f}%)  "
                f"|  Device: {device_label}  |  DepthTrack")
        cv2.putText(frame, text, (10, h - 8), _FONT, 0.42, (0, 220, 80), 1, cv2.LINE_AA)

    # ---- private helpers -----------------------------------------------------

    @staticmethod
    def _draw_lane(frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        pts = np.array([
            [int(w * 0.35), int(h * 0.55)],
            [int(w * 0.65), int(h * 0.55)],
            [int(w * 0.85), h - 1],
            [int(w * 0.15), h - 1],
        ], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 80, 0))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 255, 60), 1, cv2.LINE_AA)

    @staticmethod
    def _glow_box(frame, x1, y1, x2, y2, color, thickness: int = 2) -> None:
        for t, a in [(thickness + 4, 0.15), (thickness + 2, 0.30), (thickness, 1.0)]:
            c = tuple(int(v * a) for v in color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, t, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)