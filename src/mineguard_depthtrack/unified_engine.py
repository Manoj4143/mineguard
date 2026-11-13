"""Unified Perception & Safety Engine combining DepthTrack and MineGuard AI."""

import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import torch

from depthtrack.config import AppConfig, DistanceConfig
from depthtrack.models import DepthEstimator, ObjectTracker, VehicleDetector
from depthtrack.perception.distance import estimate_distance
from depthtrack.visualization.panels import Renderer

# Import MineGuard modules directly from original system
import sys
MINEGUARD_DIR = Path("D:/MINI_GUARD_PROJECT/MineGuard-AI")
if str(MINEGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(MINEGUARD_DIR))

from dehaze_lib import enhance_clahe, dehaze_dcp_v5
from night_vision import NightVisionProcessor
from risk_analyzer import VehicleRiskAnalyzer


class UnifiedMineGuardPerception:
    """Integrated perception and safety engine."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        detector_weights: str = "DepthTrack/assets/yolov8n.pt",
        device: Optional[str] = None,
        enable_heavy_fog_dcp: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        default_yaml = Path("DepthTrack/config/default.yaml")
        self.cfg = config or (AppConfig.from_yaml(default_yaml) if default_yaml.exists() else AppConfig())
        self.cfg.model.device = self.device
        self.enable_heavy_fog_dcp = enable_heavy_fog_dcp

        # 1. DepthTrack Perception Components
        weights_path = Path(detector_weights)
        self.detector = VehicleDetector(self.cfg.model, self.device, weights_path)
        self.detector.load()

        self.depth_estimator = DepthEstimator(self.cfg.model, self.device)
        self.depth_estimator.load()

        self.tracker = ObjectTracker(max_history=self.cfg.model.track_history)
        self.renderer = Renderer(self.cfg)

        # 2. MineGuard Safety & Environmental Processing
        self.night_processor = NightVisionProcessor()
        self.risk_analyzer = VehicleRiskAnalyzer(
            frame_width=self.cfg.video.work_width,
            frame_height=int(self.cfg.video.work_width * 0.5625),
            fps=self.cfg.video.default_fps,
        )

        # Telemetry & Performance Tracking
        self.latencies = deque(maxlen=60)
        self.system_status = "ONLINE"
        self.frame_count = 0

    def assess_atmospheric_visibility(self, frame: np.ndarray) -> Dict[str, Any]:
        """Estimate contrast and dark channel mean to classify fog intensity."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rms_contrast = float(np.std(gray))
        small = cv2.resize(gray, (160, 90))
        min_channel = cv2.erode(small, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        dark_mean = float(np.mean(min_channel))

        if dark_mean > 120 and rms_contrast < 30.0:
            fog_condition = "DENSE_FOG"
        elif dark_mean > 85 and rms_contrast < 42.0:
            fog_condition = "MEDIUM_FOG"
        elif dark_mean > 55 and rms_contrast < 50.0:
            fog_condition = "LIGHT_FOG"
        else:
            fog_condition = "CLEAR"

        return {
            "rms_contrast": round(rms_contrast, 1),
            "dark_mean": round(dark_mean, 1),
            "fog_condition": fog_condition,
        }

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Process a single video frame through the unified perception-safety pipeline."""
        t_start = time.perf_counter()
        self.frame_count += 1
        curr_time = timestamp if timestamp is not None else time.time()

        h, w = frame.shape[:2]
        work_w = min(w, self.cfg.video.work_width)
        work_h = int(h * (work_w / w))
        work_frame = cv2.resize(frame, (work_w, work_h))

        # 1. Environmental Assessment
        illum = self.night_processor.assess_scene_illumination(work_frame)
        atmos = self.assess_atmospheric_visibility(work_frame)

        # 2. Selective Decoupled Enhancement Dispatcher (based on Phase 2.5 results)
        pipeline_mode = "CLEAR_BYPASS"
        detector_input = work_frame

        if illum["is_night"]:
            pipeline_mode = "NIGHT_CLAHE"
            detector_input, _ = enhance_clahe(work_frame)
        elif self.enable_heavy_fog_dcp and atmos["fog_condition"] == "DENSE_FOG":
            pipeline_mode = "DCP_V5_FOG"
            detector_input, _ = dehaze_dcp_v5(work_frame)
        elif atmos["fog_condition"] in ("LIGHT_FOG", "MEDIUM_FOG", "DENSE_FOG"):
            pipeline_mode = "FOG_ADAPTIVE"
            # Fast CLAHE preserves details with minimal latency
            detector_input, _ = enhance_clahe(work_frame)

        # 3. Object Detection & Tracking (DepthTrack Perception)
        raw_detections = self.detector.detect(detector_input)
        tracked = self.tracker.update(raw_detections)

        # 4. Depth Estimation on Natural Original Frame (Preserves natural depth perspective)
        depth_in = cv2.resize(
            work_frame, (self.cfg.video.depth_input_width, self.cfg.video.depth_input_height)
        )
        depth_norm = self.depth_estimator.estimate(depth_in)
        depth_norm = cv2.resize(depth_norm, (work_w, work_h))

        # 5. Safety Layer: Distance, Velocity, TTC & Risk Classification
        objects_data = []
        highest_risk = "SAFE"
        risk_priority = {"SAFE": 0, "CAUTION": 1, "CRITICAL": 2}

        active_track_ids = []

        if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                active_track_ids.append(tid)
                box = tracked.xyxy[i]
                cid = int(tracked.class_id[i]) if tracked.class_id is not None else 7
                cname = self.cfg.model.vehicle_classes.get(cid, "vehicle")
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

                # Fused monocular distance (Depth map + Pinhole projection)
                dist_m = estimate_distance(depth_norm, box, work_w, work_h, self.cfg.distance, class_id=cid)

                # Temporal tracking update (Velocity & TTC from MineGuard Safety Layer)
                safety_metrics = self.risk_analyzer.update_track(tid, box, curr_time)
                # Overwrite distance with fused depth-pinhole distance
                safety_metrics["distance_m"] = dist_m

                # Recalculate TTC with updated fused distance
                ttc_s = None
                closing_kmh = safety_metrics["closing_speed_kmh"]
                closing_mps = closing_kmh / 3.6
                if closing_mps > 0.3:
                    ttc_s = round(dist_m / closing_mps, 1)

                # Re-evaluate 3-tier risk
                if dist_m <= 15.0 or (ttc_s is not None and ttc_s <= 2.0):
                    risk_tier = "CRITICAL"
                    color_bgr = (0, 0, 255)
                elif dist_m <= 35.0 or (ttc_s is not None and ttc_s <= 4.0):
                    risk_tier = "CAUTION"
                    color_bgr = (0, 215, 255)
                else:
                    risk_tier = "SAFE"
                    color_bgr = (0, 255, 0)

                if risk_priority[risk_tier] > risk_priority[highest_risk]:
                    highest_risk = risk_tier

                obj_info = {
                    "track_id": tid,
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": round(conf, 2),
                    "bbox": [int(x) for x in box],
                    "relative_depth": round(float(np.median(depth_norm[int(box[1]):int(box[3]), int(box[0]):int(box[2])])), 3) if box[3] > box[1] and box[2] > box[0] else 0.5,
                    "distance_m": dist_m,
                    "closing_speed_kmh": closing_kmh,
                    "ttc_seconds": ttc_s,
                    "risk_tier": risk_tier,
                    "color_bgr": color_bgr,
                }
                objects_data.append(obj_info)

        self.risk_analyzer.cleanup_old_tracks(active_track_ids)

        # 6. Road Clearance Assessment
        if atmos["fog_condition"] == "DENSE_FOG" and illum["is_night"]:
            road_status = "INSUFFICIENT VISIBILITY"
            road_detail = "Severe atmospheric and lighting degradation: Halt operations."
        elif not objects_data:
            road_status = "ROAD CLEAR"
            road_detail = f"Path is unobstructed. Visibility: {atmos['fog_condition']}."
        elif any(o["risk_tier"] == "CRITICAL" for o in objects_data):
            road_status = "ROAD BLOCKED / HAZARD"
            road_detail = "Immediate stopping required: Critical collision risk."
        elif any(o["risk_tier"] == "CAUTION" for o in objects_data):
            road_status = "PROXIMITY WARNING"
            road_detail = "Caution: Approaching vehicle detected."
        else:
            road_status = "ROAD CLEAR"
            road_detail = "Vehicles at safe distance."

        t_elapsed = time.perf_counter() - t_start
        self.latencies.append(t_elapsed * 1000)
        mean_lat = float(np.mean(self.latencies))
        fps = 1000.0 / max(mean_lat, 1e-6)

        # 7. Render Visualization Composite (Left: annotated, Right Top: Depth, Right Bottom: BEV)
        annotated_frame = work_frame.copy()
        for o in objects_data:
            x1, y1, x2, y2 = o["bbox"]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), o["color_bgr"], 2)
            ttc_str = f"| TTC:{o['ttc_seconds']}s" if o["ttc_seconds"] else ""
            lbl = f"ID:{o['track_id']} {o['class_name']} {o['distance_m']}m {ttc_str} [{o['risk_tier']}]"
            cv2.putText(annotated_frame, lbl, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, o["color_bgr"], 1)

        # Assemble unified composite frame
        right_panel = self.renderer.draw_right_panel(depth_norm, tracked, work_w, work_h)
        composite = self.renderer.compose(annotated_frame, right_panel)
        self.renderer.add_hud(composite, fps, self.frame_count, self.frame_count, self.device)

        telemetry = {
            "system_status": self.system_status,
            "frame_id": self.frame_count,
            "pipeline_mode": pipeline_mode,
            "fog_condition": atmos["fog_condition"],
            "ambient_lux": illum["raw_brightness"],
            "fps": round(fps, 1),
            "latency_ms": round(mean_lat, 1),
            "road_status": road_status,
            "road_status_detail": road_detail,
            "highest_risk": highest_risk,
            "active_vehicles_count": len(objects_data),
        }

        return {
            "telemetry": telemetry,
            "objects": objects_data,
            "composite_frame": composite,
            "depth_map": depth_norm,
        }
