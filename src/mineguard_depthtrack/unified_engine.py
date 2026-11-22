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

        # High-performance depth estimation cadence cache
        self._cached_depth_norm: Optional[np.ndarray] = None
        self._depth_step_stride = 3  # Estimate depth every 3rd frame for 3x speedup while ByteTrack tracks continuously

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

    def _detect_corridor_hazards(self, work_frame: np.ndarray, existing_boxes: List[List[int]]) -> List[Dict[str, Any]]:
        """Detects faint / low-contrast obstacles, approaching lights/headlights, or silhouettes in the driving path."""
        h, w = work_frame.shape[:2]
        # Central driving lane corridor
        cy1, cy2 = int(h * 0.54), int(h * 0.78)
        cx1, cx2 = int(w * 0.28), int(w * 0.62)

        gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
        roi = gray[cy1:cy2, cx1:cx2]
        hazards = []

        # 1. Oncoming Headlights / Approaching Bright Light Detection in Lane
        # In foggy, dusty, or dark conditions, approaching vehicles emit distinct bright luminous spots (headlights)
        roi_mean, roi_std = np.mean(roi), np.std(roi)
        light_thresh_val = max(195, int(roi_mean + 2.8 * roi_std))
        _, light_mask = cv2.threshold(roi, min(235, light_thresh_val), 255, cv2.THRESH_BINARY)
        l_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        light_cleaned = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, l_kernel)
        light_contours, _ = cv2.findContours(light_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        light_blobs = []
        for lcnt in light_contours:
            larea = cv2.contourArea(lcnt)
            if 30 <= larea <= 1800:
                lx, ly, lw, lh = cv2.boundingRect(lcnt)
                aspect = lw / float(lh)
                if 0.4 <= aspect <= 3.5:
                    light_blobs.append((lx, ly, lw, lh, larea))

        # Check if bright blobs represent headlights or an oncoming vehicle cluster
        if light_blobs:
            min_lx = min(b[0] for b in light_blobs)
            min_ly = min(b[1] for b in light_blobs)
            max_rx = max(b[0] + b[2] for b in light_blobs)
            max_ry = max(b[1] + b[3] for b in light_blobs)
            bw = max_rx - min_lx
            bh = max_ry - min_ly

            bx1, by1 = cx1 + min_lx, cy1 + min_ly
            bx2, by2 = bx1 + bw, by1 + bh

            # Check overlap with existing YOLO vehicle boxes
            overlaps = any(
                max(bx1, eb[0]) < min(bx2, eb[2]) and max(by1, eb[1]) < min(by2, eb[3])
                for eb in existing_boxes
            )
            if not overlaps and bw >= 14 and bh >= 10:
                hazards.append({
                    "bbox": [bx1, by1, bx2, by2],
                    "area": bw * bh,
                    "is_light": True,
                })

        # 2. Sobel gradient analysis to detect anomalous boundaries in fog/dust
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        mean_g, std_g = np.mean(mag), np.std(mag)
        thresh = (mag > (mean_g + 3.2 * std_g)).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clustered = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clustered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 80 <= area <= 2500:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / float(bh)
                if 0.6 <= aspect <= 3.0 and bw >= 12 and bh >= 10:
                    bx1, by1 = cx1 + x, cy1 + y
                    bx2, by2 = bx1 + bw, by1 + bh

                    overlaps = any(
                        max(bx1, eb[0]) < min(bx2, eb[2]) and max(by1, eb[1]) < min(by2, eb[3])
                        for eb in (existing_boxes + [h["bbox"] for h in hazards])
                    )
                    if not overlaps:
                        hazards.append({
                            "bbox": [bx1, by1, bx2, by2],
                            "area": area,
                            "is_light": False,
                        })
        return hazards

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

        # 4. Depth Estimation on Natural Original Frame with intelligent cadence
        if self._cached_depth_norm is None or (self.frame_count % self._depth_step_stride == 0):
            depth_in = cv2.resize(
                work_frame, (self.cfg.video.depth_input_width, self.cfg.video.depth_input_height)
            )
            depth_raw = self.depth_estimator.estimate(depth_in)
            self._cached_depth_norm = cv2.resize(depth_raw, (work_w, work_h))
        
        depth_norm = self._cached_depth_norm

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

                # Lateral lane position analysis:
                # box center x relative to image center
                box_center_x = (box[0] + box[2]) / 2.0
                lane_offset = abs(box_center_x - (work_w / 2.0)) / (work_w / 2.0)
                is_same_lane = lane_offset < 0.38  # Central driving corridor
                
                # Rule 1: Moving away / forward in same direction (closing_kmh < -1.5)
                # Rule 2: Approaching in adjacent / different lane (closing_kmh > 1.5, not same lane)
                # Rule 3: Approaching in the same lane (closing_kmh > 1.5, is_same_lane) -> CRITICAL DANGER
                # Rule 4: Sudden stop / stationary in path (abs(closing_kmh) <= 1.5 and dist_m < 35.0 in lane)

                hazard_state = "NORMAL"
                if dist_m <= 12.0:
                    risk_tier = "CRITICAL"
                    color_bgr = (0, 0, 255)
                    hazard_state = "CRITICAL PROXIMITY"
                elif is_same_lane and closing_kmh > 1.5:
                    # RULE 3: Approaching in the same lane -> DANGER TO MOVE
                    risk_tier = "CRITICAL"
                    color_bgr = (0, 0, 255)
                    hazard_state = "DANGER: HEAD-ON APPROACH"
                elif is_same_lane and abs(closing_kmh) <= 1.5 and dist_m <= 35.0:
                    # RULE 4: Stopped in lane ahead -> CAR IS STOPPED
                    risk_tier = "CRITICAL" if dist_m <= 18.0 else "CAUTION"
                    color_bgr = (0, 0, 255) if dist_m <= 18.0 else (0, 215, 255)
                    hazard_state = "VEHICLE STOPPED IN LANE"
                elif (not is_same_lane) and closing_kmh > 1.5:
                    # RULE 2: Approaching in different / adjacent lane -> CAUTION WARNING
                    risk_tier = "CAUTION"
                    color_bgr = (0, 215, 255)
                    hazard_state = "WARNING: ADJACENT APPROACH"
                elif closing_kmh < -1.5:
                    # RULE 1: Front vehicle moving forward / away -> SAFE
                    risk_tier = "SAFE"
                    color_bgr = (0, 255, 0)
                    hazard_state = "MOVING FORWARD (SAFE)"
                elif dist_m <= 25.0 or (ttc_s is not None and ttc_s <= 3.0):
                    risk_tier = "CRITICAL"
                    color_bgr = (0, 0, 255)
                    hazard_state = "CRITICAL DISTANCE"
                elif dist_m <= 40.0 or (ttc_s is not None and ttc_s <= 5.0):
                    risk_tier = "CAUTION"
                    color_bgr = (0, 215, 255)
                    hazard_state = "CAUTION DISTANCE"
                else:
                    risk_tier = "SAFE"
                    color_bgr = (0, 255, 0)
                    hazard_state = "CLEAR PATH"

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
                    "hazard_state": hazard_state,
                    "is_same_lane": is_same_lane,
                    "color_bgr": color_bgr,
                }
                objects_data.append(obj_info)

        self.risk_analyzer.cleanup_old_tracks(active_track_ids)

        # 5b. Driving Corridor Anomaly & Unusual Obstacle Detection
        # When standard YOLO does not identify faint/distant vehicles in the road corridor,
        # detect high-frequency silhouettes/anomalies in the driving path as "UNUSUAL OBJECT"
        # or oncoming headlights / light approaching as "DANGER: HEAD-ON LIGHT / VEHICLE APPROACHING".
        existing_boxes = [o["bbox"] for o in objects_data]
        unusual_hazards = self._detect_corridor_hazards(work_frame, existing_boxes)
        for uh in unusual_hazards:
            ubox = uh["bbox"]
            is_light = uh.get("is_light", False)
            udist_m = estimate_distance(depth_norm, ubox, work_w, work_h, self.cfg.distance, class_id=2)

            if is_light:
                # Approaching light / oncoming headlights in path indicates imminent collision danger
                urisk = "CRITICAL"
                ucolor = (0, 0, 255)
                cname = "APPROACHING LIGHT / HEADLIGHTS"
                h_state = "DANGER: ONCOMING LIGHT IN LANE"
            else:
                # Assign risk level based on proximity
                urisk = "CRITICAL" if udist_m <= 20.0 else ("CAUTION" if udist_m <= 45.0 else "SAFE")
                ucolor = (0, 0, 255) if urisk == "CRITICAL" else ((0, 215, 255) if urisk == "CAUTION" else (0, 255, 0))
                cname = "UNUSUAL OBJECT / OBSTACLE"
                h_state = "ROAD CORRIDOR OBSTACLE"

            if risk_priority[urisk] > risk_priority[highest_risk]:
                highest_risk = urisk

            objects_data.append({
                "track_id": 99 if not is_light else 98,
                "class_id": 99 if not is_light else 98,
                "class_name": cname,
                "confidence": 0.75 if is_light else 0.65,
                "bbox": ubox,
                "relative_depth": round(float(np.median(depth_norm[ubox[1]:ubox[3], ubox[0]:ubox[2]])), 3) if ubox[3] > ubox[1] and ubox[2] > ubox[0] else 0.5,
                "distance_m": udist_m,
                "closing_speed_kmh": 0.0,
                "ttc_seconds": None,
                "risk_tier": urisk,
                "hazard_state": h_state,
                "color_bgr": ucolor,
            })

        # 6. Road Clearance Assessment
        head_on_threats = [o for o in objects_data if o.get("hazard_state") in ("DANGER: HEAD-ON APPROACH", "DANGER: ONCOMING LIGHT IN LANE")]
        stopped_threats = [o for o in objects_data if o.get("hazard_state") == "VEHICLE STOPPED IN LANE"]
        adjacent_threats = [o for o in objects_data if o.get("hazard_state") == "WARNING: ADJACENT APPROACH"]

        if atmos["fog_condition"] == "DENSE_FOG" and illum["is_night"]:
            road_status = "INSUFFICIENT VISIBILITY"
            road_detail = "Severe atmospheric and lighting degradation: Halt operations."
        elif head_on_threats:
            # Rule 3: Approaching in same lane or head-on light
            road_status = "ROAD BLOCKED / DANGER"
            if head_on_threats[0].get("hazard_state") == "DANGER: ONCOMING LIGHT IN LANE":
                road_detail = f"DANGER: Oncoming headlights / approaching light detected in your lane at {head_on_threats[0]['distance_m']}m! Stop immediately."
            else:
                road_detail = f"DANGER: Vehicle approaching in SAME LANE at {head_on_threats[0]['distance_m']}m! Immediate stop required."
        elif stopped_threats:
            # Rule 4
            road_status = "ROAD BLOCKED / HAZARD"
            road_detail = f"WARNING: Vehicle stopped ahead in your lane at {stopped_threats[0]['distance_m']}m."
        elif any(o["risk_tier"] == "CRITICAL" for o in objects_data):
            road_status = "ROAD BLOCKED / HAZARD"
            road_detail = "Immediate stopping required: Critical collision risk."
        elif adjacent_threats:
            # Rule 2
            road_status = "PROXIMITY WARNING"
            road_detail = f"Caution: Vehicle approaching in adjacent lane beside you ({adjacent_threats[0]['distance_m']}m)."
        elif any(o["risk_tier"] == "CAUTION" for o in objects_data):
            road_status = "PROXIMITY WARNING"
            road_detail = "Caution: Approaching vehicle / unusual obstacle detected."
        elif any(o.get("hazard_state") == "MOVING FORWARD (SAFE)" for o in objects_data):
            # Rule 1
            road_status = "ROAD CLEAR"
            road_detail = "Vehicle ahead is moving forward in safe direction."
        elif not objects_data:
            road_status = "ROAD CLEAR"
            road_detail = f"Path is unobstructed. Visibility: {atmos['fog_condition']}."
        else:
            road_status = "ROAD CLEAR"
            road_detail = "Vehicles at safe distance."

        t_elapsed = time.perf_counter() - t_start
        self.latencies.append(t_elapsed * 1000)
        # Use instantaneous exponential/short rolling average (last 10 frames) so warmup spike doesn't drag down FPS
        recent_latencies = list(self.latencies)[-10:]
        mean_lat = float(np.mean(recent_latencies))
        fps = 1000.0 / max(mean_lat, 1e-6)

        # 7. Render Visualization (Single Expanded Live AI Vision Feed, No 3D BEV Panel)
        annotated_frame = work_frame.copy()
        for o in objects_data:
            x1, y1, x2, y2 = o["bbox"]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), o["color_bgr"], 2)
            ttc_str = f"| TTC:{o['ttc_seconds']}s" if o.get("ttc_seconds") else ""
            h_state = f" [{o['hazard_state']}]" if o.get("hazard_state") else f" [{o['risk_tier']}]"
            lbl = f"ID:{o['track_id']} {o['class_name']} {o['distance_m']}m {ttc_str}{h_state}"
            cv2.putText(annotated_frame, lbl, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, o["color_bgr"], 1)

        # Single expanded panel HUD overlay
        hud_text = f"FPS: {fps:.1f} | Frame: {self.frame_count} | Mode: {pipeline_mode} | Dev: {self.device}"
        cv2.putText(annotated_frame, hud_text, (10, work_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)
        composite = annotated_frame

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
