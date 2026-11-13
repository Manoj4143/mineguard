"""Perception Service and Adapter Layer for MineGuard Dashboard Integration.

Decouples dashboard HTTP handlers and MJPEG streaming from the underlying
perception engines (UnifiedMineGuardPerception and fallback MineGuardIntegratedSystem).
"""

import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger("MineGuardDashboardAdapter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MINEGUARD_DIR = Path("D:/MINI_GUARD_PROJECT/MineGuard-AI")
if str(MINEGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(MINEGUARD_DIR))

# Import engines
try:
    from mineguard_depthtrack.unified_engine import UnifiedMineGuardPerception
except ImportError:
    UnifiedMineGuardPerception = None

try:
    from mineguard_core import MineGuardIntegratedSystem
except ImportError:
    MineGuardIntegratedSystem = None


class PerceptionAdapter:
    """Manages perception engine lifecycles, execution, and output normalization."""

    ENGINE_UNIFIED = "UnifiedMineGuardPerception"
    ENGINE_FALLBACK = "MineGuardIntegratedSystem"

    def __init__(self, preferred_engine: str = ENGINE_UNIFIED, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preferred_engine = preferred_engine
        self.active_engine_name = None
        self.engine_instance = None
        self.is_fallback_active = False
        self.initialization_error = None
        self._init_lock = threading.Lock()

        self._initialize_engine(preferred_engine)

    def _initialize_engine(self, engine_choice: str) -> None:
        """Initializes the selected engine once with graceful fallback."""
        with self._init_lock:
            self.initialization_error = None
            if engine_choice == self.ENGINE_UNIFIED:
                if UnifiedMineGuardPerception is None:
                    err = "UnifiedMineGuardPerception module could not be imported"
                    logger.error(err)
                    self._init_fallback(err)
                    return

                try:
                    logger.info("Initializing UnifiedMineGuardPerception on %s...", self.device)
                    detector_path = "DepthTrack/assets/yolov8n.pt"
                    self.engine_instance = UnifiedMineGuardPerception(
                        detector_weights=detector_path,
                        device=self.device
                    )
                    self.active_engine_name = self.ENGINE_UNIFIED
                    self.is_fallback_active = False
                    logger.info("UnifiedMineGuardPerception initialized successfully.")
                except Exception as exc:
                    err = f"Failed to initialize UnifiedMineGuardPerception: {exc}"
                    logger.exception(err)
                    self._init_fallback(err)

            elif engine_choice == self.ENGINE_FALLBACK:
                self._init_fallback("Manual selection of original MineGuardIntegratedSystem")
            else:
                self._init_fallback(f"Unknown engine '{engine_choice}'")

    def _init_fallback(self, reason: str) -> None:
        """Safely initializes the original MineGuardIntegratedSystem fallback engine."""
        self.initialization_error = reason
        if MineGuardIntegratedSystem is None:
            logger.critical("Original MineGuardIntegratedSystem fallback is unavailable!")
            self.active_engine_name = "NONE"
            self.engine_instance = None
            self.is_fallback_active = True
            return

        try:
            logger.warning("Activating fallback engine (MineGuardIntegratedSystem). Reason: %s", reason)
            weights_local = MINEGUARD_DIR / "yolo11n.pt"
            if not weights_local.exists():
                weights_local = Path("DepthTrack/assets/yolo11n.pt")
            self.engine_instance = MineGuardIntegratedSystem(
                model_path=str(weights_local),
                frame_width=1280,
                frame_height=720,
                fps=24.0
            )
            self.active_engine_name = self.ENGINE_FALLBACK
            self.is_fallback_active = True
            logger.info("Fallback engine initialized successfully.")
        except Exception as exc:
            logger.critical("Fallback engine initialization failed: %s", exc)
            self.active_engine_name = "ERROR"
            self.engine_instance = None
            self.is_fallback_active = True

    def set_engine(self, engine_name: str) -> bool:
        """Switches the active engine cleanly."""
        if engine_name not in (self.ENGINE_UNIFIED, self.ENGINE_FALLBACK):
            logger.warning("Invalid engine name requested: %s", engine_name)
            return False

        if engine_name == self.active_engine_name and not self.is_fallback_active and getattr(self, "engine_instance", None) is not None:
            logger.info("Engine %s is already active. Skipping re-initialization.", engine_name)
            return True

        logger.info("Switching engine from %s to %s", self.active_engine_name, engine_name)
        self._initialize_engine(engine_name)
        return self.active_engine_name == engine_name

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Processes a frame and returns standard (display_frame_bgr, telemetry_dict)."""
        if self.engine_instance is None:
            # Generate black placeholder frame with error overlay
            err_frame = np.zeros((540, 960, 3), dtype=np.uint8)
            cv2.putText(err_frame, "ENGINE ERROR: NO ACTIVE PERCEPTION ENGINE", (50, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            telemetry = {
                "active_engine": self.active_engine_name or "ERROR",
                "is_fallback": self.is_fallback_active,
                "engine_error": self.initialization_error,
                "system_status": "OFFLINE",
                "pipeline_mode": "ERROR",
                "fog_condition": "UNKNOWN",
                "ambient_lux": 0.0,
                "road_status": "SYSTEM ERROR",
                "road_status_detail": self.initialization_error or "Engine unavailable",
                "road_status_class": "blocked",
                "highest_risk": "CRITICAL",
                "fps": 0.0,
                "latency_ms": 0.0,
                "active_vehicles_count": 0,
                "vehicles": [],
            }
            return err_frame, telemetry

        if self.active_engine_name == self.ENGINE_UNIFIED:
            out = self.engine_instance.process_frame(frame)
            comp = out["composite_frame"]
            telem = out["telemetry"]
            objs = out.get("objects", [])

            # Standardize vehicles list for dashboard UI
            vehicles = []
            for o in objs:
                vehicles.append({
                    "track_id": int(o["track_id"]),
                    "class_name": str(o.get("class_name", "VEHICLE")).upper(),
                    "confidence": float(o.get("confidence", 0.0)),
                    "bbox": [int(b) for b in o.get("bbox", [0, 0, 0, 0])],
                    "distance_m": float(o["distance_m"]) if o.get("distance_m") is not None else None,
                    "relative_depth": float(o.get("relative_depth", 0.0)),
                    "speed_kmh": float(o.get("closing_speed_kmh", 0.0)),
                    "ttc_s": float(o["ttc_seconds"]) if o.get("ttc_seconds") is not None else None,
                    "risk_tier": str(o.get("risk_tier", "SAFE")),
                })

            road_status_class = "clear"
            if telem.get("road_status") == "ROAD BLOCKED / HAZARD":
                road_status_class = "blocked"
            elif "WARNING" in telem.get("road_status", "") or "CAUTION" in telem.get("road_status", ""):
                road_status_class = "caution"

            normalized_telem = {
                "active_engine": self.ENGINE_UNIFIED,
                "is_fallback": False,
                "engine_error": None,
                "system_status": telem.get("system_status", "ONLINE"),
                "pipeline_mode": telem.get("pipeline_mode", "CLEAR_BYPASS"),
                "fog_condition": telem.get("fog_condition", "CLEAR"),
                "ambient_lux": float(telem.get("ambient_lux", 100.0)),
                "road_status": telem.get("road_status", "ROAD CLEAR"),
                "road_status_detail": telem.get("road_status_detail", "Path unobstructed."),
                "road_status_class": road_status_class,
                "highest_risk": telem.get("highest_risk", "SAFE"),
                "fps": float(telem.get("fps", 0.0)),
                "latency_ms": float(telem.get("latency_ms", 0.0)),
                "active_vehicles_count": len(vehicles),
                "vehicles": vehicles,
            }
            return comp, normalized_telem

        else:
            # Fallback MineGuardIntegratedSystem
            result = self.engine_instance.process_frame(frame, frame_id=frame_id)
            annotated = result["annotated_frame"]
            vehicles = []
            for v in result.get("vehicles", []):
                vehicles.append({
                    "track_id": int(v["track_id"]),
                    "class_name": "VEHICLE",
                    "confidence": float(v.get("confidence", 0.0)),
                    "bbox": [int(b) for b in v.get("bbox", [0, 0, 0, 0])],
                    "distance_m": float(v["distance_m"]) if v.get("distance_m") is not None else None,
                    "relative_depth": None,
                    "speed_kmh": float(v.get("speed_kmh", 0.0)),
                    "ttc_s": float(v["ttc_s"]) if v.get("ttc_s") is not None else None,
                    "risk_tier": str(v.get("risk_tier", "SAFE")),
                })

            # Road clearance assessment
            min_dist = min([v["distance_m"] for v in vehicles if v["distance_m"] is not None], default=999.0)
            critical_vehicles = [v for v in vehicles if v["risk_tier"] == "CRITICAL"]
            caution_vehicles = [v for v in vehicles if v["risk_tier"] == "CAUTION"]

            if critical_vehicles or min_dist <= 15.0:
                road_status = "ROAD BLOCKED / HAZARD"
                road_detail = f"Immediate vehicle obstruction at {min_dist}m! Stop."
                road_class = "blocked"
            elif caution_vehicles or min_dist <= 35.0:
                road_status = "CAUTION — VEHICLE AHEAD"
                road_detail = f"Active vehicle crossing at {min_dist}m. Reduce speed."
                road_class = "caution"
            else:
                road_status = "ROAD CLEAR"
                road_detail = "Path is unobstructed."
                road_class = "clear"

            normalized_telem = {
                "active_engine": self.ENGINE_FALLBACK,
                "is_fallback": True,
                "engine_error": self.initialization_error,
                "system_status": "ONLINE (FALLBACK)",
                "pipeline_mode": str(result.get("pipeline_mode", "FALLBACK")),
                "fog_condition": str(result.get("fog_condition", "CLEAR")),
                "ambient_lux": float(result.get("ambient_lux", 100.0)),
                "road_status": road_status,
                "road_status_detail": road_detail,
                "road_status_class": road_class,
                "highest_risk": str(result.get("highest_risk", "SAFE")),
                "fps": float(result.get("fps", 0.0)),
                "latency_ms": float(result.get("latency_ms", 0.0)),
                "active_vehicles_count": len(vehicles),
                "vehicles": vehicles,
            }
            return annotated, normalized_telem


class BackgroundVideoWorker:
    """Dedicated background worker that reads video frames, manages bounded buffer,

    invokes perception inference, and stores the latest JPEG frame and telemetry.
    """

    def __init__(self, perception_adapter: PerceptionAdapter, video_source: str, jpeg_quality: int = 80):
        self.adapter = perception_adapter
        self.video_source = str(video_source)
        self.jpeg_quality = jpeg_quality

        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._latest_jpeg: Optional[bytes] = None
        self._latest_telemetry: Dict[str, Any] = {}
        self._frame_id = 0
        self._dropped_frames = 0
        self._total_processed_frames = 0
        self._source_switch_requested = False
        self._new_source: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        """Starts the worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._run_loop, name="MineGuardInferenceWorker", daemon=True)
        self._worker_thread.start()
        logger.info("BackgroundVideoWorker started with source: %s", self.video_source)

    def stop(self, timeout: float = 3.0) -> None:
        """Stops the worker thread cleanly and releases video capture."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        logger.info("BackgroundVideoWorker stopped.")

    def switch_source(self, new_source: str) -> None:
        """Requests switching the video source without terminating the thread."""
        with self._lock:
            self._new_source = str(new_source)
            self._source_switch_requested = True
        logger.info("Source switch requested to: %s", new_source)

    def get_latest_data(self) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """Returns thread-safe latest JPEG frame and telemetry copy."""
        with self._lock:
            return self._latest_jpeg, dict(self._latest_telemetry)

    def _run_loop(self) -> None:
        """Main worker loop."""
        self._cap = cv2.VideoCapture(self.video_source)
        self._frame_id = 0

        while self._running:
            # Check source switch
            with self._lock:
                if self._source_switch_requested:
                    if self._cap is not None:
                        self._cap.release()
                    self.video_source = self._new_source
                    self._cap = cv2.VideoCapture(self.video_source)
                    self._frame_id = 0
                    self._source_switch_requested = False
                    logger.info("Active video capture switched to: %s", self.video_source)

            if self._cap is None or not self._cap.isOpened():
                with self._lock:
                    self._latest_telemetry["system_status"] = "SOURCE ERROR"
                    self._latest_telemetry["road_status"] = "SOURCE ERROR"
                    self._latest_telemetry["road_status_detail"] = f"Failed to open video source: {self.video_source}"
                    self._latest_telemetry["road_status_class"] = "blocked"
                time.sleep(0.05)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Loop video seamlessly at end-of-stream
                total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    # Video source may have failed or closed
                    with self._lock:
                        self._latest_telemetry["system_status"] = "STREAM END"
                    time.sleep(0.05)
                    continue

            self._frame_id += 1

            try:
                display_frame, telemetry = self.adapter.process_frame(frame, self._frame_id)

                # Validate display frame
                if not isinstance(display_frame, np.ndarray) or display_frame.size == 0:
                    logger.error("Invalid display frame returned from perception adapter")
                    continue

                # Encode to JPEG
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                success, encoded_buf = cv2.imencode(".jpg", display_frame, encode_params)
                if not success or encoded_buf is None:
                    logger.error("Failed to encode frame to JPEG")
                    continue

                jpeg_bytes = encoded_buf.tobytes()

                # Add worker stats to telemetry
                telemetry["dropped_frames"] = self._dropped_frames
                telemetry["worker_frame_id"] = self._frame_id

                with self._lock:
                    self._latest_jpeg = jpeg_bytes
                    self._latest_telemetry = telemetry
                    self._total_processed_frames += 1

            except Exception as exc:
                logger.exception("Error in worker frame processing: %s", exc)
                time.sleep(0.02)

            time.sleep(0.005)

        if self._cap is not None:
            self._cap.release()
            self._cap = None
