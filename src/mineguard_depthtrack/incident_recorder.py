"""Automatic Incident Recorder Engine for MineGuard AI.

Implements:
- Thread-safe rolling circular video buffer (bounded RAM)
- 15s pre-event buffer + 15s post-event buffer
- Configurable event triggers (CRITICAL_COLLISION_RISK, NEAR_MISS, ROAD_HAZARD, etc.)
- Asynchronous video clip encoding (.mp4)
- SQLite metadata persistence and JSON export
- Duplicate suppression for continuing safety events
"""

from __future__ import annotations

import collections
import json
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Deque

import cv2
import numpy as np


class IncidentEventType:
    CRITICAL_COLLISION_RISK = "CRITICAL_COLLISION_RISK"
    NEAR_MISS = "NEAR_MISS"
    ROAD_OBSTRUCTION = "ROAD_OBSTRUCTION"
    ROAD_HAZARD = "ROAD_HAZARD"
    VEHICLE_PROXIMITY = "VEHICLE_PROXIMITY"
    CAMERA_FAILURE = "CAMERA_FAILURE"
    PERCEPTION_FAILURE = "PERCEPTION_FAILURE"
    COMMUNICATION_FAILURE = "COMMUNICATION_FAILURE"


@dataclass
class IncidentRecord:
    incident_id: str
    event_type: str
    severity: str
    timestamp: float
    vehicle_id: str
    ttc_seconds: Optional[float]
    distance_m: Optional[float]
    distance_method: str
    fog_condition: str
    road_condition: str
    video_path: str
    metadata_json: str
    recording_status: str = "COMPLETED"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        try:
            d["metadata"] = json.loads(self.metadata_json)
        except Exception:
            d["metadata"] = {}
        return d


class RollingFrameBuffer:
    """Thread-safe bounded memory rolling video buffer indexed by timestamp."""

    def __init__(self, max_seconds: float = 16.0, fps: float = 20.0):
        self.max_frames = int(max_seconds * fps)
        self._buffer: Deque[Tuple[float, np.ndarray]] = collections.deque(maxlen=self.max_frames)
        self._lock = threading.Lock()

    def append(self, timestamp: float, frame: np.ndarray) -> None:
        with self._lock:
            # Downsample copy to save memory if frame is huge
            self._buffer.append((timestamp, frame.copy()))

    def get_frames_since(self, start_time: float) -> List[Tuple[float, np.ndarray]]:
        with self._lock:
            return [(t, f) for t, f in self._buffer if t >= start_time]

    def get_all_frames(self) -> List[Tuple[float, np.ndarray]]:
        with self._lock:
            return list(self._buffer)


class AutomaticIncidentRecorder:
    """Monitors telemetry and automatically encodes incident videos on trigger."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        storage_dir: Optional[str] = None,
        pre_event_seconds: float = 15.0,
        post_event_seconds: float = 15.0,
        fps: float = 20.0,
    ):
        self.storage_dir = Path(storage_dir or "data/incidents").resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(self.storage_dir / "incidents.db")
        self._conn = None
        if self.db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)

        self.pre_event_s = pre_event_seconds
        self.post_event_s = post_event_seconds
        self.fps = fps

        self.frame_buffer = RollingFrameBuffer(max_seconds=pre_event_seconds + 2.0, fps=fps)
        self.active_recordings: Dict[str, Dict[str, Any]] = {}
        self._record_queue: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._last_trigger_times: Dict[str, float] = {}
        self.cooldown_s = 10.0  # Prevent duplicate recordings of the same continuing event

        self._init_db()
        self.worker_thread = threading.Thread(target=self._encoding_worker, daemon=True, name="IncidentEncoder")
        self.worker_thread.start()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        return sqlite3.connect(self.db_path)

    def _close_conn(self, conn: sqlite3.Connection) -> None:
        if self._conn is None:
            conn.close()

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    ttc_seconds REAL,
                    distance_m REAL,
                    distance_method TEXT,
                    fog_condition TEXT,
                    road_condition TEXT,
                    video_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    recording_status TEXT NOT NULL,
                    description TEXT
                )
            """)
            conn.commit()
            self._close_conn(conn)

    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Pushes incoming live video frames into the rolling buffer and active post-recorders."""
        t = timestamp if timestamp is not None else time.time()
        self.frame_buffer.append(t, frame)

        with self._lock:
            # Feed post-event frames to any in-progress incident recordings
            completed_ids = []
            for inc_id, rec_ctx in self.active_recordings.items():
                rec_ctx["frames"].append(frame.copy())
                if t >= rec_ctx["end_time"]:
                    completed_ids.append(inc_id)

            for c_id in completed_ids:
                ctx = self.active_recordings.pop(c_id)
                self._record_queue.put(ctx)

    def trigger_incident(
        self,
        event_type: str,
        severity: str,
        vehicle_id: str,
        telemetry: Dict[str, Any],
        description: str = "",
        ttc: Optional[float] = None,
        distance_m: Optional[float] = None,
    ) -> Optional[str]:
        """Evaluates and initiates an incident recording session."""
        now = time.time()
        with self._lock:
            # Check duplicate suppression
            last_t = self._last_trigger_times.get(event_type, 0.0)
            if now - last_t < self.cooldown_s:
                return None  # Suppress duplicate

            self._last_trigger_times[event_type] = now
            inc_id = f"INC_{int(now)}_{event_type[:6]}"
            date_str = time.strftime("%Y%m%d", time.localtime(now))
            inc_dir = self.storage_dir / date_str
            inc_dir.mkdir(parents=True, exist_ok=True)
            video_filename = f"{inc_id}.mp4"
            target_video_path = str(inc_dir / video_filename)

            # Grab pre-event buffer
            pre_frames = [f for _, f in self.frame_buffer.get_frames_since(now - self.pre_event_s)]

            context = {
                "incident_id": inc_id,
                "event_type": event_type,
                "severity": severity,
                "timestamp": now,
                "vehicle_id": vehicle_id,
                "ttc": ttc,
                "distance_m": distance_m,
                "distance_method": "FUSED_MONOCULAR_DEPTH",
                "fog_condition": telemetry.get("fog_condition", "UNKNOWN"),
                "road_condition": telemetry.get("road_status", "UNKNOWN"),
                "telemetry": telemetry,
                "description": description or f"Automatic recording triggered by {event_type}",
                "target_video_path": target_video_path,
                "frames": pre_frames,
                "end_time": now + self.post_event_s,
            }

            self.active_recordings[inc_id] = context
            return inc_id

    def _encoding_worker(self) -> None:
        """Asynchronously writes queued frames to MP4 and commits metadata to SQLite."""
        while True:
            ctx = self._record_queue.get()
            try:
                frames = ctx["frames"]
                vid_path = ctx["target_video_path"]

                snapshot_path = None
                if frames:
                    # Find canonical dimension from trigger frame
                    trigger_frame_idx = min(len(frames) - 1, int(self.pre_event_s * self.fps))
                    trigger_frame = frames[trigger_frame_idx]
                    target_h, target_w = trigger_frame.shape[:2]

                    # Save trigger snapshot
                    snap_filename = f"{ctx['incident_id']}_snap.jpg"
                    snapshot_path = str(Path(vid_path).parent / snap_filename)
                    cv2.imwrite(snapshot_path, trigger_frame)

                    # Encode MP4 with strictly normalized frame dimensions
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(vid_path, fourcc, self.fps, (target_w, target_h))
                    if writer.isOpened():
                        for f in frames:
                            if f.shape[0] != target_h or f.shape[1] != target_w:
                                f = cv2.resize(f, (target_w, target_h))
                            writer.write(f)
                        writer.release()

                rec = IncidentRecord(
                    incident_id=ctx["incident_id"],
                    event_type=ctx["event_type"],
                    severity=ctx["severity"],
                    timestamp=ctx["timestamp"],
                    vehicle_id=ctx["vehicle_id"],
                    ttc_seconds=ctx["ttc"],
                    distance_m=ctx["distance_m"],
                    distance_method=ctx["distance_method"],
                    fog_condition=ctx["fog_condition"],
                    road_condition=ctx["road_condition"],
                    video_path=snapshot_path or vid_path,
                    metadata_json=json.dumps(ctx["telemetry"]),
                    recording_status="COMPLETED",
                    description=ctx["description"]
                )

                with self._lock:
                    conn = self._get_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO incidents (
                            incident_id, event_type, severity, timestamp, vehicle_id,
                            ttc_seconds, distance_m, distance_method, fog_condition,
                            road_condition, video_path, metadata_json, recording_status, description
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        rec.incident_id, rec.event_type, rec.severity, rec.timestamp,
                        rec.vehicle_id, rec.ttc_seconds, rec.distance_m, rec.distance_method,
                        rec.fog_condition, rec.road_condition, rec.video_path,
                        rec.metadata_json, rec.recording_status, rec.description
                    ))
                    conn.commit()
                    self._close_conn(conn)

            except Exception as e:
                print(f"[IncidentRecorder] Error encoding incident: {e}")
            finally:
                self._record_queue.task_done()

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM incidents")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM incidents WHERE severity = 'CRITICAL'")
            crit = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM incidents WHERE event_type = 'NEAR_MISS'")
            near_miss = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM incidents WHERE event_type LIKE '%HAZARD%'")
            hazards = cur.fetchone()[0]
            now = time.time()
            start_of_day = now - (now % 86400)
            cur.execute("SELECT COUNT(*) FROM incidents WHERE timestamp >= ?", (start_of_day,))
            today = cur.fetchone()[0]
            self._close_conn(conn)

            return {
                "total_incidents": total,
                "critical_incidents": crit,
                "near_misses": near_miss,
                "road_hazard_incidents": hazards,
                "today_incidents": today,
            }

    def get_all_incidents(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM incidents ORDER BY timestamp DESC LIMIT 50")
            rows = cur.fetchall()
            self._close_conn(conn)
            results = []
            for r in rows:
                d = dict(r)
                try:
                    d["metadata"] = json.loads(d["metadata_json"])
                except Exception:
                    d["metadata"] = {}
                results.append(d)
            return results
