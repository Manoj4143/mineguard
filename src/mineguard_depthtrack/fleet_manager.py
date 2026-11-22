"""Fleet Manager Engine for MineGuard AI.

Handles vehicle registry, telemetry ingestion, thread-safe SQLite persistence,
V2V safety checks (proximity, convoy spacing, intersection conflicts, stale communication),
and shared road hazard management.
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .fleet_models import (
    FleetVehicle,
    VehicleType,
    VehicleStatus,
    TelemetryMessage,
    FleetAlert,
    FleetAlertType,
    AlertSeverity,
    SharedRoadHazard,
)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c


class FleetManager:
    """Coordinates fleet vehicles, ingests telemetry, and calculates V2V fleet safety alerts."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path("data/fleet/fleet.db").resolve())
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.vehicles: Dict[str, FleetVehicle] = {}
        self.alerts: List[FleetAlert] = []
        self.hazards: Dict[str, SharedRoadHazard] = {}
        self.stale_timeout_s = 30.0  # Telemetry considered stale after 30s
        self.comm_loss_timeout_s = 60.0  # Communication loss after 60s
        self._conn = None
        if self.db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_db()

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
                CREATE TABLE IF NOT EXISTS vehicles (
                    vehicle_id TEXT PRIMARY KEY,
                    vehicle_name TEXT NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    registration_status TEXT NOT NULL,
                    connection_status TEXT NOT NULL,
                    camera_status TEXT,
                    perception_status TEXT,
                    latitude REAL,
                    longitude REAL,
                    speed_kmh REAL,
                    heading_deg REAL,
                    last_seen_timestamp REAL,
                    active_incidents INTEGER DEFAULT 0,
                    active_hazards INTEGER DEFAULT 0,
                    is_simulated INTEGER DEFAULT 0
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    speed_kmh REAL,
                    heading_deg REAL,
                    camera_status TEXT,
                    perception_status TEXT,
                    is_simulated INTEGER DEFAULT 0
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS shared_hazards (
                    hazard_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    reporting_vehicle_id TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    confidence REAL,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.commit()
            if self._conn is None:
                conn.close()

    def register_vehicle(
        self,
        vehicle_id: str,
        vehicle_name: str,
        vehicle_type: str = "HAUL_TRUCK",
        is_simulated: bool = False
    ) -> Tuple[bool, str]:
        """Registers a vehicle into the fleet registry."""
        with self._lock:
            clean_id = vehicle_id.strip().upper()
            if not clean_id:
                return False, "Vehicle ID cannot be empty"

            try:
                v_type = VehicleType(vehicle_type)
            except ValueError:
                v_type = VehicleType.HAUL_TRUCK

            if clean_id in self.vehicles:
                # Update existing registration
                self.vehicles[clean_id].vehicle_name = vehicle_name
                self.vehicles[clean_id].vehicle_type = v_type
                return True, f"Vehicle {clean_id} updated successfully"

            v = FleetVehicle(
                vehicle_id=clean_id,
                vehicle_name=vehicle_name,
                vehicle_type=v_type,
                connection_status=VehicleStatus.CONNECTED,
                is_simulated=is_simulated,
                last_seen_timestamp=time.time()
            )
            self.vehicles[clean_id] = v

            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO vehicles (
                    vehicle_id, vehicle_name, vehicle_type, registration_status,
                    connection_status, camera_status, perception_status,
                    latitude, longitude, speed_kmh, heading_deg,
                    last_seen_timestamp, is_simulated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                v.vehicle_id, v.vehicle_name, v.vehicle_type.value, v.registration_status,
                v.connection_status.value, v.camera_status, v.perception_status,
                v.latitude, v.longitude, v.speed_kmh, v.heading_deg,
                v.last_seen_timestamp, 1 if is_simulated else 0
            ))
            conn.commit()
            self._close_conn(conn)
            return True, f"Vehicle {clean_id} registered successfully"

    def ingest_telemetry(self, telem: TelemetryMessage) -> Tuple[bool, str]:
        """Validates and stores incoming telemetry, evaluating V2V proximity."""
        valid, err = telem.validate()
        if not valid:
            return False, f"Validation error: {err}"

        with self._lock:
            v_id = telem.vehicle_id.strip().upper()
            if v_id not in self.vehicles:
                # Auto-register if not present
                self.register_vehicle(v_id, f"Vehicle {v_id}", is_simulated=telem.is_simulated)

            v = self.vehicles[v_id]
            # Handle out of order messages
            if telem.timestamp < v.last_seen_timestamp - 5.0:
                return False, "Out of order telemetry timestamp rejected"

            v.latitude = telem.latitude
            v.longitude = telem.longitude
            v.speed_kmh = telem.speed_kmh
            v.heading_deg = telem.heading_deg
            v.camera_status = telem.camera_status
            v.perception_status = telem.perception_status
            v.connection_status = VehicleStatus.CONNECTED
            v.last_seen_timestamp = time.time()
            v.is_simulated = telem.is_simulated

            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                UPDATE vehicles SET
                    latitude = ?, longitude = ?, speed_kmh = ?, heading_deg = ?,
                    camera_status = ?, perception_status = ?, connection_status = ?,
                    last_seen_timestamp = ?, is_simulated = ?
                WHERE vehicle_id = ?
            """, (
                v.latitude, v.longitude, v.speed_kmh, v.heading_deg,
                v.camera_status, v.perception_status, v.connection_status.value,
                v.last_seen_timestamp, 1 if v.is_simulated else 0, v_id
            ))
            cur.execute("""
                INSERT INTO telemetry_history (
                    vehicle_id, timestamp, latitude, longitude, speed_kmh,
                    heading_deg, camera_status, perception_status, is_simulated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                v_id, telem.timestamp, telem.latitude, telem.longitude, telem.speed_kmh,
                telem.heading_deg, telem.camera_status, telem.perception_status, 1 if telem.is_simulated else 0
            ))
            conn.commit()
            self._close_conn(conn)

            # Evaluate Fleet Safety Rules
            self._evaluate_fleet_safety()
            return True, "Telemetry processed successfully"

    def _evaluate_fleet_safety(self) -> None:
        """Evaluates V2V proximity, convoy spacing, intersection conflict, and comms loss."""
        now = time.time()
        active_alerts: List[FleetAlert] = []

        # 1. Check for stale and communication-loss vehicles
        for v_id, v in self.vehicles.items():
            dt = now - v.last_seen_timestamp
            if dt > self.comm_loss_timeout_s:
                v.connection_status = VehicleStatus.COMMUNICATION_LOSS
                active_alerts.append(FleetAlert(
                    alert_id=f"COMM_LOSS_{v_id}_{int(now)}",
                    alert_type=FleetAlertType.COMMUNICATION_LOSS,
                    severity=AlertSeverity.CRITICAL,
                    primary_vehicle_id=v_id,
                    secondary_vehicle_id=None,
                    description=f"Communication lost with {v_id} ({int(dt)}s elapsed without heartbeat)",
                    latitude=v.latitude,
                    longitude=v.longitude
                ))
            elif dt > self.stale_timeout_s:
                v.connection_status = VehicleStatus.STALE

        # 2. V2V Proximity and Convoy Spacing
        v_list = list(self.vehicles.values())
        for i in range(len(v_list)):
            for j in range(i + 1, len(v_list)):
                v1, v2 = v_list[i], v_list[j]
                if v1.connection_status in (VehicleStatus.OFFLINE, VehicleStatus.COMMUNICATION_LOSS):
                    continue
                if v2.connection_status in (VehicleStatus.OFFLINE, VehicleStatus.COMMUNICATION_LOSS):
                    continue

                if v1.latitude is not None and v1.longitude is not None and v2.latitude is not None and v2.longitude is not None:
                    dist_m = haversine_distance_m(v1.latitude, v1.longitude, v2.latitude, v2.longitude)

                    # Convoy Spacing Check (< 25m while travelling in same heading direction)
                    headings_aligned = False
                    if v1.heading_deg is not None and v2.heading_deg is not None:
                        diff_h = abs(v1.heading_deg - v2.heading_deg)
                        if diff_h <= 30.0 or diff_h >= 330.0:
                            headings_aligned = True

                    if dist_m <= 15.0:
                        # Critical Collision Risk
                        active_alerts.append(FleetAlert(
                            alert_id=f"PROX_CRIT_{v1.vehicle_id}_{v2.vehicle_id}",
                            alert_type=FleetAlertType.PROXIMITY_HAZARD,
                            severity=AlertSeverity.CRITICAL,
                            primary_vehicle_id=v1.vehicle_id,
                            secondary_vehicle_id=v2.vehicle_id,
                            description=f"CRITICAL V2V Proximity: {v1.vehicle_id} and {v2.vehicle_id} are only {dist_m:.1f}m apart!",
                            latitude=v1.latitude,
                            longitude=v1.longitude
                        ))
                    elif dist_m <= 35.0:
                        if headings_aligned:
                            active_alerts.append(FleetAlert(
                                alert_id=f"CONVOY_{v1.vehicle_id}_{v2.vehicle_id}",
                                alert_type=FleetAlertType.CONVOY_SPACING,
                                severity=AlertSeverity.WARNING,
                                primary_vehicle_id=v1.vehicle_id,
                                secondary_vehicle_id=v2.vehicle_id,
                                description=f"Convoy spacing warning: {v1.vehicle_id} and {v2.vehicle_id} following at {dist_m:.1f}m (min 40m required)",
                                latitude=v1.latitude,
                                longitude=v1.longitude
                            ))
                        else:
                            active_alerts.append(FleetAlert(
                                alert_id=f"INTERSECT_{v1.vehicle_id}_{v2.vehicle_id}",
                                alert_type=FleetAlertType.INTERSECTION_CONFLICT,
                                severity=AlertSeverity.WARNING,
                                primary_vehicle_id=v1.vehicle_id,
                                secondary_vehicle_id=v2.vehicle_id,
                                description=f"Approaching intersection conflict between {v1.vehicle_id} and {v2.vehicle_id} ({dist_m:.1f}m)",
                                latitude=v1.latitude,
                                longitude=v1.longitude
                            ))

        # 3. Check Shared Hazard Zones
        for v in v_list:
            if v.latitude is None or v.longitude is None:
                continue
            for h in self.hazards.values():
                if h.status == "ACTIVE" and h.latitude is not None and h.longitude is not None:
                    h_dist = haversine_distance_m(v.latitude, v.longitude, h.latitude, h.longitude)
                    if h_dist <= 50.0:
                        active_alerts.append(FleetAlert(
                            alert_id=f"HAZARD_ZONE_{v.vehicle_id}_{h.hazard_id}",
                            alert_type=FleetAlertType.HAZARD_ZONE_ENTRY,
                            severity=AlertSeverity.WARNING,
                            primary_vehicle_id=v.vehicle_id,
                            secondary_vehicle_id=None,
                            description=f"{v.vehicle_id} entered danger zone of shared hazard {h.category} ({h_dist:.1f}m)",
                            latitude=v.latitude,
                            longitude=v.longitude
                        ))

        self.alerts = active_alerts

    def add_shared_hazard(
        self,
        category: str,
        severity: str,
        latitude: Optional[float],
        longitude: Optional[float],
        reporting_vehicle_id: str,
        description: str,
        confidence: float = 0.85
    ) -> SharedRoadHazard:
        """Creates or updates a shared road hazard report."""
        with self._lock:
            # Check for existing proximate hazard within 10m to prevent duplication
            for h in self.hazards.values():
                if h.status == "ACTIVE" and h.category == category and h.latitude is not None and h.longitude is not None and latitude is not None and longitude is not None:
                    dist = haversine_distance_m(h.latitude, h.longitude, latitude, longitude)
                    if dist <= 12.0:
                        h.updated_at = time.time()
                        h.confidence = max(h.confidence, confidence)
                        return h

            h_id = f"HAZ_{category}_{int(time.time() * 1000) % 100000}"
            hazard = SharedRoadHazard(
                hazard_id=h_id,
                category=category,
                severity=severity,
                latitude=latitude,
                longitude=longitude,
                reporting_vehicle_id=reporting_vehicle_id,
                description=description,
                confidence=confidence
            )
            self.hazards[h_id] = hazard

            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO shared_hazards (
                    hazard_id, category, severity, latitude, longitude,
                    reporting_vehicle_id, description, status, confidence,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hazard.hazard_id, hazard.category, hazard.severity,
                hazard.latitude, hazard.longitude, hazard.reporting_vehicle_id,
                hazard.description, hazard.status, hazard.confidence,
                hazard.created_at, hazard.updated_at
            ))
            conn.commit()
            self._close_conn(conn)
            return hazard

    def get_fleet_summary(self) -> Dict[str, Any]:
        """Returns fleet dashboard summary metrics."""
        with self._lock:
            total = len(self.vehicles)
            connected = sum(1 for v in self.vehicles.values() if v.connection_status == VehicleStatus.CONNECTED)
            offline = sum(1 for v in self.vehicles.values() if v.connection_status in (VehicleStatus.OFFLINE, VehicleStatus.COMMUNICATION_LOSS))
            stale = sum(1 for v in self.vehicles.values() if v.connection_status == VehicleStatus.STALE)
            simulated = sum(1 for v in self.vehicles.values() if v.is_simulated)
            active_alerts_count = len(self.alerts)
            shared_hazards_count = sum(1 for h in self.hazards.values() if h.status == "ACTIVE")

            return {
                "total_vehicles": total,
                "connected_vehicles": connected,
                "offline_vehicles": offline,
                "stale_vehicles": stale,
                "simulated_vehicles": simulated,
                "active_fleet_alerts": active_alerts_count,
                "shared_road_hazards": shared_hazards_count,
            }

    def get_all_vehicles(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [v.to_dict() for v in self.vehicles.values()]

    def get_vehicle(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            v = self.vehicles.get(vehicle_id.strip().upper())
            return v.to_dict() if v else None

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [a.to_dict() for a in self.alerts]

    def get_shared_hazards(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [h.to_dict() for h in self.hazards.values()]
