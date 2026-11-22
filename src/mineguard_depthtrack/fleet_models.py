"""Fleet Coordination Data Models and Validation Schemas.

Provides typed models for vehicles, telemetry messages, V2V proximity alerts,
and shared hazards with strict coordinate, timestamp, and speed validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List


class VehicleType(str, Enum):
    HAUL_TRUCK = "HAUL_TRUCK"
    EXCAVATOR = "EXCAVATOR"
    WHEEL_LOADER = "WHEEL_LOADER"
    DRILL_RIG = "DRILL_RIG"
    SERVICE_VEHICLE = "SERVICE_VEHICLE"
    LIGHT_VEHICLE = "LIGHT_VEHICLE"
    UNKNOWN = "UNKNOWN"


class VehicleStatus(str, Enum):
    CONNECTED = "CONNECTED"
    OFFLINE = "OFFLINE"
    STALE = "STALE"
    COMMUNICATION_LOSS = "COMMUNICATION_LOSS"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class FleetAlertType(str, Enum):
    PROXIMITY_HAZARD = "PROXIMITY_HAZARD"
    CONVOY_SPACING = "CONVOY_SPACING"
    INTERSECTION_CONFLICT = "INTERSECTION_CONFLICT"
    COMMUNICATION_LOSS = "COMMUNICATION_LOSS"
    HAZARD_ZONE_ENTRY = "HAZARD_ZONE_ENTRY"


@dataclass
class FleetVehicle:
    vehicle_id: str
    vehicle_name: str
    vehicle_type: VehicleType = VehicleType.HAUL_TRUCK
    registration_status: str = "REGISTERED"
    connection_status: VehicleStatus = VehicleStatus.CONNECTED
    camera_status: str = "ONLINE"
    perception_status: str = "ONLINE"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    speed_kmh: Optional[float] = None
    heading_deg: Optional[float] = None
    last_seen_timestamp: float = field(default_factory=time.time)
    active_incidents: int = 0
    active_hazards: int = 0
    is_simulated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["vehicle_type"] = self.vehicle_type.value if isinstance(self.vehicle_type, VehicleType) else str(self.vehicle_type)
        d["connection_status"] = self.connection_status.value if isinstance(self.connection_status, VehicleStatus) else str(self.connection_status)
        return d


@dataclass
class TelemetryMessage:
    vehicle_id: str
    timestamp: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    speed_kmh: Optional[float] = None
    heading_deg: Optional[float] = None
    camera_status: str = "ONLINE"
    perception_status: str = "ONLINE"
    is_simulated: bool = False

    def validate(self) -> Tuple[bool, Optional[str]]:
        if not self.vehicle_id or not isinstance(self.vehicle_id, str):
            return False, "Invalid or missing vehicle_id"
        if not (0 < self.timestamp < time.time() + 3600):
            return False, "Invalid timestamp"
        if self.latitude is not None and not (-90.0 <= self.latitude <= 90.0):
            return False, f"Invalid latitude: {self.latitude}"
        if self.longitude is not None and not (-180.0 <= self.longitude <= 180.0):
            return False, f"Invalid longitude: {self.longitude}"
        if self.speed_kmh is not None and not (0.0 <= self.speed_kmh <= 160.0):
            return False, f"Invalid speed_kmh: {self.speed_kmh}"
        if self.heading_deg is not None and not (0.0 <= self.heading_deg <= 360.0):
            return False, f"Invalid heading_deg: {self.heading_deg}"
        return True, None


@dataclass
class FleetAlert:
    alert_id: str
    alert_type: FleetAlertType
    severity: AlertSeverity
    primary_vehicle_id: str
    secondary_vehicle_id: Optional[str]
    description: str
    timestamp: float = field(default_factory=time.time)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["alert_type"] = self.alert_type.value
        d["severity"] = self.severity.value
        return d


@dataclass
class SharedRoadHazard:
    hazard_id: str
    category: str
    severity: str
    latitude: Optional[float]
    longitude: Optional[float]
    reporting_vehicle_id: str
    description: str
    status: str = "ACTIVE"
    confidence: float = 0.85
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
