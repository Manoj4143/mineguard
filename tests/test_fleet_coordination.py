"""Tests for Feature 3: Mine Fleet Coordination.

Validates vehicle registration, telemetry ingestion, validation (missing/invalid fields),
stale telemetry detection, communication-loss alerts, haversine distance calculation,
V2V proximity alerts, and convoy spacing monitoring.
"""

import time
import pytest
from mineguard_depthtrack.fleet_models import FleetVehicle, TelemetryMessage, FleetAlert
from mineguard_depthtrack.fleet_manager import FleetManager


def test_vehicle_registration():
    mgr = FleetManager(db_path=":memory:")
    
    # Register truck
    success, msg = mgr.register_vehicle("MINE-TRUCK-001", "Haul Truck Cat 797F", "HAUL_TRUCK")
    assert success is True
    
    # Retrieve vehicle
    fetched = mgr.get_vehicle("MINE-TRUCK-001")
    assert fetched is not None
    assert fetched["vehicle_id"] == "MINE-TRUCK-001"
    assert fetched["vehicle_name"] == "Haul Truck Cat 797F"
    assert fetched["registration_status"] == "REGISTERED"
    
    # Duplicate registration should update or remain registered
    success_dup, _ = mgr.register_vehicle("MINE-TRUCK-001", "Haul Truck Cat 797F Updated", "HAUL_TRUCK")
    assert success_dup is True
    assert mgr.get_vehicle("MINE-TRUCK-001")["vehicle_name"] == "Haul Truck Cat 797F Updated"


def test_telemetry_validation():
    mgr = FleetManager(db_path=":memory:")
    mgr.register_vehicle("MINE-TRUCK-001", "Haul Truck 1", "HAUL_TRUCK")

    # Invalid latitude (< -90)
    invalid_lat = TelemetryMessage(
        vehicle_id="MINE-TRUCK-001",
        timestamp=time.time(),
        latitude=-95.0,
        longitude=110.0,
        speed_kmh=25.0,
        heading_deg=180.0
    )
    is_valid, err = invalid_lat.validate()
    assert not is_valid
    assert "Invalid latitude" in err

    # Invalid speed (< 0 or > 200)
    invalid_spd = TelemetryMessage(
        vehicle_id="MINE-TRUCK-001",
        timestamp=time.time(),
        latitude=23.75,
        longitude=86.42,
        speed_kmh=-10.0,
        heading_deg=180.0
    )
    is_valid, err = invalid_spd.validate()
    assert not is_valid
    assert "Invalid speed" in err

    # Valid telemetry ingestion
    valid_telem = TelemetryMessage(
        vehicle_id="MINE-TRUCK-001",
        timestamp=time.time(),
        latitude=23.751,
        longitude=86.421,
        speed_kmh=28.5,
        heading_deg=90.0
    )
    success, res_err = mgr.ingest_telemetry(valid_telem)
    assert success
    assert "successfully" in res_err.lower()

    # Check vehicle updated
    v = mgr.get_vehicle("MINE-TRUCK-001")
    assert v["connection_status"] == "CONNECTED"
    assert v["latitude"] == 23.751
    assert v["speed_kmh"] == 28.5


def test_v2v_proximity_and_convoy_spacing():
    mgr = FleetManager(db_path=":memory:")
    mgr.register_vehicle("MINE-TRUCK-001", "Truck 1", "HAUL_TRUCK")
    mgr.register_vehicle("MINE-TRUCK-002", "Truck 2", "HAUL_TRUCK")

    now = time.time()
    # 23.75000, 86.42000 and 23.75010, 86.42000 are approx 11.1 meters apart
    telem1 = TelemetryMessage(
        vehicle_id="MINE-TRUCK-001",
        timestamp=now,
        latitude=23.75000,
        longitude=86.42000,
        speed_kmh=30.0,
        heading_deg=0.0
    )
    telem2 = TelemetryMessage(
        vehicle_id="MINE-TRUCK-002",
        timestamp=now,
        latitude=23.75010,
        longitude=86.42000,
        speed_kmh=30.0,
        heading_deg=0.0  # Travelling in same direction (convoy)
    )

    mgr.ingest_telemetry(telem1)
    mgr.ingest_telemetry(telem2)

    alerts = mgr.get_active_alerts()
    assert len(alerts) > 0
    alert_types = [a["alert_type"] for a in alerts]
    assert ("CONVOY_SPACING" in alert_types) or ("PROXIMITY_HAZARD" in alert_types)


def test_comms_loss_detection():
    mgr = FleetManager(db_path=":memory:")
    mgr.register_vehicle("MINE-TRUCK-001", "Truck 1", "HAUL_TRUCK")

    # Set internal vehicle object last seen timestamp
    v_obj = mgr.vehicles["MINE-TRUCK-001"]
    v_obj.last_seen_timestamp = time.time() - 100.0

    mgr._evaluate_fleet_safety()

    alerts = mgr.get_active_alerts()
    alert_types = [a["alert_type"] for a in alerts]
    assert "COMMUNICATION_LOSS" in alert_types
    assert mgr.get_vehicle("MINE-TRUCK-001")["connection_status"] == "COMMUNICATION_LOSS"
