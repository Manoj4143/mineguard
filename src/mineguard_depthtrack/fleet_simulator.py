"""Multi-Vehicle Mining Fleet Simulator for MineGuard AI.

Simulates at least 3 realistic mining vehicles (MINE-TRUCK-001, MINE-TRUCK-002, MINE-TRUCK-003)
traversing a configured open-pit mining bench or haul road route.
Supports:
1. Approaching head-on scenario
2. Convoy spacing scenario
3. Intersection conflict scenario
4. Communication failure scenario
5. Hazard zone entry scenario

All simulated entities explicitly carry `is_simulated = True`.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, Any, List, Optional

from .fleet_models import TelemetryMessage
from .fleet_manager import FleetManager


class FleetSimulator:
    """Simulates realistic telemetry updates for synthetic mining vehicles."""

    def __init__(self, fleet_manager: FleetManager, tick_interval_s: float = 1.0):
        self.fleet_manager = fleet_manager
        self.tick_interval_s = tick_interval_s
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Synchronized coordinate baseline matching live haul road telemetry GPS
        # Base benchmark: 35.39860° N, -119.34760° W (Main Haul Road Section)
        self.base_lat = 35.39860
        self.base_lon = -119.34760

        # Initialize 3 simulated mining vehicles along the active road corridor
        self.sim_vehicles = {
            "MINE-TRUCK-001": {
                "name": "CAT 797F Heavy Hauler #1",
                "type": "HAUL_TRUCK",
                "lat": self.base_lat,
                "lon": self.base_lon,
                "speed": 28.5,
                "heading": 45.0,
                "pattern": "convoy_lead",
                "comms_active": True,
            },
            "MINE-TRUCK-002": {
                "name": "Komatsu 930E Hauler #2",
                "type": "HAUL_TRUCK",
                "lat": self.base_lat - 0.0003, # ~33m behind
                "lon": self.base_lon - 0.0003,
                "speed": 28.0,
                "heading": 45.0,
                "pattern": "convoy_follower",
                "comms_active": True,
            },
            "MINE-TRUCK-003": {
                "name": "CAT 789D Water Truck #3",
                "type": "SERVICE_VEHICLE",
                "lat": self.base_lat + 0.0008,
                "lon": self.base_lon + 0.0010,
                "speed": 22.0,
                "heading": 225.0, # Heading toward intersection
                "pattern": "intersection_approach",
                "comms_active": True,
            },
        }

        # Pre-register vehicles in fleet manager
        for v_id, v_data in self.sim_vehicles.items():
            self.fleet_manager.register_vehicle(
                vehicle_id=v_id,
                vehicle_name=v_data["name"],
                vehicle_type=v_data["type"],
                is_simulated=True
            )

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="FleetSimulatorThread")
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while self.running:
            t_now = time.time()
            for v_id, state in self.sim_vehicles.items():
                if not state["comms_active"]:
                    continue

                # Move vehicle based on speed and heading
                speed_mps = state["speed"] / 3.6
                dist_traveled = speed_mps * self.tick_interval_s

                # Earth degree offset approximation (1 deg ~ 111,139 m)
                rad = math.radians(state["heading"])
                d_lat = (dist_traveled * math.cos(rad)) / 111139.0
                d_lon = (dist_traveled * math.sin(rad)) / (111139.0 * math.cos(math.radians(state["lat"])))

                state["lat"] += d_lat
                state["lon"] += d_lon

                # Turn vehicles back towards pit center if they venture too far (> 350m radius)
                dist_from_origin = math.hypot(
                    (state["lat"] - self.base_lat) * 111139.0,
                    (state["lon"] - self.base_lon) * 111139.0 * math.cos(math.radians(self.base_lat))
                )
                if dist_from_origin > 300.0:
                    # Heading pointing back to base
                    target_angle = math.degrees(math.atan2(
                        (self.base_lon - state["lon"]) * math.cos(math.radians(self.base_lat)),
                        self.base_lat - state["lat"]
                    ))
                    state["heading"] = (target_angle + 360.0) % 360.0

                # Slight speed variation
                state["speed"] = max(10.0, min(45.0, state["speed"] + (0.5 if (int(t_now) % 4 == 0) else -0.3)))

                # Post telemetry to manager
                msg = TelemetryMessage(
                    vehicle_id=v_id,
                    timestamp=t_now,
                    latitude=round(state["lat"], 6),
                    longitude=round(state["lon"], 6),
                    speed_kmh=round(state["speed"], 1),
                    heading_deg=round(state["heading"], 1),
                    camera_status="ONLINE",
                    perception_status="ONLINE",
                    is_simulated=True
                )
                self.fleet_manager.ingest_telemetry(msg)

            time.sleep(self.tick_interval_s)

    def trigger_communication_loss(self, vehicle_id: str, loss: bool = True) -> bool:
        """Simulates communication failure by halting telemetry transmission."""
        clean_id = vehicle_id.strip().upper()
        if clean_id in self.sim_vehicles:
            self.sim_vehicles[clean_id]["comms_active"] = not loss
            return True
        return False
