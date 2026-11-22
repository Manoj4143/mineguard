# Feature 3: Mine Fleet Coordination Report

## 1. Architectural Overview
Mine Fleet Coordination extends MineGuard AI from single-vehicle forward perception into a multi-vehicle open-pit haulage coordination and collision avoidance safety platform.

### Core Modules
- `src/mineguard_depthtrack/fleet_models.py`: Defines data schemas, strict validation rules, coordinate boundaries, and alert structures.
- `src/mineguard_depthtrack/fleet_manager.py`: Implements vehicle registry, telemetry ingestion, thread-safe SQLite persistence (`data/fleet/fleet.db`), and safety rule engines.
- `src/mineguard_depthtrack/fleet_simulator.py`: Coordinates a background thread simulating 3 mining vehicles (`MINE-TRUCK-001`, `MINE-TRUCK-002`, `MINE-TRUCK-003`) along simulated haul road paths with explicit `is_simulated=True` tagging.

---

## 2. Safety Logic & Collision Avoidance Rules
1. **Vehicle-to-Vehicle (V2V) Proximity Alerts**:
   - Computes real geodesic distances using the haversine formula between all active vehicle pairs.
   - Triggers `PROXIMITY_HAZARD` (`CRITICAL`) if separation $\le 15.0\text{ m}$.
2. **Convoy Spacing Violation**:
   - Evaluates vehicles traveling in aligned headings ($\pm 30^\circ$).
   - Triggers `CONVOY_SPACING` warning if following distance drops $\le 35.0\text{ m}$ (safety threshold $40\text{ m}$).
3. **Intersection Conflict Detection**:
   - Assesses vehicles on converging cross-trajectories approaching intersection zones simultaneously.
4. **Communication Loss & Stale Telemetry**:
   - Vehicles with heartbeats older than $30\text{ s}$ transition to `STALE`.
   - Heartbeats exceeding $60\text{ s}$ trigger a critical `COMMUNICATION_LOSS` alert.

---

## 3. REST API Endpoints
- `GET /api/fleet/summary`: Aggregate counts for vehicles, connected state, active alerts, and shared hazards.
- `GET /api/fleet/vehicles`: Real-time vehicle registry including speed, heading, and GPS coordinates.
- `GET /api/fleet/alerts`: Active V2V proximity, convoy, and comms alerts.
- `GET /api/fleet/hazards`: Distributed hazard registry across fleet members.
- `POST /api/fleet/register`: Register new equipment.
- `POST /api/fleet/telemetry`: Validate and ingest vehicle motion telemetry.

---

## 4. Test Verification
All 4 unit tests in `tests/test_fleet_coordination.py` passed with 100% compliance:
- `test_vehicle_registration` [PASSED]
- `test_telemetry_validation` [PASSED]
- `test_v2v_proximity_and_convoy_spacing` [PASSED]
- `test_comms_loss_detection` [PASSED]
