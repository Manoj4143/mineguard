# Feature 1: Automatic Incident Recorder Report

## 1. Architectural Overview
The Automatic Incident Recorder serves as an automated digital flight data and video black box for mining safety events.

### Core Modules
- `src/mineguard_depthtrack/incident_recorder.py`: Implements a bounded rolling circular video buffer (`RollingFrameBuffer`), automated triggers, asynchronous video encoding worker, and SQLite metadata cataloging (`data/incidents/incidents.db`).

---

## 2. Circular Rolling Buffer & Pre/Post Windows
- **Pre-Event Buffer**: 15 seconds of live footage continuously buffered in a thread-safe circular memory queue.
- **Post-Event Buffer**: 15 seconds of post-trigger footage captured immediately following an event trigger.
- **Async Video Worker**: Dedicated background worker encodes the combined 30-second sequence into standardized MP4 (`data/incidents/<YYYYMMDD>/INC_<timestamp>_<type>.mp4`) without dropping frames or stalling the real-time perception pipeline.

---

## 3. Automated Trigger Criteria & Duplicate Suppression
Automatic recordings trigger under the following verifiable safety conditions:
1. `CRITICAL_COLLISION_RISK`: When TTC $\le 2.0\text{ s}$ or distance $\le 15.0\text{ m}$.
2. `ROAD_HAZARD`: When a traversable corridor obstruction or critical hazard is confirmed.
3. `NEAR_MISS`: Deceleration exceeding emergency thresholds while in close proximity.
4. `PERCEPTION_FAILURE` / `CAMERA_FAILURE`: Critical degradation in sensor stream integrity.

### Duplicate Suppression
Implements a 10-second cooldown per event category to avoid spamming multiple incident files for a single continuing hazard event.

---

## 4. Test Verification
All unit tests in `tests/test_incident_recorder.py` passed with 100% compliance:
- `test_rolling_frame_buffer` [PASSED]
- `test_incident_trigger_and_metadata` [PASSED]
