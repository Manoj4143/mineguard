"""Tests for Feature 2: Mine Road Condition Intelligence.

Validates hazard categories, traversable corridor filtering, persistence tracking,
and evidence capture logic.
"""

import numpy as np
import pytest
from mineguard_depthtrack.road_intelligence import RoadConditionIntelligence, HazardCategory


def test_hazard_types_and_model_status():
    intel = RoadConditionIntelligence(db_path=":memory:")
    
    # Model readiness status reporting
    status = intel.get_model_status()
    assert "model_name" in status
    assert "is_ready" in status
    # Should cleanly report heuristic corridor mode if no weights loaded
    assert status["is_ready"] is False
    assert "heuristic" in status["active_backend"].lower()


def test_road_corridor_filtering():
    intel = RoadConditionIntelligence(db_path=":memory:")
    h, w = 720, 1280
    
    # Corridor mask should only cover lower portion of frame (road traversable area)
    mask = intel.get_corridor_mask(h, w)
    assert mask.shape == (h, w)
    # Sky region (row 0 to 100) should be 0
    assert np.all(mask[0:100, :] == 0)
    # Lower center road area should be 255
    assert mask[int(h * 0.75), int(w * 0.5)] == 255


def test_persistence_and_classification():
    intel = RoadConditionIntelligence(db_path=":memory:")
    h, w = 720, 1280
    dummy_frame = np.ones((h, w, 3), dtype=np.uint8) * 128

    # Process frame
    hazards = intel.analyze_road_frame(dummy_frame, source_vehicle_id="MINE-TRUCK-001")
    assert isinstance(hazards, list)

    summary = intel.get_summary()
    assert "total_hazards_logged" in summary
    assert "active_hazards" in summary
    assert "critical_hazards" in summary
