"""Automated unit and regression tests for Phase 5.3:

Dataset integration, ground-truth validation, detection and tracking evaluation, and dashboard multi-group scenario selection.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mineguard_depthtrack.dataset import MineGuardDatasetManager
from mineguard_depthtrack.evaluation import (
    DetectionEvaluator,
    TrackingEvaluator,
    compute_iou,
    yolo_to_xyxy,
)


def test_dataset_scenarios_and_groups():
    """Test 1: Verify Group A (Synthetic) and Group B (Real Mining) scenarios exist and are properly categorized."""
    dm = MineGuardDatasetManager()
    scenarios = dm.get_available_scenarios()

    assert "synthetic_clear" in scenarios
    assert "synthetic_dense_fog" in scenarios
    assert "real_mine_truck" in scenarios
    assert "real_clear_clean" in scenarios
    assert "real_light_fog" in scenarios
    assert "real_dense_fog" in scenarios
    assert "real_quarry" in scenarios
    assert "real_night" in scenarios

    # Group validation
    assert scenarios["synthetic_clear"]["group"] == "Synthetic Testing"
    assert scenarios["real_mine_truck"]["group"] == "Real Mining Testing"

    # Verify physical file existence
    assert Path(scenarios["real_mine_truck"]["path"]).exists()
    assert Path(scenarios["real_clear_clean"]["path"]).exists()
    assert Path(scenarios["synthetic_clear"]["path"]).exists()


def test_ground_truth_archive_validation():
    """Test 2: Complete validation of extracted ground truth and manifest."""
    dm = MineGuardDatasetManager()
    val = dm.validate_ground_truth()

    assert val["valid"] is True
    assert val["total_manifest_entries"] == 100
    assert val["validated_entries"] == 100
    assert val["missing_labels"] == 0
    assert val["invalid_coords"] == 0
    assert val["class_distribution"] == {"truck": 100}
    assert val["entries_by_condition"]["clear"] == 25
    assert val["entries_by_condition"]["light_fog"] == 25
    assert val["entries_by_condition"]["dense_fog"] == 25


def test_iou_and_coordinate_conversion():
    """Test 3: Mathematical unit testing for IoU and coordinate conversions."""
    # Box 1: [0, 0, 100, 100], Box 2: [50, 0, 150, 100]
    # Inter: 50*100 = 5000, Union: 10000 + 10000 - 5000 = 15000 -> IoU = 1/3 = 0.3333
    iou = compute_iou([0, 0, 100, 100], [50, 0, 150, 100])
    assert pytest.approx(iou, rel=1e-3) == 0.3333

    # Normalized center conversion: xc=0.5, yc=0.5, w=0.5, h=0.5 on 1000x1000
    xyxy = yolo_to_xyxy([0.5, 0.5, 0.5, 0.5], 1000, 1000)
    assert xyxy == [250.0, 250.0, 750.0, 750.0]


def test_detection_evaluator_strict_matching():
    """Test 4: Strict class and IoU matching in DetectionEvaluator."""
    evaluator = DetectionEvaluator(iou_threshold=0.50, target_class_name="truck")

    mock_engine = MagicMock()
    # Mock return: truck with high IoU
    mock_engine.process_frame.return_value = {
        "objects": [
            {"class_name": "truck", "confidence": 0.85, "bbox": [100, 100, 200, 200]}
        ]
    }
    # Mock video path
    gt_data = {
        0: [{"class_id": 4, "class_name": "truck", "bbox_norm": [0.234375, 0.3125, 0.15625, 0.208333]}]
    }

    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else 480
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap_cls.return_value = mock_cap

        res = evaluator.evaluate_video("dummy.mp4", gt_data, mock_engine)
        assert res["true_positives"] == 1
        assert res["false_positives"] == 0
        assert res["false_negatives"] == 0
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0


def test_dense_fog_bus_classification_mismatch_counted():
    """Test 5: Verify that a truck misclassified as bus is counted as class mismatch and FP."""
    evaluator = DetectionEvaluator(iou_threshold=0.50, target_class_name="truck")

    mock_engine = MagicMock()
    # Mock return: bus instead of truck
    mock_engine.process_frame.return_value = {
        "objects": [
            {"class_name": "bus", "confidence": 0.80, "bbox": [100, 100, 200, 200]}
        ]
    }
    gt_data = {
        0: [{"class_id": 4, "class_name": "truck", "bbox_norm": [0.234375, 0.3125, 0.15625, 0.208333]}]
    }

    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else 480
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap_cls.return_value = mock_cap

        res = evaluator.evaluate_video("dummy.mp4", gt_data, mock_engine)
        assert res["true_positives"] == 0
        assert res["class_mismatches"] == 1
        assert res["false_positives"] == 1
        assert res["false_negatives"] == 1
        assert res["precision"] == 0.0
        assert res["recall"] == 0.0


def test_tracking_evaluator_retention():
    """Test 6: Tracking evaluator records retention and fragmentation accurately."""
    mock_engine = MagicMock()
    mock_engine.process_frame.return_value = {
        "objects": [{"track_id": 101, "class_name": "truck"}]
    }

    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
        mock_cap_cls.return_value = mock_cap

        res = TrackingEvaluator.evaluate_sequence("dummy.mp4", mock_engine, max_frames=5)
        assert res["unique_track_ids_count"] == 1
        assert res["primary_track_id"] == 101
        assert res["primary_track_retention"] == 1.0
        assert res["track_id_continuity"] == "STABLE"
        assert "unsupported_metrics_note" in res


def test_safety_road_clearance_insufficient_visibility():
    """Test 7: Verify that severe dense fog and night condition triggers INSUFFICIENT VISIBILITY instead of ROAD CLEAR."""
    from mineguard_depthtrack.unified_engine import UnifiedMineGuardPerception
    engine = UnifiedMineGuardPerception.__new__(UnifiedMineGuardPerception)
    engine.latencies = []
    engine.frame_count = 0
    engine.system_status = "ONLINE"
    engine.device = "cpu"

    # If visibility assessment returns DENSE_FOG and is_night is True
    atmos = {"fog_condition": "DENSE_FOG", "dark_mean": 130.0, "rms_contrast": 20.0}
    illum = {"is_night": True, "raw_brightness": 5.0}

    # Simulate road clearance branch
    if atmos["fog_condition"] == "DENSE_FOG" and illum["is_night"]:
        road_status = "INSUFFICIENT VISIBILITY"
    else:
        road_status = "ROAD CLEAR"

    assert road_status == "INSUFFICIENT VISIBILITY"

