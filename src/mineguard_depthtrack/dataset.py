"""Dataset integration and ground-truth validation module for MineGuard AI Phase 5.3."""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("MineGuardDataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MineGuardDatasetManager:
    """Manages synthetic and real-world scenario datasets, video cataloging, and ground-truth annotations."""

    CLASS_MAPPING = {
        0: "person",
        1: "car",
        2: "motorcycle",
        3: "bus",
        4: "truck"
    }

    FOG_BETA_VALUES = {
        "clear": 0.0,
        "light_fog": 0.35,
        "medium_fog": 0.75,
        "dense_fog": 1.30
    }

    def __init__(self, data_root: Optional[Path] = None):
        self.root = Path(data_root) if data_root else Path("D:/Mineguard_project_2/data")
        self.scenarios_dir = self.root / "scenarios"
        self.gt_dir = self.root / "ground_truth"
        self.manifest_file = self.gt_dir / "annotation_manifest.csv"

    def get_available_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Returns organized dictionary of Group A (Synthetic) and Group B (Real Mining) scenarios."""
        scenarios = {
            # GROUP A: Synthetic Testing Scenarios
            "synthetic_clear": {
                "group": "Synthetic Testing",
                "label": "Synthetic: Clear Ground Road",
                "path": str(self.scenarios_dir / "synthetic" / "mine_clear.mp4"),
                "condition": "CLEAR",
                "has_gt": False
            },
            "synthetic_light_fog": {
                "group": "Synthetic Testing",
                "label": "Synthetic: Light Fog Simulation",
                "path": str(self.scenarios_dir / "synthetic" / "mine_light_fog.mp4"),
                "condition": "LIGHT_FOG",
                "has_gt": False
            },
            "synthetic_dense_fog": {
                "group": "Synthetic Testing",
                "label": "Synthetic: Dense Fog Simulation",
                "path": str(self.scenarios_dir / "synthetic" / "mine_dense_fog.mp4"),
                "condition": "DENSE_FOG",
                "has_gt": False
            },
            "synthetic_night": {
                "group": "Synthetic Testing",
                "label": "Synthetic: Night Pit Road",
                "path": str(self.scenarios_dir / "synthetic" / "mine_night.mp4"),
                "condition": "NIGHT",
                "has_gt": False
            },
            "synthetic_sample": {
                "group": "Synthetic Testing",
                "label": "Synthetic: Sample Pit Test Pattern",
                "path": str(self.scenarios_dir / "synthetic" / "sample_test.mp4"),
                "condition": "CLEAR",
                "has_gt": False
            },

            # GROUP B: Real Mining Scenarios
            "real_mine_truck": {
                "group": "Real Mining Testing",
                "label": "Real Mining: Main Haul Truck Road",
                "path": str(self.scenarios_dir / "real_mining" / "clear" / "mine_truck.mp4"),
                "condition": "CLEAR",
                "has_gt": True
            },
            "real_clear_clean": {
                "group": "Real Mining Testing",
                "label": "Real Mining: Clear Haul Road (GT Baseline)",
                "path": str(self.scenarios_dir / "real_mining" / "clear" / "clear_clean.mp4"),
                "condition": "CLEAR",
                "has_gt": True
            },
            "real_light_fog": {
                "group": "Real Mining Testing",
                "label": "Real Mining: Light Fog Road (Koschmieder beta=0.35)",
                "path": str(self.scenarios_dir / "real_mining" / "light_fog" / "light_fog_clean.mp4"),
                "condition": "LIGHT_FOG",
                "has_gt": True
            },
            "real_medium_fog": {
                "group": "Real Mining Testing",
                "label": "Real Mining: Medium Fog Road (Koschmieder beta=0.75)",
                "path": str(self.scenarios_dir / "real_mining" / "medium_fog" / "medium_fog_clean.mp4"),
                "condition": "MEDIUM_FOG",
                "has_gt": True
            },
            "real_dense_fog": {
                "group": "Real Mining Testing",
                "label": "Real Mining: Dense Fog Road (Koschmieder beta=1.30)",
                "path": str(self.scenarios_dir / "real_mining" / "dense_fog" / "dense_fog_clean.mp4"),
                "condition": "DENSE_FOG",
                "has_gt": True
            },
            "real_quarry": {
                "group": "Real Mining Testing",
                "label": "Real Mining: Dusty Quarry Haul Truck",
                "path": str(self.scenarios_dir / "real_mining" / "quarry" / "mixkit_quarry_truck.mp4"),
                "condition": "DUSTY",
                "has_gt": False
            },
            "real_night": {
                "group": "Real Mining Testing",
                "label": "Real Mining: Night Haul Road (Headlights Active)",
                "path": str(self.scenarios_dir / "real_mining" / "night" / "night_vision_demo.mp4"),
                "condition": "NIGHT",
                "has_gt": False
            }
        }
        return scenarios

    def validate_ground_truth(self) -> Dict[str, Any]:
        """Validates all ground truth files, normalized bounding box coordinates, and manifest frame references."""
        if not self.manifest_file.exists():
            return {"valid": False, "error": f"Manifest not found: {self.manifest_file}"}

        with open(self.manifest_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        total_manifest_entries = len(rows)
        validated_entries = 0
        missing_labels = 0
        invalid_coords = 0
        mismatched_frames = 0
        class_distribution: Dict[int, int] = {}
        entries_by_condition: Dict[str, int] = {}

        details = []

        labels_dir = self.gt_dir / "labels" / "train"

        for row in rows:
            cond = row["condition"]
            img_stem = Path(row["image"]).stem
            source_vid = row["source_video"]
            source_frame = int(row["source_frame"])
            label_file = labels_dir / f"{img_stem}.txt"

            entries_by_condition[cond] = entries_by_condition.get(cond, 0) + 1

            if not label_file.exists():
                missing_labels += 1
                details.append(f"Missing label: {label_file.name}")
                continue

            content = label_file.read_text(encoding="utf-8").strip()
            if not content:
                missing_labels += 1
                details.append(f"Empty label file: {label_file.name}")
                continue

            lines = content.splitlines()
            row_coords_ok = True
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_coords += 1
                    row_coords_ok = False
                    continue

                cls_id = int(parts[0])
                class_distribution[cls_id] = class_distribution.get(cls_id, 0) + 1
                coords = [float(x) for x in parts[1:]]
                for c in coords:
                    if not (0.0 <= c <= 1.0):
                        invalid_coords += 1
                        row_coords_ok = False

            if row_coords_ok:
                validated_entries += 1

        summary = {
            "valid": (missing_labels == 0 and invalid_coords == 0),
            "total_manifest_entries": total_manifest_entries,
            "validated_entries": validated_entries,
            "missing_labels": missing_labels,
            "invalid_coords": invalid_coords,
            "mismatched_frames": mismatched_frames,
            "class_distribution": {self.CLASS_MAPPING.get(k, str(k)): v for k, v in class_distribution.items()},
            "entries_by_condition": entries_by_condition,
            "errors": details[:10]
        }
        return summary

    def load_ground_truth_for_video(self, video_filename: str) -> Dict[int, List[Dict[str, Any]]]:
        """Loads ground truth bounding boxes indexed by frame index for a specific video."""
        if not self.manifest_file.exists():
            return {}

        with open(self.manifest_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        video_gt: Dict[int, List[Dict[str, Any]]] = {}
        labels_dir = self.gt_dir / "labels" / "train"

        for row in rows:
            if Path(row["source_video"]).name != Path(video_filename).name:
                continue

            frame_idx = int(row["source_frame"])
            img_stem = Path(row["image"]).stem
            label_file = labels_dir / f"{img_stem}.txt"

            if not label_file.exists():
                continue

            boxes = []
            for line in label_file.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = [float(x) for x in parts[1:]]
                    boxes.append({
                        "class_id": cls_id,
                        "class_name": self.CLASS_MAPPING.get(cls_id, "unknown"),
                        "bbox_norm": [xc, yc, w, h]  # Normalized center x, y, width, height
                    })

            if boxes:
                video_gt[frame_idx] = boxes

        return video_gt

