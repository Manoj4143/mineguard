#!/usr/bin/env python3
"""inspect_repository.py - Codebase audit and architecture inspection for DepthTrack."""

import os
import sys
import importlib
from pathlib import Path

def audit_codebase():
    base_dir = Path("DepthTrack")
    src_dir = base_dir / "src" / "depthtrack"
    
    print("=" * 60)
    print("DEPTH TRACK CODEBASE AUDIT")
    print("=" * 60)
    
    # 1. Check directories
    print("\n[1] Directory Structure:")
    for root, dirs, files in os.walk(base_dir):
        rel = os.path.relpath(root, base_dir)
        if ".git" in rel or ".venv" in rel or "__pycache__" in rel:
            continue
        level = rel.count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            if not f.endswith(".pyc"):
                print(f"{subindent}{f}")

    # 2. Key Modules Inspection
    print("\n[2] Key Modules & Entry Points:")
    modules = [
        "depthtrack.cli",
        "depthtrack.config",
        "depthtrack.pipeline",
        "depthtrack.models.detector",
        "depthtrack.models.depth",
        "depthtrack.models.tracker",
        "depthtrack.perception.distance",
        "depthtrack.perception.risk",
        "depthtrack.visualization.bev",
        "depthtrack.visualization.colormap",
        "depthtrack.visualization.panels",
    ]
    for mod in modules:
        try:
            m = importlib.import_module(mod)
            print(f"  [OK] {mod} loaded successfully")
        except Exception as e:
            print(f"  [FAIL] {mod}: {e}")

if __name__ == "__main__":
    audit_codebase()

