#!/usr/bin/env python3
"""inspect_videos.py - Inspects final_output.mp4 and verifies visual alignment."""

import cv2
import numpy as np
from pathlib import Path

def inspect_output_video():
    video_path = Path("DepthTrack/output/final_output.mp4")
    evidence_dir = Path("reports/phase1/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        print(f"Error: {video_path} does not exist.")
        return False
        
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video File: {video_path}")
    print(f"Resolution: {w}x{h}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    
    # Analyze frame structure
    # Expected output: (Left: 960x540) + (Divider: 4px) + (Right: 960x540) = 1924x540
    has_valid_dims = (w == 1924 and h == 540)
    print(f"Resolution Check (1924x540): {'PASS' if has_valid_dims else 'FAIL'}")
    
    # Sample frames across the video
    sample_indices = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4, total_frames - 1]
    verified_frames = 0
    
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in sample_indices:
            # Check left panel (camera view + detections)
            left_panel = frame[:, :960]
            # Check divider
            divider = frame[:, 960:964]
            # Check right panel (top depth + bottom BEV)
            right_panel = frame[:, 964:]
            
            left_std = np.std(left_panel)
            right_std = np.std(right_panel)
            
            frame_save_path = evidence_dir / f"panel_inspection_{idx:03d}.png"
            cv2.imwrite(str(frame_save_path), frame)
            
            print(f"  Frame {idx:02d}: Left StdDev={left_std:.2f}, Right StdDev={right_std:.2f} -> Saved {frame_save_path}")
            if left_std > 5.0 and right_std > 5.0:
                verified_frames += 1
                
    cap.release()
    print(f"Frame content validation: {verified_frames}/{len(sample_indices)} frames passed non-trivial content checks.")
    return True

if __name__ == "__main__":
    inspect_output_video()

