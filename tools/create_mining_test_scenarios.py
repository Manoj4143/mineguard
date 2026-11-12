#!/usr/bin/env python3
"""create_mining_test_scenarios.py - Generate synthetic mine condition videos representing mining road challenges."""

import cv2
import numpy as np
from pathlib import Path

def generate_mine_scenarios():
    data_dir = Path("DepthTrack/data/scenarios")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = ["clear", "light_fog", "dense_fog", "night"]
    
    # 640x480 resolution, 40 frames, 10 fps
    w, h, n_frames, fps = 640, 480, 40, 10.0
    
    for sc in scenarios:
        video_path = data_dir / f"mine_{sc}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        
        for i in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Ground road (dirt/gravel color)
            road_color = (45, 60, 75) if sc != "night" else (15, 20, 25)
            cv2.rectangle(frame, (0, 220), (w, h), road_color, -1)
            
            # Simulated Mining Haul Truck (Large, moving forward/closer)
            truck_w = int(120 + i * 2.5)
            truck_h = int(100 + i * 2.0)
            truck_x = int((w - truck_w) / 2)
            truck_y = int(240 - (truck_h // 2) + i * 1.5)
            
            # Truck body
            cv2.rectangle(frame, (truck_x, truck_y), (truck_x + truck_w, truck_y + truck_h), (0, 140, 255), -1)
            # Cabin
            cv2.rectangle(frame, (truck_x + int(truck_w*0.2), truck_y - int(truck_h*0.3)), (truck_x + int(truck_w*0.8), truck_y), (40, 40, 40), -1)
            # Huge tires
            cv2.circle(frame, (truck_x + int(truck_w*0.15), truck_y + truck_h), int(truck_w*0.18), (10, 10, 10), -1)
            cv2.circle(frame, (truck_x + int(truck_w*0.85), truck_y + truck_h), int(truck_w*0.18), (10, 10, 10), -1)
            
            # Secondary vehicle on the right (approaching pickup truck)
            p_w = int(60 + i * 1.0)
            p_h = int(45 + i * 0.8)
            p_x = int(w * 0.72)
            p_y = int(250 + i * 1.0)
            cv2.rectangle(frame, (p_x, p_y), (p_x + p_w, p_y + p_h), (180, 50, 50), -1)
            
            # Environmental degradation
            if sc == "light_fog":
                # Add uniform haze
                frame = cv2.addWeighted(frame, 0.7, np.full_like(frame, 200), 0.3, 0)
            elif sc == "dense_fog":
                # Heavy dust / dense fog
                frame = cv2.addWeighted(frame, 0.4, np.full_like(frame, 220), 0.6, 0)
            elif sc == "night":
                # Low light with headlight cone
                mask = np.zeros((h, w), dtype=np.uint8)
                pts = np.array([[w//2, h//2], [0, h], [w, h]], np.int32)
                cv2.fillPoly(mask, [pts], 255)
                headlight = cv2.bitwise_and(frame, frame, mask=mask)
                ambient = (frame * 0.15).astype(np.uint8)
                frame = cv2.add(headlight, ambient)
                
            writer.write(frame)
        writer.release()
        print(f"Generated {video_path}")

if __name__ == "__main__":
    generate_mine_scenarios()

