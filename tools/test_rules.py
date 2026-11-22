import cv2
from mineguard_depthtrack.unified_engine import UnifiedMineGuardPerception

engine = UnifiedMineGuardPerception()
cap = cv2.VideoCapture("DepthTrack/data/user_uploads/video_1795341969.mp4")

for fid in [100, 700, 925, 1142, 1164]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if not ret: continue
    out = engine.process_frame(frame)
    tele = out["telemetry"]
    objs = out["objects"]
    print(f"Frame {fid}: status={tele['road_status']}")
    print(f"         detail={tele['road_status_detail']}")
    for o in objs:
        print(f"   -> ID:{o['track_id']} {o['class_name']} at {o['distance_m']}m, closing={o['closing_speed_kmh']} km/h -> {o.get('hazard_state')}")
