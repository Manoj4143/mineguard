import json
import threading
import time
import urllib.request
from pathlib import Path

from mineguard_depthtrack.dashboard import start_dashboard_server

def run_test():
    server_ready = threading.Event()

    def server_runner():
        from mineguard_depthtrack.dashboard import adapter, worker, server_instance, server_running, PerceptionAdapter, BackgroundVideoWorker, ThreadingHTTPServer, DashboardHTTPRequestHandler, current_source_path
        import mineguard_depthtrack.dashboard as dash
        dash.adapter = PerceptionAdapter(preferred_engine=PerceptionAdapter.ENGINE_UNIFIED)
        dash.worker = BackgroundVideoWorker(dash.adapter, current_source_path)
        dash.worker.start()
        dash.server_instance = ThreadingHTTPServer(('127.0.0.1', 8092), DashboardHTTPRequestHandler)
        server_ready.set()
        dash.server_instance.serve_forever()

    t = threading.Thread(target=server_runner, daemon=True)
    t.start()
    assert server_ready.wait(timeout=30.0), 'Server took too long to bind'
    time.sleep(1.0)

    base_url = 'http://127.0.0.1:8092'

    stream_open = threading.Event()
    mjpeg_chunks = 0
    stop_stream = threading.Event()

    def consume_mjpeg():
        nonlocal mjpeg_chunks
        try:
            req = urllib.request.urlopen(f'{base_url}/stream.mjpg', timeout=10)
            stream_open.set()
            while not stop_stream.is_set():
                data = req.read(2048)
                if data:
                    mjpeg_chunks += 1
            req.close()
        except Exception as e:
            print('MJPEG stream ended:', e)

    mjpeg_thread = threading.Thread(target=consume_mjpeg, daemon=True)
    mjpeg_thread.start()
    assert stream_open.wait(timeout=5.0), 'Failed to open MJPEG stream'
    print('1. Live MJPEG stream connected successfully!')

    print('2. Polling telemetry while MJPEG is actively streaming...')
    for i in range(5):
        t_req = time.time()
        req = urllib.request.urlopen(f'{base_url}/api/telemetry?t={t_req}')
        assert req.status == 200
        data = json.loads(req.read().decode('utf-8'))
        print(f'   Poll {i+1}: Engine={data.get("active_engine")} | Road={data.get("road_status")} | FPS={data.get("fps")} | WorkerFrameID={data.get("worker_frame_id")}')
        time.sleep(0.2)

    sample_video = Path('DepthTrack/data/scenarios/mine_light_fog.mp4')
    video_bytes = sample_video.read_bytes()
    print(f'3. Uploading {len(video_bytes)} bytes of real video while streaming...')
    req = urllib.request.Request(
        f'{base_url}/api/upload_video?filename=live_light_fog.mp4',
        data=video_bytes,
        headers={'Content-Length': str(len(video_bytes))},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode('utf-8'))
        print('   Upload response received:', res)
        assert res['success'] is True

    time.sleep(1.0)
    req = urllib.request.urlopen(f'{base_url}/api/telemetry?t={time.time()}')
    post_upload_telem = json.loads(req.read().decode('utf-8'))
    print(f'4. Post-upload worker telemetry: FrameID={post_upload_telem.get("worker_frame_id")} | FPS={post_upload_telem.get("fps")}')
    assert post_upload_telem.get('worker_frame_id', 0) > 0

    print('5. Switching engine to fallback (MineGuardIntegratedSystem)...')
    req = urllib.request.urlopen(f'{base_url}/api/set_engine?engine=MineGuardIntegratedSystem')
    assert req.status == 200
    engine_res = json.loads(req.read().decode('utf-8'))
    print('   Engine switch response:', engine_res)
    assert engine_res['active_engine'] == 'MineGuardIntegratedSystem'

    print('6. Switching back to UnifiedMineGuardPerception...')
    req = urllib.request.urlopen(f'{base_url}/api/set_engine?engine=UnifiedMineGuardPerception')
    assert req.status == 200
    engine_res = json.loads(req.read().decode('utf-8'))
    print('   Engine switch response:', engine_res)
    assert engine_res['active_engine'] == 'UnifiedMineGuardPerception'

    stop_stream.set()
    print(f'ALL TESTS PASSED! Total MJPEG chunks received during concurrency test: {mjpeg_chunks}')

if __name__ == '__main__':
    run_test()
