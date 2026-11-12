"""End-to-end perception pipeline.

Owns the per-frame loop: read frame -> detect -> track -> estimate depth ->
render panels -> write output, followed by an optional ffmpeg re-encode pass.
Models and renderer are injected/constructed from an :class:`AppConfig`.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import cv2

from .config import AppConfig
from .logging_utils import get_logger
from .models import DepthEstimator, ObjectTracker, VehicleDetector
from .visualization import Renderer

logger = get_logger(__name__)


class VideoSource:
    """Context-managed OpenCV video reader exposing basic stream metadata."""

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class PerceptionPipeline:
    """Runs the full detection + depth + BEV perception pipeline on a video."""

    def __init__(self, cfg: AppConfig, asset_dir: Path | None = None):
        self.cfg = cfg
        self.asset_dir = asset_dir or Path("assets")
        self.device = self._resolve_device(cfg.model.device)

        weights = VehicleDetector.resolve_weights(
            cfg.model.detector_weights, self.asset_dir
        )
        self.detector = VehicleDetector(cfg.model, self.device, weights)
        self.depth = DepthEstimator(cfg.model, self.device)
        self.tracker = ObjectTracker(max_history=cfg.model.track_history)
        self.renderer = Renderer(cfg)

    # ---- public API ----------------------------------------------------------

    def run(self, input_path: str) -> Path:
        """Process ``input_path`` and return the path to the final output video."""
        if not Path(input_path).is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info("Device: %s", self.device)
        self.detector.load()
        self.depth.load()

        with VideoSource(input_path) as src:
            src_fps = src.fps or self.cfg.video.default_fps
            logger.info(
                "Input: %s  %dx%d  %.1ffps  %d frames",
                input_path, src.width, src.height, src_fps, src.total_frames,
            )
            tmp_out, final_out = self._output_paths()
            writer = self._make_writer(tmp_out, src_fps)
            try:
                self._process_loop(src, writer, src_fps)
            finally:
                writer.release()

        return self._finalize(tmp_out, final_out)

    # ---- per-frame processing ------------------------------------------------

    def _process_loop(self, src: VideoSource, writer, src_fps: float) -> None:
        from tqdm import tqdm

        work_w = min(src.width, self.cfg.video.work_width)
        work_h = int(src.height * (work_w / src.width)) if src.width else self.cfg.video.work_width

        pbar = tqdm(total=src.total_frames, desc="Processing", unit="fr")
        t0 = time.time()
        frame_no = 0
        fps_avg = 0.0

        for bgr in src:
            frame_no += 1
            work = cv2.resize(bgr, (work_w, work_h))

            detections = self.detector.detect(work)
            tracked = self.tracker.update(detections)

            depth_input = cv2.resize(
                work, (self.cfg.video.depth_input_width, self.cfg.video.depth_input_height)
            )
            depth_norm = self.depth.estimate(depth_input)
            depth_norm = cv2.resize(depth_norm, (work_w, work_h))

            left = self.renderer.draw_left_panel(work, tracked, depth_norm, self.tracker)
            right = self.renderer.draw_right_panel(depth_norm, tracked, work_w, work_h)
            composed = self.renderer.compose(left, right)

            elapsed = time.time() - t0
            fps_now = frame_no / max(elapsed, 1e-6)
            fps_avg = 0.9 * fps_avg + 0.1 * fps_now if fps_avg > 0 else fps_now

            self.renderer.add_hud(composed, fps_avg, frame_no, src.total_frames, self.device)
            writer.write(composed)
            pbar.update(1)

        pbar.close()
        total_elapsed = time.time() - t0
        logger.info(
            "Processed %d frames in %.1fs (%.1f fps avg)",
            frame_no, total_elapsed, frame_no / max(total_elapsed, 1e-6),
        )

    # ---- output handling -----------------------------------------------------

    def _output_paths(self) -> tuple[Path, Path]:
        out_dir = Path(self.cfg.video.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "_tmp_raw.mp4", out_dir / self.cfg.video.output_name

    def _make_writer(self, tmp_out: Path, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        size = (
            self.cfg.video.output_width + self.cfg.video.divider_width,
            self.cfg.video.output_height,
        )
        return cv2.VideoWriter(str(tmp_out), fourcc, fps, size)

    def _finalize(self, tmp_out: Path, final_out: Path) -> Path:
        if self._ffmpeg_available():
            logger.info("Re-encoding with ffmpeg -> %s", final_out)
            cmd = [
                "ffmpeg", "-y", "-i", str(tmp_out),
                "-c:v", self.cfg.video.encode_codec,
                "-preset", self.cfg.video.encode_preset,
                "-crf", str(self.cfg.video.encode_crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(final_out),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                tmp_out.unlink(missing_ok=True)
                size_mb = final_out.stat().st_size / (1024 ** 2)
                logger.info("Output saved: %s (%.1f MB)", final_out, size_mb)
                return final_out
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode()[:300] if exc.stderr else ""
                logger.warning("ffmpeg failed (%s); writing uncompressed output", stderr)

        shutil.move(str(tmp_out), str(final_out))
        size_mb = final_out.stat().st_size / (1024 ** 2)
        logger.info("Output saved (uncompressed): %s (%.1f MB)", final_out, size_mb)
        logger.info("Install ffmpeg for smaller files: https://ffmpeg.org/download.html")
        return final_out

    # ---- helpers -------------------------------------------------------------

    @staticmethod
    def _ffmpeg_available() -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (OSError, subprocess.CalledProcessError):
            return False

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    logger.info("CUDA available: %s", name)
                    return "cuda"
                logger.warning("CUDA requested but unavailable; falling back to CPU")
            except ImportError:
                logger.warning("torch not importable; falling back to CPU")
        return "cpu"
