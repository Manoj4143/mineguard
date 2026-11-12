"""Command-line interface for DepthTrack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import AppConfig
from .logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

BANNER = r"""
  DepthTrack
  YOLO . Depth Anything V2 . BEV . Collision Risk
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depthtrack",
        description="DepthTrack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str, help="Path to input dashcam video (mp4)")
    parser.add_argument(
        "-c", "--config", type=str, default=None,
        help="Path to a YAML config file (overrides defaults)",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Inference device (overrides config)",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="YOLO confidence threshold (overrides config)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Directory for the output video (overrides config)",
    )
    parser.add_argument(
        "--assets-dir", type=str, default="assets",
        help="Directory containing bundled model weights",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
        help="Logging verbosity",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def build_config(args: argparse.Namespace) -> AppConfig:
    """Build the config from defaults/YAML, then apply CLI overrides."""
    cfg = AppConfig.from_yaml(args.config) if args.config else AppConfig()
    if args.device is not None:
        cfg.model.device = args.device
    if args.conf is not None:
        cfg.model.confidence = args.conf
    if args.output_dir is not None:
        cfg.video.output_dir = args.output_dir
    # Re-validate after overrides.
    cfg.model.__post_init__()
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    logger.info(BANNER.strip())

    if not Path(args.input).is_file():
        logger.error("Input file not found: %s", args.input)
        return 1

    try:
        cfg = build_config(args)
    except (ValueError, TypeError, OSError) as exc:
        logger.error("Configuration error: %s", exc)
        return 2

    # Imported here so `--help`/`--version` don't pay the heavy import cost.
    from .pipeline import PerceptionPipeline

    try:
        pipeline = PerceptionPipeline(cfg, asset_dir=Path(args.assets_dir))
        output = pipeline.run(args.input)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130

    logger.info("Done: %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
