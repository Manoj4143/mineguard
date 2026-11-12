"""DepthTrack.

DepthTrack: YOLO detection, Depth Anything V2 depth estimation, and ByteTrack
Depth Anything V2 depth estimation, ByteTrack multi-object tracking, distance
and collision-risk estimation, and a bird's-eye-view visualisation.
"""

__version__ = "1.2.0"

from .config import AppConfig

__all__ = ["AppConfig", "__version__"]
