"""Detectors and the cached runner (optional install: pip install tssa[detect]); design section 6."""

from .detectors import CuboidDetector as CuboidDetector
from .detectors import Detector as Detector
from .detectors import KeypointDetector as KeypointDetector
from .detectors import MaskSegmenter as MaskSegmenter
from .runner import iter_frames as iter_frames
from .runner import run as run
