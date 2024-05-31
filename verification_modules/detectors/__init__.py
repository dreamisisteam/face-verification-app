from typing import Literal

from .opencv import OpenCVDetector

DETECTORS_MAPPING = {
    "opencv": OpenCVDetector(),
}
DETECTORS_LITERAL = Literal["opencv"]

DEFAULT_DETECTOR_NAME = "opencv"
DEFAULT_DETECTOR = DETECTORS_MAPPING[DEFAULT_DETECTOR_NAME]
