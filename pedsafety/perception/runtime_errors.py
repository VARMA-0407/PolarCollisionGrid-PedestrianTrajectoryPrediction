"""Errors raised by the real (OpenCV / YOLO / ByteTrack) perception components. Stdlib only."""

from ..errors import PedsafetyError


class VideoSourceError(PedsafetyError):
    """A video file is missing, unreadable, or reports unusable properties (size, FPS)."""


class DetectorLoadError(PedsafetyError):
    """The detector model could not be located or loaded, or the requested device is unavailable."""


class TrackerInitError(PedsafetyError):
    """The ByteTrack implementation is not importable or rejected its configuration."""
