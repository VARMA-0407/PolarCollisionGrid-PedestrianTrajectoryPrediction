"""Object-detector interface (Stage 2A). A real YOLO wrapper is a Stage 2B deliverable.

Contract: ``detect`` receives one frame's metadata and opaque payload and returns ``RawDetection`` objects with
boxes in the pixel space of that frame (see ``types`` for the convention). It must be deterministic for fixed
weights/settings if it claims to be, must not keep state that depends on future frames, and must not raise for an
empty frame (return an empty sequence)."""

from abc import ABC, abstractmethod
from typing import Any, Sequence

from .types import ComponentInfo, FrameMeta, RawDetection


class Detector(ABC):
    @property
    @abstractmethod
    def info(self) -> ComponentInfo:
        """Identity (kind 'detector'); real detectors record weights hash and thresholds in ``parameters``."""

    @abstractmethod
    def detect(self, frame: FrameMeta, payload: Any) -> Sequence[RawDetection]:
        """Detections for this frame only."""
