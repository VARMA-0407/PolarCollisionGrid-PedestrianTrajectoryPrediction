"""Multi-object-tracker interface (Stage 2A). A real ByteTrack wrapper is a Stage 2B deliverable.

Contract: ``update`` is called once per accepted frame, in order, with the VALIDATED detections of that frame
(possibly none - the tracker must still age its tracks). It returns ``TrackObservation`` objects. It is causal:
it may use only the current and earlier frames. Track ids are positive integers, unique within a sequence, and
never reused for another object. ``reset`` starts a new sequence (ids may restart)."""

from abc import ABC, abstractmethod
from typing import Sequence

from .types import ComponentInfo, DetectionRecord, FrameMeta, TrackObservation


class Tracker(ABC):
    @property
    @abstractmethod
    def info(self) -> ComponentInfo:
        """Identity (kind 'tracker')."""

    @abstractmethod
    def update(self, frame: FrameMeta, detections: Sequence[DetectionRecord]) -> Sequence[TrackObservation]:
        """Track states for this frame."""

    @abstractmethod
    def reset(self) -> None:
        """Forget all tracks (new sequence)."""
