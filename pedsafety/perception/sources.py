"""Frame-source interface (Stage 2A). No video decoding happens here.

A real source (OpenCV/FFmpeg based) is a Stage 2B deliverable and must live behind this interface so that no
other module imports a video library. Pixels are opaque: the pipeline passes ``payload`` to the detector and
never inspects it, so tests can use ``None`` and a real source can use an array."""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Tuple

from .types import ComponentInfo, FrameMeta


class FrameSource(ABC):
    @property
    @abstractmethod
    def info(self) -> ComponentInfo:
        """Identity of this source (kind must be 'frame_source')."""

    @abstractmethod
    def frames(self) -> Iterator[Tuple[FrameMeta, Any]]:
        """Yield ``(metadata, payload)`` in presentation order. Must not mutate earlier frames. Ordering and timestamp
        validity are CHECKED by the pipeline, not trusted."""
