"""Performance MEASUREMENT scaffolding (Stage 2A). No performance has been measured.

``PerformanceRecorder`` records durations of named stages per frame using an injectable clock and summarises them.
Nothing is reported unless a real run records it. A summary is only allowed to back a performance CLAIM when
(a) no component was a mock, (b) the clock is the real monotonic clock, and (c) the hardware/runtime was declared.
Unit tests use a fake clock to check the arithmetic; those summaries are marked ``clock_is_real=False`` and can
never support a claim.

Definitions (for Stage 2B): per-frame latency = wall time of one stage for one frame; end-to-end latency = wall time
from frame hand-over to the last stage's output for that frame; processing FPS = accepted frames / total elapsed wall
time of the run (NOT 1 / mean latency); memory = peak Python-heap bytes by ``tracemalloc`` when enabled, which excludes
native, GPU and pinned memory, so GPU memory must be measured separately by the Stage 2B runtime.
"""

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .types import ComponentInfo

REAL_CLOCK = time.perf_counter


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]) of a non-empty sequence."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    s = sorted(values)
    rank = max(1, int(-(-q * len(s) // 100)))       # ceil(q/100 * n), at least 1
    return s[min(rank, len(s)) - 1]


@dataclass(frozen=True)
class StageStats:
    count: int
    mean_s: float
    p50_s: float
    p95_s: float
    max_s: float


@dataclass(frozen=True)
class PerformanceSummary:
    n_frames: int
    elapsed_s: float
    processing_fps: Optional[float]
    stages: Dict[str, StageStats]
    end_to_end: Optional[StageStats]
    peak_python_heap_bytes: Optional[int]
    hardware: Optional[str]
    components: Tuple[ComponentInfo, ...]
    clock_is_real: bool

    @property
    def any_mock(self) -> bool:
        return any(c.is_mock for c in self.components)

    @property
    def supports_performance_claims(self) -> bool:
        """False for mocks, fake clocks, undeclared hardware, or an empty run."""
        return (not self.any_mock and self.clock_is_real and bool(self.hardware and self.hardware.strip()) and self.n_frames > 0
                and bool(self.components))


class PerformanceRecorder:
    def __init__(self, clock: Callable[[], float] = REAL_CLOCK, *, track_memory: bool = False):
        self._clock = clock
        self._real = clock is REAL_CLOCK
        self._stage: Dict[str, List[float]] = {}
        self._e2e: List[float] = []
        self._frames = 0
        self._t_first: Optional[float] = None
        self._t_last: Optional[float] = None
        self._track_memory = track_memory
        self._started_tracemalloc = False

    def start_run(self) -> None:
        if self._track_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracemalloc = True
        if self._track_memory:
            tracemalloc.reset_peak()

    @contextmanager
    def frame(self) -> Iterator[None]:
        """Wrap ONE accepted frame's complete processing (gives end-to-end latency)."""
        t0 = self._clock()
        if self._t_first is None:
            self._t_first = t0
        try:
            yield
        finally:
            t1 = self._clock()
            self._e2e.append(t1 - t0)
            self._frames += 1
            self._t_last = t1

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = self._clock()
        try:
            yield
        finally:
            self._stage.setdefault(name, []).append(self._clock() - t0)

    @staticmethod
    def _stats(v: Sequence[float]) -> StageStats:
        return StageStats(len(v), sum(v) / len(v), percentile(v, 50), percentile(v, 95), max(v))

    def summary(self, components: Sequence[ComponentInfo], hardware: Optional[str] = None) -> PerformanceSummary:
        elapsed = 0.0 if self._t_first is None or self._t_last is None else self._t_last - self._t_first
        fps = (self._frames / elapsed) if (self._frames and elapsed > 0) else None
        peak = None
        if self._track_memory and tracemalloc.is_tracing():
            peak = tracemalloc.get_traced_memory()[1]
            if self._started_tracemalloc:
                tracemalloc.stop()
                self._started_tracemalloc = False
        return PerformanceSummary(self._frames, elapsed, fps, {k: self._stats(v) for k, v in sorted(self._stage.items()) if v},
                                  self._stats(self._e2e) if self._e2e else None, peak, hardware, tuple(components), self._real)
