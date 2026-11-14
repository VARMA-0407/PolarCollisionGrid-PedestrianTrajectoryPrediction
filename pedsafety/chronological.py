"""Leakage-safe chronological window assignment.

This module is deliberately separate from the historical group splitter.  It
operates on raw, timestamped rows and returns window metadata for a later
window-generation step; it never creates model tensors or metrics.

Boundaries are half-open: train is ``(-inf, train_end)``, validation is
``[train_end, validation_end)``, and test is ``[validation_end, test_end)``.
The final test end may be omitted.  A candidate is retained only when its
complete observation/target interval and its causal feature dependency
interval fit inside one period.  Otherwise it is recorded as excluded.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .errors import ConfigError, PedsafetyError
from .windows import WindowConfig

CHRONOLOGICAL_VERSION = "chronological-evaluation-v1"
PARTITIONS = ("train", "validation", "test")


class ChronologicalSplitError(PedsafetyError):
    """A chronological split cannot be constructed safely."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__("[%s] %s" % (code, message))


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: float
    validation_end: float
    test_end: Optional[float] = None

    def __post_init__(self):
        values = (self.train_end, self.validation_end)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ConfigError("chronological boundaries must be finite numeric timestamps")
        if not self.train_end < self.validation_end:
            raise ConfigError("train_end must be earlier than validation_end")
        if self.test_end is not None:
            if isinstance(self.test_end, bool) or not isinstance(self.test_end, (int, float)) or not math.isfinite(self.test_end):
                raise ConfigError("test_end must be a finite numeric timestamp or None")
            if not self.validation_end < self.test_end:
                raise ConfigError("test_end must be later than validation_end")

    def partition(self, start: float, end: float) -> Optional[str]:
        if start < self.train_end and end <= self.train_end:
            return "train"
        if start >= self.train_end and end <= self.validation_end:
            return "validation"
        if start >= self.validation_end and (self.test_end is None or end <= self.test_end):
            return "test"
        return None


@dataclass(frozen=True)
class RawSequence:
    """Raw rows for one track/sequence, before trajectory or window processing."""

    sequence_id: str
    timestamps: Tuple[float, ...]
    frame_indices: Tuple[int, ...]
    scene_id: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ConfigError("sequence_id must be a non-empty string")
        if len(self.timestamps) != len(self.frame_indices):
            raise ConfigError("timestamps and frame_indices must have equal lengths")
        if not self.timestamps:
            raise ChronologicalSplitError("EMPTY_SEQUENCE", "sequence %r has no rows" % self.sequence_id)
        for i, (ts, frame) in enumerate(zip(self.timestamps, self.frame_indices)):
            if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
                raise ChronologicalSplitError("INVALID_TIMESTAMP", "sequence %r row %d has timestamp %r" % (self.sequence_id, i, ts))
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise ChronologicalSplitError("INVALID_FRAME", "sequence %r row %d has frame %r" % (self.sequence_id, i, frame))
            if i and (self.timestamps[i] <= self.timestamps[i - 1] or self.frame_indices[i] <= self.frame_indices[i - 1]):
                raise ChronologicalSplitError("NON_MONOTONIC_SEQUENCE", "sequence %r is not strictly ordered at row %d" % (self.sequence_id, i))


@dataclass(frozen=True)
class ChronologicalWindow:
    sequence_id: str
    start_index: int
    history_indices: Tuple[int, ...]
    target_indices: Tuple[int, ...]
    partition: str
    observation_start: float
    observation_end: float
    target_end: float
    dependency_start: float


@dataclass(frozen=True)
class ExcludedWindow:
    sequence_id: str
    start_index: int
    reason: str
    observation_start: float
    target_end: float


@dataclass(frozen=True)
class ChronologicalSplitResult:
    windows: Mapping[str, Tuple[ChronologicalWindow, ...]]
    excluded: Tuple[ExcludedWindow, ...]
    purge_lookback_frames: int
    purge_lookback_seconds_by_sequence: Mapping[str, float]
    single_scene_warning: Optional[str]

    @property
    def all_windows(self) -> Tuple[ChronologicalWindow, ...]:
        return tuple(w for p in PARTITIONS for w in self.windows[p])


def derive_purge_lookback_frames(window_config: WindowConfig, feature_lookback_frames: int) -> int:
    """Return the exact dependency look-back in native frames.

    ``WindowConfig.frame_stride`` spaces model steps, while trajectory features
    are computed at native rate.  The caller must state the feature look-back;
    this avoids inventing a fixed purge duration.  A causal velocity needs one
    prior native frame and acceleration needs two.  Offline centred smoothing
    is rejected because its future dependency is not a one-sided purge.
    """
    if isinstance(feature_lookback_frames, bool) or not isinstance(feature_lookback_frames, int) or feature_lookback_frames < 0:
        raise ConfigError("feature_lookback_frames must be an integer >= 0")
    return feature_lookback_frames


def split_raw_sequences(sequences: Iterable[RawSequence], boundaries: SplitBoundaries, window_config: WindowConfig,
                        *, feature_lookback_frames: int = 0, frame_step: int = 1) -> ChronologicalSplitResult:
    """Assign complete raw windows chronologically and record every exclusion."""
    if isinstance(frame_step, bool) or not isinstance(frame_step, int) or frame_step < 1:
        raise ConfigError("frame_step must be an integer >= 1")
    lookback = derive_purge_lookback_frames(window_config, feature_lookback_frames)
    seqs = list(sequences)
    if not seqs:
        raise ChronologicalSplitError("NO_SEQUENCES", "at least one raw sequence is required")
    out: Dict[str, List[ChronologicalWindow]] = {p: [] for p in PARTITIONS}
    excluded: List[ExcludedWindow] = []
    purge_seconds: Dict[str, float] = {}
    seen = set()
    for seq in seqs:
        if seq.sequence_id in seen:
            raise ChronologicalSplitError("DUPLICATE_SEQUENCE", "sequence_id %r appears more than once" % seq.sequence_id)
        seen.add(seq.sequence_id)
        step = window_config.frame_stride * frame_step
        span = window_config.obs_len + window_config.pred_len
        required = (span - 1) * step
        if len(seq.timestamps) < span:
            raise ChronologicalSplitError("INSUFFICIENT_SEQUENCE_LENGTH", "sequence %r has %d rows; need at least %d"
                                           % (seq.sequence_id, len(seq.timestamps), span))
        max_lookback = 0.0
        for start in range(0, len(seq.timestamps) - span + 1, window_config.window_stride * frame_step):
            h = tuple(start + i * step for i in range(window_config.obs_len))
            t = tuple(start + (window_config.obs_len + i) * step for i in range(window_config.pred_len))
            indices = h + t
            if seq.frame_indices[indices[-1]] - seq.frame_indices[indices[0]] != required:
                excluded.append(ExcludedWindow(seq.sequence_id, start, "non_contiguous_frames",
                                               seq.timestamps[h[0]], seq.timestamps[t[-1]]))
                continue
            obs_start, obs_end, target_end = seq.timestamps[h[0]], seq.timestamps[h[-1]], seq.timestamps[t[-1]]
            dependency_start = obs_start
            if start >= lookback:
                dependency_start = seq.timestamps[start - lookback]
            max_lookback = max(max_lookback, obs_start - dependency_start)
            partition = boundaries.partition(dependency_start, target_end)
            if partition is None:
                reason = "crosses_split_boundary"
                if dependency_start < boundaries.train_end <= target_end or dependency_start < boundaries.validation_end <= target_end:
                    reason = "purged_dependency_overlap"
                excluded.append(ExcludedWindow(seq.sequence_id, start, reason, obs_start, target_end))
                continue
            out[partition].append(ChronologicalWindow(seq.sequence_id, start, h, t, partition, obs_start, obs_end,
                                                      target_end, dependency_start))
        purge_seconds[seq.sequence_id] = max_lookback
    warning = None
    scenes = {s.scene_id for s in seqs if s.scene_id is not None}
    if len(scenes) == 1:
        warning = "temporal generalization within one scene only; scene-independent generalization is not established"
    return ChronologicalSplitResult({p: tuple(out[p]) for p in PARTITIONS}, tuple(excluded), lookback, purge_seconds, warning)

