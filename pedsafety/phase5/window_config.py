"""Strict window configuration (unknown keys rejected; keys starting with '_' are documentation).

For anchor frame ``t0`` (the last history sample), stride ``s``, history length ``H`` and horizon ``F``::

    history frames  h_k = t0 - (H - 1 - k) * s,   k = 0 .. H-1
    future frames   f_m = t0 + (m + 1) * s,       m = 0 .. F-1

With the approved values (30 fps, s = 3, H = 6, F = 15) the history covers t0-0.5 s .. t0 and the future
t0+0.1 s .. t0+1.5 s. The dependency interval of a window (every source frame its inputs or targets can depend on) is
``[t0 - (H-1)*s - 1, t0 + F*s]``; the extra ``-1`` is the Task 2 one-frame backward difference behind velocities.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np

from ..errors import ConfigError

TARGET_SOURCE_MODES = ("PILOT_SMOKE", "ANNOTATED_TARGET")
IMPLEMENTED_TARGET_SOURCE_MODES = ("PILOT_SMOKE",)
KNOWN_NODE_TYPES = ("pedestrian", "vehicle", "bicyclist")
MISSING_OBSERVATION_POLICIES = ("TARGET_STRICT_CONTEXT_MASKED",)
TRACK_CONTINUITY_POLICIES = ("NO_INTERPOLATION_NO_FORWARD_FILL_NO_ID_MERGE",)
INPUT_SOURCES = ("phase4_task3_interaction_representation",)


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class WindowConfig:
    config_version: str
    source_fps: float
    target_sampling_hz: float
    sampling_stride_frames: int
    history_length: int
    prediction_length: int
    history_span_s: float
    future_horizon_s: float
    anchor_stride_frames: int
    anchor_phase_frames: int
    timestamp_tolerance_s: float
    coordinate_system: str
    coordinate_unit: str
    target_source_mode: str
    input_source: str
    expected_feature_dictionary_version: str
    target_node_types: Tuple[str, ...]
    default_context_node_types: Tuple[str, ...]
    optional_context_node_types: Tuple[str, ...]
    semantically_uncertain_node_types: Tuple[str, ...]
    missing_observation_policy: str
    track_continuity_policy: str
    pilot_split_label: str

    def __post_init__(self):
        for n in ("config_version", "coordinate_system", "coordinate_unit", "expected_feature_dictionary_version", "pilot_split_label"):
            v = getattr(self, n)
            if not (isinstance(v, str) and v.strip()):
                raise ConfigError("%s must be a non-empty string" % n)
        for n in ("source_fps", "target_sampling_hz", "history_span_s", "future_horizon_s", "timestamp_tolerance_s"):
            if not (_num(getattr(self, n)) and getattr(self, n) > 0):
                raise ConfigError("%s must be a positive finite number" % n)
        for n in ("sampling_stride_frames", "history_length", "prediction_length", "anchor_stride_frames"):
            if not (_int(getattr(self, n)) and getattr(self, n) >= 1):
                raise ConfigError("%s must be an integer >= 1" % n)
        if self.history_length < 2:
            raise ConfigError("history_length must be >= 2 (a span needs two samples)")
        if not (_int(self.anchor_phase_frames) and 0 <= self.anchor_phase_frames < self.anchor_stride_frames):
            raise ConfigError("anchor_phase_frames must be an integer in [0, anchor_stride_frames)")
        tol = self.timestamp_tolerance_s
        # stride must convert source fps to the target frequency exactly
        if abs(self.source_fps / self.sampling_stride_frames - self.target_sampling_hz) > 1e-9:
            raise ConfigError("source_fps / sampling_stride_frames (%r) must equal target_sampling_hz (%r)"
                              % (self.source_fps / self.sampling_stride_frames, self.target_sampling_hz))
        if abs((self.history_length - 1) * self.sampling_stride_frames / self.source_fps - self.history_span_s) > tol:
            raise ConfigError("history_span_s must equal (history_length - 1) * stride / fps = %r"
                              % ((self.history_length - 1) * self.sampling_stride_frames / self.source_fps))
        if abs(self.prediction_length * self.sampling_stride_frames / self.source_fps - self.future_horizon_s) > tol:
            raise ConfigError("future_horizon_s must equal prediction_length * stride / fps = %r"
                              % (self.prediction_length * self.sampling_stride_frames / self.source_fps))
        if self.anchor_stride_frames % self.sampling_stride_frames:
            raise ConfigError("anchor_stride_frames must be a multiple of sampling_stride_frames (shared frame grid)")
        if self.target_source_mode not in TARGET_SOURCE_MODES:
            raise ConfigError("target_source_mode must be one of %s" % (TARGET_SOURCE_MODES,))
        if self.input_source not in INPUT_SOURCES:
            raise ConfigError("input_source must be one of %s" % (INPUT_SOURCES,))
        if self.missing_observation_policy not in MISSING_OBSERVATION_POLICIES:
            raise ConfigError("missing_observation_policy must be one of %s" % (MISSING_OBSERVATION_POLICIES,))
        if self.track_continuity_policy not in TRACK_CONTINUITY_POLICIES:
            raise ConfigError("track_continuity_policy must be one of %s" % (TRACK_CONTINUITY_POLICIES,))
        if self.coordinate_unit != "pixel":
            raise ConfigError("coordinate_unit must be 'pixel' (image space only; no metric frame is authorised)")
        for n in ("target_node_types", "default_context_node_types", "optional_context_node_types", "semantically_uncertain_node_types"):
            v = getattr(self, n)
            if not isinstance(v, (list, tuple)) or len(set(v)) != len(v) or not all(t in KNOWN_NODE_TYPES for t in v):
                raise ConfigError("%s must be a list of distinct node types from %s" % (n, KNOWN_NODE_TYPES))
            object.__setattr__(self, n, tuple(v))
        if self.target_node_types != ("pedestrian",):
            raise ConfigError("P5.3 supports pedestrian prediction targets only")
        if not self.default_context_node_types:
            raise ConfigError("default_context_node_types must not be empty")
        if set(self.default_context_node_types) & set(self.optional_context_node_types):
            raise ConfigError("a node type cannot be both default and optional context")
        if set(self.semantically_uncertain_node_types) & set(self.default_context_node_types):
            raise ConfigError("semantically uncertain node types must not be default context")
        if "pedestrian" not in self.default_context_node_types:
            raise ConfigError("pedestrian must be a default context type (targets are pedestrian nodes of the same graph)")

    # ----- derived quantities -----
    @property
    def sampling_interval_s(self) -> float:
        return self.sampling_stride_frames / self.source_fps

    @property
    def history_offsets(self) -> np.ndarray:
        """Frame offsets relative to t0, oldest first, ending at 0."""
        s, h = self.sampling_stride_frames, self.history_length
        return np.array([-(h - 1 - k) * s for k in range(h)], np.int64)

    @property
    def future_offsets(self) -> np.ndarray:
        s = self.sampling_stride_frames
        return np.array([(m + 1) * s for m in range(self.prediction_length)], np.int64)

    @property
    def dependency_offsets(self) -> Tuple[int, int]:
        """Inclusive [lo, hi] frame offsets relative to t0 of every source frame a window depends on."""
        return int(self.history_offsets[0]) - 1, int(self.future_offsets[-1])

    @property
    def dependency_length_frames(self) -> int:
        lo, hi = self.dependency_offsets
        return hi - lo + 1

    @property
    def minimum_purge_gap_frames(self) -> int:
        """Minimum gap between chronological partitions so no dependency interval spans two partitions:
        (H - 1 + F) * s + 2 (= dependency interval length)."""
        return (self.history_length - 1 + self.prediction_length) * self.sampling_stride_frames + 2

    def is_anchor(self, frame: int) -> bool:
        return frame % self.anchor_stride_frames == self.anchor_phase_frames

    def active_context_types(self, include_optional: bool = False) -> Tuple[str, ...]:
        return self.default_context_node_types + (self.optional_context_node_types if include_optional else ())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WindowConfig":
        if not isinstance(d, Mapping):
            raise ConfigError("configuration must be a JSON object")
        data = {k: v for k, v in d.items() if not k.startswith("_")}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError("unknown configuration keys: %s" % sorted(unknown))
        missing = set(cls.__dataclass_fields__) - set(data)
        if missing:
            raise ConfigError("configuration is missing required keys: %s" % sorted(missing))
        return cls(**data)

    @classmethod
    def load(cls, path) -> "WindowConfig":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError("cannot read configuration %s: %s" % (path, exc))
        return cls.from_dict(d)

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def derived(self) -> dict:
        lo, hi = self.dependency_offsets
        return {"sampling_interval_s": self.sampling_interval_s,
                "history_offsets_frames": self.history_offsets.tolist(),
                "future_offsets_frames": self.future_offsets.tolist(),
                "history_offsets_s": [round(x / self.source_fps, 9) for x in self.history_offsets.tolist()],
                "future_offsets_s": [round(x / self.source_fps, 9) for x in self.future_offsets.tolist()],
                "dependency_interval_offsets_frames": [lo, hi],
                "dependency_interval_length_frames": self.dependency_length_frames,
                "minimum_purge_gap_frames": self.minimum_purge_gap_frames,
                "note": "dependency interval [t0-(H-1)s-1, t0+F*s]; the -1 is the Task 2 one-frame backward difference"}


__all__ = ["WindowConfig", "TARGET_SOURCE_MODES", "IMPLEMENTED_TARGET_SOURCE_MODES", "KNOWN_NODE_TYPES"]
