"""Stage 5: split-aware trajectory windows.

Windows are built from Stage 4 trajectories that were computed INSIDE one
partition (see ``dataset_builder``), so a window can never contain data from
another partition. Each window keeps three things apart:

* HISTORY: positions and kinematic features for the observation steps, with a
  per-step validity mask, an imputation mask and a per-channel availability mask;
* TARGET: the RAW future positions (never features, never smoothed);
* PROVENANCE: dataset, raw sequence / scene / track, group, partition, frame
  indices, timestamps, coordinate space and units, and source row indices.

Windows are defined by absolute frames, not by array rows: step ``i`` sits at
``start + i * frame_stride * frame_step``. Start frames advance by
``window_stride * frame_step`` from the first frame of the trajectory. A candidate
window that violates the configured gap / validity policy is rejected and counted
under a named reason; nothing is repaired or silently shortened.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import ConfigError
from .trajectory import (
    ORIGIN_IMPUTED, ORIGIN_OBSERVED, ST_OK, VIEW_CAUSAL, TrajectoryConfig, TrajectoryFeatures,
)

WINDOW_VERSION = "windows-0.5.0-stage5"
HISTORY_CHANNELS = ("xs", "ys", "vx", "vy", "speed", "ax", "ay", "heading")
_CHANNEL_STATE = {"vx": "velocity_state", "vy": "velocity_state", "speed": "velocity_state", "ax": "acceleration_state",
                  "ay": "acceleration_state", "heading": "heading_state"}
SKIP_REASONS = ("unusable_target", "last_history_step_unusable", "insufficient_history", "gap_in_native_frames",
                "mixed_coordinate_identity", "required_feature_unavailable")


@dataclass(frozen=True)
class WindowConfig:
    """obs_len / pred_len: number of history / target STEPS.
    frame_stride: native-frame spacing between steps (features stay native-rate).
    window_stride: spacing between window starts, in native grid steps.
    min_valid_history_steps: usable history steps required (default obs_len = none
        may be missing). Fewer is allowed only as LEFT padding when
        ``require_contiguous_native_frames`` is True.
    require_contiguous_native_frames: every native frame from the first usable
        history step to the last target step must exist and be usable.
    allow_imputed: imputed rows (OFFLINE views) may count as usable.
    allow_offline_view: needed to build windows from an OFFLINE view.
    require_last_step_features: channels that must be available at the last history step."""

    obs_len: int
    pred_len: int
    frame_stride: int = 1
    window_stride: int = 1
    min_valid_history_steps: Optional[int] = None
    require_contiguous_native_frames: bool = True
    allow_imputed: bool = False
    allow_offline_view: bool = False
    require_last_step_features: Tuple[str, ...] = ()

    def __post_init__(self):
        for n in ("obs_len", "pred_len", "frame_stride", "window_stride"):
            v = getattr(self, n)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s must be an integer >= 1, got %r" % (n, v))
        m = self.min_valid_history_steps
        if m is not None and (isinstance(m, bool) or not isinstance(m, int) or not 1 <= m <= self.obs_len):
            raise ConfigError("min_valid_history_steps must be an integer in [1, obs_len=%d], got %r" % (self.obs_len, m))
        bad = [c for c in self.require_last_step_features if c not in HISTORY_CHANNELS]
        if bad:
            raise ConfigError("unknown require_last_step_features %s; allowed: %s" % (bad, HISTORY_CHANNELS))
        object.__setattr__(self, "require_last_step_features", tuple(self.require_last_step_features))

    @property
    def min_valid(self) -> int:
        return self.obs_len if self.min_valid_history_steps is None else self.min_valid_history_steps

    def to_dict(self) -> Dict[str, Any]:
        return {"obs_len": self.obs_len, "pred_len": self.pred_len, "frame_stride": self.frame_stride, "window_stride": self.window_stride,
                "min_valid_history_steps": self.min_valid, "require_contiguous_native_frames": self.require_contiguous_native_frames,
                "allow_imputed": self.allow_imputed, "allow_offline_view": self.allow_offline_view,
                "require_last_step_features": list(self.require_last_step_features)}


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.array(a)
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class SplitWindow:
    window_id: str
    partition: str
    group_id: str
    group_unit: str
    # provenance
    dataset: str
    raw_sequence_id: str
    raw_scene_id: str
    raw_track_id: str
    track_id: str
    agent_class: str
    view: str
    history_may_use_future: bool
    # coordinate space
    coordinate_mode: str
    unit_status: str
    physical: bool
    physical_basis: Optional[str]
    frame_name: Optional[str]
    calibration_id: Optional[str]
    manifest_sha256: Optional[str]
    position_unit: str
    velocity_unit: str
    acceleration_unit: str
    heading_convention: str
    # shape / sampling
    obs_len: int
    pred_len: int
    frame_stride: int
    frame_step: int
    # history
    history: Mapping[str, np.ndarray]
    history_available: Mapping[str, np.ndarray]
    history_valid: np.ndarray
    history_origin: np.ndarray
    history_imputed: np.ndarray
    history_frame_index: np.ndarray
    history_timestamp: np.ndarray
    history_source_index: np.ndarray
    # target (raw positions only)
    target_xy: np.ndarray
    target_origin: np.ndarray
    target_imputed: np.ndarray
    target_frame_index: np.ndarray
    target_timestamp: np.ndarray
    target_source_index: np.ndarray

    def channel_units(self) -> Dict[str, str]:
        return {"xs": self.position_unit, "ys": self.position_unit, "vx": self.velocity_unit, "vy": self.velocity_unit,
                "speed": self.velocity_unit, "ax": self.acceleration_unit, "ay": self.acceleration_unit, "heading": "rad"}

    def norm_key(self) -> str:
        """Statistics must never be shared across datasets, coordinate spaces, units or calibrations."""
        return "|".join(str(x) for x in (self.dataset, self.coordinate_mode, self.unit_status, self.position_unit, self.frame_name,
                                          self.calibration_id, self.manifest_sha256))


def _usable_origin(origin: str, allow_imputed: bool) -> bool:
    return origin == ORIGIN_OBSERVED or (allow_imputed and origin == ORIGIN_IMPUTED)


def generate_windows(trajectories: Sequence[TrajectoryFeatures], *, partition: str, group_of_track: Mapping[str, str], group_unit: str,
                     wcfg: WindowConfig, identities: Mapping[int, Tuple[Any, ...]], global_index: Sequence[int]
                     ) -> Tuple[List[SplitWindow], Dict[str, int]]:
    """Windows for trajectories that were all computed inside ``partition``.

    ``identities`` maps a PARTITION-LOCAL input index to the coordinate identity of
    that observation (frame, unit, calibration, class); ``global_index`` maps local
    index -> index in the caller's original record list."""
    skipped = {r: 0 for r in SKIP_REASONS}
    out: List[SplitWindow] = []
    for traj in trajectories:
        offline = traj.VIEW != VIEW_CAUSAL
        if offline and not wcfg.allow_offline_view:
            raise ConfigError("trajectory %s is an %s view: its history features may use future samples. Set "
                              "WindowConfig.allow_offline_view=True to accept that leakage risk." % (traj.track_id, traj.VIEW))
        a = traj.arrays
        n = len(a["frame_index"])
        if n == 0:
            continue
        step, fs = traj.frame_step, wcfg.frame_stride
        span = fs * step
        n_steps = wcfg.obs_len + wcfg.pred_len
        frames = a["frame_index"]
        row_of = {int(f): k for k, f in enumerate(frames)}
        first, last = int(frames[0]), int(frames[-1])
        f0 = first
        while f0 + (n_steps - 1) * span <= last:
            start = f0
            f0 += wcfg.window_stride * step
            wf = [start + i * span for i in range(n_steps)]
            rows = [row_of.get(f) for f in wf]

            def hist_ok(r):
                return r is not None and _usable_origin(str(a["origin"][r]), wcfg.allow_imputed) and bool(np.isfinite(a["xs"][r]) and np.isfinite(a["ys"][r]))

            def tgt_ok(r):
                return r is not None and _usable_origin(str(a["origin"][r]), wcfg.allow_imputed) and bool(np.isfinite(a["x"][r]) and np.isfinite(a["y"][r]))

            h_rows, t_rows = rows[:wcfg.obs_len], rows[wcfg.obs_len:]
            if not all(tgt_ok(r) for r in t_rows):
                skipped["unusable_target"] += 1
                continue
            hmask = np.array([hist_ok(r) for r in h_rows], dtype=bool)
            if not hmask[-1]:
                skipped["last_history_step_unusable"] += 1
                continue
            if int(hmask.sum()) < wcfg.min_valid:
                skipped["insufficient_history"] += 1
                continue
            first_use = int(np.argmax(hmask))
            if wcfg.require_contiguous_native_frames:
                native = range(wf[first_use], wf[-1] + 1, step)
                if not all(f in row_of and tgt_ok(row_of[f]) for f in native):
                    skipped["gap_in_native_frames"] += 1
                    continue
                if not hmask[first_use:].all():
                    skipped["gap_in_native_frames"] += 1
                    continue
                used_rows = [row_of[f] for f in native]
            else:
                used_rows = [r for r in rows if r is not None]
            idents = set()
            for r in used_rows:
                li = int(a["input_index"][r])
                if li >= 0:
                    idents.add(identities.get(li))
            if len(idents) > 1 or (idents and None in idents):
                skipped["mixed_coordinate_identity"] += 1
                continue
            last_h = h_rows[-1]
            if any(not _channel_ok(a, last_h, c) for c in wcfg.require_last_step_features):
                skipped["required_feature_unavailable"] += 1
                continue
            out.append(_build(traj, a, rows, hmask, wcfg, partition, group_of_track[traj.track_id], group_unit, offline, start, global_index))
    return out, skipped


def _channel_ok(a: Mapping[str, np.ndarray], r: int, ch: str) -> bool:
    if ch in ("xs", "ys"):
        return bool(np.isfinite(a[ch][r]))
    return str(a[_CHANNEL_STATE[ch]][r]) == ST_OK and bool(np.isfinite(a[ch][r]))


def _build(traj, a, rows, hmask, wcfg, partition, group_id, group_unit, offline, start, global_index) -> SplitWindow:
    o, p = wcfg.obs_len, wcfg.pred_len
    h_rows, t_rows = rows[:o], rows[o:]
    hist: Dict[str, np.ndarray] = {}
    avail: Dict[str, np.ndarray] = {}
    for ch in HISTORY_CHANNELS:
        ok = np.array([bool(hmask[i]) and h_rows[i] is not None and _channel_ok(a, h_rows[i], ch) for i in range(o)], dtype=bool)
        vals = np.array([float(a[ch][r]) if (ok[i] and r is not None) else np.nan for i, r in enumerate(h_rows)])
        hist[ch], avail[ch] = _ro(vals), _ro(ok)

    def pick(name, rs, fill, dtype):
        return _ro(np.array([a[name][r] if r is not None else fill for r in rs], dtype=dtype))

    def gsrc(rs):
        out = []
        for r in rs:
            li = int(a["input_index"][r]) if r is not None else -1
            out.append(int(global_index[li]) if li >= 0 else -1)
        return _ro(np.array(out, dtype=np.int64))

    h_origin = _ro(np.array([str(a["origin"][r]) if (r is not None and hmask[i]) else "masked" for i, r in enumerate(h_rows)], dtype="<U8"))
    t_origin = _ro(np.array([str(a["origin"][r]) for r in t_rows], dtype="<U8"))
    h_frames = _ro(np.array([start + i * wcfg.frame_stride * traj.frame_step for i in range(o)], dtype=np.int64))
    t_frames = _ro(np.array([start + (o + i) * wcfg.frame_stride * traj.frame_step for i in range(p)], dtype=np.int64))
    h_ts = _ro(np.array([float(a["timestamp"][r]) if (r is not None and hmask[i]) else np.nan for i, r in enumerate(h_rows)]))
    wid = "%s|f%d|o%d|p%d|fs%d" % (traj.track_id, start, o, p, wcfg.frame_stride)
    return SplitWindow(
        window_id=wid, partition=partition, group_id=group_id, group_unit=group_unit, dataset=traj.dataset,
        raw_sequence_id=traj.raw_sequence_id, raw_scene_id=traj.raw_scene_id, raw_track_id=traj.raw_track_id, track_id=traj.track_id,
        agent_class=traj.agent_class, view=traj.VIEW, history_may_use_future=offline, coordinate_mode=traj.coordinate_mode,
        unit_status=traj.unit_status, physical=traj.physical, physical_basis=traj.physical_basis, frame_name=traj.frame_name,
        calibration_id=traj.calibration_id, manifest_sha256=traj.manifest_sha256, position_unit=traj.position_unit,
        velocity_unit=traj.velocity_unit, acceleration_unit=traj.acceleration_unit, heading_convention=traj.heading_convention,
        obs_len=o, pred_len=p, frame_stride=wcfg.frame_stride, frame_step=traj.frame_step,
        history=hist, history_available=avail, history_valid=_ro(hmask), history_origin=h_origin,
        history_imputed=_ro(h_origin == ORIGIN_IMPUTED), history_frame_index=h_frames, history_timestamp=h_ts, history_source_index=gsrc(h_rows),
        target_xy=_ro(np.column_stack([pick("x", t_rows, np.nan, float), pick("y", t_rows, np.nan, float)])), target_origin=t_origin,
        target_imputed=_ro(t_origin == ORIGIN_IMPUTED), target_frame_index=t_frames,
        target_timestamp=pick("timestamp", t_rows, np.nan, float), target_source_index=gsrc(t_rows))
