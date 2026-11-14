"""Stage 4 core: trajectory grouping, coordinate-mode selection, the
timestamp-aware kinematic engine and window preparation.

Three kinds of data stay distinct:

* RAW observations: the Stage 2 records / sidecars / Stage 3 projections. Never modified.
* PROCESSED trajectories: positions on a frame grid with explicit masks
  (``origin`` = observed / imputed / missing) and untouched raw coordinates.
* DERIVED features: displacement, velocity, speed, acceleration, heading, each with
  an explicit validity STATE and a declared unit. An unavailable value is NaN plus a
  state; it is never replaced by zero or by another coordinate mode.

The feature engine only ever reads the current and PAST samples. Whether a
pipeline is causal is decided by what feeds the engine: see
``causal_trajectory.py`` (streaming, past only) and ``offline_trajectory.py``
(interpolation / centred smoothing, may use future samples, labelled OFFLINE).

This module deliberately knows nothing about train/validation/test splits.
"""

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .agent_state import RecordProvenance
from .errors import ConfigError, PedsafetyError
from .geometry import EPS
from .io.csv_adapter import PHYSICAL_UNITS
from .projection import ProjectionRecord, ProjectionStatus

TRAJECTORY_VERSION = "trajectory-0.4.0-stage4"

# ----------------------------------------------------------------- states
ST_OK = "valid"
ST_INSUFFICIENT = "insufficient_samples"
ST_GAP = "gap_before"
ST_IDENTITY = "frame_identity_changed"
ST_NOT_INCREASING = "frame_not_increasing"
ST_POSITION_UNAVAILABLE = "position_unavailable"
ST_MISSING = "missing_position"
ST_NONPOSITIVE_DT = "nonpositive_dt"
ST_NON_FINITE_DT = "non_finite_dt"
ST_DT_TOO_LARGE = "dt_too_large"
ST_IMPUTED_DISALLOWED = "imputed_support_disallowed"
ST_BELOW_MIN_SPEED = "below_min_speed"
ST_NON_FINITE = "non_finite_result"
ST_PREV_VEL = "previous_velocity_unavailable"

ORIGIN_OBSERVED, ORIGIN_IMPUTED, ORIGIN_MISSING = "observed", "imputed", "missing"
VIEW_CAUSAL, VIEW_OFFLINE = "CAUSAL", "OFFLINE_NON_CAUSAL"

HEADING_RANGE = "[-pi, pi)"


class CoordinateMode(str, Enum):
    IMAGE_PIXEL = "image_pixel"
    PROJECTIVE_UNVERIFIED = "projective_unverified"
    METRIC_MANIFEST_ELIGIBLE = "metric_manifest_eligible"
    DECLARED_WORLD_UNVERIFIED = "declared_world_unverified"
    UNVERIFIED_PLANAR = "unverified_planar"


def wrap_angle(a: float) -> float:
    """Wrap an angle in radians to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angular_difference(a: float, b: float) -> float:
    """Signed shortest difference a - b in [-pi, pi)."""
    return wrap_angle(a - b)


# ----------------------------------------------------------------- config
@dataclass(frozen=True)
class SmoothingPolicy:
    """kind: 'none' | 'causal_ema' (past only; alpha in (0,1]) |
    'offline_centered_moving_average' (OFFLINE only; odd window >= 3, needs a
    complete window of usable samples, so edge samples become unavailable)."""

    kind: str = "none"
    alpha: Optional[float] = None
    window: Optional[int] = None

    def __post_init__(self):
        if self.kind not in ("none", "causal_ema", "offline_centered_moving_average"):
            raise ConfigError("smoothing kind must be none/causal_ema/offline_centered_moving_average, got %r" % (self.kind,))
        if self.kind == "causal_ema" and not (isinstance(self.alpha, (int, float)) and not isinstance(self.alpha, bool)
                                              and 0.0 < self.alpha <= 1.0):
            raise ConfigError("causal_ema needs alpha in (0, 1], got %r" % (self.alpha,))
        if self.kind == "offline_centered_moving_average" and not (
                isinstance(self.window, int) and not isinstance(self.window, bool) and self.window >= 3 and self.window % 2 == 1):
            raise ConfigError("offline_centered_moving_average needs an odd integer window >= 3, got %r" % (self.window,))

    @property
    def offline_only(self) -> bool:
        return self.kind == "offline_centered_moving_average"

    def describe(self) -> str:
        if self.kind == "causal_ema":
            return "causal_ema(alpha=%r)" % (self.alpha,)
        if self.kind == "offline_centered_moving_average":
            return "offline_centered_moving_average(window=%d)" % self.window
        return "none"


@dataclass(frozen=True)
class TrajectoryConfig:
    """Every policy is explicit. Defaults are the SAFE choice (no interpolation,
    no smoothing, no imputed support), not tuned values.

    coordinate_mode: which coordinate space features are computed in. A sample
        lacking that space is marked unavailable; modes are never mixed silently.
    frame_step: expected spacing of source frame indices (grid step).
    max_interpolation_gap_frames: OFFLINE only. 0 = never interpolate. A gap of
        N absent frames is filled only if N <= this (and <= the seconds limit).
    allow_imputed_support: if False (default) no feature uses an imputed position.
    min_heading_speed: heading is invalid below this speed (0.0 = only the
        exact zero vector is invalid). Same unit as the velocity.
    dt_rel_tol: relative deviation of per-step dt that counts as irregular
        sampling (default sqrt(eps): floating-point noise only).
    min_valid_observations / short_track_policy: trajectories with fewer valid
        observed positions are flagged ('flag') or dropped with an issue ('exclude').
    """

    coordinate_mode: CoordinateMode = CoordinateMode.IMAGE_PIXEL
    frame_step: int = 1
    max_interpolation_gap_frames: int = 0
    max_interpolation_gap_seconds: Optional[float] = None
    allow_imputed_support: bool = False
    smoothing: SmoothingPolicy = SmoothingPolicy()
    min_heading_speed: float = 0.0
    max_dt_seconds: Optional[float] = None
    dt_rel_tol: float = math.sqrt(EPS)
    min_valid_observations: Optional[int] = None
    short_track_policy: str = "flag"

    def __post_init__(self):
        if not isinstance(self.coordinate_mode, CoordinateMode):
            try:
                object.__setattr__(self, "coordinate_mode", CoordinateMode(self.coordinate_mode))
            except ValueError:
                raise ConfigError("coordinate_mode must be one of %s" % [m.value for m in CoordinateMode])
        for n in ("frame_step",):
            v = getattr(self, n)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s must be an integer >= 1, got %r" % (n, v))
        g = self.max_interpolation_gap_frames
        if isinstance(g, bool) or not isinstance(g, int) or g < 0:
            raise ConfigError("max_interpolation_gap_frames must be an integer >= 0, got %r" % (g,))
        for n in ("max_interpolation_gap_seconds", "max_dt_seconds"):
            v = getattr(self, n)
            if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0):
                raise ConfigError("%s must be a positive finite number or None, got %r" % (n, v))
        if not (isinstance(self.min_heading_speed, (int, float)) and math.isfinite(self.min_heading_speed) and self.min_heading_speed >= 0):
            raise ConfigError("min_heading_speed must be a finite number >= 0")
        if not (isinstance(self.dt_rel_tol, (int, float)) and math.isfinite(self.dt_rel_tol) and self.dt_rel_tol > 0):
            raise ConfigError("dt_rel_tol must be a positive finite number")
        if self.min_valid_observations is not None and (isinstance(self.min_valid_observations, bool) or
                                                       not isinstance(self.min_valid_observations, int) or self.min_valid_observations < 1):
            raise ConfigError("min_valid_observations must be an integer >= 1 or None")
        if self.short_track_policy not in ("flag", "exclude"):
            raise ConfigError("short_track_policy must be 'flag' or 'exclude'")
        if not isinstance(self.smoothing, SmoothingPolicy):
            raise ConfigError("smoothing must be a SmoothingPolicy")


# ------------------------------------------------- coordinate-mode selection
@dataclass(frozen=True)
class PositionInfo:
    x: float
    y: float
    identity: Tuple[Any, ...]            # any change breaks differencing and interpolation
    unit: str
    velocity_unit: str
    acceleration_unit: str
    unit_status: str
    physical: bool
    frame_name: Optional[str]
    calibration_id: Optional[str] = None
    manifest_sha256: Optional[str] = None


_UNIT_STATUS = {
    CoordinateMode.IMAGE_PIXEL: "pixel_image_space",
    CoordinateMode.PROJECTIVE_UNVERIFIED: "projective_unverified_scale",
    CoordinateMode.METRIC_MANIFEST_ELIGIBLE: "metric_by_manifest_checks_only",
    CoordinateMode.DECLARED_WORLD_UNVERIFIED: "declared_world_not_calibration_verified",
    CoordinateMode.UNVERIFIED_PLANAR: "unverified_planar_scale_unknown",
}
_NON_IMAGE_HEADING = ("atan2(vy, vx) in the declared plane axes: angle from +x toward +y; its relation to north or the "
                      "road depends on a frame definition that is not verified here; range " + HEADING_RANGE)
_IMAGE_HEADING = ("atan2(vy, vx) in image axes: angle from +x toward +y; image +y points down so positive is clockwise on "
                  "screen; not a world heading; range " + HEADING_RANGE)


def heading_convention(mode: CoordinateMode) -> str:
    return _IMAGE_HEADING if mode == CoordinateMode.IMAGE_PIXEL else _NON_IMAGE_HEADING


def _finite_pair(x: Any, y: Any) -> bool:
    return all(isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y))


def _info(mode, x, y, ident, unit, physical, frame_name, cal=None, sha=None, label=None):
    label = label or unit
    return PositionInfo(float(x), float(y), ident, label, label + "/s", label + "/s^2", _UNIT_STATUS[mode], physical, frame_name, cal, sha)


def select_position(record: Mapping[str, Any], prov: RecordProvenance, projection: Optional[ProjectionRecord],
                    mode: CoordinateMode) -> Tuple[Optional[PositionInfo], str]:
    """Position for ``record`` in exactly ``mode`` or ``(None, reason)``."""
    cls = record.get("agent_class")
    if mode == CoordinateMode.IMAGE_PIXEL:
        if prov.position_semantics != "image_pixel" or record.get("image_x") is None or record.get("image_y") is None:
            return None, "no_image_position(semantics=%s)" % (prov.position_semantics,)
        if not _finite_pair(record["image_x"], record["image_y"]):
            return None, "non_finite_position"
        return _info(mode, record["image_x"], record["image_y"], ("image", record.get("image_coordinate_system"), cls),
                     "pixel", False, record.get("image_coordinate_system")), "ok"
    if mode == CoordinateMode.PROJECTIVE_UNVERIFIED:
        if projection is None:
            return None, "projection_missing"
        if projection.status != ProjectionStatus.PROJECTED or projection.plane_xy is None:
            return None, "not_projected(%s)" % projection.status.value
        if not _finite_pair(*projection.plane_xy):
            return None, "non_finite_position"
        name = projection.plane_coordinate_system or "unnamed_projective_plane"
        # Always labelled projective, even when the manifest was eligible: this mode never claims a physical unit.
        return _info(mode, projection.plane_xy[0], projection.plane_xy[1],
                     ("projective", name, projection.calibration_id, projection.manifest_sha256,
                      projection.effective_homography_sha256, cls), "projective_plane_units", False, name,
                     projection.calibration_id, projection.manifest_sha256), "ok"
    if mode == CoordinateMode.METRIC_MANIFEST_ELIGIBLE:
        wx, wy, sysname, unit = record.get("world_x"), record.get("world_y"), record.get("world_coordinate_system"), record.get("world_coordinate_unit")
        if record.get("world_position_valid") is not True:
            return None, "world_position_valid_is_false"
        if not _finite_pair(wx, wy) or not (isinstance(sysname, str) and sysname.strip()) or unit not in PHYSICAL_UNITS:
            return None, "incomplete_world_fields"
        p = projection
        if p is None:
            return None, "metric_requires_projection_provenance"
        if not (p.status == ProjectionStatus.PROJECTED and p.promoted_to_canonical and p.manifest_eligible is True
                and p.manifest_sha256 and p.calibration_id):
            return None, "projection_provenance_incomplete"
        if p.plane_coordinate_system != sysname or p.plane_unit != unit or p.plane_xy is None or (wx, wy) != tuple(p.plane_xy):
            return None, "record_and_projection_disagree"
        return _info(mode, wx, wy, ("metric", sysname, unit, p.calibration_id, p.manifest_sha256, p.effective_homography_sha256, cls),
                     unit, True, sysname, p.calibration_id, p.manifest_sha256), "ok"
    if mode == CoordinateMode.DECLARED_WORLD_UNVERIFIED:
        if prov.withheld_position_status != "declared_world_not_calibration_verified" or prov.withheld_position_xy is None:
            return None, "no_declared_world_position"
        unit, name = prov.position_unit, prov.position_coordinate_system
        if unit not in PHYSICAL_UNITS or not (isinstance(name, str) and name.strip()):
            return None, "declared_world_frame_or_unit_incomplete"
        if not _finite_pair(*prov.withheld_position_xy):
            return None, "non_finite_position"
        return _info(mode, prov.withheld_position_xy[0], prov.withheld_position_xy[1], ("declared_world", name, unit, cls),
                     unit, False, name, label="declared:" + unit), "ok"
    if prov.withheld_position_status != "unverified_planar_not_in_record" or prov.withheld_position_xy is None:
        return None, "no_unverified_planar_position"
    if not _finite_pair(*prov.withheld_position_xy):
        return None, "non_finite_position"
    name = prov.position_coordinate_system or "unnamed_planar"
    return _info(mode, prov.withheld_position_xy[0], prov.withheld_position_xy[1], ("unverified_planar", name, cls),
                 "unverified_planar_units", False, name), "ok"


# ------------------------------------------------------------- engine types
@dataclass(frozen=True)
class ObsSample:
    input_index: int
    track_id: str
    frame_index: int
    timestamp: float
    origin: str                       # observed | imputed
    pos: Optional[PositionInfo]       # position fed to the engine (after any smoothing); None = unavailable
    reason: str = "ok"
    raw_xy: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class FeatureSample:
    track_id: str
    input_index: int
    frame_index: int
    timestamp: float
    origin: str
    x: float
    y: float
    xs: float
    ys: float
    dx: float
    dy: float
    dt: float
    vx: float
    vy: float
    speed: float
    ax: float
    ay: float
    heading: float
    position_state: str
    velocity_state: str
    acceleration_state: str
    heading_state: str
    support_imputed: bool


NAN = float("nan")


class FeatureEngine:
    """Per-track, strictly sequential. ``update`` reads only the sample given and
    state built from earlier updates, so it cannot see the future by construction."""

    def __init__(self, cfg: TrajectoryConfig):
        self.cfg = cfg
        self.prev: Optional[Dict[str, Any]] = None
        self.prev_vel: Optional[Dict[str, Any]] = None

    def _reset(self):
        self.prev = None
        self.prev_vel = None

    def update(self, s: ObsSample) -> FeatureSample:
        cfg = self.cfg
        rx, ry = s.raw_xy if s.raw_xy is not None else (NAN, NAN)

        def out(state_v, state_a=None, state_h=None, pos_state="ok", **kw):
            vals = dict(xs=NAN, ys=NAN, dx=NAN, dy=NAN, dt=NAN, vx=NAN, vy=NAN, speed=NAN, ax=NAN, ay=NAN, heading=NAN,
                        support_imputed=False)
            vals.update(kw)
            return FeatureSample(s.track_id, s.input_index, s.frame_index, s.timestamp, s.origin, rx, ry,
                                 position_state=pos_state, velocity_state=state_v,
                                 acceleration_state=state_a or state_v, heading_state=state_h or state_v, **vals)

        if s.pos is None:
            self._reset()
            return out(ST_POSITION_UNAVAILABLE, pos_state=s.reason)
        imputed = s.origin == ORIGIN_IMPUTED
        prev = self.prev
        brk = None
        if prev is None:
            brk = ST_INSUFFICIENT
        elif prev["identity"] != s.pos.identity:
            brk = ST_IDENTITY
        elif s.frame_index <= prev["frame"]:
            brk = ST_NOT_INCREASING
        elif s.frame_index - prev["frame"] != cfg.frame_step:
            brk = ST_GAP
        if brk == ST_NOT_INCREASING:
            return out(brk)                                 # sample rejected: state not advanced
        dt = NAN
        if brk is None:
            dt = s.timestamp - prev["t"]
            if not math.isfinite(dt):
                return out(ST_NON_FINITE_DT)               # sample rejected: state not advanced
            if dt <= 0.0:
                return out(ST_NONPOSITIVE_DT, dt=dt)
            if cfg.max_dt_seconds is not None and dt > cfg.max_dt_seconds:
                brk = ST_DT_TOO_LARGE
        # optional causal smoothing (restarts at every break)
        x, y = s.pos.x, s.pos.y
        if cfg.smoothing.kind == "causal_ema" and brk is None:
            a = cfg.smoothing.alpha
            x, y = a * x + (1.0 - a) * prev["xs"], a * y + (1.0 - a) * prev["ys"]
        new_prev = dict(frame=s.frame_index, t=s.timestamp, xs=x, ys=y, identity=s.pos.identity, imputed=imputed)
        if brk is not None:
            self.prev, self.prev_vel = new_prev, None
            return out(brk, xs=x, ys=y, dt=dt, support_imputed=imputed)
        support_imputed = imputed or prev["imputed"]
        if support_imputed and not cfg.allow_imputed_support:
            self.prev, self.prev_vel = new_prev, None
            return out(ST_IMPUTED_DISALLOWED, xs=x, ys=y, dt=dt, support_imputed=True)
        dx, dy = x - prev["xs"], y - prev["ys"]
        vx, vy = dx / dt, dy / dt
        speed = math.hypot(vx, vy)
        if not all(math.isfinite(v) for v in (dx, dy, vx, vy, speed)):
            self.prev, self.prev_vel = new_prev, None
            return out(ST_NON_FINITE, xs=x, ys=y, dt=dt, support_imputed=support_imputed)
        ax = ay = NAN
        acc_state = ST_OK
        pv = self.prev_vel
        if pv is None or pv["frame"] != prev["frame"]:
            acc_state = ST_PREV_VEL
        else:
            ax, ay = (vx - pv["vx"]) / dt, (vy - pv["vy"]) / dt
            if not (math.isfinite(ax) and math.isfinite(ay)):
                ax = ay = NAN
                acc_state = ST_NON_FINITE
        heading, head_state = NAN, ST_OK
        if speed > 0.0 and speed >= cfg.min_heading_speed:
            heading = wrap_angle(math.atan2(vy, vx))
        else:
            head_state = ST_BELOW_MIN_SPEED
        self.prev = new_prev
        self.prev_vel = dict(frame=s.frame_index, vx=vx, vy=vy)
        return out(ST_OK, acc_state, head_state, xs=x, ys=y, dx=dx, dy=dy, dt=dt, vx=vx, vy=vy, speed=speed, ax=ax, ay=ay,
                   heading=heading, support_imputed=support_imputed)


def missing_row(track_id: str, frame_index: int) -> FeatureSample:
    return FeatureSample(track_id, -1, frame_index, NAN, ORIGIN_MISSING, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
                         ST_MISSING, ST_MISSING, ST_MISSING, ST_MISSING, False)


def rejected_row(track_id: str, input_index: int, frame_index: int, timestamp: float, reason: str) -> FeatureSample:
    """Row for a sample a stream refused (late / duplicate). Carries no features and
    does not touch any engine state."""
    return FeatureSample(track_id, input_index, frame_index, timestamp, ORIGIN_OBSERVED, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
                         NAN, NAN, NAN, reason, "rejected", "rejected", "rejected", False)


# ------------------------------------------------------------- grouping
@dataclass(frozen=True)
class TrajectoryIssue:
    code: str
    track_id: str
    input_index: int
    frame_index: Optional[int]
    message: str


@dataclass
class Grouping:
    groups: Dict[str, List[int]]          # track_id -> accepted input indices, input order
    issues: List[TrajectoryIssue]


def check_parallel(records, provenance, projections) -> None:
    if len(records) != len(provenance) or (projections is not None and len(projections) != len(records)):
        raise ValueError("records, provenance and projections must be parallel (got %d, %d, %s)"
                         % (len(records), len(provenance), None if projections is None else len(projections)))
    for i, (r, p) in enumerate(zip(records, provenance)):
        if r.get("frame_index") != p.frame_index or r.get("source_dataset") != p.source_dataset:
            raise ValueError("provenance[%d] does not describe records[%d]" % (i, i))
        if projections is not None and (projections[i].track_id != r.get("track_id") or projections[i].frame_index != r.get("frame_index")):
            raise ValueError("projections[%d] does not describe records[%d]" % (i, i))


def group_tracks(records: Sequence[Mapping[str, Any]], frame_step: int) -> Grouping:
    """Group by canonical ``track_id`` (dataset + sequence + source track), keeping
    INPUT order. Nothing is sorted. Within a track: duplicate frames quarantine ALL
    members (as in Stage 2), a frame that goes backwards is quarantined, timestamps
    must increase with frames, and frames must lie on the ``frame_step`` grid."""
    raw: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        raw.setdefault(r["track_id"], []).append(i)
    groups: Dict[str, List[int]] = {}
    issues: List[TrajectoryIssue] = []

    def bad(code, tid, i, msg):
        issues.append(TrajectoryIssue(code, tid, i, records[i].get("frame_index"), msg))

    for tid, idxs in raw.items():
        by_frame: Dict[Any, List[int]] = {}
        for i in idxs:
            by_frame.setdefault(records[i].get("frame_index"), []).append(i)
        dup = {i for m in by_frame.values() if len(m) > 1 for i in m}
        for i in sorted(dup):
            bad("duplicate_observation", tid, i, "frame %r occurs %d times in this track; all copies quarantined"
                % (records[i]["frame_index"], len(by_frame[records[i]["frame_index"]])))
        kept: List[int] = []
        last_frame = last_t = None
        origin = None
        for i in idxs:
            r = records[i]
            f, t = r.get("frame_index"), r.get("timestamp")
            if i in dup:
                continue
            if isinstance(f, bool) or not isinstance(f, int) or f < 0 or not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t):
                bad("invalid_time", tid, i, "frame_index/timestamp are not a non-negative int and a finite number")
                continue
            if last_frame is not None and f < last_frame:
                bad("out_of_order_frame", tid, i, "frame %d appears after frame %d; input is not sorted or repaired" % (f, last_frame))
                continue
            if last_t is not None and t <= last_t:
                bad("timestamp_not_monotonic", tid, i, "timestamp %r does not increase with frame_index (previous %r)" % (t, last_t))
                continue
            if origin is None:
                origin = f
            if (f - origin) % frame_step != 0:
                bad("off_grid_frame", tid, i, "frame %d is not on the frame_step=%d grid anchored at frame %d" % (f, frame_step, origin))
                continue
            kept.append(i)
            last_frame, last_t = f, t
        if kept:
            groups[tid] = kept
    return Grouping(groups, issues)


# ------------------------------------------------------------ result types
@dataclass(frozen=True)
class SamplingReport:
    n_rows: int
    n_valid_positions: int
    first_frame: Optional[int]
    last_frame: Optional[int]
    missing_frames: Tuple[int, ...]
    median_dt_per_step: Optional[float]
    min_dt_per_step: Optional[float]
    max_dt_per_step: Optional[float]
    irregular_intervals: Tuple[Tuple[int, int, float], ...]     # (frame_a, frame_b, dt per step)


def sampling_report(frames: Sequence[int], times: Sequence[float], valid: Sequence[bool], cfg: TrajectoryConfig) -> SamplingReport:
    frames, times = list(frames), list(times)
    missing: List[int] = []
    dts: List[Tuple[int, int, float]] = []
    for a in range(1, len(frames)):
        gap = (frames[a] - frames[a - 1]) // cfg.frame_step
        missing.extend(frames[a - 1] + cfg.frame_step * k for k in range(1, gap))
        dts.append((frames[a - 1], frames[a], (times[a] - times[a - 1]) / gap if gap else NAN))
    per = [d for _, _, d in dts if math.isfinite(d)]
    if per:
        med = float(np.median(per))
        irregular = tuple((fa, fb, d) for fa, fb, d in dts if math.isfinite(d) and abs(d - med) > cfg.dt_rel_tol * abs(med))
        lo, hi = min(per), max(per)
    else:
        med = lo = hi = None
        irregular = ()
    return SamplingReport(len(frames), int(sum(bool(v) for v in valid)), frames[0] if frames else None, frames[-1] if frames else None,
                          tuple(missing), med, lo, hi, irregular)


ARRAY_KEYS = ("frame_index", "timestamp", "origin", "input_index", "x", "y", "xs", "ys", "dx", "dy", "dt", "vx", "vy", "speed",
              "ax", "ay", "heading", "position_state", "velocity_state", "acceleration_state", "heading_state", "support_imputed")


def assemble_arrays(rows: Sequence[FeatureSample]) -> Dict[str, np.ndarray]:
    def col(name, dtype):
        return np.array([getattr(r, name) for r in rows], dtype=dtype)
    arrays = {
        "frame_index": col("frame_index", np.int64), "timestamp": col("timestamp", float), "origin": col("origin", "<U8"),
        "input_index": col("input_index", np.int64),
        "position_state": col("position_state", "<U48"), "velocity_state": col("velocity_state", "<U32"),
        "acceleration_state": col("acceleration_state", "<U32"), "heading_state": col("heading_state", "<U32"),
        "support_imputed": col("support_imputed", bool),
    }
    for k in ("x", "y", "xs", "ys", "dx", "dy", "dt", "vx", "vy", "speed", "ax", "ay", "heading"):
        arrays[k] = col(k, float)
    for a in arrays.values():
        a.flags.writeable = False
    return {k: arrays[k] for k in ARRAY_KEYS}


@dataclass(frozen=True)
class TrajectoryFeatures:
    """One track. ``arrays`` are read-only, aligned, and documented by ARRAY_KEYS.
    Raw coordinates are ``x``/``y``; ``xs``/``ys`` are the positions actually
    differentiated (equal to raw unless smoothing is on). Invalid values are NaN
    with a state string, never zero. Subclasses fix ``VIEW``."""

    VIEW: ClassVar[str] = "ABSTRACT"
    track_id: str
    dataset: str
    sequence_id: str
    scene_id: str
    raw_sequence_id: str
    raw_scene_id: str
    raw_track_id: str
    agent_class: str
    coordinate_mode: str
    unit_status: str
    physical: bool
    physical_basis: Optional[str]
    position_unit: str
    velocity_unit: str
    acceleration_unit: str
    heading_convention: str
    frame_name: Optional[str]
    calibration_id: Optional[str]
    manifest_sha256: Optional[str]
    frame_step: int
    smoothing: str
    interpolation: str
    sampling: SamplingReport
    n_observed: int
    n_imputed: int
    n_missing: int
    too_short: bool
    arrays: Dict[str, np.ndarray]

    @property
    def view(self) -> str:
        return self.VIEW

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def __len__(self) -> int:
        return len(self.arrays["frame_index"])


def trajectory_features_equal(a: TrajectoryFeatures, b: TrajectoryFeatures) -> bool:
    """NaN-aware exact equality of metadata and every array."""
    if type(a) is not type(b) or set(a.arrays) != set(b.arrays):
        return False
    meta = [f for f in a.__dataclass_fields__ if f != "arrays"]
    if any(getattr(a, f) != getattr(b, f) for f in meta):
        return False
    return all(np.array_equal(a.arrays[k], b.arrays[k], equal_nan=a.arrays[k].dtype.kind == "f") for k in a.arrays)


@dataclass
class TrajectorySetResult:
    trajectories: List[TrajectoryFeatures]
    issues: List[TrajectoryIssue]
    excluded_tracks: List[str] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)


def trajectory_metadata(tid: str, first_idx: int, records, provenance, infos: Sequence[PositionInfo], cfg: TrajectoryConfig) -> Dict[str, Any]:
    r, p = records[first_idx], provenance[first_idx]
    mode = cfg.coordinate_mode
    ref = infos[0] if infos else None
    if ref is not None:
        unit, vel, acc, phys, name = ref.unit, ref.velocity_unit, ref.acceleration_unit, ref.physical, ref.frame_name
    else:
        unit = vel = acc = "unavailable"
        phys, name = False, None
    return dict(
        track_id=tid, dataset=p.source_dataset, sequence_id=r["sequence_id"], scene_id=r["scene_id"],
        raw_sequence_id=p.raw_sequence_id, raw_scene_id=p.raw_scene_id, raw_track_id=p.raw_track_id,
        agent_class=r["agent_class"], coordinate_mode=mode.value, unit_status=_UNIT_STATUS[mode], physical=phys,
        physical_basis="software_manifest_checks_only" if phys else None, position_unit=unit, velocity_unit=vel,
        acceleration_unit=acc, heading_convention=heading_convention(mode), frame_name=name,
        calibration_id=ref.calibration_id if ref else None, manifest_sha256=ref.manifest_sha256 if ref else None,
        frame_step=cfg.frame_step)


# ------------------------------------------------------------- windows
@dataclass(frozen=True)
class Window:
    """History features/positions (inputs) and target positions (labels) kept
    separate. Source identity is attached. ``history_may_use_future`` is True for
    an OFFLINE view: its history features can depend on samples after the history
    end (interpolation / centred smoothing), i.e. potential leakage."""

    track_id: str
    dataset: str
    raw_sequence_id: str
    raw_scene_id: str
    raw_track_id: str
    view: str
    coordinate_mode: str
    unit_status: str
    position_unit: str
    velocity_unit: str
    obs_range: Tuple[int, int]
    pred_range: Tuple[int, int]
    obs_frames: Tuple[int, ...]
    pred_frames: Tuple[int, ...]
    history: Dict[str, np.ndarray]
    target_xy: np.ndarray
    target_origin: Tuple[str, ...]
    history_may_use_future: bool


@dataclass
class WindowSet:
    windows: List[Window]
    skipped: Dict[str, int]


def prepare_windows(traj: TrajectoryFeatures, obs_len: int, pred_len: int, stride: int = 1, *, allow_offline_view: bool = False,
                    allow_imputed: bool = False, require_valid_history_features: bool = False) -> WindowSet:
    """Slice ONE trajectory into (history, target) windows. Split-agnostic: Stage 5
    must assign whole groups to partitions first and call this per partition.

    History = positions and features at rows [s, s+obs_len); target = RAW positions
    at rows [s+obs_len, s+obs_len+pred_len). Windows need consecutive frames and
    usable positions throughout. Offline views are refused unless explicitly allowed,
    and then flagged ``history_may_use_future``."""
    for n, v in (("obs_len", obs_len), ("pred_len", pred_len), ("stride", stride)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ConfigError("%s must be an integer >= 1, got %r" % (n, v))
    offline = traj.VIEW != VIEW_CAUSAL
    if offline and not allow_offline_view:
        raise PedsafetyError("refusing to window an %s trajectory: its history features may use future samples. "
                             "Use the CAUSAL view, or pass allow_offline_view=True and accept the leakage risk." % traj.VIEW)
    a = traj.arrays
    n = len(traj)
    ok_origin = (ORIGIN_OBSERVED, ORIGIN_IMPUTED) if allow_imputed else (ORIGIN_OBSERVED,)
    windows: List[Window] = []
    skipped = {"not_contiguous": 0, "unusable_history": 0, "unusable_target": 0, "invalid_history_features": 0}
    span = obs_len + pred_len
    for s in range(0, n - span + 1, stride):
        e, t = s + obs_len, s + span
        if not np.all(np.diff(a["frame_index"][s:t]) == traj.frame_step):
            skipped["not_contiguous"] += 1
            continue
        if not (np.isin(a["origin"][s:e], ok_origin).all() and np.isfinite(a["xs"][s:e]).all() and np.isfinite(a["ys"][s:e]).all()):
            skipped["unusable_history"] += 1
            continue
        if not (np.isin(a["origin"][e:t], ok_origin).all() and np.isfinite(a["x"][e:t]).all() and np.isfinite(a["y"][e:t]).all()):
            skipped["unusable_target"] += 1
            continue
        if require_valid_history_features and not (a["velocity_state"][s:e] == ST_OK).all():
            skipped["invalid_history_features"] += 1
            continue
        hist = {k: a[k][s:e].copy() for k in ("frame_index", "timestamp", "origin", "xs", "ys", "vx", "vy", "speed", "ax", "ay", "heading",
                                              "velocity_state", "acceleration_state", "heading_state")}
        windows.append(Window(
            traj.track_id, traj.dataset, traj.raw_sequence_id, traj.raw_scene_id, traj.raw_track_id, traj.VIEW, traj.coordinate_mode,
            traj.unit_status, traj.position_unit, traj.velocity_unit, (s, e), (e, t),
            tuple(int(v) for v in a["frame_index"][s:e]), tuple(int(v) for v in a["frame_index"][e:t]), hist,
            np.column_stack([a["x"][e:t], a["y"][e:t]]), tuple(str(v) for v in a["origin"][e:t]), offline))
    return WindowSet(windows, skipped)
