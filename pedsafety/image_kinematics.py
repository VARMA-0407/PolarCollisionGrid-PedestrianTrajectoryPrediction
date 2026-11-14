"""Phase 4 Task 2: image-space kinematics for Phase 3 agent-state records (CAUSAL path).

This module is a thin layer over the Phase 2 Stage 4 engine; it does not re-implement it:

* ``trajectory.FeatureEngine`` computes the timestamp-aware backward-difference velocity, acceleration and image
  heading, with its validity states (``gap_before``, ``nonpositive_dt``, ``below_min_speed`` ...);
* ``trajectory.group_tracks`` applies the Stage 2 duplicate / out-of-order / non-monotonic quarantine rules;
* ``trajectory.SmoothingPolicy('causal_ema')`` is the existing fixed-alpha causal EMA.

What is added here:

1. ``validate_phase3_record``: the Phase 4 coordinate contract for one Phase 3 ``agent_states.jsonl`` record. A record
   that breaks it is never repaired: it keeps its row, its position is marked unavailable with a reason, and the engine
   restarts after it. Rows that cannot even be placed in a track (no track id / frame / timestamp) are quarantined.
2. A position source that reads ``image_x``/``image_y`` (the footpoint) straight from the record. No provenance sidecar
   is reconstructed: the Phase 3 output has none, and the fields the Stage 2 sidecar would need are not invented.
3. ``TimeAwareEMA``: alpha_t = 1 - exp(-dt / tau), a causal position filter whose smoothing does not depend on the
   sampling rate (the existing EMA uses one alpha per SAMPLE).
4. ``CausalImageKinematics``: streaming per-track driver. ``update`` receives one record and can only use it and state
   built from earlier records, so a future observation cannot influence an earlier output by construction.
5. Sidecar rows (``docs/phase4/schemas/image_kinematics_sidecar.schema.json``). Canonical agent-state records are
   neither modified nor extended.

Image space only: positions in pixels, velocity in pixel/s, acceleration in pixel/s^2, heading in image-plane radians.
None of these is a physical quantity; on a moving camera image motion also contains ego-motion.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import ConfigError
from .trajectory import (
    HEADING_RANGE, ORIGIN_OBSERVED, ST_OK, VIEW_CAUSAL, CoordinateMode, FeatureEngine, FeatureSample, ObsSample, PositionInfo,
    SmoothingPolicy, TrajectoryConfig, TrajectoryIssue, group_tracks, heading_convention,
)

IMAGE_KINEMATICS_VERSION = "image-kinematics-0.1.0-phase4-task2"
SIDECAR_SCHEMA_VERSION = "image-kinematics-sidecar-1.0.0"
SIDECAR_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "phase4" / "schemas" / "image_kinematics_sidecar.schema.json"

POSITION_UNIT, VELOCITY_UNIT, ACCELERATION_UNIT, HEADING_UNIT, TIME_UNIT = "pixel", "pixel/s", "pixel/s^2", "rad", "s"
IMAGE_HEADING_CONVENTION = heading_convention(CoordinateMode.IMAGE_PIXEL)          # full sentence (Stage 4), used in manifests
HEADING_CONVENTION_CODE = "image_atan2_vy_vx_from_+u_toward_+v_clockwise_on_screen_range_[-pi,pi)"   # per-row code (sidecar schema const)
CAUSAL_VELOCITY_METHOD = "backward_difference_actual_dt"
CAUSAL_ACCELERATION_METHOD = "backward_difference_of_backward_velocity_actual_dt"
VALID = "valid"   # the sidecar's word for the Stage 4 state ST_OK (identical value)
assert ST_OK == VALID

SMOOTHING_KINDS = ("none", "causal_ema", "time_aware_ema")


# ------------------------------------------------------------------- config
def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class SmoothingConfig:
    """kind 'none' (default), 'causal_ema' (existing Stage 4 fixed-alpha EMA) or 'time_aware_ema' (tau_s seconds)."""

    kind: str = "none"
    alpha: Optional[float] = None
    tau_s: Optional[float] = None

    def __post_init__(self):
        if self.kind not in SMOOTHING_KINDS:
            raise ConfigError("smoothing.kind must be one of %s, got %r" % (SMOOTHING_KINDS, self.kind))
        if self.kind == "causal_ema":
            SmoothingPolicy("causal_ema", alpha=self.alpha)               # reuse the existing validation
            if self.tau_s is not None:
                raise ConfigError("smoothing.tau_s is only used by time_aware_ema")
        elif self.kind == "time_aware_ema":
            if not (_num(self.tau_s) and self.tau_s > 0):
                raise ConfigError("time_aware_ema needs a positive finite tau_s (seconds), got %r" % (self.tau_s,))
            if self.alpha is not None:
                raise ConfigError("smoothing.alpha is only used by causal_ema")
        elif self.alpha is not None or self.tau_s is not None:
            raise ConfigError("smoothing kind 'none' takes no alpha or tau_s")

    def describe(self) -> str:
        if self.kind == "causal_ema":
            return "causal_ema(alpha=%r, per-sample, not time-aware)" % (self.alpha,)
        if self.kind == "time_aware_ema":
            return "time_aware_ema(tau_s=%r, alpha_t = 1 - exp(-dt/tau_s))" % (self.tau_s,)
        return "none"


def _num_list(v: Any, name: str, positive: bool = False, integer: bool = False) -> Tuple:
    if not isinstance(v, list) or not v:
        raise ConfigError("%s must be a non-empty list" % name)
    for x in v:
        ok = _int(x) if integer else _num(x)
        if not ok or (positive and x <= 0) or (not positive and x < 0):
            raise ConfigError("%s contains an invalid value %r" % (name, x))
    if list(v) != sorted(set(v)):
        raise ConfigError("%s must be strictly increasing" % name)
    return tuple(v)


@dataclass(frozen=True)
class ContinuityConfig:
    short_track_thresholds: Tuple[int, ...] = (1, 2, 3, 5, 10, 30)
    jump_ratio_thresholds: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    box_height_ratio_thresholds: Tuple[float, ...] = (1.5, 2.0, 3.0)
    fragment_max_gap_frames: Tuple[int, ...] = (1, 5, 15, 30)
    fragment_max_distance_box_heights: float = 1.0
    border_tolerance_px: float = 1.0
    max_listed_items: int = 50

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ContinuityConfig":
        if not isinstance(d, Mapping):
            raise ConfigError("continuity must be an object")
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError("unknown continuity keys: %s" % sorted(unknown))
        kw = dict(d)
        for k, pos, integer in (("short_track_thresholds", True, True), ("jump_ratio_thresholds", True, False),
                                ("box_height_ratio_thresholds", True, False), ("fragment_max_gap_frames", True, True)):
            if k in kw:
                kw[k] = _num_list(kw[k], "continuity." + k, positive=pos, integer=integer)
        for k in ("fragment_max_distance_box_heights", "border_tolerance_px"):
            if k in kw and not (_num(kw[k]) and kw[k] >= 0):
                raise ConfigError("continuity.%s must be a finite number >= 0" % k)
        if "max_listed_items" in kw and not (_int(kw["max_listed_items"]) and kw["max_listed_items"] >= 0):
            raise ConfigError("continuity.max_listed_items must be an integer >= 0")
        if any(r <= 1.0 for r in kw.get("box_height_ratio_thresholds", (1.5,))):
            raise ConfigError("continuity.box_height_ratio_thresholds must all be > 1")
        return cls(**kw)


@dataclass(frozen=True)
class ImageKinematicsConfig:
    """Every policy is explicit. ``max_derivative_gap_frames`` is fixed at 0 (no derivative across any missing frame):
    bridging is refused until the real gap distribution has been reviewed."""

    config_version: str
    expected_image_coordinate_system: str
    expected_image_width: int
    expected_image_height: int
    footpoint_tolerance_px: float = 1e-6
    bbox_bounds_tolerance_px: float = 1.0
    frame_step: int = 1
    max_derivative_gap_frames: int = 0
    max_dt_seconds: Optional[float] = None
    min_heading_speed_px_per_s: float = 0.0
    smoothing: SmoothingConfig = SmoothingConfig()
    offline_central_difference: bool = True
    heading_speed_sensitivity_px_per_s: Tuple[float, ...] = (0.0, 10.0, 30.0, 60.0, 120.0)
    continuity: ContinuityConfig = ContinuityConfig()

    def __post_init__(self):
        if not (isinstance(self.config_version, str) and self.config_version.strip()):
            raise ConfigError("config_version must be a non-empty string")
        if not (isinstance(self.expected_image_coordinate_system, str) and self.expected_image_coordinate_system.strip()):
            raise ConfigError("expected_image_coordinate_system must be a non-empty string")
        for n in ("expected_image_width", "expected_image_height", "frame_step"):
            v = getattr(self, n)
            if not (_int(v) and v >= 1):
                raise ConfigError("%s must be an integer >= 1, got %r" % (n, v))
        for n in ("footpoint_tolerance_px", "bbox_bounds_tolerance_px", "min_heading_speed_px_per_s"):
            v = getattr(self, n)
            if not (_num(v) and v >= 0):
                raise ConfigError("%s must be a finite number >= 0, got %r" % (n, v))
        if not _int(self.max_derivative_gap_frames) or self.max_derivative_gap_frames < 0:
            raise ConfigError("max_derivative_gap_frames must be an integer >= 0")
        if self.max_derivative_gap_frames != 0:
            raise ConfigError("max_derivative_gap_frames must be 0: derivatives across missing frames are not authorized until the "
                              "real PIE gap distribution has been reviewed (Phase 4 Task 2 approval condition 4)")
        if self.max_dt_seconds is not None and not (_num(self.max_dt_seconds) and self.max_dt_seconds > 0):
            raise ConfigError("max_dt_seconds must be a positive finite number or null")
        if not isinstance(self.smoothing, SmoothingConfig):
            raise ConfigError("smoothing must be a SmoothingConfig")
        if not isinstance(self.offline_central_difference, bool):
            raise ConfigError("offline_central_difference must be true or false")
        if not isinstance(self.continuity, ContinuityConfig):
            raise ConfigError("continuity must be a ContinuityConfig")

    # -- (de)serialisation -------------------------------------------------
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ImageKinematicsConfig":
        if not isinstance(d, Mapping):
            raise ConfigError("configuration must be a JSON object")
        data = {k: v for k, v in d.items() if not k.startswith("_")}       # "_notes" etc. are documentation only
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError("unknown configuration keys: %s" % sorted(unknown))
        kw = dict(data)
        if "smoothing" in kw:
            s = kw["smoothing"]
            if not isinstance(s, Mapping) or set(s) - {"kind", "alpha", "tau_s"}:
                raise ConfigError("smoothing must be an object with keys kind/alpha/tau_s")
            kw["smoothing"] = SmoothingConfig(**s)
        if "continuity" in kw:
            kw["continuity"] = ContinuityConfig.from_dict(kw["continuity"])
        if "heading_speed_sensitivity_px_per_s" in kw:
            kw["heading_speed_sensitivity_px_per_s"] = _num_list(kw["heading_speed_sensitivity_px_per_s"], "heading_speed_sensitivity_px_per_s")
        try:
            return cls(**kw)
        except TypeError as exc:
            raise ConfigError("configuration is missing required keys: %s" % exc)

    @classmethod
    def load(cls, path) -> "ImageKinematicsConfig":
        try:
            raw = Path(path).read_text(encoding="utf-8")
            d = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise ConfigError("cannot read configuration %s: %s" % (path, exc))
        return cls.from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("heading_speed_sensitivity_px_per_s",):
            d[k] = list(d[k])
        d["continuity"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d["continuity"].items()}
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def trajectory_config(self, smoothing: Optional[SmoothingPolicy] = None) -> TrajectoryConfig:
        """The Stage 4 engine configuration. Interpolation and imputed support stay at their safe defaults (off)."""
        return TrajectoryConfig(coordinate_mode=CoordinateMode.IMAGE_PIXEL, frame_step=self.frame_step,
                                max_interpolation_gap_frames=0, allow_imputed_support=False,
                                smoothing=smoothing or SmoothingPolicy("none"), min_heading_speed=float(self.min_heading_speed_px_per_s),
                                max_dt_seconds=self.max_dt_seconds)


# ------------------------------------------------------- record contract
REQUIRED_PLACEMENT = ("track_id", "sequence_id", "frame_index", "timestamp")

# position-state reason codes (row kept, position unavailable, engine restarted)
R_COORD_SYSTEM = "coordinate_system_mismatch"
R_UNIT = "image_unit_not_pixel"
R_WORLD = "world_fields_not_null_or_world_valid"
R_OBS_INVALID = "observation_valid_not_true"
R_INTERPOLATED = "interpolated_input_not_supported"
R_BBOX = "invalid_bbox"
R_BBOX_BOUNDS = "bbox_outside_image"
R_FOOTPOINT = "footpoint_missing_or_not_bottom_centre"
R_IMAGE_XY = "image_xy_missing_or_not_footpoint"


@dataclass(frozen=True)
class RecordCheck:
    placeable: bool                   # False -> quarantined (cannot be placed in a track)
    position: Optional[Tuple[float, float]]
    reason: str                       # 'ok' or a reason code
    touches_border: Optional[bool]
    box_height: Optional[float]


def _finite_seq(v: Any, n: int) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == n and all(_num(x) for x in v)


def placement_problem(r: Mapping[str, Any]) -> Optional[str]:
    """Why a record cannot be placed in a track at all (None if it can)."""
    if not isinstance(r, Mapping):
        return "record_not_an_object"
    for k in ("track_id", "sequence_id"):
        if not (isinstance(r.get(k), str) and r[k].strip()):
            return "missing_or_empty_%s" % k
    f = r.get("frame_index")
    if not (_int(f) and f >= 0):
        return "frame_index_not_a_non_negative_integer"
    if not _num(r.get("timestamp")):
        return "timestamp_missing_or_non_finite"
    return None


def validate_phase3_record(r: Mapping[str, Any], cfg: ImageKinematicsConfig) -> RecordCheck:
    """Apply the Phase 4 coordinate contract to one record. Nothing is repaired."""
    why = placement_problem(r)
    if why is not None:
        return RecordCheck(False, None, why, None, None)
    if r.get("image_coordinate_system") != cfg.expected_image_coordinate_system:
        return RecordCheck(True, None, R_COORD_SYSTEM, None, None)
    if r.get("image_coordinate_unit") != POSITION_UNIT:
        return RecordCheck(True, None, R_UNIT, None, None)
    if r.get("world_position_valid") is not False or r.get("world_x") is not None or r.get("world_y") is not None:
        return RecordCheck(True, None, R_WORLD, None, None)
    if r.get("observation_valid") is not True:
        return RecordCheck(True, None, R_OBS_INVALID, None, None)
    if r.get("interpolation_flag") is not False:
        return RecordCheck(True, None, R_INTERPOLATED, None, None)
    b = r.get("bbox_xyxy")
    if not _finite_seq(b, 4) or not (b[0] < b[2] and b[1] < b[3]):
        return RecordCheck(True, None, R_BBOX, None, None)
    tol, W, H = cfg.bbox_bounds_tolerance_px, cfg.expected_image_width, cfg.expected_image_height
    if b[0] < -tol or b[1] < -tol or b[2] > W + tol or b[3] > H + tol:
        return RecordCheck(True, None, R_BBOX_BOUNDS, None, None)
    ftol = cfg.footpoint_tolerance_px
    fp = r.get("footpoint")
    bc = ((b[0] + b[2]) / 2.0, b[3])
    if not _finite_seq(fp, 2) or abs(fp[0] - bc[0]) > ftol or abs(fp[1] - bc[1]) > ftol:
        return RecordCheck(True, None, R_FOOTPOINT, None, None)
    ix, iy = r.get("image_x"), r.get("image_y")
    if not (_num(ix) and _num(iy)) or abs(ix - fp[0]) > ftol or abs(iy - fp[1]) > ftol:
        return RecordCheck(True, None, R_IMAGE_XY, None, None)
    bt = cfg.continuity.border_tolerance_px
    touches = b[0] <= bt or b[1] <= bt or b[2] >= W - bt or b[3] >= H - bt
    return RecordCheck(True, (float(ix), float(iy)), "ok", bool(touches), float(b[3] - b[1]))


def position_info(r: Mapping[str, Any], xy: Tuple[float, float]) -> PositionInfo:
    """Stage 4 ``PositionInfo`` for an image-pixel footpoint. The identity mirrors ``select_position`` for image mode:
    a change of raster or agent class breaks differencing."""
    return PositionInfo(float(xy[0]), float(xy[1]), ("image", r.get("image_coordinate_system"), r.get("agent_class")), POSITION_UNIT,
                        VELOCITY_UNIT, ACCELERATION_UNIT, "pixel_image_space", False, r.get("image_coordinate_system"))


# --------------------------------------------------------- causal filter
class TimeAwareEMA:
    """Causal per-track position filter with alpha_t = 1 - exp(-dt / tau).

    It restarts (output = measurement) at the same places the Stage 4 engine breaks differencing: first sample, a frame
    step other than ``frame_step``, non-positive / non-finite dt, dt above ``max_dt``, a change of identity, or an
    unavailable position. It reads only the current sample and its own past state."""

    def __init__(self, tau_s: float, frame_step: int = 1, max_dt: Optional[float] = None):
        if not (_num(tau_s) and tau_s > 0):
            raise ConfigError("tau_s must be positive")
        self.tau, self.step, self.max_dt = float(tau_s), frame_step, max_dt
        self.state: Optional[Dict[str, Any]] = None
        self.last_alpha: Optional[float] = None

    def reset(self) -> None:
        self.state = None
        self.last_alpha = None

    def update(self, frame: int, t: float, xy: Optional[Tuple[float, float]], identity: Any) -> Optional[Tuple[float, float]]:
        if xy is None:
            self.reset()
            return None
        s = self.state
        restart = s is None or s["identity"] != identity or frame - s["frame"] != self.step
        if not restart:
            dt = t - s["t"]
            restart = not math.isfinite(dt) or dt <= 0 or (self.max_dt is not None and dt > self.max_dt)
        if restart:
            fx, fy = float(xy[0]), float(xy[1])
            self.last_alpha = None
        else:
            a = 1.0 - math.exp(-(t - s["t"]) / self.tau)
            fx, fy = s["fx"] + a * (xy[0] - s["fx"]), s["fy"] + a * (xy[1] - s["fy"])
            self.last_alpha = a
        self.state = dict(frame=frame, t=t, fx=fx, fy=fy, identity=identity)
        return fx, fy


# --------------------------------------------------------- causal driver
def _f(v: float) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


@dataclass
class CausalStep:
    """Everything the causal path derived for one accepted record."""

    raw: FeatureSample
    filtered: Optional[FeatureSample]
    frame_gap_before: Optional[int]


class CausalImageKinematics:
    """Streaming causal driver. Records must arrive in time order per track; a late, duplicate or non-increasing record is
    rejected and leaves every state untouched (a stream cannot sort or retract). Raw and filtered estimates come from two
    independent Stage 4 engines fed with the same samples, so the raw estimate never depends on the filter."""

    VIEW = VIEW_CAUSAL

    def __init__(self, cfg: ImageKinematicsConfig):
        self.cfg = cfg
        self.raw_cfg = cfg.trajectory_config()
        sm = cfg.smoothing
        self.filtered_cfg = None
        if sm.kind == "causal_ema":
            self.filtered_cfg = cfg.trajectory_config(SmoothingPolicy("causal_ema", alpha=sm.alpha))
        elif sm.kind == "time_aware_ema":
            self.filtered_cfg = cfg.trajectory_config()                   # positions are pre-filtered; engine smoothing off
        self._raw: Dict[str, FeatureEngine] = {}
        self._flt: Dict[str, FeatureEngine] = {}
        self._ema: Dict[str, TimeAwareEMA] = {}
        self._last: Dict[str, Tuple[int, float]] = {}
        self.issues: List[TrajectoryIssue] = []

    def update(self, record: Mapping[str, Any], check: RecordCheck, input_index: int) -> Optional[CausalStep]:
        """Process one record. Returns None when the record is rejected as late/duplicate."""
        tid, f, t = record["track_id"], record["frame_index"], float(record["timestamp"])
        last = self._last.get(tid)
        if last is not None and (f <= last[0] or t <= last[1]):
            code = "duplicate_or_out_of_order" if f <= last[0] else "timestamp_not_monotonic"
            self.issues.append(TrajectoryIssue(code, tid, input_index, f, "frame %d / t=%r arrives after frame %d / t=%r" % (f, t, last[0], last[1])))
            return None
        gap = None if last is None else (f - last[0]) // self.cfg.frame_step - 1
        pos = position_info(record, check.position) if check.position is not None else None
        raw_xy = check.position
        raw = self._raw.setdefault(tid, FeatureEngine(self.raw_cfg)).update(ObsSample(input_index, tid, f, t, ORIGIN_OBSERVED, pos, check.reason, raw_xy))
        flt = None
        if self.filtered_cfg is not None:
            eng = self._flt.setdefault(tid, FeatureEngine(self.filtered_cfg))
            fpos = pos
            if self.cfg.smoothing.kind == "time_aware_ema":
                ema = self._ema.setdefault(tid, TimeAwareEMA(self.cfg.smoothing.tau_s, self.cfg.frame_step, self.cfg.max_dt_seconds))
                fxy = ema.update(f, t, raw_xy, None if pos is None else pos.identity)
                fpos = None if pos is None else PositionInfo(fxy[0], fxy[1], pos.identity, pos.unit, pos.velocity_unit, pos.acceleration_unit,
                                                             pos.unit_status, pos.physical, pos.frame_name)
            flt = eng.update(ObsSample(input_index, tid, f, t, ORIGIN_OBSERVED, fpos, check.reason, raw_xy))
        self._last[tid] = (f, t)
        return CausalStep(raw, flt, gap)


# ------------------------------------------------------------ sidecar rows
def base_row(record: Mapping[str, Any], source_line: int, view: str) -> Dict[str, Any]:
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION, "view": view, "source_line": int(source_line),
        "track_id": record["track_id"], "sequence_id": record["sequence_id"], "scene_id": record.get("scene_id"),
        "source_dataset": record.get("source_dataset"), "agent_class": record.get("agent_class"),
        "frame_index": int(record["frame_index"]), "timestamp": float(record["timestamp"]),
        "coordinate_system": record.get("image_coordinate_system"), "coordinate_space": "image",
        "position_unit": POSITION_UNIT, "velocity_unit": VELOCITY_UNIT, "acceleration_unit": ACCELERATION_UNIT,
        "heading_unit": HEADING_UNIT, "time_unit": TIME_UNIT, "heading_convention": HEADING_CONVENTION_CODE,
        "position_source": "footpoint_bottom_centre", "interpolated": False, "causal_estimate": False,
        "world_x": None, "world_y": None, "world_position_valid": False,
    }


_FILTERED_NULL = {k: None for k in ("filtered_position_x", "filtered_position_y", "filtered_velocity_x", "filtered_velocity_y", "filtered_speed",
                                    "filtered_velocity_state", "filtered_acceleration_x", "filtered_acceleration_y",
                                    "filtered_acceleration_state", "filtered_heading", "filtered_heading_state")}


def _velocity_fields(s: FeatureSample, prefix: str = "") -> Dict[str, Any]:
    vok = s.velocity_state == VALID
    aok = s.acceleration_state == VALID
    hok = s.heading_state == VALID
    return {prefix + "velocity_x": _f(s.vx) if vok else None, prefix + "velocity_y": _f(s.vy) if vok else None,
            prefix + "speed": _f(s.speed) if vok else None, prefix + "velocity_state": s.velocity_state,
            prefix + "acceleration_x": _f(s.ax) if aok else None, prefix + "acceleration_y": _f(s.ay) if aok else None,
            prefix + "acceleration_state": s.acceleration_state,
            prefix + "heading": _f(s.heading) if hok else None, prefix + "heading_state": s.heading_state}


def causal_row(record: Mapping[str, Any], check: RecordCheck, source_line: int, step: CausalStep, filter_kind: str) -> Dict[str, Any]:
    row = base_row(record, source_line, VIEW_CAUSAL)
    s = step.raw
    ok = check.position is not None
    row.update({"position_x": check.position[0] if ok else None, "position_y": check.position[1] if ok else None,
                "position_state": check.reason, "observation_type": "observed" if ok else "invalid",
                "bbox_touches_border": check.touches_border, "dt_s": _f(s.dt), "frame_gap_before": step.frame_gap_before,
                "velocity_method": CAUSAL_VELOCITY_METHOD, "acceleration_method": CAUSAL_ACCELERATION_METHOD, "filter_kind": filter_kind})
    row.update(_velocity_fields(s))
    row.update(_FILTERED_NULL)
    if step.filtered is not None:
        fs = step.filtered
        row.update({"filtered_position_x": _f(fs.xs) if ok else None, "filtered_position_y": _f(fs.ys) if ok else None})
        row.update(_velocity_fields(fs, "filtered_"))
    return row


def rejected_causal_row(record: Mapping[str, Any], check: RecordCheck, source_line: int, code: str, filter_kind: str) -> Dict[str, Any]:
    """Row for a record refused by the Stage 2 grouping rules or by the stream. It carries no kinematics."""
    row = base_row(record, source_line, VIEW_CAUSAL)
    state = "rejected_" + code
    row.update({"position_x": None, "position_y": None, "position_state": state, "observation_type": "rejected",
                "bbox_touches_border": check.touches_border, "dt_s": None, "frame_gap_before": None,
                "velocity_x": None, "velocity_y": None, "speed": None, "velocity_state": state, "velocity_method": CAUSAL_VELOCITY_METHOD,
                "acceleration_x": None, "acceleration_y": None, "acceleration_state": state,
                "acceleration_method": CAUSAL_ACCELERATION_METHOD, "heading": None, "heading_state": state, "filter_kind": filter_kind})
    row.update(_FILTERED_NULL)
    if filter_kind != "none":
        row.update({"filtered_velocity_state": state, "filtered_acceleration_state": state, "filtered_heading_state": state})
    return row


class NonCausalRowError(ValueError):
    """An offline (or unlabelled) row was about to enter a causal output."""


def require_causal_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for r in rows:
        if r.get("view") != VIEW_CAUSAL or r.get("velocity_method") != CAUSAL_VELOCITY_METHOD:
            raise NonCausalRowError("row for %s frame %s is view=%r method=%r; only CAUSAL backward-difference rows may enter a "
                                    "causal output" % (r.get("track_id"), r.get("frame_index"), r.get("view"), r.get("velocity_method")))


# --------------------------------------------------------- batch processing
@dataclass
class CausalBatchResult:
    rows: List[Dict[str, Any]]                    # one per placeable record, input order
    quarantined: List[Dict[str, Any]]             # records that could not be placed in a track
    issues: List[TrajectoryIssue]                 # grouping + stream issues
    checks: List[Optional[RecordCheck]]           # parallel to the input records
    accepted: Dict[str, List[int]]                # track_id -> accepted input indices (input order)
    steps: Dict[int, CausalStep] = field(default_factory=dict)


def track_sequence_conflicts(records: Sequence[Mapping[str, Any]], placeable: Sequence[int]) -> Dict[str, str]:
    """Track ids that appear with more than one sequence_id: identities are sequence-scoped, so such a track is never joined."""
    seen: Dict[str, set] = {}
    for i in placeable:
        seen.setdefault(records[i]["track_id"], set()).add(records[i]["sequence_id"])
    return {t: "track_id appears in sequences %s" % sorted(s) for t, s in seen.items() if len(s) > 1}


def process_causal_records(records: Sequence[Mapping[str, Any]], source_lines: Sequence[int], cfg: ImageKinematicsConfig) -> CausalBatchResult:
    """Batch convenience: validate, apply the Stage 2/4 grouping rules, then feed accepted records one at a time, in input
    order, through :class:`CausalImageKinematics`. Input order is preserved; nothing is sorted."""
    checks: List[Optional[RecordCheck]] = [validate_phase3_record(r, cfg) for r in records]
    quarantined, placeable = [], []
    for i, (r, c) in enumerate(zip(records, checks)):
        if c.placeable:
            placeable.append(i)
        else:
            quarantined.append({"source_line": int(source_lines[i]), "reason": c.reason,
                                "track_id": r.get("track_id") if isinstance(r, Mapping) else None,
                                "frame_index": r.get("frame_index") if isinstance(r, Mapping) else None})
    conflicts = track_sequence_conflicts(records, placeable)
    issues: List[TrajectoryIssue] = []
    for i in list(placeable):
        tid = records[i]["track_id"]
        if tid in conflicts:
            issues.append(TrajectoryIssue("track_sequence_conflict", tid, i, records[i]["frame_index"], conflicts[tid]))
    grouping = group_tracks([records[i] for i in placeable if records[i]["track_id"] not in conflicts], cfg.frame_step)
    sub = [i for i in placeable if records[i]["track_id"] not in conflicts]
    accepted = {t: [sub[j] for j in idxs] for t, idxs in grouping.groups.items()}
    reject_code: Dict[int, str] = {}
    for iss in grouping.issues:
        orig = sub[iss.input_index]
        issues.append(TrajectoryIssue(iss.code, iss.track_id, orig, iss.frame_index, iss.message))
        reject_code[orig] = iss.code
    for i in placeable:
        if records[i]["track_id"] in conflicts:
            reject_code[i] = "track_sequence_conflict"
    driver = CausalImageKinematics(cfg)
    acc_set = sorted(i for v in accepted.values() for i in v)
    steps: Dict[int, CausalStep] = {}
    for i in acc_set:
        st = driver.update(records[i], checks[i], i)
        if st is None:
            reject_code[i] = driver.issues[-1].code
        else:
            steps[i] = st
    issues.extend(driver.issues)
    fk = cfg.smoothing.kind
    rows = []
    for i in placeable:
        if i in steps:
            rows.append(causal_row(records[i], checks[i], source_lines[i], steps[i], fk))
        else:
            rows.append(rejected_causal_row(records[i], checks[i], source_lines[i], reject_code.get(i, "unaccepted"), fk))
    require_causal_rows(rows)
    return CausalBatchResult(rows, quarantined, issues, checks, accepted, steps)


__all__ = [
    "IMAGE_KINEMATICS_VERSION", "SIDECAR_SCHEMA_VERSION", "SIDECAR_SCHEMA_PATH", "ImageKinematicsConfig", "SmoothingConfig", "ContinuityConfig",
    "RecordCheck", "validate_phase3_record", "placement_problem", "position_info", "TimeAwareEMA", "CausalImageKinematics", "CausalStep",
    "causal_row", "rejected_causal_row", "require_causal_rows", "NonCausalRowError", "process_causal_records", "CausalBatchResult",
    "HEADING_RANGE", "IMAGE_HEADING_CONVENTION",
]
