"""Stage 6: declarative, strict pipeline configuration (JSON).

One file describes the whole supported pipeline. Unknown keys are errors,
nothing is guessed, and ``grouping`` has NO default: a split needs a stated
grouping basis (recording/scene metadata is never invented). Relative paths
are resolved against the directory of the config file.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .agent_state import AgentStateConfig, VelocityContract
from .calibration import CalibrationManifest, RasterSpec, RasterTransform
from .errors import ConfigError
from .io.csv_adapter import CsvAdapterConfig
from .normalization import NormalizationConfig
from .projection import TransformConfig
from .splitting import GroupingPolicy, SplitConfig
from .trajectory import SmoothingPolicy, TrajectoryConfig
from .windows import WindowConfig

CONFIG_VERSION = 1
SECTIONS = ("config_version", "input", "agent_state", "calibration", "trajectory", "grouping", "split", "windows", "view", "normalization")


def _only(d: Any, allowed, section: str) -> Mapping[str, Any]:
    if not isinstance(d, Mapping):
        raise ConfigError("config section %r must be an object, got %s" % (section, type(d).__name__))
    extra = sorted(set(d) - set(allowed))
    if extra:
        raise ConfigError("unknown key(s) %s in config section %r; allowed: %s" % (extra, section, sorted(allowed)))
    return d


def _need(d: Mapping[str, Any], keys, section: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ConfigError("config section %r is missing required key(s) %s" % (section, missing))


def _build(cls, d: Mapping[str, Any], section: str, **extra):
    try:
        return cls(**d, **extra)
    except TypeError as exc:
        raise ConfigError("config section %r: %s" % (section, exc))


@dataclass(frozen=True)
class CalibrationSetup:
    manifest: CalibrationManifest
    observed_raster: RasterSpec
    raster_transform: Optional[RasterTransform]
    transform: TransformConfig


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved configuration. ``resolved`` is the canonical JSON form that is
    hashed into every run manifest (paths made absolute)."""

    config_path: Optional[str]
    input_csv: str
    adapter: CsvAdapterConfig
    agent_state: AgentStateConfig
    calibration: Optional[CalibrationSetup]
    trajectory: TrajectoryConfig
    grouping: Optional[GroupingPolicy]
    split: Optional[SplitConfig]
    windows: Optional[WindowConfig]
    view: str
    normalization: NormalizationConfig
    resolved: Dict[str, Any]

    def sha256(self) -> str:
        blob = json.dumps(self.resolved, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def require_build_sections(self) -> None:
        missing = [n for n, v in (("grouping", self.grouping), ("split", self.split), ("windows", self.windows)) if v is None]
        if missing:
            raise ConfigError("the 'build' command needs config section(s) %s. There is no default grouping: state which "
                              "groups are independent (and why) instead of letting the tool guess" % missing)


def load_config(path, *, input_csv_override: Optional[str] = None) -> PipelineConfig:
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError("config file not found: %s" % p)
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError("cannot read config %s: %s" % (p, exc))
    except json.JSONDecodeError as exc:
        raise ConfigError("config %s is not valid JSON: %s" % (p, exc))
    return config_from_dict(raw, base_dir=p.resolve().parent, config_path=str(p.resolve()), input_csv_override=input_csv_override)


def _resolve_path(value: str, base: Path) -> str:
    q = Path(value)
    return str(q if q.is_absolute() else (base / q).resolve())


def config_from_dict(raw: Mapping[str, Any], *, base_dir: Path, config_path: Optional[str] = None,
                     input_csv_override: Optional[str] = None) -> PipelineConfig:
    _only(raw, SECTIONS, "<root>")
    if raw.get("config_version") != CONFIG_VERSION:
        raise ConfigError("config_version must be %d, got %r" % (CONFIG_VERSION, raw.get("config_version")))
    if "input" not in raw:
        raise ConfigError("config is missing the 'input' section")

    inp = dict(_only(raw["input"], ("csv", "source_dataset", "columns", "sequence_id", "scene_id", "fps", "timestamp_unit",
                                     "position_semantics", "position_unit", "position_coordinate_system", "delimiter", "strict"), "input"))
    _need(inp, ("csv", "source_dataset", "columns"), "input")
    csv_path = input_csv_override if input_csv_override is not None else _resolve_path(str(inp.pop("csv")), base_dir)
    inp.pop("csv", None)
    adapter = _build(CsvAdapterConfig, inp, "input")

    a = dict(_only(raw.get("agent_state", {}), ("class_map", "agent_id_policy", "velocity_contract", "strict"), "agent_state"))
    if a.get("velocity_contract") is not None:
        a["velocity_contract"] = _build(VelocityContract, _only(a["velocity_contract"], ("space", "position_unit", "time_unit"),
                                                                 "agent_state.velocity_contract"), "agent_state.velocity_contract")
    agent_state = _build(AgentStateConfig, a, "agent_state")

    calibration, cal_resolved = None, None
    if raw.get("calibration") is not None:
        c = _only(raw["calibration"], ("manifest", "observed_raster", "raster_transform", "promote_eligible_to_canonical",
                                       "footpoint_tolerance_px", "position_source"), "calibration")
        _need(c, ("manifest", "observed_raster"), "calibration")
        mpath = _resolve_path(str(c["manifest"]), base_dir)
        try:
            manifest = CalibrationManifest.from_dict(json.loads(Path(mpath).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError("cannot load calibration manifest %s: %s" % (mpath, exc))
        raster = _build(RasterSpec, c["observed_raster"], "calibration.observed_raster")
        rt = None if c.get("raster_transform") is None else _build(RasterTransform, c["raster_transform"], "calibration.raster_transform")
        tkw = {k: c[k] for k in ("promote_eligible_to_canonical", "footpoint_tolerance_px", "position_source") if k in c}
        calibration = CalibrationSetup(manifest, raster, rt, _build(TransformConfig, tkw, "calibration"))
        cal_resolved = {"manifest": mpath, "manifest_sha256": manifest.sha256(), "observed_raster": raster.to_dict(),
                        "raster_transform": c.get("raster_transform"), "options": tkw}

    t = dict(_only(raw.get("trajectory", {}), ("coordinate_mode", "frame_step", "max_interpolation_gap_frames", "max_interpolation_gap_seconds",
                                               "allow_imputed_support", "smoothing", "min_heading_speed", "max_dt_seconds", "dt_rel_tol",
                                               "min_valid_observations", "short_track_policy"), "trajectory"))
    if "smoothing" in t:
        t["smoothing"] = _build(SmoothingPolicy, _only(t["smoothing"], ("kind", "alpha", "window"), "trajectory.smoothing"), "trajectory.smoothing")
    trajectory = _build(TrajectoryConfig, t, "trajectory")

    grouping = None
    if raw.get("grouping") is not None:
        g = dict(_only(raw["grouping"], ("unit", "independence_basis", "recording_map", "isolate_scenes"), "grouping"))
        _need(g, ("unit", "independence_basis"), "grouping")
        if g.get("recording_map") is not None:
            rm = g["recording_map"]
            if not isinstance(rm, list) or any(not isinstance(e, (list, tuple)) or len(e) != 3 for e in rm):
                raise ConfigError("grouping.recording_map must be a list of [source_dataset, raw_sequence_id, recording_id] triples")
            g["recording_map"] = _dup_map(rm)
        grouping = _build(GroupingPolicy, g, "grouping")

    split = None
    if raw.get("split") is not None:
        s = dict(_only(raw["split"], ("seed", "ratios", "explicit_assignments"), "split"))
        _need(s, ("seed",), "split")
        if s.get("explicit_assignments") is not None:
            s["explicit_assignments"] = [tuple(e) for e in s["explicit_assignments"]]
        split = _build(SplitConfig, s, "split")

    windows = None
    if raw.get("windows") is not None:
        w = dict(_only(raw["windows"], ("obs_len", "pred_len", "frame_stride", "window_stride", "min_valid_history_steps",
                                        "require_contiguous_native_frames", "allow_imputed", "allow_offline_view",
                                        "require_last_step_features"), "windows"))
        _need(w, ("obs_len", "pred_len"), "windows")
        windows = _build(WindowConfig, w, "windows")

    view = raw.get("view", "causal")
    if view not in ("causal", "offline"):
        raise ConfigError("view must be 'causal' or 'offline', got %r" % (view,))
    n = dict(_only(raw.get("normalization", {}), ("channels", "min_count", "constant_policy", "constant_rel_tol"), "normalization"))
    normalization = _build(NormalizationConfig, n, "normalization")

    resolved = json.loads(json.dumps({
        "config_version": CONFIG_VERSION,
        "input": {"csv": csv_path, **{k: v for k, v in dict(raw["input"]).items() if k != "csv"}},
        "agent_state": raw.get("agent_state", {}), "calibration": cal_resolved, "trajectory": raw.get("trajectory", {}),
        "grouping": raw.get("grouping"), "split": raw.get("split"), "windows": raw.get("windows"), "view": view,
        "normalization": raw.get("normalization", {}),
    }, sort_keys=True))
    return PipelineConfig(config_path, csv_path, adapter, agent_state, calibration, trajectory, grouping, split, windows, view,
                          normalization, resolved)


def _dup_map(rm: List[Any]):
    from .splitting import SplitError
    seen: Dict[Tuple[str, str], str] = {}
    for ds, seq, rec in rm:
        if (ds, seq) in seen:
            kind = "CONFLICTING_RECORDING_MAP" if seen[(ds, seq)] != rec else "DUPLICATE_RECORDING_MAP"
            raise SplitError(kind, "sequence %r is listed more than once in grouping.recording_map" % ((ds, seq),))
        seen[(ds, seq)] = rec
    return seen
