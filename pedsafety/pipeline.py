"""Stage 6: integration layer that wires Stages 1-5 into one auditable pipeline.

This module adds NO new science. It calls the existing stages in their documented
order, writes deterministic output files, records provenance, and turns failures
into structured errors with stable exit codes. Nothing here modifies a source file.

Commands (see ``pedsafety.cli``): ingest, features, build, hbs-smoke, plot.
"""

import dataclasses
import functools
import hashlib
import itertools
import json
import os
import platform
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import __version__
from .agent_state import BUILDER_VERSION, BuildResult, RecordProvenance, build_agent_states
from .geometry import CalibrationError
from .causal_trajectory import process_causal
from .dataset_builder import DATASET_BUILDER_VERSION, SplitDataset, build_split_dataset
from .errors import ConfigError, ParseError, PedsafetyError, SchemaLoadError, SchemaViolation
from .io.base import ParseResult
from .normalization import NormalizationError
from .output_dir import OutputDirectoryError, check_output_dir, output_dir_state
from .io.csv_adapter import CsvAnnotationAdapter
from .pipeline_config import PipelineConfig
from .projection import PROJECTION_VERSION, ProjectionError, TransformResult, transform_agent_states
from .schema import schema_sha256
from .splitting import GroupingPolicy, SplitConfig, SplitError
from .trajectory import CoordinateMode, TrajectoryConfig
from .validation import ObservationValidationError
from .windows import HISTORY_CHANNELS, SplitWindow, WindowConfig

PIPELINE_VERSION = "pipeline-0.6.0-stage6"
MANIFEST_VERSION = "pedsafety-run-manifest-v1"
MAX_LISTED_ISSUES = 500

EXIT_OK, EXIT_CHECK_FAILED, EXIT_CONFIG, EXIT_DATA, EXIT_SPLIT, EXIT_CALIBRATION, EXIT_OTHER, EXIT_OUTPUT, EXIT_INTERNAL = 0, 1, 2, 3, 4, 5, 6, 7, 70


class PipelineFailure(PedsafetyError):
    """A stage-tagged failure. ``exit_code`` is stable and documented in the CLI help."""

    def __init__(self, stage: str, code: str, message: str, exit_code: int, details: Optional[Sequence[str]] = None):
        self.stage, self.code, self.exit_code = stage, code, exit_code
        self.details = list(details or [])
        super().__init__("[%s/%s] %s" % (stage, code, message))


def classify_error(exc: BaseException, stage: str = "unknown") -> Dict[str, Any]:
    """Map any exception to ``{stage, code, message, exit_code, details}`` (no traceback)."""
    if isinstance(exc, PipelineFailure):
        return {"stage": exc.stage, "code": exc.code, "message": str(exc), "exit_code": exc.exit_code, "details": exc.details}
    if isinstance(exc, OutputDirectoryError):
        return {"stage": "output", "code": exc.code, "message": str(exc), "exit_code": EXIT_OUTPUT, "details": [exc.path]}
    if isinstance(exc, SplitError):
        return {"stage": "split", "code": exc.code, "message": str(exc), "exit_code": EXIT_SPLIT, "details": list(getattr(exc, "details", []) or [])}
    if isinstance(exc, (CalibrationError, ProjectionError)):
        return {"stage": "calibration", "code": getattr(exc, "code", "PROJECTION_FAILED"), "message": str(exc), "exit_code": EXIT_CALIBRATION, "details": []}
    if isinstance(exc, ObservationValidationError):
        return {"stage": "agent_state", "code": "OBSERVATION_VALIDATION_FAILED", "message": str(exc), "exit_code": EXIT_DATA,
                "details": [str(i) for i in exc.report.issues[:20]]}
    if isinstance(exc, ParseError):
        return {"stage": "ingest", "code": "PARSE_ERROR", "message": str(exc), "exit_code": EXIT_DATA, "details": [str(i) for i in exc.issues[:20]]}
    if isinstance(exc, SchemaViolation):
        return {"stage": "agent_state", "code": "SCHEMA_VIOLATION", "message": str(exc), "exit_code": EXIT_DATA, "details": []}
    if isinstance(exc, (ConfigError, SchemaLoadError)):
        return {"stage": stage if stage != "unknown" else "config", "code": "CONFIG_ERROR", "message": str(exc), "exit_code": EXIT_CONFIG, "details": []}
    if isinstance(exc, FileNotFoundError):
        return {"stage": "ingest", "code": "FILE_NOT_FOUND", "message": str(exc), "exit_code": EXIT_DATA, "details": []}
    if isinstance(exc, NormalizationError):
        return {"stage": "normalization", "code": getattr(exc, "code", "NORMALIZATION_ERROR"), "message": str(exc), "exit_code": EXIT_DATA, "details": []}
    if isinstance(exc, PedsafetyError):
        return {"stage": stage, "code": type(exc).__name__.upper(), "message": str(exc), "exit_code": EXIT_OTHER, "details": []}
    return {"stage": stage, "code": "INTERNAL_ERROR", "message": "%s: %s" % (type(exc).__name__, exc), "exit_code": EXIT_INTERNAL, "details": []}


# ----------------------------------------------------------------- file utils
def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "value") and isinstance(getattr(o, "value"), str):
        return o.value
    raise TypeError("not JSON serialisable: %r" % (type(o),))


def write_json(path, obj: Any) -> str:
    """Deterministic JSON (sorted keys, LF, no NaN), written atomically (temp file + replace)
    so a reader never sees a half-written file. Returns the file SHA-256."""
    text = json.dumps(obj, sort_keys=True, indent=2, allow_nan=False, default=_jsonable) + "\n"
    data = text.encode("utf-8")
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(str(tmp), str(p))
    return hashlib.sha256(data).hexdigest()


def arrays_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Content digest independent of any container (npz files embed timestamps)."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(("%s|%s|%s;" % (k, a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ ingestion
@dataclass
class IngestResult:
    parse: ParseResult
    build: BuildResult
    records: List[Dict[str, Any]]
    provenance: List[RecordProvenance]
    transform: Optional[TransformResult]
    projections: Optional[list]
    input_sha256: str
    input_bytes: int


def ingest(cfg: PipelineConfig, *, strict: bool = False) -> IngestResult:
    """CSV -> RawObservation -> schema-valid records (-> optional calibration projection).

    ``strict=True`` only ever TIGHTENS the config (parse and agent-state strictness)."""
    path = Path(cfg.input_csv)
    if not path.is_file():
        raise PipelineFailure("ingest", "FILE_NOT_FOUND", "input CSV not found: %s" % path, EXIT_DATA)
    before = file_sha256(path)
    adapter_cfg = dataclasses.replace(cfg.adapter, strict=True) if strict else cfg.adapter
    parsed = CsvAnnotationAdapter(path, adapter_cfg).parse()
    agent_cfg = dataclasses.replace(cfg.agent_state, strict=True) if strict else cfg.agent_state
    built = build_agent_states(parsed.observations, agent_cfg)
    records, prov = built.records, built.provenance
    transform, projections = None, None
    if cfg.calibration is not None:
        cal = cfg.calibration
        transform = transform_agent_states(records, prov, cal.manifest, cal.observed_raster, cal.transform, cal.raster_transform)
        if transform.preflight is not None and transform.preflight.error:
            raise PipelineFailure("calibration", "CALIBRATION_UNUSABLE", transform.preflight.error, EXIT_CALIBRATION)
        records, projections = transform.records, transform.projections
    after = file_sha256(path)
    if before != after:
        raise PipelineFailure("ingest", "INPUT_CHANGED_DURING_RUN", "input file changed while the pipeline was reading it", EXIT_DATA)
    return IngestResult(parsed, built, records, prov, transform, projections, before, path.stat().st_size)


def _cap(items: list) -> Dict[str, Any]:
    return {"total": len(items), "listed": min(len(items), MAX_LISTED_ISSUES), "truncated": len(items) > MAX_LISTED_ISSUES,
            "items": items[:MAX_LISTED_ISSUES]}


def parse_report(ing: IngestResult) -> Dict[str, Any]:
    p = ing.parse
    return {"rows_read": p.rows_read, "rows_rejected": p.rows_rejected, "observations": len(p.observations),
            "issue_counts": dict(sorted(Counter(i.code.value for i in p.issues).items())),
            "issues": _cap([{"source_line": i.source_line, "column": i.column, "code": i.code.value, "message": i.message,
                             "raw_value": i.raw_value} for i in p.issues])}


def validation_report(ing: IngestResult) -> Dict[str, Any]:
    r = ing.build.report.to_dict()
    r["issues"] = _cap(r["issues"])
    return r


def projection_report(ing: IngestResult) -> Optional[Dict[str, Any]]:
    t = ing.transform
    if t is None or t.preflight is None:
        return None
    pf = t.preflight
    return {"calibration_id": pf.calibration_id, "calibration_version": pf.calibration_version, "manifest_sha256": pf.manifest_sha256,
            "eligibility": pf.eligibility.to_dict(), "effective_homography_sha256": pf.effective_homography_sha256,
            "raster_adapted": pf.raster_adapted, "counters": dict(sorted(t.counters.items())),
            "note": "manifest_eligible reports software checks only; it is NOT independent physical verification"}


# ------------------------------------------------------------------- features
def describe_features(ing: IngestResult, tcfg: TrajectoryConfig) -> Dict[str, Any]:
    """Causal features over ALL records, no partitioning. DESCRIPTIVE only: this view
    must never feed training because it is not split."""
    res = process_causal(ing.records, ing.provenance, ing.projections, tcfg)
    by_ident: Dict[str, Dict[str, Any]] = {}
    states: Counter = Counter()
    for t in res.trajectories:
        key = "%s | %s | physical=%s" % (t.coordinate_mode, t.position_unit, t.physical)
        d = by_ident.setdefault(key, {"coordinate_mode": t.coordinate_mode, "unit_status": t.unit_status, "position_unit": t.position_unit,
                                      "velocity_unit": t.velocity_unit, "physical": t.physical, "physical_basis": t.physical_basis,
                                      "n_tracks": 0, "n_observed": 0, "n_short_tracks": 0})
        d["n_tracks"] += 1
        d["n_observed"] += t.n_observed
        d["n_short_tracks"] += int(t.too_short)
        states.update(str(s) for s in t.arrays["velocity_state"])
    return {"view": "CAUSAL", "partitioned": False, "use": "descriptive only; not a training/evaluation input",
            "coordinate_mode_requested": tcfg.coordinate_mode.value, "n_tracks": len(res.trajectories),
            "n_trajectory_issues": len(res.issues), "trajectory_issue_counts": dict(sorted(Counter(i.code for i in res.issues).items())),
            "identities": dict(sorted(by_ident.items())), "velocity_state_counts": dict(sorted(states.items()))}, res


# ---------------------------------------------------------------------- build
@dataclass
class BuildOutcome:
    ingest: IngestResult
    dataset: SplitDataset
    normalization: Any
    manifest: Dict[str, Any]
    out_dir: Path


def _export_windows(windows: List[SplitWindow], wcfg: WindowConfig, out_dir: Path, partition: str):
    n = len(windows)
    arrays: Dict[str, np.ndarray] = {}
    for ch in HISTORY_CHANNELS:
        arrays["history_" + ch] = np.stack([w.history[ch] for w in windows]) if n else np.zeros((0, wcfg.obs_len))
        arrays["available_" + ch] = np.stack([w.history_available[ch] for w in windows]) if n else np.zeros((0, wcfg.obs_len), dtype=bool)
    def stack(name, shape, dtype):
        return np.stack([getattr(w, name) for w in windows]) if n else np.zeros(shape, dtype=dtype)
    arrays["history_valid"] = stack("history_valid", (0, wcfg.obs_len), bool)
    arrays["history_frame_index"] = stack("history_frame_index", (0, wcfg.obs_len), np.int64)
    arrays["history_timestamp"] = stack("history_timestamp", (0, wcfg.obs_len), float)
    arrays["target_xy"] = stack("target_xy", (0, wcfg.pred_len, 2), float)
    arrays["target_frame_index"] = stack("target_frame_index", (0, wcfg.pred_len), np.int64)
    arrays["target_timestamp"] = stack("target_timestamp", (0, wcfg.pred_len), float)
    index = [{"row": i, "window_id": w.window_id, "partition": w.partition, "group_id": w.group_id, "dataset": w.dataset,
              "raw_sequence_id": w.raw_sequence_id, "raw_scene_id": w.raw_scene_id, "raw_track_id": w.raw_track_id, "track_id": w.track_id,
              "agent_class": w.agent_class, "view": w.view, "history_may_use_future": w.history_may_use_future,
              "coordinate_mode": w.coordinate_mode, "unit_status": w.unit_status, "physical": w.physical, "physical_basis": w.physical_basis,
              "frame_name": w.frame_name, "calibration_id": w.calibration_id, "manifest_sha256": w.manifest_sha256,
              "position_unit": w.position_unit, "velocity_unit": w.velocity_unit, "acceleration_unit": w.acceleration_unit,
              "heading_convention": w.heading_convention, "norm_key": w.norm_key()} for i, w in enumerate(windows)]
    np.savez(out_dir / ("windows_%s.npz" % partition), **arrays)
    idx_sha = write_json(out_dir / ("windows_%s_index.json" % partition), {"partition": partition, "n_windows": n, "windows": index})
    return arrays_digest(arrays), idx_sha


def _claims(dataset: SplitDataset, ing: IngestResult) -> Dict[str, Any]:
    wins = [w for p in dataset.partitions.values() for w in p.windows]
    modes = sorted({(w.coordinate_mode, w.unit_status, w.position_unit, w.physical) for w in wins})
    return {
        "coordinate_identities_in_windows": [{"coordinate_mode": m, "unit_status": s, "position_unit": u, "physical": ph} for m, s, u, ph in modes],
        "any_window_physical": any(w.physical for w in wins),
        "physical_basis": sorted({w.physical_basis for w in wins if w.physical_basis}),
        "calibration_supplied": ing.transform is not None,
        "software_eligibility_is_physical_verification": False,
        "statement": "units are reported exactly as each coordinate identity declares; 'physical' is true only for a calibration manifest "
                     "that passed software checks (basis software_manifest_checks_only), never for independent verification",
    }


def tool_versions() -> Dict[str, str]:
    return {"pedsafety": __version__, "pipeline": PIPELINE_VERSION, "agent_state": BUILDER_VERSION, "projection": PROJECTION_VERSION,
            "dataset_builder": DATASET_BUILDER_VERSION, "python": platform.python_version(), "numpy": np.__version__}


def base_manifest(command: str, cfg: PipelineConfig, ing: IngestResult) -> Dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION, "command": command, "tool_versions": tool_versions(),
        "config": {"path": cfg.config_path, "sha256": cfg.sha256()},
        "input": {"csv": cfg.input_csv, "sha256": ing.input_sha256, "bytes": ing.input_bytes, "read_only": True},
        "schema_sha256": schema_sha256(),
        "counts": {"rows_read": ing.parse.rows_read, "rows_rejected": ing.parse.rows_rejected, "observations": len(ing.parse.observations),
                   "records": len(ing.records), "records_quarantined": ing.build.report.n_quarantined},
        "calibration": None if ing.transform is None else {"manifest_sha256": cfg.calibration.manifest.sha256(),
                                                           "eligible_by_software_checks": ing.transform.preflight.eligibility.manifest_eligible,
                                                           "physically_verified_by_software": False},
    }


def _warnings(ing: IngestResult) -> List[str]:
    w = []
    if ing.parse.issues:
        w.append("%d source row(s) rejected during parsing" % ing.parse.rows_rejected)
    if not ing.build.report.ok:
        w.append("%d observation(s) quarantined: %s" % (ing.build.report.n_quarantined, ing.build.report.counts_by_code()))
    return w


def _prepare_out(out_dir) -> Path:
    """mkdir only: the policy check is done by ``_owns_output`` BEFORE the command body runs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _owns_output(fn):
    """Apply the output-directory policy (absent or empty only; see ``output_dir``) before the
    command runs. An exception raised after the check is tagged ``pedsafety_output_owned`` so the
    CLI may write ``error.json`` into the directory this run claimed, and only into that one."""
    @functools.wraps(fn)
    def wrapper(cfg, out_dir, *args, **kwargs):
        check_output_dir(out_dir)                      # refuses non-empty / non-directory paths; touches nothing
        try:
            return fn(cfg, out_dir, *args, **kwargs)
        except BaseException as exc:                   # noqa: BLE001 - re-raised unchanged
            try:
                exc.pedsafety_output_owned = str(out_dir)
            except Exception:                          # exceptions that forbid attributes
                pass
            raise
    return wrapper


def _require_clean(stage: str, warnings: List[str], fail_on_issues: bool) -> None:
    """``--fail-on-issues`` is decided BEFORE any result file is written, so a run that is
    reported as failed never leaves a complete-looking set of outputs."""
    if fail_on_issues and warnings:
        raise PipelineFailure(stage, "ISSUES_PRESENT", "; ".join(warnings), EXIT_DATA)


@_owns_output
def run_ingest(cfg: PipelineConfig, out_dir, *, strict: bool = False, fail_on_issues: bool = False) -> Dict[str, Any]:
    out = _prepare_out(out_dir)
    ing = ingest(cfg, strict=strict)
    warnings = _warnings(ing)
    _require_clean("ingest", warnings, fail_on_issues)
    files = {"parse_report.json": write_json(out / "parse_report.json", parse_report(ing)),
             "validation_report.json": write_json(out / "validation_report.json", validation_report(ing))}
    pr = projection_report(ing)
    if pr is not None:
        files["projection_report.json"] = write_json(out / "projection_report.json", pr)
    files["config_resolved.json"] = write_json(out / "config_resolved.json", cfg.resolved)
    m = base_manifest("ingest", cfg, ing)
    m.update({"status": "completed_with_warnings" if warnings else "completed", "warnings": warnings, "outputs": dict(sorted(files.items()))})
    write_json(out / "run_manifest.json", m)        # completion file: always last
    return m


@_owns_output
def run_features(cfg: PipelineConfig, out_dir, *, strict: bool = False, fail_on_issues: bool = False) -> Dict[str, Any]:
    out = _prepare_out(out_dir)
    ing = ingest(cfg, strict=strict)
    summary, _ = describe_features(ing, cfg.trajectory)
    warnings = _warnings(ing)
    if summary["n_tracks"] == 0:
        warnings.append("no trajectories were produced for coordinate mode %r" % cfg.trajectory.coordinate_mode.value)
    _require_clean("features", warnings, fail_on_issues)
    files = {"features_summary.json": write_json(out / "features_summary.json", summary),
             "parse_report.json": write_json(out / "parse_report.json", parse_report(ing)),
             "validation_report.json": write_json(out / "validation_report.json", validation_report(ing)),
             "config_resolved.json": write_json(out / "config_resolved.json", cfg.resolved)}
    pr = projection_report(ing)
    if pr is not None:
        files["projection_report.json"] = write_json(out / "projection_report.json", pr)
    m = base_manifest("features", cfg, ing)
    m.update({"status": "completed_with_warnings" if warnings else "completed", "warnings": warnings, "outputs": dict(sorted(files.items()))})
    write_json(out / "run_manifest.json", m)        # completion file: always last
    return m


def build_dataset_from_ingest(cfg: PipelineConfig, ing: IngestResult) -> SplitDataset:
    cfg.require_build_sections()
    return build_split_dataset(ing.records, ing.provenance, ing.projections, grouping=cfg.grouping, split=cfg.split, windows=cfg.windows,
                               trajectory=cfg.trajectory, view=cfg.view)


@_owns_output
def run_build(cfg: PipelineConfig, out_dir, *, strict: bool = False, fail_on_issues: bool = False) -> BuildOutcome:
    out = _prepare_out(out_dir)
    ing = ingest(cfg, strict=strict)
    ds = build_dataset_from_ingest(cfg, ing)
    norm = ds.fit_normalization(cfg.normalization)
    warnings = _warnings(ing) + list(ds.report["warnings"])
    _require_clean("build", warnings, fail_on_issues)
    files: Dict[str, str] = {}
    files["config_resolved.json"] = write_json(out / "config_resolved.json", cfg.resolved)
    files["parse_report.json"] = write_json(out / "parse_report.json", parse_report(ing))
    files["validation_report.json"] = write_json(out / "validation_report.json", validation_report(ing))
    pr = projection_report(ing)
    if pr is not None:
        files["projection_report.json"] = write_json(out / "projection_report.json", pr)
    files["split_plan.json"] = write_json(out / "split_plan.json", ds.plan.to_dict())
    files["split_report.json"] = write_json(out / "split_report.json", ds.report)
    files["normalization.json"] = write_json(out / "normalization.json", norm.to_dict())
    content: Dict[str, str] = {}
    for part in ("train", "validation", "test"):
        arr_sha, idx_sha = _export_windows(ds.partitions[part].windows, cfg.windows, out, part)
        content["windows_%s.npz (array content digest)" % part] = arr_sha
        files["windows_%s_index.json" % part] = idx_sha
    m = base_manifest("build", cfg, ing)
    m.update({
        "status": "completed_with_warnings" if warnings else "completed", "warnings": warnings,
        "split": {"plan_sha256": ds.plan.sha256(), "seed": ds.plan.seed, "method": ds.plan.method, "grouping_unit": ds.plan.grouping_unit,
                  "independence_basis": ds.plan.independence_basis, "view": ds.view,
                  "windows": {p: len(ds.partitions[p].windows) for p in ("train", "validation", "test")}},
        "normalization_sha256": norm.sha256(), "claims": _claims(ds, ing),
        "outputs": dict(sorted(files.items())), "output_array_digests": dict(sorted(content.items())),
    })
    write_json(out / "run_manifest.json", m)        # completion file: always last
    return BuildOutcome(ing, ds, norm, m, out)


# ------------------------------------------------------------------ HBS smoke
@_owns_output
def run_hbs_smoke(cfg: PipelineConfig, out_dir, *, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Read-only HBS check. Verifies (does not assume) that: positions stay
    ``unverified_planar`` with no unit, no world or image coordinates are produced, no
    feature is physical, and a split is REFUSED because HBS has one scene."""
    out = _prepare_out(out_dir)
    src = Path(cfg.input_csv)
    if not src.is_file():
        raise PipelineFailure("ingest", "FILE_NOT_FOUND", "HBS CSV not found: %s" % src, EXIT_DATA)
    before = file_sha256(src)
    use = cfg
    if max_rows is not None:
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
            raise ConfigError("max_rows must be an integer >= 1")
        sample = out / "hbs_sample.csv"                 # a COPY; the source is only read
        with open(src, "r", encoding="utf-8", newline="") as fh:
            lines = list(itertools.islice(fh, max_rows + 1))      # header + first max_rows data rows
        sample.write_bytes("".join(lines).encode("utf-8"))   # byte-exact: line endings preserved
        use = dataclasses.replace(cfg, input_csv=str(sample))
    tcfg = dataclasses.replace(cfg.trajectory, coordinate_mode=CoordinateMode.UNVERIFIED_PLANAR)
    ing = ingest(use)
    summary, res = describe_features(ing, tcfg)
    recs = ing.records
    scenes = sorted({r["scene_id"] for r in recs})
    cls = Counter(r["agent_class"] for r in recs)
    split_outcome: Dict[str, Any]
    try:
        build_split_dataset(ing.records, ing.provenance, None,
                            grouping=GroupingPolicy("scene", "smoke test only: demonstrates that a single scene cannot be split"),
                            split=SplitConfig(seed=0), windows=WindowConfig(obs_len=4, pred_len=4), trajectory=tcfg)
        split_outcome = {"refused": False, "code": None}
    except SplitError as exc:
        split_outcome = {"refused": True, "code": exc.code, "message": str(exc)}
    dts = sorted({round(float(d), 9) for t in res.trajectories for d in t.arrays["dt"] if np.isfinite(d)})
    checks = {
        "positions_unverified_planar_only": all(p.position_semantics == "unverified_planar" and p.position_unit is None for p in ing.provenance),
        "no_world_fields": all(r.get("world_x") is None and r.get("world_y") is None for r in recs),
        "world_position_valid_never_true": not any(r.get("world_position_valid") for r in recs),
        "no_image_fields": all(r.get("image_x") is None and r.get("image_y") is None for r in recs),
        "no_physical_features": all(t.physical is False for t in res.trajectories),
        "velocity_units_unverified": all(t.velocity_unit.startswith("unverified_planar") for t in res.trajectories),
        "no_metre_unit_anywhere": all("m/s" not in t.velocity_unit and t.position_unit != "m" for t in res.trajectories),
        "single_scene": len(scenes) == 1,
        "split_refused_insufficient_groups": split_outcome["refused"] and split_outcome["code"] == "INSUFFICIENT_GROUPS",
        "source_file_unchanged": file_sha256(src) == before,
    }
    report = {
        "kind": "hbs_read_only_smoke_test", "tool_versions": tool_versions(), "source": {"path": str(src), "sha256": before},
        "rows_limit": max_rows, "counts": {"rows_read": ing.parse.rows_read, "records": len(recs), "quarantined": ing.build.report.n_quarantined,
                                           "scenes": scenes, "records_by_agent_class": dict(sorted(cls.items())),
                                           "n_tracks": summary["n_tracks"]},
        "observed_dt_seconds": dts[:20], "features": summary, "split_attempt": split_outcome, "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "not_claimed": ["metric coordinates", "position or velocity units", "camera calibration", "recording-level independence",
                        "a train/validation/test partition", "dataset provenance or licence"],
        "assumptions": ["label 'bike' maps to schema class 'bicyclist' (default class map assumption)",
                        "timestamps are used as seconds exactly as the CSV provides them; original video FPS is not established"],
    }
    write_json(out / "hbs_smoke_report.json", report)
    return report


# ----------------------------------------------------------------- plotting
@_owns_output
def run_plot(cfg: PipelineConfig, out_dir, *, which: Optional[Sequence[str]] = None, strict: bool = False) -> Dict[str, Any]:
    from . import viz
    out = _prepare_out(out_dir)
    ing = ingest(cfg, strict=strict)
    summary, res = describe_features(ing, cfg.trajectory)
    wanted = list(which) if which else list(viz.FIGURES)
    bad = [w for w in wanted if w not in viz.FIGURES]
    if bad:
        raise ConfigError("unknown figure(s) %s; choose from %s" % (bad, list(viz.FIGURES)))
    written: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}
    ds = None
    needs_ds = {"split", "windows"} & set(wanted)
    if needs_ds:
        try:
            ds = build_dataset_from_ingest(cfg, ing)
        except (SplitError, ConfigError) as exc:
            for k in needs_ds:
                skipped[k] = "%s" % exc
    for name in wanted:
        if name in skipped:
            continue
        path = out / ("%s.png" % name)
        try:
            ok, why = viz.FIGURES[name](path, trajectories=res.trajectories, ingest=ing, dataset=ds, config=cfg)
        except ImportError as exc:
            raise PipelineFailure("plot", "MATPLOTLIB_MISSING", "plotting needs matplotlib: %s" % exc, EXIT_CONFIG)
        if ok:
            written[name] = str(path.name)
        else:
            skipped[name] = why
    m = {"command": "plot", "tool_versions": tool_versions(), "written": written, "skipped": skipped, "config_sha256": cfg.sha256(),
         "note": "figure bytes are not claimed to be reproducible; the data behind them is"}
    write_json(out / "plot_manifest.json", m)
    return m


def write_error(out_dir, info: Mapping[str, Any], *, owned: bool = False) -> Optional[Path]:
    """Write ``error.json`` only into a directory that is new/empty or that this run claimed
    (``owned``). A refused (non-empty) directory is never touched, so existing results stay intact.
    Returns None when nothing was written."""
    try:
        out = Path(out_dir)
        if not owned and output_dir_state(out) not in ("absent", "empty"):
            return None
        out.mkdir(parents=True, exist_ok=True)
        p = out / "error.json"
        write_json(p, {"status": "failed", **info})
        return p
    except OSError:
        return None
