"""Phase 4 Task 2 runner: Phase 3 ``agent_states.jsonl`` -> image-space kinematic sidecars + diagnostics.

Output policy (shared ``output_dir`` rules): the output directory must be absent or empty; nothing is overwritten or
deleted; every file is written to a temporary name and renamed into place; ``run_manifest.json`` is written last and is
the completion marker. The input file is opened read-only and its SHA-256 is checked before and after processing.

Files written:
    causal_kinematics.jsonl       CAUSAL sidecar rows (backward differences, optional causal filter), one per placeable record
    offline_kinematics.jsonl      OFFLINE_NON_CAUSAL rows (central differences), only if enabled; never mixed into the causal file
    quarantine.jsonl              records that could not be placed in a track (or malformed JSON lines)
    track_continuity_report.json  continuity facts and anomaly indicators
    kinematic_quality_report.json counts, states and distributions per view and class
    config_resolved.json          the configuration actually used
    processing_log.txt            human-readable log
    run_manifest.json             provenance, hashes, counts, claims, limitations (written last)
"""

import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from jsonschema import Draft202012Validator

from .errors import PedsafetyError
from .image_kinematics import (
    IMAGE_HEADING_CONVENTION, IMAGE_KINEMATICS_VERSION, SIDECAR_SCHEMA_PATH, SIDECAR_SCHEMA_VERSION, VALID, ImageKinematicsConfig,
    process_causal_records, require_causal_rows,
)
from .image_kinematics_offline import process_offline_records
from .output_dir import check_output_dir
from .track_continuity import continuity_report
from .trajectory import TRAJECTORY_VERSION

MANIFEST_VERSION = "phase4-task2-run-manifest-1.0.0"
COMPLETION_FILE = "run_manifest.json"

CLAIMS = {
    "coordinate_space": "image (captured raster pixels)",
    "metric_world_coordinates": False, "physical_speed": False, "physical_acceleration": False, "world_heading": False,
    "metric_ttc_or_distance": False, "ego_motion_compensated": False,
    "note": "Image-space derivatives of a moving vehicle-mounted camera mix agent motion and camera ego-motion; they are not physical quantities.",
}
LIMITATIONS = [
    "Image space only (pixels, pixel/s, pixel/s^2, image-plane radians); no metric or world interpretation is supported (Phase 4 Task 1).",
    "Camera ego-motion is not removed: image velocity is not agent velocity.",
    "max_derivative_gap_frames = 0: no derivative across any missing frame; gap bridging awaits review of the gap distribution.",
    "Positions are bottom-centre footpoints, a ground-contact proxy affected by occlusion, truncation and box looseness.",
    "Causal acceleration divides by the current step dt (the specified Stage 4 formula); on unequal spacing it is not centred.",
    "Anomaly indicators (suspicious jumps, possible fragment pairs) are not confirmed ID switches.",
    "No interpolation is performed in either view.",
]


class RunError(PedsafetyError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path) -> Iterator[Tuple[int, Optional[Any], Optional[str]]]:
    """Yield (line_number, parsed_object_or_None, error_or_None) for every non-blank line, streaming."""
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield n, json.loads(line), None
            except ValueError as exc:
                yield n, None, "malformed_json: %s" % exc


def _atomic_write_text(path: Path, text: str) -> str:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(r, separators=(",", ":"), allow_nan=False, sort_keys=False) + "\n" for r in rows)


def _json_text(obj: Any) -> str:
    return json.dumps(obj, indent=2, allow_nan=False, sort_keys=False) + "\n"


def load_sidecar_validator() -> Draft202012Validator:
    schema = json.loads(SIDECAR_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_rows(rows: Sequence[Mapping[str, Any]], validator: Draft202012Validator, what: str) -> None:
    for r in rows:
        errs = sorted(validator.iter_errors(r), key=lambda e: list(e.path))
        if errs:
            raise RunError("%s row for %s frame %s violates the sidecar schema: %s" % (what, r.get("track_id"), r.get("frame_index"), errs[0].message))


def _pct(values) -> Optional[Dict[str, float]]:
    a = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "min": float(a.min()), "p05": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "mean": float(a.mean())}


def _view_quality(rows: Sequence[Mapping[str, Any]], prefix: str, cfg: ImageKinematicsConfig) -> Dict[str, Any]:
    vs = Counter(r[prefix + "velocity_state"] for r in rows)
    acs = Counter(r[prefix + "acceleration_state"] for r in rows)
    hs = Counter(r[prefix + "heading_state"] for r in rows)
    vrows = [r for r in rows if r[prefix + "velocity_state"] == VALID]
    arows = [r for r in rows if r[prefix + "acceleration_state"] == VALID]
    speeds = [r[prefix + "speed"] for r in vrows]
    acc = [math.hypot(r[prefix + "acceleration_x"], r[prefix + "acceleration_y"]) for r in arows]
    dts = [r["dt_s"] for r in vrows]
    per_class: Dict[str, Any] = {}
    for c in sorted({str(r["agent_class"]) for r in rows}):
        cr = [r for r in rows if str(r["agent_class"]) == c]
        cv = [r for r in cr if r[prefix + "velocity_state"] == VALID]
        ca = [r for r in cr if r[prefix + "acceleration_state"] == VALID]
        per_class[c] = {"rows": len(cr), "valid_velocity": len(cv), "valid_acceleration": len(ca),
                        "valid_heading": sum(1 for r in cr if r[prefix + "heading_state"] == VALID),
                        "speed_px_per_s": _pct([r[prefix + "speed"] for r in cv]),
                        "acceleration_magnitude_px_per_s2": _pct([math.hypot(r[prefix + "acceleration_x"], r[prefix + "acceleration_y"]) for r in ca])}
    sens = [{"min_speed_px_per_s": t, "rows_with_heading": sum(1 for s in speeds if s > 0 and s >= t)} for t in cfg.heading_speed_sensitivity_px_per_s]
    return {"rows": len(rows), "velocity_states": dict(sorted(vs.items())), "acceleration_states": dict(sorted(acs.items())),
            "heading_states": dict(sorted(hs.items())), "valid_velocity": len(vrows), "valid_acceleration": len(arows),
            "valid_heading": hs.get(VALID, 0), "dt_s_of_valid_velocity": _pct(dts), "speed_px_per_s": _pct(speeds),
            "acceleration_magnitude_px_per_s2": _pct(acc), "heading_count_sensitivity": sens, "per_class": per_class}


def quality_report(causal_rows, offline_rows, n_input_rows: int, n_quarantined: int, cfg: ImageKinematicsConfig) -> Dict[str, Any]:
    pos = Counter(r["position_state"] for r in causal_rows)
    obs = Counter(r["observation_type"] for r in causal_rows)
    rep = {"units": {"position": "pixel", "velocity": "pixel/s", "acceleration": "pixel/s^2", "heading": "rad", "time": "s"},
           "heading_convention": IMAGE_HEADING_CONVENTION,
           "input_rows": n_input_rows, "quarantined_rows": n_quarantined, "causal_rows": len(causal_rows),
           "observation_types": dict(sorted(obs.items())), "position_states": dict(sorted(pos.items())),
           "valid_position_rows": pos.get("ok", 0), "invalid_position_rows": len(causal_rows) - pos.get("ok", 0) - obs.get("rejected", 0),
           "rejected_rows": obs.get("rejected", 0), "interpolated_rows": 0, "causal_estimate_rows": 0,
           "filter_kind": cfg.smoothing.kind, "filtered_rows": sum(1 for r in causal_rows if r["filtered_velocity_state"] is not None),
           "bbox_touches_border_rows": sum(1 for r in causal_rows if r["bbox_touches_border"]),
           "causal_raw": _view_quality(causal_rows, "", cfg),
           "causal_filtered": _view_quality(causal_rows, "filtered_", cfg) if cfg.smoothing.kind != "none" else None,
           "offline_central_difference": _view_quality(offline_rows, "", cfg) if offline_rows is not None else None}
    return rep


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _optional_reference(path) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        raise RunError("reference file not found: %s" % p)
    info: Dict[str, Any] = {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
    if p.suffix == ".json":
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            info["excerpt"] = {k: m.get(k) for k in ("manifest_version", "created_utc", "video", "backend", "is_mock", "video_stats", "world_coordinates")
                               if isinstance(m, dict) and k in m}
        except ValueError:
            info["excerpt"] = "not parseable as JSON"
    return info


def run_image_kinematics(input_path, cfg: ImageKinematicsConfig, out_dir, *, config_path: Optional[str] = None,
                         run_manifest_path=None, tracked_agents_path=None, command: Optional[str] = None) -> Dict[str, Any]:
    out = check_output_dir(out_dir)                                     # absent or empty only; touches nothing
    src = Path(input_path)
    if not src.is_file():
        raise RunError("input file not found: %s" % src)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha_before = sha256_file(src)
    refs = {"phase3_run_manifest": _optional_reference(run_manifest_path), "phase3_tracked_agents": _optional_reference(tracked_agents_path)}
    log: List[str] = ["phase4-task2 image kinematics %s" % IMAGE_KINEMATICS_VERSION, "started_utc %s" % started,
                      "input %s sha256 %s" % (src, sha_before)]

    records: List[Any] = []
    lines: List[int] = []
    malformed: List[Dict[str, Any]] = []
    n_lines = 0
    for n, obj, err in iter_jsonl(src):
        n_lines += 1
        if err is not None:
            malformed.append({"source_line": n, "reason": err, "track_id": None, "frame_index": None})
        else:
            records.append(obj)
            lines.append(n)
    log.append("lines %d parsed %d malformed %d" % (n_lines, len(records), len(malformed)))

    causal = process_causal_records(records, lines, cfg)
    offline_rows = process_offline_records(records, lines, causal, cfg) if cfg.offline_central_difference else None
    require_causal_rows(causal.rows)                                    # offline rows can never enter the causal file
    validator = load_sidecar_validator()
    validate_rows(causal.rows, validator, "causal")
    if offline_rows is not None:
        validate_rows(offline_rows, validator, "offline")
    quarantine = malformed + causal.quarantined
    cont = continuity_report(records, causal, cfg)
    qual = quality_report(causal.rows, offline_rows, n_lines, len(quarantine), cfg)
    log.append("causal rows %d offline rows %s quarantined %d tracks %d" % (len(causal.rows), "n/a" if offline_rows is None else len(offline_rows),
                                                                          len(quarantine), cont["summary"]["n_tracks"]))

    out.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Dict[str, Any]] = {}

    def put(name: str, text: str, rows: Optional[int] = None):
        files[name] = {"sha256": _atomic_write_text(out / name, text), "bytes": (out / name).stat().st_size}
        if rows is not None:
            files[name]["rows"] = rows

    put("causal_kinematics.jsonl", _jsonl_text(causal.rows), len(causal.rows))
    if offline_rows is not None:
        put("offline_kinematics.jsonl", _jsonl_text(offline_rows), len(offline_rows))
    put("quarantine.jsonl", _jsonl_text(quarantine), len(quarantine))
    put("track_continuity_report.json", _json_text(cont))
    put("kinematic_quality_report.json", _json_text(qual))
    put("config_resolved.json", _json_text(cfg.to_dict()))
    sha_after = sha256_file(src)
    if sha_after != sha_before:
        raise RunError("input file changed during processing (sha256 %s -> %s)" % (sha_before, sha_after))
    finished = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log.append("input sha256 unchanged after processing")
    log.append("finished_utc %s" % finished)
    put("processing_log.txt", "\n".join(log) + "\n")

    import jsonschema as _js
    try:
        from importlib.metadata import version as _ver
        js_version = _ver("jsonschema")
    except Exception:                                                   # noqa: BLE001
        js_version = getattr(_js, "__version__", None)
    manifest = {
        "manifest_version": MANIFEST_VERSION, "task": "phase4_task2_image_space_kinematics", "status": "completed",
        "processing_version": IMAGE_KINEMATICS_VERSION, "engine": {"name": "pedsafety.trajectory.FeatureEngine", "version": TRAJECTORY_VERSION},
        "sidecar_schema": {"version": SIDECAR_SCHEMA_VERSION, "path": str(SIDECAR_SCHEMA_PATH.relative_to(SIDECAR_SCHEMA_PATH.parents[3])),
                           "sha256": sha256_file(SIDECAR_SCHEMA_PATH)},
        "started_utc": started, "finished_utc": finished, "command": command,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "jsonschema": js_version,
                        "git_commit": _git_commit()},
        "input": {"path": str(src), "sha256_before": sha_before, "sha256_after": sha_after, "bytes": src.stat().st_size, "non_blank_lines": n_lines,
                  "read_only": True},
        "references": refs,
        "config": {"path": config_path, "sha256_resolved": cfg.sha256(), "config_version": cfg.config_version},
        "modes": {"causal": True, "offline_central_difference": cfg.offline_central_difference,
                  "offline_rows_in_causal_output": False},
        "conventions": {"coordinate_system": cfg.expected_image_coordinate_system, "coordinate_space": "image", "origin": "top-left",
                        "u_positive": "right", "v_positive": "down", "position_unit": "pixel", "velocity_unit": "pixel/s",
                        "acceleration_unit": "pixel/s^2", "time_unit": "s", "heading_unit": "rad", "heading_convention": IMAGE_HEADING_CONVENTION,
                        "position_source": "footpoint = (bbox centre x, bbox bottom y), taken from image_x/image_y after a consistency check"},
        "policies": {"timestamp": "actual record timestamps; dt must be finite and > 0; nominal FPS is never substituted",
                     "ordering": "input order per track; duplicate frames quarantine all copies; out-of-order frames and non-monotonic timestamps quarantined (Stage 2/4 rules)",
                     "gaps": "max_derivative_gap_frames = %d: any missing frame breaks velocity and acceleration" % cfg.max_derivative_gap_frames,
                     "max_dt_seconds": cfg.max_dt_seconds, "min_heading_speed_px_per_s": cfg.min_heading_speed_px_per_s,
                     "smoothing": cfg.smoothing.describe(), "interpolation": "none",
                     "provenance": "the Phase 3 records carry no Stage 2 provenance sidecar; none is reconstructed. Fields absent from a record stay null."},
        "counts": {"input_non_blank_lines": n_lines, "parsed_records": len(records), "quarantined": len(quarantine), "causal_rows": len(causal.rows),
                   "offline_rows": None if offline_rows is None else len(offline_rows), "tracks": cont["summary"]["n_tracks"],
                   "valid_positions": qual["valid_position_rows"], "valid_velocity": qual["causal_raw"]["valid_velocity"],
                   "valid_acceleration": qual["causal_raw"]["valid_acceleration"], "valid_heading": qual["causal_raw"]["valid_heading"]},
        "outputs": dict(sorted(files.items())),
        "claims": CLAIMS, "limitations": LIMITATIONS,
    }
    put(COMPLETION_FILE, _json_text(manifest))
    return manifest


__all__ = ["run_image_kinematics", "RunError", "sha256_file", "iter_jsonl", "load_sidecar_validator", "validate_rows", "quality_report",
           "MANIFEST_VERSION", "COMPLETION_FILE", "CLAIMS", "LIMITATIONS"]
