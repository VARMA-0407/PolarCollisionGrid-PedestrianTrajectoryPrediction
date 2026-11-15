"""T3.1: input adapter and contract validation for Task 3.

Reads exactly three inputs and nothing else:
    * Task 2 ``causal_kinematics.jsonl`` (CAUSAL sidecar rows)
    * that run's ``run_manifest.json`` (and, through it, the run's ``config_resolved.json``)
    * Phase 3 ``agent_states.jsonl`` (box geometry and source label, joined through ``source_line``)

Every check either passes or raises :class:`InteractionContractError` with a stable code. Nothing is repaired, dropped
silently or matched by row order. The Task 2 offline file is refused by name and by content.

Checks (decision record P7, P9-P14):
    hashes      causal file == Task 2 manifest output hash and row count; Phase 3 file == Task 2 manifest input hash
                (before and after); sidecar schema file == manifest schema hash and version; config_resolved.json ==
                manifest output hash, with max_derivative_gap_frames == 0 and smoothing 'none'
    rows        every causal row validated against the full Task 2 sidecar schema, plus explicit view / method /
                coordinate space / coordinate system / unit / world-field checks
    join        source_line is a one-based PHYSICAL line number (Task 2 iter_jsonl: enumerate(fh, start=1), blank lines
                skipped but counted); it must be an int in [1, physical lines], used once, point to a non-blank line that
                parses as a JSON object, and that record must have identical track_id, frame_index and timestamp; the
                joined bbox must be four finite numbers with positive width and height
    frames      all causal rows of one frame share exactly one timestamp
    nodes       observed rows need a boolean bbox_touches_border
    P7          frame_gap_before recomputed from the causal rows equals the stored value for every row
"""

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..image_kinematics import (
    ACCELERATION_UNIT, CAUSAL_ACCELERATION_METHOD, CAUSAL_VELOCITY_METHOD, HEADING_UNIT, POSITION_UNIT, SIDECAR_SCHEMA_PATH,
    SIDECAR_SCHEMA_VERSION, TIME_UNIT, VELOCITY_UNIT,
)
from ..image_kinematics_run import load_sidecar_validator, sha256_file
from ..trajectory import VIEW_CAUSAL
from .config import NODE_TYPES, InteractionConfig, InteractionContractError

ADAPTER_VERSION = "interaction-input-adapter-0.1.0"
OFFLINE_FILENAME = "offline_kinematics.jsonl"


@dataclass
class JoinedRow:
    """One causal row plus the provenance-joined Phase 3 fields needed by Task 3 (nothing else is kept)."""

    source_line: int
    track_id: str
    sequence_id: str
    frame_index: int
    timestamp: float
    agent_class: Optional[str]
    observation_type: str
    position_state: str
    velocity_state: str
    heading_state: str
    position_x: Optional[float]
    position_y: Optional[float]
    velocity_x: Optional[float]
    velocity_y: Optional[float]
    speed: Optional[float]
    heading: Optional[float]
    bbox_touches_border: Optional[bool]
    frame_gap_before: Optional[int]
    coordinate_system: Optional[str]
    bbox_xyxy: Tuple[float, float, float, float]
    source_label: Optional[str]


@dataclass
class AdapterResult:
    rows: List[JoinedRow]                         # every causal row, causal-file order
    input_hashes: Dict[str, Dict[str, Any]]
    task2_manifest: Dict[str, Any]
    task2_config: Dict[str, Any]
    counts: Dict[str, Any]
    timings_s: Dict[str, float] = field(default_factory=dict)


def _fail(code: str, msg: str):
    raise InteractionContractError(code, msg)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _load_json(path: Path, what: str) -> Dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail("UNREADABLE_" + what.upper(), "cannot read %s %s: %s" % (what, path, exc))
    if not isinstance(d, dict):
        _fail("BAD_" + what.upper(), "%s %s is not a JSON object" % (what, path))
    return d


# ------------------------------------------------------------ manifest checks
def verify_task2_manifest(causal_path: Path, manifest_path: Path, phase3_path: Path, cfg: InteractionConfig) -> Tuple[Dict, Dict, Dict]:
    if causal_path.name == OFFLINE_FILENAME:
        _fail("OFFLINE_INPUT_REFUSED", "the Task 2 offline file %s is never a Task 3 input" % causal_path)
    for p, what in ((causal_path, "causal sidecar"), (manifest_path, "Task 2 manifest"), (phase3_path, "Phase 3 agent_states")):
        if not p.is_file():
            _fail("INPUT_MISSING", "%s not found: %s" % (what, p))
    m = _load_json(manifest_path, "task2_manifest")
    if m.get("task") != "phase4_task2_image_space_kinematics" or m.get("status") != "completed":
        _fail("TASK2_MANIFEST_IDENTITY", "manifest is not a completed Task 2 run (task=%r status=%r)" % (m.get("task"), m.get("status")))
    outs = m.get("outputs") or {}
    entry = outs.get("causal_kinematics.jsonl")
    if not isinstance(entry, dict):
        _fail("TASK2_MANIFEST_NO_CAUSAL", "manifest lists no causal_kinematics.jsonl output")
    hashes: Dict[str, Dict[str, Any]] = {}
    h_causal = sha256_file(causal_path)
    if h_causal != entry.get("sha256"):
        _fail("CAUSAL_HASH_MISMATCH", "causal sidecar sha256 %s != Task 2 manifest %s" % (h_causal, entry.get("sha256")))
    hashes["causal_kinematics"] = {"path": str(causal_path), "sha256": h_causal, "bytes": causal_path.stat().st_size,
                                   "manifest_rows": entry.get("rows")}
    inp = m.get("input") or {}
    h_p3 = sha256_file(phase3_path)
    if not (h_p3 == inp.get("sha256_before") == inp.get("sha256_after")):
        _fail("PHASE3_HASH_MISMATCH", "Phase 3 agent_states sha256 %s != Task 2 manifest input (before %s, after %s)"
              % (h_p3, inp.get("sha256_before"), inp.get("sha256_after")))
    hashes["phase3_agent_states"] = {"path": str(phase3_path), "sha256": h_p3, "bytes": phase3_path.stat().st_size,
                                     "task2_manifest_path_field": inp.get("path")}
    sch = m.get("sidecar_schema") or {}
    h_schema = sha256_file(SIDECAR_SCHEMA_PATH)
    if sch.get("version") != SIDECAR_SCHEMA_VERSION or sch.get("sha256") != h_schema:
        _fail("SIDECAR_SCHEMA_MISMATCH", "Task 2 run used schema %r/%r; repository schema is %r/%s"
              % (sch.get("version"), sch.get("sha256"), SIDECAR_SCHEMA_VERSION, h_schema))
    hashes["sidecar_schema"] = {"path": str(SIDECAR_SCHEMA_PATH), "sha256": h_schema, "version": SIDECAR_SCHEMA_VERSION}
    if (m.get("modes") or {}).get("offline_rows_in_causal_output") is not False:
        _fail("TASK2_MODES", "Task 2 manifest does not state offline_rows_in_causal_output = false")
    conv = m.get("conventions") or {}
    expect = {"coordinate_system": cfg.expected_coordinate_system, "coordinate_space": "image", "position_unit": POSITION_UNIT,
              "velocity_unit": VELOCITY_UNIT, "acceleration_unit": ACCELERATION_UNIT, "time_unit": TIME_UNIT, "heading_unit": HEADING_UNIT}
    bad = {k: conv.get(k) for k, v in expect.items() if conv.get(k) != v}
    if bad:
        _fail("TASK2_CONVENTIONS", "Task 2 manifest conventions differ from the Task 3 contract: %s" % bad)
    cfg_entry = outs.get("config_resolved.json")
    cfg_path = manifest_path.parent / "config_resolved.json"
    if not isinstance(cfg_entry, dict) or not cfg_path.is_file() or sha256_file(cfg_path) != cfg_entry.get("sha256"):
        _fail("TASK2_CONFIG_HASH", "config_resolved.json missing next to the manifest or its hash differs from the manifest")
    t2cfg = _load_json(cfg_path, "task2_config")
    if t2cfg.get("max_derivative_gap_frames") != 0:
        _fail("TASK2_GAP_BRIDGING", "Task 2 run used max_derivative_gap_frames=%r; Task 3 requires 0" % t2cfg.get("max_derivative_gap_frames"))
    if (t2cfg.get("smoothing") or {}).get("kind") != "none":
        _fail("TASK2_SMOOTHING", "Task 2 run used smoothing %r; Task 3 consumes the unsmoothed causal output (Q7)" % t2cfg.get("smoothing"))
    if t2cfg.get("expected_image_coordinate_system") != cfg.expected_coordinate_system:
        _fail("TASK2_RASTER", "Task 2 raster %r != %r" % (t2cfg.get("expected_image_coordinate_system"), cfg.expected_coordinate_system))
    hashes["task2_manifest"] = {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
    hashes["task2_config_resolved"] = {"path": str(cfg_path), "sha256": cfg_entry["sha256"]}
    return m, t2cfg, hashes


# ------------------------------------------------------------- row checks
_EXPLICIT = (("view", VIEW_CAUSAL), ("velocity_method", CAUSAL_VELOCITY_METHOD), ("acceleration_method", CAUSAL_ACCELERATION_METHOD),
             ("coordinate_space", "image"), ("position_unit", POSITION_UNIT), ("velocity_unit", VELOCITY_UNIT),
             ("acceleration_unit", ACCELERATION_UNIT), ("time_unit", TIME_UNIT), ("heading_unit", HEADING_UNIT),
             ("world_x", None), ("world_y", None), ("world_position_valid", False), ("schema_version", SIDECAR_SCHEMA_VERSION))


def read_causal_rows(causal_path: Path, cfg: InteractionConfig) -> Tuple[List[Dict[str, Any]], float]:
    validator = load_sidecar_validator()
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    with open(causal_path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                _fail("CAUSAL_BLANK_LINE", "causal sidecar line %d is blank" % n)
            try:
                r = json.loads(line)
            except ValueError as exc:
                _fail("CAUSAL_MALFORMED_LINE", "causal sidecar line %d is not JSON: %s" % (n, exc))
            if not isinstance(r, dict):
                _fail("CAUSAL_MALFORMED_LINE", "causal sidecar line %d is not a JSON object" % n)
            for k, v in _EXPLICIT:                         # explicit checks first: clearer error than a schema message
                if k not in r or r[k] != v or (v is not None and type(r[k]) is not type(v)):
                    code = "OFFLINE_ROW_REFUSED" if k in ("view", "velocity_method", "acceleration_method") else (
                        "WORLD_FIELDS_POPULATED" if k.startswith("world") else "CAUSAL_ROW_CONTRACT")
                    _fail(code, "causal sidecar line %d: %s=%r, required %r" % (n, k, r.get(k, "<missing>"), v))
            if r.get("coordinate_system") != cfg.expected_coordinate_system:
                _fail("COORDINATE_SYSTEM_MISMATCH", "causal sidecar line %d: coordinate_system %r != %r"
                      % (n, r.get("coordinate_system"), cfg.expected_coordinate_system))
            err = next(validator.iter_errors(r), None)
            if err is not None:
                _fail("SIDECAR_SCHEMA_VIOLATION", "causal sidecar line %d violates the Task 2 sidecar schema: %s" % (n, err.message))
            rows.append(r)
    return rows, time.perf_counter() - t0


def read_phase3_lines(phase3_path: Path) -> Tuple[Dict[int, Optional[Dict[str, Any]]], int, int]:
    """Physical line number -> parsed object (None for a blank line). Any malformed non-blank line fails the run."""
    out: Dict[int, Optional[Dict[str, Any]]] = {}
    blank = 0
    n = 0
    with open(phase3_path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                out[n] = None
                blank += 1
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                _fail("PHASE3_MALFORMED_LINE", "Phase 3 agent_states physical line %d is not JSON: %s" % (n, exc))
            if not isinstance(obj, dict):
                _fail("PHASE3_MALFORMED_LINE", "Phase 3 agent_states physical line %d is not a JSON object" % n)
            out[n] = obj
    return out, n, blank


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def join_rows(causal: List[Dict[str, Any]], phase3: Dict[int, Optional[Dict[str, Any]]], n_physical: int) -> List[JoinedRow]:
    used: Dict[int, int] = {}
    out: List[JoinedRow] = []
    for k, r in enumerate(causal, start=1):
        sl = r["source_line"]
        if not _is_int(sl) or sl < 1 or sl > n_physical:
            _fail("SOURCE_LINE_OUT_OF_RANGE", "causal row %d: source_line %r outside [1, %d]" % (k, sl, n_physical))
        if sl in used:
            _fail("SOURCE_LINE_DUPLICATE", "causal rows %d and %d both reference Phase 3 physical line %d" % (used[sl], k, sl))
        used[sl] = k
        rec = phase3[sl]
        if rec is None:
            _fail("SOURCE_LINE_BLANK", "causal row %d: source_line %d is a blank line in the Phase 3 file" % (k, sl))
        for f in ("track_id", "frame_index", "timestamp"):
            a, b = rec.get(f), r[f]
            same = (a == b and type(a) is type(b)) if f != "timestamp" else (_finite(a) and float(a) == float(b))
            if not same:
                _fail("SOURCE_LINE_IDENTITY_MISMATCH", "causal row %d (source_line %d): %s %r != Phase 3 %r" % (k, sl, f, b, a))
        bb = rec.get("bbox_xyxy")
        if not (isinstance(bb, list) and len(bb) == 4 and all(_finite(x) for x in bb)):
            _fail("BBOX_NON_FINITE", "Phase 3 line %d: bbox_xyxy %r is not four finite numbers" % (sl, bb))
        if not (bb[3] - bb[1] > 0) or not (bb[2] - bb[0] > 0):
            _fail("BBOX_NON_POSITIVE", "Phase 3 line %d: bbox_xyxy %r has non-positive height or width" % (sl, bb))
        out.append(JoinedRow(sl, r["track_id"], r["sequence_id"], r["frame_index"], float(r["timestamp"]), r.get("agent_class"),
                             r["observation_type"], r["position_state"], r["velocity_state"], r["heading_state"],
                             r["position_x"], r["position_y"], r["velocity_x"], r["velocity_y"], r["speed"], r["heading"],
                             r["bbox_touches_border"], r["frame_gap_before"], r["coordinate_system"],
                             (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])), rec.get("source_label")))
    return out


def check_frames_and_nodes(rows: List[JoinedRow], frame_step: int) -> Dict[str, Any]:
    ts: Dict[int, float] = {}
    for r in rows:
        if r.frame_index in ts and ts[r.frame_index] != r.timestamp:
            _fail("FRAME_TIMESTAMP_INCONSISTENT", "frame %d has timestamps %r and %r" % (r.frame_index, ts[r.frame_index], r.timestamp))
        ts.setdefault(r.frame_index, r.timestamp)
    for r in rows:
        if r.observation_type == "observed" and r.position_state == "ok":
            if not isinstance(r.bbox_touches_border, bool):
                _fail("BORDER_FLAG_INVALID", "observed row (%s, frame %d) has bbox_touches_border=%r" % (r.track_id, r.frame_index, r.bbox_touches_border))
            if r.agent_class not in NODE_TYPES:
                _fail("UNKNOWN_NODE_TYPE", "observed row (%s, frame %d) has agent_class %r" % (r.track_id, r.frame_index, r.agent_class))
            if not (_finite(r.position_x) and _finite(r.position_y)):
                _fail("POSITION_NON_FINITE", "observed row (%s, frame %d) has a non-finite position" % (r.track_id, r.frame_index))
    # P7: frame_gap_before = (f - f_prev) // frame_step - 1 over the same track's previous ACCEPTED row (observed or invalid);
    # null for the first accepted row and for rejected rows.
    last: Dict[str, int] = {}
    verified = 0
    for r in rows:
        if r.observation_type == "rejected":
            expected = None
        else:
            expected = None if r.track_id not in last else (r.frame_index - last[r.track_id]) // frame_step - 1
            last[r.track_id] = r.frame_index
        if r.frame_gap_before != expected:
            _fail("FRAME_GAP_SEMANTICS", "(%s, frame %d): frame_gap_before %r, recomputed %r" % (r.track_id, r.frame_index, r.frame_gap_before, expected))
        verified += 1
    return {"frames": len(ts), "frame_gap_before_rows_verified": verified}


def load_inputs(causal_path, manifest_path, phase3_path, cfg: InteractionConfig) -> AdapterResult:
    causal_path, manifest_path, phase3_path = Path(causal_path), Path(manifest_path), Path(phase3_path)
    t0 = time.perf_counter()
    m, t2cfg, hashes = verify_task2_manifest(causal_path, manifest_path, phase3_path, cfg)
    t1 = time.perf_counter()
    causal, t_rows = read_causal_rows(causal_path, cfg)
    if causal and hashes["causal_kinematics"]["manifest_rows"] != len(causal):
        _fail("CAUSAL_ROW_COUNT", "causal sidecar has %d rows; Task 2 manifest says %r" % (len(causal), hashes["causal_kinematics"]["manifest_rows"]))
    t2 = time.perf_counter()
    p3, n_phys, n_blank = read_phase3_lines(phase3_path)
    joined = join_rows(causal, p3, n_phys)
    t3 = time.perf_counter()
    fs = t2cfg.get("frame_step")
    if not (_is_int(fs) and fs >= 1):
        _fail("TASK2_CONFIG", "Task 2 frame_step %r is not a positive integer" % fs)
    fr = check_frames_and_nodes(joined, fs)
    t4 = time.perf_counter()
    obs = Counter(r.observation_type for r in joined)
    counts = {"causal_rows": len(joined), "phase3_physical_lines": n_phys, "phase3_blank_lines": n_blank,
              "phase3_lines_referenced": len(joined), "phase3_lines_not_referenced": n_phys - n_blank - len(joined),
              "observation_types": dict(sorted(obs.items())),
              "invalid_rows_by_position_state": dict(sorted(Counter(r.position_state for r in joined if r.observation_type == "invalid").items())),
              "rejected_rows_by_state": dict(sorted(Counter(r.velocity_state for r in joined if r.observation_type == "rejected").items())),
              "task2_quarantined_rows": (m.get("counts") or {}).get("quarantined"),
              "source_line_base": "one-based physical line numbers (verified by identity match on every row)",
              **fr}
    return AdapterResult(joined, hashes, m, t2cfg, counts, {"manifest_and_hashes": t1 - t0, "causal_rows_read_and_schema_validated": t2 - t1,
                                                              "schema_validation_loop": t_rows, "phase3_read_and_join": t3 - t2,
                                                              "frame_and_node_checks": t4 - t3})


__all__ = ["ADAPTER_VERSION", "JoinedRow", "AdapterResult", "load_inputs", "verify_task2_manifest", "read_causal_rows",
           "read_phase3_lines", "join_rows", "check_frames_and_nodes"]
