"""Verify the provenance chain Phase 3 -> Task 2 (causal view) -> Task 3 and the node source-line join.

Read-only. The Task 2 *offline* sidecar is never opened: only its manifest entry is reported, to document that it was
not consumed. Every mismatch raises :class:`WindowContractError` with a stable code.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..image_kinematics_run import sha256_file
from .errors import WindowContractError
from .task3_source import NODE_TYPE_NAMES, Task3Run
from .window_config import WindowConfig

REPO = Path(__file__).resolve().parents[2]
POS_TOL_PX = 1e-3     # float32 storage of float64 positions


def _fail(code: str, msg: str):
    raise WindowContractError(code, msg)


def _resolve(p: str) -> Path:
    q = Path(p.replace("\\", "/"))
    return q if q.is_absolute() else REPO / q


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def verify_upstream(run: Task3Run, cfg: WindowConfig, phase3_manifest: Optional[Path] = None,
                    tracked_agents: Optional[Path] = None, check_causal_rows: bool = True) -> Dict[str, Any]:
    inp = run.manifest["inputs"]
    causal = _resolve(inp["causal_kinematics"]["path"])
    phase3 = _resolve(inp["phase3_agent_states"]["path"])
    t2m = _resolve(inp["task2_manifest"]["path"])
    t2c = _resolve(inp["task2_config_resolved"]["path"])
    schema = _resolve(inp["sidecar_schema"]["path"])
    if causal.name != "causal_kinematics.jsonl" or "offline" in causal.name:
        _fail("NON_CAUSAL_INPUT", "Task 3 consumed %s; only the Task 2 causal sidecar is acceptable" % causal)
    files: Dict[str, Dict[str, Any]] = {}
    for key, path, expect in (("task2_causal_kinematics", causal, inp["causal_kinematics"]["sha256"]),
                              ("phase3_agent_states", phase3, inp["phase3_agent_states"]["sha256"]),
                              ("task2_manifest", t2m, inp["task2_manifest"]["sha256"]),
                              ("task2_config_resolved", t2c, inp["task2_config_resolved"]["sha256"]),
                              ("task2_sidecar_schema", schema, inp["sidecar_schema"]["sha256"])):
        if not path.is_file():
            _fail("UPSTREAM_MISSING_FILE", "%s not found at %s" % (key, path))
        h = sha256_file(path)
        if h != expect:
            _fail("UPSTREAM_HASH_MISMATCH", "%s sha256 %s != Task 3 manifest %s" % (key, h, expect))
        files[key] = {"path": _rel(path), "sha256": h, "bytes": path.stat().st_size, "matches_task3_manifest": True}

    # ----- Task 2 manifest -----
    t2 = json.loads(t2m.read_text(encoding="utf-8"))
    if t2.get("status") != "completed":
        _fail("UPSTREAM_TASK2_STATUS", "Task 2 status %r" % t2.get("status"))
    if t2["outputs"]["causal_kinematics.jsonl"]["sha256"] != files["task2_causal_kinematics"]["sha256"]:
        _fail("UPSTREAM_HASH_MISMATCH", "Task 2 manifest causal sidecar hash differs from the file")
    if t2["input"]["sha256_before"] != files["phase3_agent_states"]["sha256"] or t2["input"]["sha256_after"] != files["phase3_agent_states"]["sha256"]:
        _fail("UPSTREAM_HASH_MISMATCH", "Task 2 input hash differs from the Phase 3 agent_states file")
    for k, v in t2.get("claims", {}).items():
        if isinstance(v, bool) and v:
            _fail("UPSTREAM_CLAIM_TRUE", "Task 2 claim %r is true" % k)
    offline = t2["outputs"].get("offline_kinematics.jsonl")

    # ----- Phase 3 manifest (optional; fps, frame range, record count) -----
    p3info: Dict[str, Any] = {"provided": False}
    if phase3_manifest is not None:
        p3 = json.loads(Path(phase3_manifest).read_text(encoding="utf-8"))
        fs = p3["components"]["frame_source"]["parameters"]
        fps = float(fs["fps"])
        if abs(fps - cfg.source_fps) > 1e-9:
            _fail("SOURCE_FPS_MISMATCH", "Phase 3 fps %r != configured source_fps %r" % (fps, cfg.source_fps))
        if p3.get("is_mock") is not False:
            _fail("UPSTREAM_PHASE3_MOCK", "Phase 3 run is a mock run")
        if p3["agent_states"]["records"] != run.manifest["counts"]["causal_rows"]:
            _fail("UPSTREAM_COUNT_MISMATCH", "Phase 3 agent_states records != Task 3 causal_rows")
        res = p3["results"]
        if not (res["first_frame_index"] <= int(run.frame.min()) and int(run.frame.max()) <= res["last_frame_index"]):
            _fail("UPSTREAM_FRAME_RANGE", "Task 3 frames outside the Phase 3 frame range")
        p3info = {"provided": True, "path": _rel(Path(phase3_manifest)), "sha256": sha256_file(Path(phase3_manifest)),
                  "fps": fps, "fps_origin": fs.get("fps_origin"), "timestamp_source": fs.get("timestamp_source"),
                  "width": int(fs["width"]), "height": int(fs["height"]), "frames_delivered": p3["video_stats"]["frames_delivered"],
                  "first_frame_index": res["first_frame_index"], "last_frame_index": res["last_frame_index"],
                  "detector_label_map": p3["config"]["detector"]["label_map"], "tracker_class_flips": p3.get("tracker_class_flips"),
                  "git_commit": p3.get("git_commit"), "world_coordinates": p3.get("world_coordinates"),
                  "outputs_listed": p3.get("outputs"), "note": "Phase 3 manifest records no per-file output hashes"}
    if tracked_agents is not None:
        files["phase3_tracked_agents"] = {"path": _rel(Path(tracked_agents)), "sha256": sha256_file(Path(tracked_agents)),
                                          "bytes": Path(tracked_agents).stat().st_size, "consumed": False,
                                          "note": "hashed for provenance only; not a Task 3 or Phase 5 input"}

    # ----- source-line join against Phase 3 agent_states -----
    lines = phase3.read_text(encoding="utf-8").splitlines()
    n = run.n_nodes
    label_by_type: Dict[str, Dict[str, int]] = {t: {} for t in NODE_TYPE_NAMES}
    for i in range(n):
        sl = int(run.source_line[i])
        if sl > len(lines):
            _fail("SOURCE_LINE_JOIN", "node %d source_line %d beyond Phase 3 file" % (i, sl))
        r = json.loads(lines[sl - 1])
        tid = run.track_id(int(run.track_index[i]))
        if r.get("track_id") != tid or r.get("frame_index") != int(run.frame[i]):
            _fail("SOURCE_LINE_JOIN", "node %d: Phase 3 line %d is (%r, %r), node is (%r, %r)"
                  % (i, sl, r.get("track_id"), r.get("frame_index"), tid, int(run.frame[i])))
        if r.get("image_coordinate_system") != cfg.coordinate_system or r.get("image_coordinate_unit") != cfg.coordinate_unit:
            _fail("COORDINATE_SYSTEM_MISMATCH", "Phase 3 line %d coordinate metadata %r/%r" % (sl, r.get("image_coordinate_system"), r.get("image_coordinate_unit")))
        if r.get("world_position_valid") is not False or r.get("world_x") is not None or r.get("world_y") is not None:
            _fail("WORLD_COORDINATE_PRESENT", "Phase 3 line %d carries world coordinates" % sl)
        if r.get("interpolation_flag") is not False:
            _fail("INTERPOLATED_INPUT", "Phase 3 line %d is interpolated" % sl)
        fx, fy = r["footpoint"]
        if abs(fx - float(run.features[i, 0])) > POS_TOL_PX or abs(fy - float(run.features[i, 1])) > POS_TOL_PX:
            _fail("SOURCE_LINE_JOIN", "node %d position differs from the Phase 3 footpoint" % i)
        d = label_by_type[NODE_TYPE_NAMES[int(run.node_type[i])]]
        d[str(r.get("source_label"))] = d.get(str(r.get("source_label")), 0) + 1

    # ----- causal sidecar rows behind the nodes -----
    causal_info: Dict[str, Any] = {"checked": False}
    if check_causal_rows:
        node_by_line = {int(s): i for i, s in enumerate(run.source_line.tolist())}
        seen = 0
        with open(causal, encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                r = json.loads(raw)
                if r.get("view") != "CAUSAL":
                    _fail("NON_CAUSAL_INPUT", "Task 2 sidecar row with view %r" % r.get("view"))
                i = node_by_line.get(int(r["source_line"]))
                if i is None:
                    continue
                seen += 1
                if r.get("interpolated") is not False or r.get("world_position_valid") is not False or r.get("coordinate_system") != cfg.coordinate_system:
                    _fail("CAUSAL_ROW_CONTRACT", "causal row for source_line %d violates interpolation/world/coordinate checks" % r["source_line"])
                if r.get("filter_kind") != "none":
                    _fail("CAUSAL_ROW_CONTRACT", "causal row for source_line %d is filtered (%r)" % (r["source_line"], r.get("filter_kind")))
                if abs(r["position_x"] - float(run.features[i, 0])) > POS_TOL_PX or abs(r["position_y"] - float(run.features[i, 1])) > POS_TOL_PX:
                    _fail("CAUSAL_ROW_CONTRACT", "causal row position differs from node %d" % i)
                vel_ok = r.get("velocity_state") == "valid"
                if vel_ok != bool(run.masks[i, 1]):
                    _fail("CAUSAL_ROW_CONTRACT", "velocity validity differs between causal row and node %d" % i)
                if vel_ok and (abs(r["velocity_x"] - float(run.features[i, 2])) > 1e-2 or abs(r["velocity_y"] - float(run.features[i, 3])) > 1e-2):
                    _fail("CAUSAL_ROW_CONTRACT", "causal velocity differs from node %d" % i)
        if seen != n:
            _fail("CAUSAL_ROW_CONTRACT", "%d of %d nodes found in the causal sidecar" % (seen, n))
        causal_info = {"checked": True, "rows_matched_to_nodes": seen, "all_rows_view_causal": True,
                       "checks": "view == CAUSAL, interpolated false, world_position_valid false, coordinate system, filter_kind none, "
                                 "position and velocity equal to the node within float32 tolerance, velocity validity equal to the node mask"}

    return {"files": files, "phase3_manifest": p3info, "task2": {
                "manifest_status": t2["status"], "config_version": t2["config"].get("config_version"), "git": t2.get("git"),
                "offline_sidecar_consumed": False,
                "offline_sidecar_manifest_entry": offline,
                "note": "the offline (non-causal) sidecar is never opened by Task 3 or Phase 5"},
            "source_line_join": {"nodes_checked": n, "all_match": True,
                                 "checks": "Phase 3 line track_id/frame_index equal node identity; coordinate system/unit; world fields null; "
                                           "interpolation_flag false; footpoint equal to node (u, v)"},
            "source_labels_by_node_type": label_by_type,
            "causal_rows": causal_info}


__all__ = ["verify_upstream"]
