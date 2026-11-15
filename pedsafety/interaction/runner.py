"""T3.5: Task 3 runner (inputs -> validated nodes/edges -> deterministic files -> manifest last).

Output policy: the output directory must be absent or empty (shared ``output_dir`` rule); nothing is overwritten or
deleted; each file is written to a temporary name and renamed; ``run_manifest.json`` is written last, does not hash
itself, and is the completion marker. All inputs are re-hashed after processing and must be unchanged.
"""

import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from jsonschema import Draft202012Validator

from . import INTERACTION_VERSION
from ..image_kinematics_run import sha256_file
from ..output_dir import check_output_dir
from .config import NODE_TYPES, InteractionConfig, InteractionContractError
from .diagnostics import build_diagnostics
from .feature_dictionary import build_feature_dictionary
from .input_adapter import ADAPTER_VERSION, load_inputs
from .nodes import NODE_BOX_COLUMNS, NODE_FLOAT_FEATURES, NODE_MASKS, build_node_table
from .pairwise import EDGE_FEATURES, build_edges, flag_names
from .storage import write_json, write_npz, write_text

MANIFEST_VERSION = "phase4-task3-run-manifest-1.0.0"
REPO = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO / "docs" / "phase4" / "schemas"
DECISION_RECORD = REPO / "docs" / "phase4" / "PHASE4_TASK3_DECISION_RECORD.md"
DESIGN_DOCUMENT = REPO / "docs" / "phase4" / "PHASE4_TASK3_INTERACTION_REPRESENTATION_DESIGN.md"

CLAIMS = {"coordinate_space": "image (captured raster pixels)", "physical_features": False, "metric_world_coordinates": False,
          "world_position_valid": False, "ttc_pet_or_min_separation": False, "collision_or_conflict_probability": False,
          "ego_motion_compensated": False, "learned_components": False, "real_time_claim": False,
          "note": "Pixel separations, relative image velocities and closing rates are image-plane quantities affected by perspective and by "
                  "moving-camera ego-motion; box-height normalisation is a heuristic proxy, not calibrated depth."}
LIMITATIONS = [
    "Image space only (captured_1920x1080 pixels, px/s, s, image-plane radians); no metric geometry is authorised (Phase 4 Task 1).",
    "Relative image velocity and closing rate include depth-dependent ego-motion parallax and perspective; not physical relative motion.",
    "sep_over_h and log_h_ratio use box height as a heuristic scale proxy; affected by truncation (flagged), pose and class size differences.",
    "Node velocities inherit Task 2 per-frame jitter (~48 px/s per axis per agent, single-stride estimate); no smoothing applied (Q7).",
    "Missing agents are not interpolated or carried forward; track fragments are not merged; class-flip holes from Phase 3 remain.",
    "Radius and kNN flags are exploratory sensitivity diagnostics with unvalidated values (P5); they never remove edges.",
    "Bicyclist->bicyclist pairs are not represented: the approved relation list has no such relation (counted in diagnostics).",
    "Runtime is offline preprocessing time on one machine; it is not evidence of real-time performance.",
]
DEVIATIONS = [
    "dv_row_px is numerically identical to dv_px (both v_j - v_i) as specified in the design; kept for semantic labelling.",
    "same_track_family (design §6 table, 'always false in Task 3') is not stored because it is constant; recorded under excluded.",
    "Bicyclist->bicyclist pairs are neither approved nor mentioned in the design; conservatively not represented and counted.",
    "P10 applied conservatively: non-positive bbox WIDTH is also treated as a contract failure (positive height is the approved minimum).",
    "Any malformed non-blank line in the Phase 3 file fails the run, even if no causal row references it (conservative reading of P13(c)).",
]


def peak_memory_bytes() -> Optional[int]:
    """Peak working set of this process (Windows, via psapi) or peak RSS (POSIX); None if unavailable."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE          # explicit types: avoid 64-bit handle truncation
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return int(pmc.PeakWorkingSetSize) if ok else None
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r * 1024) if sys.platform != "darwin" else int(r)
    except Exception:                                                  # noqa: BLE001 - measurement is best effort
        return None


def _git() -> Dict[str, Optional[str]]:
    def run(*a):
        try:
            o = subprocess.run(["git"] + list(a), cwd=str(REPO), capture_output=True, text=True, timeout=10)
            return o.stdout.strip() if o.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    return {"branch": run("branch", "--show-current"), "commit": run("rev-parse", "HEAD")}


def _validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _require_valid(validator: Draft202012Validator, obj: Any, what: str) -> None:
    err = next(validator.iter_errors(obj), None)
    if err is not None:
        raise InteractionContractError("OUTPUT_SCHEMA_VIOLATION", "%s violates its schema: %s at %s" % (what, err.message, list(err.path)))


def run_interaction_representation(causal_path, task2_manifest_path, phase3_path, cfg: InteractionConfig, out_dir, *,
                                   config_path: Optional[str] = None, command: Optional[str] = None) -> Dict[str, Any]:
    out = check_output_dir(out_dir)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    res = load_inputs(causal_path, task2_manifest_path, phase3_path, cfg)
    t1 = time.perf_counter()
    nt = build_node_table(res.rows)
    t2 = time.perf_counter()
    edges, not_rep = build_edges(nt, cfg)
    t3 = time.perf_counter()
    diag = build_diagnostics(nt, edges, not_rep, cfg, res.counts)
    fdict = build_feature_dictionary(cfg)
    _require_valid(_validator("interaction_feature_dictionary.schema.json"), fdict, "feature dictionary")
    t4 = time.perf_counter()

    out.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Dict[str, Any]] = {}

    def record(name: str, sha: str):
        files[name] = {"sha256": sha, "bytes": (out / name).stat().st_size}

    record("nodes.npz", write_npz(out / "nodes.npz", {
        "frame": nt.frame, "timestamp": nt.timestamp, "track_index": nt.track_index, "node_type": nt.node_type, "source_line": nt.source_line,
        "features": nt.features, "feature_names": np.array(NODE_FLOAT_FEATURES), "boxes": nt.boxes, "box_names": np.array(NODE_BOX_COLUMNS),
        "border": nt.border, "masks": nt.masks, "mask_names": np.array(NODE_MASKS), "node_type_names": np.array(NODE_TYPES)}))
    fn = np.array(flag_names(cfg))
    for name, e in edges.items():
        record("edges_%s.npz" % name, write_npz(out / ("edges_%s.npz" % name), {
            "relation": np.array(e.relation), "frame": e.frame, "edge_index": e.edge_index, "edge_attr": e.edge_attr, "edge_mask": e.edge_mask,
            "flags": e.flags, "feature_names": np.array(EDGE_FEATURES), "flag_names": fn}))
    tracks = []
    for ti, tid in enumerate(nt.track_ids):
        sel = nt.track_index == ti
        fr = nt.frame[sel]
        tracks.append({"track_index": ti, "track_id": tid, "node_type": NODE_TYPES[int(nt.node_type[sel][0])],
                       "source_labels": dict(sorted(nt.track_source_labels[ti].items())), "n_nodes": int(sel.sum()),
                       "first_frame": int(fr.min()), "last_frame": int(fr.max())})
    record("tracks.json", write_json(out / "tracks.json", {"tracks": tracks}))
    record("feature_dictionary.json", write_json(out / "feature_dictionary.json", fdict))
    record("interaction_diagnostics.json", write_json(out / "interaction_diagnostics.json", diag))
    record("config_resolved.json", write_json(out / "config_resolved.json", cfg.to_dict()))
    t5 = time.perf_counter()
    after = {"causal_kinematics": sha256_file(Path(causal_path)), "phase3_agent_states": sha256_file(Path(phase3_path)),
             "task2_manifest": sha256_file(Path(task2_manifest_path)), "task2_config_resolved": sha256_file(Path(task2_manifest_path).parent / "config_resolved.json")}
    changed = [k for k, h in after.items() if res.input_hashes[k]["sha256"] != h]
    if changed:
        raise InteractionContractError("INPUT_CHANGED_DURING_RUN", "inputs changed during processing: %s" % changed)
    counts = {"causal_rows": res.counts["causal_rows"], "frames_with_nodes": diag["frames_with_nodes"], "nodes": nt.n, "tracks": len(nt.track_ids),
              "nodes_by_type": diag["nodes_by_type"], "edges_total": diag["edges_total"],
              "edges_by_relation": {k: v["edges"] for k, v in diag["relations"].items()},
              "candidate_pairs_not_represented": not_rep}
    summary = ["# Phase 4 Task 3 interaction representation run", "",
               "Image space only (%s; px, px/s, s, image-plane rad). Every feature physical: false. Not a safety measure." % cfg.expected_coordinate_system, "",
               "| Item | Value |", "|---|---|", "| causal rows | %d |" % counts["causal_rows"], "| frames with nodes | %d |" % counts["frames_with_nodes"],
               "| nodes | %d (%s) |" % (nt.n, ", ".join("%s %d" % kv for kv in counts["nodes_by_type"].items())),
               "| edges | %d |" % counts["edges_total"]] + ["| edges %s | %d |" % kv for kv in counts["edges_by_relation"].items()] + \
              ["| pairs not represented | %s |" % (", ".join("%s %d" % kv for kv in not_rep.items()) or "none"), "",
               "All edges equal the complete-graph bound for their relation: %s." % all(v["equals_theoretical"] for v in diag["relations"].values()), "",
               "See interaction_diagnostics.json, feature_dictionary.json and run_manifest.json."]
    record("interaction_summary.md", write_text(out / "interaction_summary.md", "\n".join(summary) + "\n"))
    finished = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        from importlib.metadata import version as _v
        jsv = _v("jsonschema")
    except Exception:                                                  # noqa: BLE001
        jsv = None
    manifest = {
        "manifest_version": MANIFEST_VERSION, "task": "phase4_task3_image_space_interaction_representation", "status": "completed",
        "versions": {"interaction": INTERACTION_VERSION, "input_adapter": ADAPTER_VERSION, "python": platform.python_version(),
                     "numpy": np.__version__, "jsonschema": jsv, "platform": platform.platform()},
        "git": _git(), "started_utc": started, "finished_utc": finished, "command": command,
        "inputs": dict(res.input_hashes, unchanged_after_run=not changed),
        "config": {"path": config_path, "sha256_resolved": cfg.sha256(), "content": cfg.to_dict()},
        "decision_record": {"path": str(DECISION_RECORD.relative_to(REPO)), "sha256": sha256_file(DECISION_RECORD) if DECISION_RECORD.is_file() else None},
        "design_document": {"path": str(DESIGN_DOCUMENT.relative_to(REPO)), "sha256": sha256_file(DESIGN_DOCUMENT) if DESIGN_DOCUMENT.is_file() else None},
        "conventions": fdict["conventions"] | {"coordinate_system": cfg.expected_coordinate_system, "relations": fdict["relations"]},
        "counts": counts,
        "validation": {"all_input_contracts_passed": True, "checks": [
            "Task 2 manifest identity, hashes (causal, Phase 3 input before/after, sidecar schema, config_resolved)",
            "Task 2 config: max_derivative_gap_frames == 0, smoothing none, raster matches",
            "every causal row: full sidecar-schema validation + explicit view/method/space/system/unit/world checks",
            "source_line: one-based physical line, in range, unique, non-blank, JSON object, track_id/frame_index/timestamp identity",
            "Phase 3 bbox: four finite numbers, positive height and width",
            "one timestamp per frame; observed rows have boolean bbox_touches_border and a known node type",
            "frame_gap_before recomputed and equal on every row (P7)",
            "feature dictionary validated against its schema; edges equal the complete-graph bound per relation"],
            "adapter_counts": res.counts, "edges_equal_theoretical_bound": all(v["equals_theoretical"] for v in diag["relations"].values())},
        "runtime_s": {"input_validation_total": t1 - t0, **{"input_" + k: v for k, v in res.timings_s.items()},
                      "node_table": t2 - t1, "edges": t3 - t2, "diagnostics_and_dictionary": t4 - t3, "writing_outputs": t5 - t4,
                      "total_before_manifest": time.perf_counter() - t0,
                      "note": "offline preprocessing time on one machine; not a real-time claim"},
        "peak_memory_bytes": peak_memory_bytes(),
        "outputs": dict(sorted(files.items())), "output_total_bytes": int(sum(f["bytes"] for f in files.values())),
        "claims": CLAIMS, "limitations": LIMITATIONS, "deviations_and_notes": DEVIATIONS,
    }
    _require_valid(_validator("interaction_run_manifest.schema.json"), manifest, "run manifest")
    write_json(out / "run_manifest.json", manifest)
    return manifest


__all__ = ["run_interaction_representation", "MANIFEST_VERSION", "CLAIMS", "LIMITATIONS", "DEVIATIONS", "peak_memory_bytes"]
