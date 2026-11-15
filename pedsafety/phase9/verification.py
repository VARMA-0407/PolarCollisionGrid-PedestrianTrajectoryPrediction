"""Post-run verification of a Phase 9 run against the committed reference artifacts, and input-immutability hashing.

Everything here is read-only. Comparisons use EXACT equality (hashes, ``np.array_equal`` with NaN equal, list equality). The deterministic contracts that justify exactness:
the Phase 4 / 5 stages are pure functions of their inputs (data files compared by sha256), inference runs with one torch thread on CPU (bitwise reproducible, verified by the Phase 5
determinism rerun), and the Phase 6-8 stages are numpy / pure python. Only the free-form ``provenance`` block of an event legitimately differs between runs (it names the producing
stage), and it is reported separately instead of being ignored silently.
"""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .contracts import sha256_file


def _p(repo: Path, x) -> Path:
    q = Path(str(x).replace("\\", "/"))
    return q if q.is_absolute() else repo / q


def hash_inputs(cfg_dict: Mapping[str, Any], repo: Path) -> Dict[str, str]:
    """sha256 of every input artifact the pipeline reads (Phase 3 files, reference manifests and the data files consumed from them)."""
    files: List[Path] = [_p(repo, cfg_dict["phase3"]["dir"]) / n for n in ("tracked_agents.jsonl", "agent_states.jsonl", "run_manifest.json")]
    r = {k: _p(repo, v) for k, v in cfg_dict["reference_runs"].items()}
    for key, names in (("task2_dir", ["run_manifest.json", "causal_kinematics.jsonl"]), ("task3_dir", ["run_manifest.json"]),
                       ("windows_dir", ["run_manifest.json", "windows.npz", "graph_bank.npz", "rejected_candidates.npz"]),
                       ("training_dir", ["run_manifest.json", "scaler_statistics.json", "config_resolved.json"]),
                       ("neural_eval_dir", ["run_manifest.json", "predictions_neural.npz"]), ("phase7_dir", ["risk_arrays.npz", "risk_report.json"]),
                       ("phase8_dir", ["alert_events.jsonl.gz", "alerts.json", "phase7_records_input.jsonl.gz", "run_manifest.json"])):
        if key in r:
            files += [r[key] / n for n in names]
    if "training_dir" in r and (r["training_dir"] / "checkpoints").is_dir():
        files += sorted((r["training_dir"] / "checkpoints").glob("*.pt"))
    files += [_p(repo, v) for k, v in cfg_dict["configs"].items() if v]
    out = {}
    for f in files:
        if f.is_file():
            try:
                out[f.relative_to(repo).as_posix()] = sha256_file(f)
            except ValueError:
                out[str(f)] = sha256_file(f)
    return out


def _read_jsonl_gz(p: Path) -> List[Dict[str, Any]]:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _npz_equal(a: Path, b: Path) -> Dict[str, Any]:
    x, y = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    keys_equal = sorted(x.files) == sorted(y.files)
    bad = [k for k in x.files if k in y.files and not np.array_equal(x[k], y[k], equal_nan=(x[k].dtype.kind == "f"))]
    return {"same_keys": keys_equal, "n_arrays": len(x.files), "unequal_arrays": bad, "equal": keys_equal and not bad}


def compare_run_with_references(run_dir, cfg_dict: Mapping[str, Any], repo: Path) -> Dict[str, Any]:
    """Compare the outputs present in ``run_dir`` with the committed Phase 4-8 reference artifacts; stages absent from the run are reported as not compared."""
    run, ref = Path(run_dir), {k: _p(repo, v) for k, v in cfg_dict["reference_runs"].items()}
    out: Dict[str, Any] = {}
    s5b = run / "S5B_forecast" / "predictions_neural.npz"
    if s5b.is_file():
        out["phase5_predictions_npz"] = {"sha256_equal": sha256_file(s5b) == sha256_file(ref["neural_eval_dir"] / "predictions_neural.npz"),
                                         "arrays": _npz_equal(s5b, ref["neural_eval_dir"] / "predictions_neural.npz")}
    s5a = run / "S5A_windows"
    if s5a.is_dir():
        out["phase5_windows"] = {n: _npz_equal(s5a / n, ref["windows_dir"] / n) for n in ("windows.npz", "graph_bank.npz", "rejected_candidates.npz")}
    s8 = run / "S8_alerts"
    if s8.is_dir() and "phase8_dir" in ref:
        r8 = ref["phase8_dir"]
        mine_recs, ref_recs = _read_jsonl_gz(s8 / "phase7_records_input.jsonl.gz"), _read_jsonl_gz(r8 / "phase7_records_input.jsonl.gz")
        out["phase7_records"] = {"n": len(mine_recs), "n_reference": len(ref_recs), "exactly_equal_lists": mine_recs == ref_recs,
                                 "covers": "pedestrian/vehicle ids, anchor frames, window ids, risk scores, risk categories, category status, masks, probabilities (all null), reasons"}
        arr = np.load(ref["phase7_dir"] / "risk_arrays.npz")
        rs = np.array([np.nan if r["risk_score"] is None else r["risk_score"] for r in mine_recs])
        out["phase7_arrays_vs_records"] = {"risk_score_equal": bool(np.array_equal(arr["risk_score"], rs, equal_nan=True)),
                                           "risk_category_equal": bool(np.array_equal(arr["risk_category"], np.array([r["risk_category"] for r in mine_recs]))),
                                           "anchor_frame_equal": bool(np.array_equal(arr["anchor_frame"], np.array([r["anchor_frame"] for r in mine_recs]))),
                                           "window_id_equal": bool(np.array_equal(arr["window_id"], np.array([r["window_id"] for r in mine_recs]))),
                                           "probabilities_all_null": all(r["calibrated_probability"] is None for r in mine_recs),
                                           "calibration_status_all_not_calibrated": all(r["calibration_status"] == "NOT_CALIBRATED" for r in mine_recs)}
        me, re = _read_jsonl_gz(s8 / "alert_events.jsonl.gz"), _read_jsonl_gz(r8 / "alert_events.jsonl.gz")
        strip = lambda ev: [{k: v for k, v in e.items() if k != "provenance"} for e in ev]                    # noqa: E731
        prov_diff = sorted({k for a, b in zip(me, re) for k in set(a["provenance"]) | set(b["provenance"]) if a["provenance"].get(k) != b["provenance"].get(k)})
        out["phase8_events"] = {"n": len(me), "n_reference": len(re), "event_ids_equal_in_order": [e["event_id"] for e in me] == [e["event_id"] for e in re],
                                "alert_ids_equal_in_order": [e["alert_id"] for e in me] == [e["alert_id"] for e in re],
                                "pairs_equal_in_order": [(e["pedestrian_id"], e["vehicle_id"], e["frame"]) for e in me] == [(e["pedestrian_id"], e["vehicle_id"], e["frame"]) for e in re],
                                "transitions_equal_in_order": [(e["from_state"], e["to_state"], e["trigger_reason"]) for e in me] == [(e["from_state"], e["to_state"], e["trigger_reason"]) for e in re],
                                "all_fields_equal_except_provenance": strip(me) == strip(re), "provenance_keys_that_differ": prov_diff,
                                "note": "the free-form provenance block names the producing stage; all other event fields, including hashes and source references, are exactly equal"}
        out["phase8_alert_table"] = {"exactly_equal": json.loads((s8 / "alerts.json").read_text(encoding="utf-8")) == json.loads((r8 / "alerts.json").read_text(encoding="utf-8"))}
    out["all_compared_checks_pass"] = _all_true(out)
    return out


def _all_true(o: Any) -> bool:
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("n", "n_reference", "n_arrays", "provenance_keys_that_differ", "note", "covers", "unequal_arrays", "same_keys") or isinstance(v, (str, int)) and not isinstance(v, bool):
                continue
            if k == "unequal_arrays":
                continue
            if not _all_true(v):
                return False
        return True
    if isinstance(o, bool):
        return o
    return True


def inventory_run(run_dir) -> Dict[str, Any]:
    """sha256 of every output file of a run and the status / hashes of every per-stage manifest."""
    run = Path(run_dir)
    files = {f.relative_to(run).as_posix(): sha256_file(f) for f in sorted(run.rglob("*")) if f.is_file()}
    stages = {}
    for m in sorted(run.glob("*/run_manifest.json")):
        d = json.loads(m.read_text(encoding="utf-8"))
        cfg = d.get("config") or {}
        stages[m.parent.name] = {"manifest_sha256": sha256_file(m), "status": d.get("status"), "task": d.get("task"),
                                 "config_sha256": (cfg.get("sha256_resolved") or cfg.get("sha256") or (d.get("alert_config") or {}).get("sha256")) if isinstance(cfg, dict) else None}
    return {"output_sha256": files, "stage_manifests": stages}


__all__ = ["hash_inputs", "compare_run_with_references", "inventory_run"]
