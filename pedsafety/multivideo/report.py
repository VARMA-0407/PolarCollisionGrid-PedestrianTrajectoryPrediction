"""Experiment report (JSON + Markdown) and final validation.

The final validation never reports success by default: the overall status is ``COMPLETED`` only if every required stage completed for
every video of the frozen split, every required file exists (and matches its recorded SHA-256 when full verification is enabled), the
schemas validate, the split is intact and the checkpoint verifies. Otherwise it is ``INCOMPLETE`` (something was not run / skipped) or
``FAILED`` (a stage or check failed). Scientific items that are blocked by missing evidence are listed separately and are never presented
as completed.
"""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import COORDINATE_SYSTEM, MULTIVIDEO_LABEL, TARGET_SOURCE
from .common import MARKER, check_marker, read_json, sha256_file, utc_now, write_json

BLOCKED_SCIENTIFIC_ITEMS = {
    "annotation_ground_truth_trajectory_evaluation": "BLOCKED: ANNOTATED_TARGET adapter (verified PIE-id <-> track-id mapping and frame alignment) is not implemented",
    "independent_evaluation_phase9_gate": "BLOCKED unless pedsafety.phase9.evaluation reports READY (annotation provenance, identity mapping, timestamp alignment, licence)",
    "phase6_supervised_conflict_training": "BLOCKED: no verified conflict labels",
    "phase7_probability_calibration": "BLOCKED: no verified conflict labels (calibrated_probability = null, NOT_CALIBRATED)",
    "phase7_category_validation": "BLOCKED: categories are PROVISIONAL engineering thresholds",
    "phase8_lead_time_and_false_alarm_evaluation": "BLOCKED: no ground-truth event timestamps",
    "metric_geometry_ttc_pet_distance": "BLOCKED: image-space pixels only; no validated calibration / homography / ego-motion compensation",
    "real_time_claim": "NOT SUPPORTED: offline batch replay; the window builder needs future frames (no causal online mode); stages are timed separately",
}


def _sample_jsonl(path: Path, every: int, first: int):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip() and (i < first or i % every == 0):
                yield i, json.loads(line)


def validate_schemas(phase3_dir: Path, alerts_dir: Optional[Path], risk_dir: Optional[Path], *, sample_every: int = 200, sample_first: int = 2000) -> Dict[str, Any]:
    from ..schema import collect_violations
    out: Dict[str, Any] = {}
    bad = []
    n = 0
    for i, r in _sample_jsonl(phase3_dir / "agent_states.jsonl", sample_every, sample_first):
        n += 1
        v = collect_violations(r)
        if v:
            bad.append({"line": i, "violations": [str(x) for x in v[:3]]})
    out["agent_states_schema"] = {"rows_checked": n, "rows_invalid": len(bad), "examples": bad[:5], "sampling": "first %d rows + every %d-th row" % (sample_first, sample_every)}
    if risk_dir is not None and (risk_dir / "risk_records.jsonl.gz").is_file():
        nn, badr = 0, 0
        for _, r in _sample_jsonl(risk_dir / "risk_records.jsonl.gz", 1, 0):
            nn += 1
            badr += int(r.get("calibrated_probability") is not None or r.get("calibration_status") == "CALIBRATED" or "risk_category" not in r)
        out["risk_records"] = {"rows_checked": nn, "rows_invalid": badr, "rule": "calibrated_probability must be null and status not CALIBRATED; risk_category present"}
    if alerts_dir is not None and (alerts_dir / "alert_events.jsonl.gz").is_file():
        from ..phase8.schema import validate_event
        ne, bade = 0, []
        for i, e in _sample_jsonl(alerts_dir / "alert_events.jsonl.gz", 1, 0):
            ne += 1
            try:
                validate_event(e)
            except Exception as exc:                                                                  # noqa: BLE001
                bade.append({"line": i, "error": str(exc)[:200]})
        out["alert_events_schema"] = {"events_checked": ne, "events_invalid": len(bade), "examples": bade[:5]}
    out["ok"] = (out["agent_states_schema"]["rows_invalid"] == 0 and out.get("risk_records", {}).get("rows_invalid", 0) == 0
                 and out.get("alert_events_schema", {}).get("events_invalid", 0) == 0)
    return out


def marker_table(stage_dirs: Dict[str, Path], videos: Dict[str, str], *, verify: str = "size") -> Dict[str, Dict[str, str]]:
    """``stage_dirs``: {stage_name: root}; ``videos``: {key: set}. Returns {stage: {key: marker state}} (config hash not checked here: any
    STALE marker is caught when the stage runs; here only completeness and file integrity are checked)."""
    table: Dict[str, Dict[str, str]] = {}
    for stage, root in stage_dirs.items():
        table[stage] = {}
        for key, s in videos.items():
            d = Path(root) / s / key
            if not (d / MARKER).is_file():
                table[stage][key] = "ABSENT" if not d.exists() else "INCOMPLETE"
                continue
            m = read_json(d / MARKER)
            st, _ = check_marker(d, stage=m.get("stage", stage), config_sha256=m.get("config_sha256"), verify=verify)
            table[stage][key] = st
    return table


def realtime_statement(phase3_details: List[Dict[str, Any]], fps_target: float = 30.0) -> Dict[str, Any]:
    from ..phase9.benchmark import realtime_assessment
    frames = sum(int(d.get("frames_processed") or 0) for d in phase3_details)
    wall = sum(float(d.get("wall_time_s_perception_run") or 0) for d in phase3_details)
    ra = realtime_assessment(fps_target=fps_target, frames=frames, measured_seconds=wall, perception_executed=bool(frames), streaming_capable=False)
    ra["scope"] = "Phase 3 perception only (decode + YOLO + ByteTrack + JSONL writing), offline, per video; Phases 4-8 are timed separately and are batch replays"
    ra["claim_supported"] = False
    return ra


def build_report(*, experiment_id: str, exp_dir: Path, lock: Dict[str, Any], inventory: Dict[str, Any], split: Dict[str, Any], tracker_data: Dict[str, Any],
                 environment: Dict[str, Any], git: Dict[str, Any], markers: Dict[str, Dict[str, str]], phase3: Dict[str, Any], training: Optional[Dict[str, Any]],
                 evaluation: Optional[Dict[str, Any]], downstream: Dict[str, Any], tests: Optional[Dict[str, Any]], validation: Optional[Dict[str, Any]],
                 execution: Dict[str, Any], deviations: List[str]) -> Dict[str, Any]:
    p3d = [r.get("detail", {}) for r in (phase3.get("results") or {}).values() if r.get("status") in ("COMPLETED", "SKIPPED_ALREADY_COMPLETE", "REUSED_VERIFIED")]
    cats: Dict[str, int] = {}
    alerts = {"n_alerts": 0, "n_events": 0, "alerts_escalated": 0, "alerts_by_clearance_kind": {}}
    pairs = {"pairs": 0, "pairs_valid": 0}
    for key, r in (downstream.get("results") or {}).items():
        st = r.get("stages", {})
        for c, n in (st.get("S7_RISK", {}).get("detail", {}).get("category_counts") or {}).items():
            cats[c] = cats.get(c, 0) + n
        a = st.get("S8_ALERTS", {}).get("detail", {})
        for k in ("n_alerts", "n_events", "alerts_escalated"):
            alerts[k] += int(a.get(k) or 0)
        for k, n in (a.get("alerts_by_clearance_kind") or {}).items():
            alerts["alerts_by_clearance_kind"][k] = alerts["alerts_by_clearance_kind"].get(k, 0) + n
        c6 = st.get("S6_CONFLICT", {}).get("detail", {})
        pairs["pairs"] += int(c6.get("pairs") or 0)
        pairs["pairs_valid"] += int(c6.get("pairs_valid") or 0)
    stage_status = {k: {kk: v.get(kk) for kk in ("status", "runtime_s", "error_code", "error", "reason", "detail_summary") if v.get(kk) is not None}
                    for k, v in tracker_data["stages"].items()}
    rep = {"report_version": "multivideo-experiment-report-1.0.0", "experiment_id": experiment_id, "created_utc": utc_now(), "label": MULTIVIDEO_LABEL,
           "target_source": TARGET_SOURCE, "coordinate_system": COORDINATE_SYSTEM, "git": git, "environment": environment, "locked_config": lock.get("locked_config"),
           "dataset": {"status": inventory.get("status"), "n_videos": inventory.get("n_videos"), "total_gb": inventory.get("total_gb"),
                       "videos_per_set": {s: v.get("n_videos") for s, v in inventory.get("sets", {}).items()}, "problems": inventory.get("problems"),
                       "warnings": inventory.get("warnings"), "videos_with_annotation_xml": inventory.get("annotations", {}).get("n_videos_with_annt_xml")},
           "split": {"split_id": split.get("split_id"), "scheme": split.get("scheme"), "partitions": split.get("partitions"),
                     "official_split_status": split.get("official_split", {}).get("status"), "deviations": split.get("deviations_from_official"),
                     "videos_per_partition": {p: sum(len(split["groups"][s]["videos"]) for s in sets) for p, sets in split.get("partitions", {}).items()}},
           "execution": execution, "stage_status": stage_status, "per_video_markers": markers,
           "phase3": {"by_status": phase3.get("by_status"), "failed": phase3.get("failed"), "frames_processed_total": sum(int(d.get("frames_processed") or 0) for d in p3d),
                      "track_rows_total": sum(int(d.get("track_rows") or 0) for d in p3d),
                      "pie_annotation_evaluations_completed": sum(1 for d in p3d if (d.get("pie_annotation_evaluation") or {}).get("status") == "COMPLETED")},
           "training": None if not training else {k: training.get(k) for k in ("status", "training", "inputs", "run_metadata", "deviations_from_single_video_harness")},
           "evaluation": None if not evaluation else {"per_partition": evaluation.get("per_partition"), "metric_policy": evaluation.get("metric_policy"),
                                                       "phase9_gate": evaluation.get("phase9_independent_evaluation_gate"), "interpretation": evaluation.get("interpretation")},
           "conflict": pairs, "risk_category_distribution": cats, "alerts": alerts,
           "failures": {"per_stage_failed_videos": {k: v.get("failed") for k, v in (("phase3", phase3), ("downstream", downstream)) if v}},
           "realtime": realtime_statement(p3d), "tests": tests, "final_validation": validation, "blocked_scientific_items": BLOCKED_SCIENTIFIC_ITEMS,
           "deviations_from_original_implementation": deviations,
           "claims": {"independent_generalization": False, "annotation_ground_truth_metrics": False, "calibrated_probability": False, "physical_ttc_pet": False,
                      "validated_early_warning_time": False, "real_time_performance": False, "deployment_ready": False}}
    write_json(exp_dir / "experiment_report.json", rep)
    (exp_dir / "EXPERIMENT_REPORT.md").write_text(render_markdown(rep), encoding="utf-8")
    return rep


def render_markdown(rep: Dict[str, Any]) -> str:
    f = lambda x: "—" if x is None else ("%.2f" % x if isinstance(x, float) else str(x))               # noqa: E731
    L = ["# PIE multi-video pedestrian-safety experiment `%s`" % rep["experiment_id"], "",
         "Label `%s`. Targets: %s. Coordinates: `%s` pixels only." % (rep["label"], rep["target_source"], rep["coordinate_system"]), "",
         "**Final status: %s**" % ((rep.get("final_validation") or {}).get("overall_status", "NOT_VALIDATED")), "",
         "## Provenance", "", "- Git: `%s` on `%s` (clean working tree: %s)" % (rep["git"].get("commit"), rep["git"].get("branch"), rep["git"].get("working_tree_clean")),
         "- GPU: %s; Python %s; torch %s" % (rep["environment"].get("gpu"), rep["environment"].get("python"), rep["environment"].get("torch")), "",
         "## Dataset and split", "", "- Inventory status: %s; %s videos, %s GB; per set: %s" % (rep["dataset"]["status"], rep["dataset"]["n_videos"], rep["dataset"]["total_gb"],
                                                                                           rep["dataset"]["videos_per_set"]),
         "- Split `%s` (%s, official split %s): %s; videos per partition %s" % (rep["split"]["split_id"], rep["split"]["scheme"], rep["split"]["official_split_status"],
                                                                                rep["split"]["partitions"], rep["split"]["videos_per_partition"]),
         "- Deviations from the official split: %s" % (rep["split"]["deviations"] or "none"), "",
         "## Execution", "", "- %s" % rep["execution"].get("statement"), "", "| stage | status | runtime (s) | note |", "|---|---|---|---|"]
    for k, v in rep["stage_status"].items():
        L.append("| %s | %s | %s | %s |" % (k, v.get("status"), f(v.get("runtime_s")), (v.get("error") or v.get("reason") or "")[:160].replace("|", "/")))
    ev = rep.get("evaluation") or {}
    if ev.get("per_partition"):
        L += ["", "## Trajectory forecasting (pixels; primary = most-probable mode; oracle reported separately)", "",
              "| partition | sample status | videos | ADE | FDE | minADE@K (oracle) | minFDE@K (oracle) |", "|---|---|---|---|---|---|---|"]
        for p in ("train", "validation", "calibration", "test"):
            x = ev["per_partition"].get(p)
            if not x or not x.get("primary_most_probable_mode"):
                continue
            s = x["primary_most_probable_mode"]
            L.append("| %s | %s | %d | %s | %s | %s | %s |" % (p, x["sample_status"], x["n_videos"], f(s["window_weighted"]["ade_px"]["mean"]), f(s["window_weighted"]["fde_px"]["mean"]),
                                                            f(s["oracle_best_of_k"]["min_ade_px"]["mean"]), f(s["oracle_best_of_k"]["min_fde_px"]["mean"])))
        L += ["", "Held-out rows are separate recordings scored against perceived-track pseudo-targets; they are not independent validation against annotations."]
    L += ["", "## Conflict / risk / alerts (descriptive counts only)", "", "- Pairs: %s (valid %s)" % (rep["conflict"]["pairs"], rep["conflict"]["pairs_valid"]),
          "- Risk categories (uncalibrated heuristic, provisional thresholds): %s" % rep["risk_category_distribution"],
          "- Alerts: %s; events: %s; escalated: %s; clearance kinds: %s" % (rep["alerts"]["n_alerts"], rep["alerts"]["n_events"], rep["alerts"]["alerts_escalated"],
                                                                           rep["alerts"]["alerts_by_clearance_kind"]), "",
          "## Real-time", "", "- Claim supported: **no**. Unmet conditions: %s" % "; ".join(rep["realtime"]["unmet_conditions"]),
          "- Measured offline Phase 3 rate: %s frames/s (%s)" % (f(rep["realtime"]["measured_frames_per_s_of_measured_stages"]), rep["realtime"]["scope"]), "",
          "## Blocked scientific items", ""] + ["- **%s**: %s" % (k, v) for k, v in rep["blocked_scientific_items"].items()]
    L += ["", "## Deviations from the original implementation", ""] + ["- %s" % d for d in rep["deviations_from_original_implementation"]]
    t = rep.get("tests") or {}
    L += ["", "## Tests", "", "- %s" % (t.get("summary_line") or "not run")]
    fv = rep.get("final_validation") or {}
    if fv:
        L += ["", "## Final validation", "", "| check | result |", "|---|---|"] + ["| %s | %s |" % (c["name"], c["result"]) for c in fv.get("checks", [])]
    L += ["", "Claims: " + ", ".join("%s=%s" % kv for kv in rep["claims"].items()), ""]
    return "\n".join(L)


def final_status(checks: List[Dict[str, Any]]) -> str:
    res = [c["result"] for c in checks]
    if any(r == "FAIL" for r in res):
        return "FAILED"
    if any(r in ("NOT_RUN", "INCOMPLETE", "SKIPPED") for r in res):
        return "INCOMPLETE"
    return "COMPLETED"


__all__ = ["build_report", "render_markdown", "validate_schemas", "marker_table", "realtime_statement", "final_status", "BLOCKED_SCIENTIFIC_ITEMS"]
