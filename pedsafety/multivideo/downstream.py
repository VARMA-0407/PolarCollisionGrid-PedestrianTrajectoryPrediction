"""Phase 6 (conflict representation), Phase 7 (risk assessment) and Phase 8 (early-warning events) per video.

The sequence and the checks are those of the Phase 9 orchestrator stages S6 / S7 / S8, through the same public APIs:
``validate_predictions`` -> ``build_conflict_pairs`` (pedestrian-vehicle pairs, mode-aggregated image-space indicators, pair validity masks)
-> ``HeuristicProximityScorer`` -> ``assess_risk`` (no calibration artifact: ``calibrated_probability`` stays null, status NOT_CALIBRATED)
-> ``validate_risk_records`` -> ``run_alerts`` (persistence, hysteresis, escalation, de-escalation, deduplication, cooldown, evidence loss,
end-of-stream) -> ``validate_alert_outputs``. Configurations, thresholds and units are the repository files, unchanged.

Phase 6 keeps no persistent pair-batch format, so S7 rebuilds the (deterministic) pair batch and verifies that its summary hash equals the
one recorded by S6; S8 reads the Phase 7 records written by S7. No stage reads labels, future observations or ground-truth events.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import MULTIVIDEO_LABEL
from .common import StageError, canonical_sha256, check_marker, publish_dir, read_json, sha256_file, write_json, write_marker
from .evaluation import SAMPLE_STATUS

STAGES = ("S6_CONFLICT", "S7_RISK", "S8_ALERTS")
DIRNAMES = {"S6_CONFLICT": "phase6_conflict", "S7_RISK": "phase7_risk", "S8_ALERTS": "phase8_alerts"}


def downstream_config(repo, configs: Dict[str, str]) -> Dict[str, Any]:
    repo = Path(repo)
    return {"configs": dict(configs), "config_file_sha256": {k: sha256_file(repo / configs[k]) for k in ("conflict", "risk", "alert", "windows")}}


def _pairs(repo, windows_dir, predictions_path, dcfg):
    from ..phase5.evaluation.experiment import load_baseline_config
    from ..phase5.window_config import WindowConfig
    from ..phase5.window_storage import read_dataset
    from ..phase6.config import ConflictConfig
    from ..phase6.pairs import build_conflict_pairs
    from ..phase6.runner import load_predictions
    from ..phase9.contracts import validate_predictions
    repo = Path(repo)
    wcfg = WindowConfig.load(repo / dcfg["configs"]["windows"])
    ccfg = ConflictConfig.load(repo / dcfg["configs"]["conflict"])
    ws, bank, extra = read_dataset(windows_dir)
    pred = load_predictions(predictions_path)
    validate_predictions(pred, ws, wcfg, ccfg)
    bcfg, bsha = load_baseline_config(repo / ccfg.baseline_evaluation_config_path)
    tid = [str(i) for i in range(int(bank.nodes["track_index"].max()) + 1)]
    batch = build_conflict_pairs(pred, ws, bank, extra["history_target_bank_node"], wcfg, ccfg, bcfg, track_ids=tid,
                                 provenance={"baseline_config_sha256_canonical": bsha, "predictions_sha256": sha256_file(predictions_path),
                                             "windows_npz_sha256": sha256_file(Path(windows_dir) / "windows.npz"), "stage": "multivideo S6"})
    if batch.provenance.get("future_observations_used") or batch.provenance.get("labels_used"):
        raise StageError("S6_PROVENANCE", "conflict stage reports use of future observations or labels")
    return batch, ws, wcfg


def run_downstream_video(repo, key: str, set_id: str, partition: str, windows_dir, predictions_dir, artifacts_root, work_root, quarantine_root, dcfg: Dict[str, Any], *,
                         stages=STAGES, git_commit: Optional[str] = None, verify_only: bool = False) -> Dict[str, Any]:
    from ..phase6.runner import summarize as conflict_summary, supervision_status
    from ..phase6.scoring import HeuristicProximityScorer
    from ..phase7.risk import RiskConfig, assess_risk, summarize_risk
    from ..phase7.runner import calibration_status
    from ..phase8.config import AlertConfig
    from ..phase8.runner import read_records, run_alerts, write_jsonl
    from ..phase9.contracts import validate_alert_outputs, validate_risk_records

    repo = Path(repo)
    pred_path = Path(predictions_dir) / "predictions_neural.npz"
    ev_marker = read_json(Path(predictions_dir) / "_STAGE_COMPLETE.json")
    ck = read_json(Path(predictions_dir) / "evaluation_results.json")["verification"]["checkpoint_file_sha256"]
    up = {"predictions_sha256": sha256_file(pred_path), "windows_manifest_sha256": sha256_file(Path(windows_dir) / "run_manifest.json"),
          "evaluation_marker_config_sha256": ev_marker["config_sha256"], "checkpoint_file_sha256": ck}
    finals = {s: Path(artifacts_root) / DIRNAMES[s] / set_id / key for s in STAGES}
    sha = {s: canonical_sha256({"stage": s, **dcfg, **up}) for s in STAGES}
    out: Dict[str, Any] = {"key": key, "partition": partition, "stages": {}}
    sample_status = SAMPLE_STATUS[partition]
    from ..phase5.window_config import WindowConfig
    fps = WindowConfig.load(repo / dcfg["configs"]["windows"]).source_fps
    for s in STAGES:
        if s not in stages:
            continue
        st, m = check_marker(finals[s], stage=s, config_sha256=sha[s])
        if st == "STALE":
            raise StageError("STALE_ARTIFACT", "%s holds %s outputs of different inputs/configuration" % (finals[s], s))
        if st == "COMPLETE":
            out["stages"][s] = {"status": "SKIPPED_ALREADY_COMPLETE", "detail": m.get("detail", {})}
            continue
        if verify_only:
            out["stages"][s] = {"status": "MISSING_VERIFY_ONLY", "marker_state": st}
            continue
        loc = Path(work_root) / "downstream" / key / s
        if loc.exists():
            import shutil
            shutil.rmtree(loc)
        loc.mkdir(parents=True)
        t0 = time.perf_counter()
        if s == "S6_CONFLICT":
            batch, ws, wcfg = _pairs(repo, windows_dir, pred_path, dcfg)
            summ = conflict_summary(batch)
            rep = {"status": "completed", "label": MULTIVIDEO_LABEL, "video": key, "set": set_id, "partition": partition, "sample_status": sample_status,
                   "coordinate_system": batch.coordinate_system, "coordinate_unit": batch.coordinate_unit, "config_sha256": batch.config_sha256, "provenance": batch.provenance,
                   "exclusions": batch.exclusions, "summary": summ, "summary_sha256": canonical_sha256(summ), "supervision": supervision_status(),
                   "pair_counts": {"pairs": batch.n_pairs, "pairs_valid": int(batch.masks["pair_valid"].sum()), "pairs_invalid": int((~batch.masks["pair_valid"]).sum())},
                   "claims": {"physical_ttc_pet": False, "collision_probability": False, "calibrated": False, "verified_conflict_labels": False}}
            write_json(loc / "conflict_report.json", rep)
            det = {"pairs": batch.n_pairs, "pairs_valid": rep["pair_counts"]["pairs_valid"], "summary_sha256": rep["summary_sha256"]}
        elif s == "S7_RISK":
            s6 = finals["S6_CONFLICT"]
            if check_marker(s6, stage="S6_CONFLICT", config_sha256=sha["S6_CONFLICT"])[0] != "COMPLETE":
                raise StageError("S7_UPSTREAM", "S6 conflict output of %s is not complete" % key)
            batch, ws, wcfg = _pairs(repo, windows_dir, pred_path, dcfg)
            if canonical_sha256(conflict_summary(batch)) != read_json(s6 / "conflict_report.json")["summary_sha256"]:
                raise StageError("S6_S7_MISMATCH", "rebuilt pair batch differs from the recorded Phase 6 output")
            rcfg = RiskConfig.load(repo / dcfg["configs"]["risk"])
            risk = assess_risk(batch, HeuristicProximityScorer().predict(batch), rcfg, None,
                               provenance={"dataset": "PIE %s/%s" % (set_id, key.split("_", 1)[1]), "video_key": key, "label": MULTIVIDEO_LABEL, "sample_status": sample_status,
                                           "partition": partition, "stage": "multivideo S7"},
                               checkpoint_file_sha256=ck, phase6_config_sha256=batch.config_sha256)
            recs = risk.to_records()
            val = validate_risk_records(recs, ws, fps)
            if any(r["calibrated_probability"] is not None for r in recs):
                raise StageError("UNCALIBRATED_PROBABILITY", "a calibrated probability was emitted without a calibration artifact")
            rec_sha = write_jsonl(loc / "risk_records.jsonl.gz", recs, gz=True)
            np.savez_compressed(loc / "risk_arrays.npz", window_id=risk.pair["window_id"], anchor_frame=risk.pair["anchor_frame"], conflict_score=risk.conflict_score,
                                calibrated_probability=risk.calibrated_probability, temporal_urgency=risk.temporal_urgency, risk_score=risk.risk_score,
                                risk_category=risk.risk_category.astype(str), category_status=risk.category_status.astype(str))
            cal = calibration_status()
            cal["available_now"] = "multi-video experiment: no verified conflict labels; the calibration partition (if any) holds no labels"
            rep = {"status": "completed", "label": MULTIVIDEO_LABEL, "video": key, "partition": partition, "sample_status": sample_status, "summary": summarize_risk(risk),
                   "calibration": cal, "calibration_status": risk.calibration_status, "records_sha256_uncompressed": rec_sha, "validation": val, "provenance": risk.provenance,
                   "claims": {"calibrated_probability": False, "validated_risk_categories": False, "physical_ttc_pet": False, "severity": False, "deployment_ready": False}}
            write_json(loc / "risk_report.json", rep)
            det = {"records": len(recs), "category_counts": rep["summary"]["category_counts"], "calibration_status": risk.calibration_status}
        else:
            s7 = finals["S7_RISK"]
            if check_marker(s7, stage="S7_RISK", config_sha256=sha["S7_RISK"])[0] != "COMPLETE":
                raise StageError("S8_UPSTREAM", "S7 risk output of %s is not complete" % key)
            recs = read_records(s7 / "risk_records.jsonl.gz")
            r7 = read_json(s7 / "risk_report.json")
            acfg = AlertConfig.load(repo / dcfg["configs"]["alert"])
            from ..phase5.window_storage import read_dataset
            ws, _, _ = read_dataset(windows_dir)
            rp = r7["provenance"]
            prov = {"dataset": rp.get("dataset"), "label": MULTIVIDEO_LABEL, "sample_status": sample_status, "phase7_schema": "phase7-risk-output-1.0.0",
                    "phase7_category_status": rp.get("category_status"), "phase7_threshold_evidence": rp.get("threshold_evidence"), "coordinate_system": rp.get("coordinate_system"),
                    "phase6_config_sha256": rp.get("phase6_config_sha256"), "phase7_risk_report_sha256": sha256_file(s7 / "risk_report.json"),
                    "phase7_calibration_status": "BLOCKED_NO_VERIFIED_CONFLICT_LABELS", "phase5_predictions_sha256": up["predictions_sha256"], "stage": "multivideo S8"}
            hashes = {"phase7_risk_config_sha256": rp.get("risk_config_sha256"), "source_checkpoint_file_sha256": ck, "calibration_artifact_sha256": None}
            res = run_alerts(recs, acfg, prov, hashes, loc)
            val = validate_alert_outputs(res["events"], res["alerts"], recs, fps)
            sm = res["summary"]
            det = {"n_alerts": sm["n_alerts"], "n_events": sm["n_events"], "alerts_escalated": sm["alerts_reached_escalated"], "alerts_by_clearance_kind": sm["alerts_by_clearance_kind"],
                   "transitions": sm["transitions"], "lead_time": sm["lead_time"], "validation": val}
        det["wall_s"] = time.perf_counter() - t0
        files = publish_dir(loc, finals[s], quarantine_root)
        write_marker(finals[s], stage=s, key=key, config_sha256=sha[s], upstream=up, detail=det, files=files, git_commit=git_commit)
        out["stages"][s] = {"status": "COMPLETED", "detail": det}
    return out


def downstream_batch(repo, items, *, stages=STAGES, progress=print, **kw) -> Dict[str, Any]:
    """``items``: [{"key", "set", "partition", "windows_dir", "predictions_dir"}]."""
    from .common import utc_now
    results, failed = {}, []
    for i, it in enumerate(items, 1):
        try:
            r = run_downstream_video(repo, it["key"], it["set"], it["partition"], it["windows_dir"], it["predictions_dir"], stages=stages, **kw)
            progress("[%s] %s %d/%d %s: %s" % (utc_now(), "/".join(stages), i, len(items), it["key"], {s: v["status"] for s, v in r["stages"].items()}))
        except Exception as exc:                                                                      # noqa: BLE001 - recorded per video
            r = {"key": it["key"], "status": "FAILED", "error_code": getattr(exc, "code", type(exc).__name__), "error": str(exc)[:2000]}
            failed.append(it["key"])
            progress("[%s] %s %s FAILED: %s" % (utc_now(), "/".join(stages), it["key"], r["error"][:400]))
        results[it["key"]] = r
    missing = sorted(k for k, r in results.items() if any(v.get("status") == "MISSING_VERIFY_ONLY" for v in r.get("stages", {}).values()))
    return {"n_videos": len(items), "failed": sorted(failed), "missing": missing, "results": results, "all_complete": not failed and not missing}


__all__ = ["run_downstream_video", "downstream_batch", "downstream_config", "STAGES", "DIRNAMES"]
