"""Run the early-warning module on actual Phase 7 outputs and write the event log, alert table and reports.

Phase 7 persisted only arrays and a 25-record sample (no pedestrian/vehicle ids for all pairs), so the full per-pair records are regenerated through the PUBLIC
Phase 7 API (``build_conflict_pairs`` -> ``assess_risk`` -> ``RiskOutput.to_records``), deterministically, from the same committed artifacts, and cross-checked
field by field against the committed ``risk_arrays.npz``. A mismatch aborts the run. Phase 7 code is not modified.
"""

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..phase5.evaluation.experiment import load_baseline_config
from ..phase5.window_config import WindowConfig
from ..phase5.window_storage import read_dataset
from ..phase6.config import ConflictConfig
from ..phase6.pairs import build_conflict_pairs
from ..phase6.runner import REPO, file_sha256, load_predictions
from ..phase6.scoring import HeuristicProximityScorer
from ..phase7.risk import RiskConfig, assess_risk
from .config import AlertConfig
from .errors import AlertContractError
from .lead_time import blocked_status
from .manager import AlertManager
from .phase7_input import PHASE7_SCHEMA, parse_record
from .schema import EVENT_SCHEMA_VERSION, schema_sha256

RUN_LABEL = "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT"


def records_sha256(records: List[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps(r, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def regenerate_phase7_records(windows_run, neural_eval_dir, conflict_cfg, risk_cfg, window_cfg, phase7_arrays=None):
    ccfg, rcfg, wcfg = ConflictConfig.load(conflict_cfg), RiskConfig.load(risk_cfg), WindowConfig.load(window_cfg)
    ws, bank, extra = read_dataset(windows_run)
    ndir = Path(neural_eval_dir)
    manifest = json.loads((ndir / "run_manifest.json").read_text(encoding="utf-8"))
    pred_path = ndir / "predictions_neural.npz"
    if file_sha256(pred_path) != manifest["data_artifacts"]["predictions_neural.npz"]["sha256"]:
        raise AlertContractError("PHASE5_HASH", "predictions_neural.npz differs from its Phase 5 run manifest")
    bcfg, bsha = load_baseline_config(REPO / ccfg.baseline_evaluation_config_path)
    tid = [str(i) for i in range(int(bank.nodes["track_index"].max()) + 1)]
    batch = build_conflict_pairs(load_predictions(pred_path), ws, bank, extra["history_target_bank_node"], wcfg, ccfg, bcfg, track_ids=tid,
                                 provenance={"predictions_sha256": file_sha256(pred_path), "windows_npz_sha256": file_sha256(Path(windows_run) / "windows.npz"),
                                             "graph_bank_sha256": file_sha256(Path(windows_run) / "graph_bank.npz"), "baseline_config_sha256_canonical": bsha})
    out = assess_risk(batch, HeuristicProximityScorer().predict(batch), rcfg, None,
                      provenance={"dataset": "PIE set01/video_0004", "label": RUN_LABEL, "sample_status": "IN_SAMPLE_PHASE5_MODEL",
                                  "phase5_run_manifest_sha256": file_sha256(ndir / "run_manifest.json")},
                      checkpoint_file_sha256=manifest["inputs"]["checkpoint_file_sha256"], phase6_config_sha256=batch.config_sha256)
    check = None
    if phase7_arrays is not None:
        a = np.load(phase7_arrays, allow_pickle=False)
        ok = {"window_id": np.array_equal(a["window_id"], out.pair["window_id"]), "anchor_frame": np.array_equal(a["anchor_frame"], out.pair["anchor_frame"]),
              "risk_score": np.array_equal(a["risk_score"], out.risk_score, equal_nan=True),
              "conflict_score": np.array_equal(a["conflict_score"], out.conflict_score, equal_nan=True),
              "risk_category": np.array_equal(a["risk_category"], out.risk_category.astype(str)),
              "category_status": np.array_equal(a["category_status"], out.category_status.astype(str)),
              "calibrated_probability_all_nan": bool(np.isnan(a["calibrated_probability"]).all())}
        check = {"file_sha256": file_sha256(phase7_arrays), "n": int(a["window_id"].shape[0]), "fields_equal": ok}
        if not all(ok.values()):
            raise AlertContractError("PHASE7_ARTIFACT_MISMATCH", "regenerated Phase 7 records differ from the committed risk_arrays.npz: %s" % ok)
    return out.to_records(), out.provenance, check


def read_records(path) -> List[Dict[str, Any]]:
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows, gz=False):
    p = Path(path)
    data = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    if gz:
        with open(p, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:             # mtime=0 and no name: byte-reproducible
            f.write(data)
    else:
        p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def run_alerts(records, cfg: AlertConfig, provenance=None, hashes=None, out_dir=None, extra_manifest=None) -> Dict[str, Any]:
    obs = [parse_record(r, i, policy=cfg.conflicting_indicators_policy) for i, r in enumerate(records)]
    src = records_sha256(records)
    mgr = AlertManager(cfg, provenance, source_records_sha256=src, hashes=hashes)
    t0 = time.perf_counter()
    mgr.process_stream(obs)
    dt = time.perf_counter() - t0
    alerts = mgr.alert_table()
    ev_types, trans, kinds, by_cat = {}, {}, {}, {}
    for e in mgr.events:
        ev_types[e["event_type"]] = ev_types.get(e["event_type"], 0) + 1
        if e["event_type"] == "STATE_TRANSITION":
            k = "%s->%s" % (e["from_state"], e["to_state"])
            trans[k] = trans.get(k, 0) + 1
    for a in alerts:
        kinds[str(a["clearance_kind"])] = kinds.get(str(a["clearance_kind"]), 0) + 1
    for o in obs:
        by_cat[o.risk_category] = by_cat.get(o.risk_category, 0) + 1
    summary = {"label": RUN_LABEL, "prototype_notice": "RESEARCH PROTOTYPE: not validated, not for operational use",
               "n_records": len(records), "phase7_category_counts_input": by_cat, "n_frames": mgr.stats["steps"], "n_events": len(mgr.events), "event_types": ev_types,
               "transitions": dict(sorted(trans.items())), "n_alerts": len(alerts), "alerts_reached_escalated": sum(a["peak_state"] == "ESCALATED_WARNING" for a in alerts),
               "alerts_by_clearance_kind": kinds, "alerts_with_evidence_loss": sum(a["evidence_lost_steps"] > 0 for a in alerts),
               "unique_pedestrians_alerted": len({a["pedestrian_id"] for a in alerts}), "stats": mgr.stats,
               "lead_time": blocked_status(), "calibration": "Phase 7 probabilities are null (NOT_CALIBRATED); no alert decision uses a probability",
               "claims": {"verified_collision": False, "calibrated_probability": False, "physical_ttc_pet": False, "validated_early_warning_time": False,
                          "real_time_performance": False, "deployment_ready": False}}
    result = {"summary": summary, "alerts": alerts, "events": mgr.events, "events_sha256": records_sha256(mgr.events)}
    if out_dir:
        o = Path(out_dir)
        o.mkdir(parents=True, exist_ok=True)
        ev_sha = write_jsonl(o / "alert_events.jsonl.gz", mgr.events, gz=True)
        rec_sha = write_jsonl(o / "phase7_records_input.jsonl.gz", records, gz=True)
        (o / "alerts.json").write_text(json.dumps({"schema": "phase8-alert-table-1.0.0", "alerts": alerts}, indent=1, sort_keys=True), encoding="utf-8")
        (o / "summary_report.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
        manifest = {"manifest_version": "phase8-run-manifest-1.0.0", "label": RUN_LABEL, "event_schema_version": EVENT_SCHEMA_VERSION, "event_schema_sha256": schema_sha256(),
                    "alert_config": {"sha256": cfg.sha256(), "content": cfg.to_dict()}, "phase7_input_records_sha256": src, "phase7_input_file_sha256_uncompressed": rec_sha,
                    "alert_events_sha256_uncompressed": ev_sha, "phase7_provenance": provenance, "hashes": hashes,
                    "component_throughput": {"observations_per_s_state_machine_only": len(obs) / dt if dt > 0 else None, "seconds": dt,
                                             "caveat": "microbenchmark of the alert manager alone on one machine; excludes video, perception, forecasting and risk "
                                                       "computation; NOT an end-to-end latency or real-time claim"},
                    **(extra_manifest or {})}
        (o / "run_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return result


def run_on_phase7_artifacts(windows_run, neural_eval_dir, phase7_dir, conflict_cfg, risk_cfg, window_cfg, alert_cfg, out_dir) -> Dict[str, Any]:
    recs, prov, check = regenerate_phase7_records(windows_run, neural_eval_dir, conflict_cfg, risk_cfg, window_cfg, Path(phase7_dir) / "risk_arrays.npz")
    hashes = {"phase7_risk_config_sha256": prov.get("risk_config_sha256"), "source_checkpoint_file_sha256": prov.get("source_checkpoint_file_sha256"),
              "calibration_artifact_sha256": prov.get("calibration_artifact_sha256")}
    p7 = json.loads((Path(phase7_dir) / "risk_report.json").read_text(encoding="utf-8"))
    pv = {"dataset": prov.get("dataset"), "label": prov.get("label"), "sample_status": prov.get("sample_status"), "phase7_schema": PHASE7_SCHEMA,
          "phase7_category_status": prov.get("category_status"), "phase7_threshold_evidence": prov.get("threshold_evidence"),
          "phase5_run_manifest_sha256": prov.get("phase5_run_manifest_sha256"), "coordinate_system": prov.get("coordinate_system"),
          "phase6_config_sha256": prov.get("phase6_config_sha256"), "phase7_risk_report_sha256": file_sha256(Path(phase7_dir) / "risk_report.json"),
          "phase7_calibration_status": p7["calibration"]["status"]}
    return run_alerts(recs, AlertConfig.load(alert_cfg), pv, hashes, out_dir, {"phase7_artifact_cross_check": check})


__all__ = ["run_alerts", "run_on_phase7_artifacts", "regenerate_phase7_records", "read_records", "write_jsonl", "records_sha256"]
