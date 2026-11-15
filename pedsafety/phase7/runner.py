"""Integration: Phase 6 pairs from real Phase 5 artifacts -> heuristic scores -> risk output (no calibration: blocked) -> report + arrays."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..interaction.storage import write_json
from ..phase5.evaluation.experiment import load_baseline_config
from ..phase5.window_config import WindowConfig
from ..phase5.window_storage import read_dataset
from ..phase6.config import ConflictConfig
from ..phase6.pairs import build_conflict_pairs
from ..phase6.runner import REPO, file_sha256, load_predictions
from ..phase6.scoring import HeuristicProximityScorer
from .calibration import MIN_NEG, MIN_POS
from .risk import RiskConfig, assess_risk, summarize_risk


def calibration_status() -> Dict[str, Any]:
    return {"status": "BLOCKED_NO_VERIFIED_CONFLICT_LABELS", "fitted": False, "calibrated_probabilities_emitted": False,
            "required": ["verified conflict labels (collision / near-conflict / annotated interaction) from a documented independent source with SHA-256 and protocol",
                         "at least 3 independent scene/video groups, partitioned into disjoint train, calibration and test groups",
                         ">= %d positives and >= %d negatives in the calibration groups (isotonic: >= 1000 rows, >= 100 positives)" % (MIN_POS, MIN_NEG),
                         "context-agent realised futures or event annotations aligned to pair anchors, with label times strictly after the anchor"],
            "available_now": "one PIE video (set01/video_0004), no annotations, in-sample Phase 5 model"}


def run_risk_pipeline(windows_run, neural_eval_dir, conflict_cfg_path, risk_cfg_path, window_cfg_path, out_dir: Optional[str] = None) -> Dict[str, Any]:
    ccfg, rcfg, wcfg = ConflictConfig.load(conflict_cfg_path), RiskConfig.load(risk_cfg_path), WindowConfig.load(window_cfg_path)
    ws, bank, extra = read_dataset(windows_run)
    ndir = Path(neural_eval_dir)
    manifest = json.loads((ndir / "run_manifest.json").read_text(encoding="utf-8"))
    ck = manifest["inputs"]
    pred_path = ndir / "predictions_neural.npz"
    if file_sha256(pred_path) != manifest["data_artifacts"]["predictions_neural.npz"]["sha256"]:
        raise RuntimeError("predictions_neural.npz hash differs from its run manifest")
    bcfg, bsha = load_baseline_config(REPO / ccfg.baseline_evaluation_config_path)
    tid = [str(i) for i in range(int(bank.nodes["track_index"].max()) + 1)]
    batch = build_conflict_pairs(load_predictions(pred_path), ws, bank, extra["history_target_bank_node"], wcfg, ccfg, bcfg, track_ids=tid,
                                 provenance={"predictions_sha256": file_sha256(pred_path), "windows_npz_sha256": file_sha256(Path(windows_run) / "windows.npz"),
                                             "graph_bank_sha256": file_sha256(Path(windows_run) / "graph_bank.npz"), "baseline_config_sha256_canonical": bsha})
    scores = HeuristicProximityScorer().predict(batch)
    out = assess_risk(batch, scores, rcfg, None, provenance={"dataset": "PIE set01/video_0004", "label": "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT",
                                                          "sample_status": "IN_SAMPLE_PHASE5_MODEL", "phase5_run_manifest_sha256": file_sha256(ndir / "run_manifest.json")},
                      checkpoint_file_sha256=ck["checkpoint_file_sha256"], phase6_config_sha256=batch.config_sha256)
    report = {"status": "completed", "label": "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT", "summary": summarize_risk(out), "calibration": calibration_status(),
              "provenance": out.provenance, "claims": {"calibrated_probability": False, "validated_risk_categories": False, "physical_ttc_pet": False,
                                                       "severity": False, "deployment_ready": False}}
    if out_dir:
        o = Path(out_dir)
        o.mkdir(parents=True, exist_ok=True)
        write_json(o / "risk_report.json", report)
        recs = out.to_records()
        (o / "risk_output_sample.json").write_text(json.dumps({"schema": out.schema, "n_total": out.n, "first_records": recs[:25]}, indent=1), encoding="utf-8")
        np.savez_compressed(o / "risk_arrays.npz", window_id=out.pair["window_id"], anchor_frame=out.pair["anchor_frame"], conflict_score=out.conflict_score,
                            calibrated_probability=out.calibrated_probability, temporal_urgency=out.temporal_urgency, risk_score=out.risk_score,
                            risk_category=out.risk_category.astype(str), category_status=out.category_status.astype(str))
    return report


if __name__ == "__main__":
    import sys
    print(json.dumps(run_risk_pipeline(*sys.argv[1:7]), indent=1)[:3500])
