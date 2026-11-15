"""Integration runner: real Phase 5 artifacts -> conflict pair batch -> heuristic indicator summary + provenance manifest.

No labels are read or created. Supervised training / evaluation stays blocked (see ``supervision_status``)."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..interaction.storage import write_json
from ..phase5.evaluation.experiment import load_baseline_config
from ..phase5.window_config import WindowConfig
from ..phase5.window_storage import read_dataset
from .config import ConflictConfig
from .errors import ConflictContractError
from .pairs import ConflictPairBatch, Phase5Predictions, build_conflict_pairs
from .scoring import HeuristicProximityScorer

REPO = Path(__file__).resolve().parents[2]
LABEL = "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT"


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(path) -> Phase5Predictions:
    d = np.load(path, allow_pickle=False)
    return Phase5Predictions(window_id=d["window_id"], target_track_index=d["target_track_index"], anchor_frame=d["anchor_frame"],
                             positions_uv_px=d["positions_uv_px"], valid=d["valid"], mode_probabilities=d["mode_probabilities"],
                             anchor_uv_px=d["anchor_uv_px"], horizon_offsets_s=d["horizon_offsets_s"],
                             coordinate_system=str(d["coordinate_system"][0]), position_unit=str(d["position_unit"][0]))


def supervision_status() -> Dict[str, Any]:
    return {"supervised_conflict_training": "BLOCKED_NO_VERIFIED_CONFLICT_LABELS_AND_ONE_SCENE_GROUP",
            "reasons": ["no collision / near-conflict / interaction annotation exists in the repository (one PIE video, no annotation XML)",
                        "the graph bank stores only history nodes of context agents, so the realised future of a vehicle is unavailable",
                        "a single independent group exists; at least 3 are required for a train/validation/test protocol"],
            "candidate_labels_generated": False}


def summarize(batch: ConflictPairBatch) -> Dict[str, Any]:
    scores = HeuristicProximityScorer().predict(batch)
    v = scores.valid
    s = scores.scores[v]
    out = {"n_pairs": batch.n_pairs, "n_pairs_valid": int(v.sum()), "n_windows_with_pairs": int(np.unique(batch.window_id).size),
           "pairs_by_type": batch.exclusions.get("pairs_by_type"), "score_kind": scores.kind, "score_calibrated": False}
    if s.size:
        out["heuristic_score"] = {"mean": float(s.mean()), "median": float(np.median(s)), "p90": float(np.quantile(s, 0.9)), "max": float(s.max()),
                                  "fraction_nonzero": float((s > 0).mean())}
        soh = batch.features["worst_case_min_sep_over_h"][v]
        out["worst_case_min_sep_over_h"] = {"min": float(soh.min()), "median": float(np.median(soh))}
    return out


def run_conflict_pipeline(windows_run, predictions_path, config_path, window_config_path, out_dir: Optional[str] = None) -> Dict[str, Any]:
    cfg = ConflictConfig.load(config_path)
    wcfg = WindowConfig.load(window_config_path)
    ws, bank, extra = read_dataset(windows_run)
    pred = load_predictions(predictions_path)
    baseline_cfg, baseline_sha = load_baseline_config(REPO / cfg.baseline_evaluation_config_path)
    tid = [str(i) for i in range(int(bank.nodes["track_index"].max()) + 1)]
    batch = build_conflict_pairs(pred, ws, bank, extra["history_target_bank_node"], wcfg, cfg, baseline_cfg, track_ids=tid,
                                 provenance={"predictions_sha256": file_sha256(predictions_path), "windows_npz_sha256": file_sha256(Path(windows_run) / "windows.npz"),
                                             "graph_bank_sha256": file_sha256(Path(windows_run) / "graph_bank.npz"), "baseline_config_sha256_canonical": baseline_sha})
    report = {"status": "completed", "label": LABEL, "sample_status": "IN_SAMPLE_PHASE5_MODEL", "coordinate_system": cfg.coordinate_system,
              "coordinate_unit": cfg.coordinate_unit, "config_sha256": cfg.sha256(), "provenance": batch.provenance, "exclusions": batch.exclusions,
              "summary": summarize(batch), "supervision": supervision_status(),
              "claims": {"physical_ttc_pet": False, "collision_probability": False, "calibrated": False, "independent_generalization": False}}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "conflict_report.json", report)
    return report


if __name__ == "__main__":
    import sys
    print(json.dumps(run_conflict_pipeline(*sys.argv[1:6]), indent=1, default=str)[:4000])
