"""Phase 5 evaluation of the multi-video model on every video, aggregated per partition.

Per video the UNCHANGED P5.7.5 path is used: ``evaluate_neural`` (history-only ``predict_windows`` -> unchanged P5.4
``evaluate_predictions`` with the most-probable mode as PRIMARY and best-of-K minADE/minFDE as a separate ORACLE block, plus the
training-free classical baselines on the same windows). ``write_neural_predictions`` writes the ``predictions_neural.npz`` consumed by
Phase 6. A future-isolation check (scrambling the future targets must not change any prediction), as in Phase 9 S5B, is run per video.

Sample status per partition (never mixed): ``train`` = IN-SAMPLE; ``validation`` / ``calibration`` / ``test`` = HELD-OUT RECORDING SETS,
evaluated against PERCEIVED-TRACK PSEUDO-TARGETS. Because the targets are not annotation ground truth and the Phase 9 independent-evaluation
gate is BLOCKED (annotation provenance, identity mapping and timestamp alignment unverified), held-out numbers are labelled
``HELD_OUT_RECORDING_PSEUDO_TARGET_DIAGNOSTIC`` and are not reported as independent validation. Every evaluation of the test partition
is appended to ``test_access_log.jsonl``.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..phase5.evaluation.experiment import eval_config_from, load_baseline_config
from ..phase5.evaluation.metrics import EvaluationResult, summarize
from ..phase5.models.neural_evaluation import (LoadedModel, NeuralEvaluationConfig, evaluate_neural, predict_windows, write_neural_metrics,
                                               write_neural_predictions)
from ..phase5.training_data import TrainingWindowDataset
from . import MULTIVIDEO_LABEL, TARGET_SOURCE
from .common import StageError, canonical_sha256, check_marker, publish_dir, read_json, sha256_file, utc_now, write_json, write_marker
from .training import load_multivideo_model

STAGE = "S5C_EVALUATION"
SAMPLE_STATUS = {"train": "IN_SAMPLE_TRAIN_PARTITION_PSEUDO_TARGETS", "validation": "HELD_OUT_RECORDING_PSEUDO_TARGET_DIAGNOSTIC_VALIDATION",
                 "calibration": "HELD_OUT_RECORDING_PSEUDO_TARGET_DIAGNOSTIC_CALIBRATION", "test": "HELD_OUT_RECORDING_PSEUDO_TARGET_DIAGNOSTIC_TEST"}


def _summ(res: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in res.items() if k != "per_track"}


def future_isolation_check(model, ds: TrainingWindowDataset, batch_size: int, n: int = 48) -> Dict[str, Any]:
    sub = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    base = predict_windows(model, ds, batch_size, indices=sub)
    saved_uv, saved_v = ds.ws.target_future_uv_px.copy(), ds.ws.target_future_valid.copy()
    try:
        ds.ws.target_future_uv_px[...] = saved_uv[::-1] + 1000.0
        ds.ws.target_future_valid[...] = ~saved_v
        pert = predict_windows(model, ds, batch_size, indices=sub)
    finally:
        ds.ws.target_future_uv_px[...] = saved_uv
        ds.ws.target_future_valid[...] = saved_v
    ok = all(np.array_equal(base[k], pert[k]) for k in ("positions_uv_px", "valid", "mode_probabilities"))
    if not ok:
        raise StageError("FUTURE_LEAK", "predictions changed when future targets were scrambled")
    return {"passed": True, "windows_perturbed": len(sub)}


def _concat(results: List[EvaluationResult], keys: List[str]) -> EvaluationResult:
    """Partition-level result = window-level concatenation; window ids are made unique per video (video index * 10^8 + id) and
    track ids are prefixed with the video key, so track-weighted aggregation never merges tracks of different videos."""
    k = results[0].k
    wid = np.concatenate([r.window_id.astype(np.int64) + i * 100_000_000 for i, r in enumerate(results)])
    tid = np.concatenate([np.array(["%s|%s" % (kk, t) for t in r.track_id.tolist()], dtype=str) for kk, r in zip(keys, results)])
    cat = lambda f: np.concatenate([getattr(r, f) for r in results])                                    # noqa: E731
    return EvaluationResult(model_id=results[0].model_id, window_id=wid, track_id=tid, evaluated=cat("evaluated"), invalid_reason=cat("invalid_reason"), ade=cat("ade"),
                            fde=cat("fde"), step_error=cat("step_error"), min_ade=cat("min_ade"), min_fde=cat("min_fde"), oracle_evaluated=cat("oracle_evaluated"), k=k)


def evaluate_all(train_dir, window_dirs: Dict[str, Any], split: Dict[str, Any], out_root, work_root, quarantine_root, *, neural_eval_config_path, repo,
                 device: str = "cpu", batch_size: Optional[int] = None, partitions=("train", "validation", "calibration", "test"), git_commit: Optional[str] = None,
                 phase9_gate: Optional[Dict[str, Any]] = None, progress=print, verify_only: bool = False) -> Dict[str, Any]:
    repo = Path(repo)
    model, tman, ck_name, ck_rec, stats, verification, wcfg = load_multivideo_model(train_dir, device=device)
    ncfg = NeuralEvaluationConfig.load(neural_eval_config_path)
    bcfg, bsha = load_baseline_config(repo / ncfg.baseline_evaluation_config_path)
    ecfg = eval_config_from(bcfg)
    bs = batch_size or ncfg.batch_size
    parts = {v: (s, g["partition"]) for s, g in split["groups"].items() for v in g["videos"]}
    train_videos = {r["video"] for r in tman["inputs"]["train_videos"]}
    cfg_sha = canonical_sha256({"stage": STAGE, "training_run_manifest_sha256": verification["training_run_manifest_sha256"], "checkpoint": ck_name,
                                "neural_eval_config_sha256": ncfg.sha256(), "baseline_config_sha256": bsha, "split_content_sha256": split["content_sha256"]})
    if verify_only:
        sp = Path(out_root) / "evaluation_summary.json"
        if not sp.is_file():
            raise StageError("NOT_EVALUATED_VERIFY_ONLY", "evaluation is disabled and %s does not exist" % sp)
        old = read_json(sp)
        if old.get("verification", {}).get("checkpoint_file_sha256") != verification["checkpoint_file_sha256"]:
            raise StageError("STALE_EVALUATION", "the stored evaluation refers to another checkpoint")
        missing = [k for k, (s_, p_) in parts.items() if k in window_dirs and p_ in partitions and old["per_video"].get(k, {}).get("status") == "COMPLETED"
                   and check_marker(Path(out_root) / s_ / k, stage=STAGE, config_sha256=cfg_sha)[0] != "COMPLETE"]
        if missing:
            raise StageError("EVALUATION_MARKERS_MISSING", "evaluation outputs missing or modified for %s" % missing)
        return dict(old, reused_verified=True)
    per_video: Dict[str, Any] = {}
    acc: Dict[str, Dict[str, Any]] = {}
    Path(out_root).mkdir(parents=True, exist_ok=True)
    test_log = Path(out_root) / "test_access_log.jsonl"
    for i, key in enumerate(sorted(window_dirs), 1):
        set_id, part = parts[key]
        if part not in partitions:
            continue
        if part != "train" and key in train_videos:
            raise StageError("TRAIN_EVAL_OVERLAP", "%s is in partition %s but was used for training" % (key, part))
        final = Path(out_root) / set_id / key
        st, mk = check_marker(final, stage=STAGE, config_sha256=cfg_sha)
        if st == "STALE":
            raise StageError("STALE_ARTIFACT", "%s holds an evaluation of a different model/configuration" % final)
        wdir = Path(window_dirs[key])
        if read_json(wdir / "run_manifest.json")["counts"]["accepted_windows"] == 0:
            per_video[key] = {"status": "NO_WINDOWS", "partition": part}
            continue
        ds = TrainingWindowDataset(wdir)
        if ds.config.sha256() != wcfg.sha256():
            raise StageError("WINDOW_CONFIG_MISMATCH", "%s uses a different window configuration than the model" % key)
        if part == "test":
            with open(test_log, "a", encoding="utf-8") as fh:
                fh.write('{"utc": "%s", "video": "%s", "checkpoint_file_sha256": "%s", "marker_state": "%s"}\n' % (utc_now(), key, verification["checkpoint_file_sha256"], st))
        progress("[%s] evaluation %d/%d %s (%s, %d windows)%s" % (utc_now(), i, len(window_dirs), key, part, len(ds), " [cached]" if st == "COMPLETE" else ""))
        t0 = time.perf_counter()
        loaded = LoadedModel(model, tman, ck_name, ck_rec, stats, ds, dict(verification, window_run_manifest_sha256=ds.provenance["run_manifest_sha256"]))
        pred, results, comparison, diag, hz = evaluate_neural(loaded, ncfg, bcfg, ecfg)
        iso = future_isolation_check(model, ds, bs)
        neural = results[ncfg.model_id]
        a = acc.setdefault(part, {"keys": [], "res": {}})
        a["keys"].append(key)
        for mid, r in results.items():
            a["res"].setdefault(mid, []).append(r)
        if st != "COMPLETE":
            loc = Path(work_root) / "evaluation" / key
            loc.mkdir(parents=True, exist_ok=True)
            write_neural_predictions(loc / "predictions_neural.npz", pred, ds, ncfg.model_id, hz, wcfg)
            write_neural_metrics(loc / "evaluation_neural.npz", neural)
            res = {"label": MULTIVIDEO_LABEL, "sample_status": SAMPLE_STATUS[part], "partition": part, "video": key, "set": set_id, "target_source": TARGET_SOURCE,
                   "primary": {"selection": "most probable mode (argmax of the model's own probabilities)", "summary": _summ(neural.summary)},
                   "oracle_note": "summary.oracle_best_of_k holds minADE@K / minFDE@K: ground truth selects the mode (diagnostic, not a single-trajectory result)",
                   "baseline_comparison": comparison, "mode_diagnostics": diag, "future_isolation": iso, "verification": loaded.verification,
                   "phase9_independent_evaluation_gate": (phase9_gate or {}).get("status"), "coordinates": {"system": wcfg.coordinate_system, "unit": wcfg.coordinate_unit},
                   "horizon_offsets_s": [round(float(x), 9) for x in hz], "wall_s": time.perf_counter() - t0}
            write_json(loc / "evaluation_results.json", res)
            files = publish_dir(loc, final, quarantine_root)
            write_marker(final, stage=STAGE, key=key, config_sha256=cfg_sha, files=files, git_commit=git_commit,
                         detail={"partition": part, "windows": len(ds), "windows_evaluated": neural.summary["windows_evaluated"]})
        per_video[key] = {"status": "COMPLETED", "partition": part, "dir": str(final), "windows": len(ds),
                          "ade_px_mean_primary": neural.summary["window_weighted"]["ade_px"]["mean"], "fde_px_mean_primary": neural.summary["window_weighted"]["fde_px"]["mean"],
                          "min_ade_px_mean_oracle": neural.summary["oracle_best_of_k"]["min_ade_px"]["mean"],
                          "min_fde_px_mean_oracle": neural.summary["oracle_best_of_k"]["min_fde_px"]["mean"]}
    hz = wcfg.future_offsets / wcfg.source_fps
    per_partition = {}
    for part, a in acc.items():
        models = {}
        for mid, rs in a["res"].items():
            cat = _concat(rs, a["keys"])
            models[mid] = _summ(summarize(cat, hz, ecfg))
        per_partition[part] = {"sample_status": SAMPLE_STATUS[part], "videos": a["keys"], "n_videos": len(a["keys"]), "neural_model_id": ncfg.model_id,
                               "primary_most_probable_mode": models.get(ncfg.model_id), "baselines_single_trajectory": {k: v for k, v in models.items() if k != ncfg.model_id},
                               "aggregation": "window-level concatenation over the partition's videos; track ids are made unique per video"}
    summary = {"label": MULTIVIDEO_LABEL, "created_utc": utc_now(), "training_run": str(train_dir), "verification": verification, "target_source": TARGET_SOURCE,
               "per_partition": per_partition, "per_video": per_video, "phase9_independent_evaluation_gate": phase9_gate,
               "metric_policy": {"primary": "most-probable mode ADE/FDE (pixels)", "oracle": "minADE@K / minFDE@K reported separately in oracle_best_of_k",
                                 "miss_rate": "NOT_CONFIGURED" if ecfg.miss_rate_threshold_px is None else ecfg.miss_rate_threshold_px},
               "interpretation": ["train partition numbers are in-sample", "held-out partitions are separate recordings but are scored against perceived-track pseudo-targets",
                                  "no number here is independent validation against annotation ground truth (Phase 9 gate BLOCKED)", "pixels only; no metric distance"]}
    write_json(Path(out_root) / "evaluation_summary.json", summary)
    return summary


def render_evaluation_markdown(summary: Dict[str, Any]) -> str:
    f = lambda x: "—" if x is None else "%.2f" % x                                                    # noqa: E731
    L = ["# Phase 5 multi-video evaluation (%s)" % summary["label"], "",
         "Targets: %s. Units: pixels of `captured_1920x1080`. Phase 9 independent-evaluation gate: **%s**." % (
             summary["target_source"], (summary.get("phase9_independent_evaluation_gate") or {}).get("status")), "",
         "| partition | sample status | videos | windows evaluated | ADE mean (primary) | FDE mean (primary) | minADE@K (oracle) | minFDE@K (oracle) |",
         "|---|---|---|---|---|---|---|---|"]
    for part in ("train", "validation", "calibration", "test"):
        p = summary["per_partition"].get(part)
        if not p or not p["primary_most_probable_mode"]:
            continue
        s = p["primary_most_probable_mode"]
        o = s["oracle_best_of_k"]
        L.append("| %s | %s | %d | %s | %s | %s | %s | %s |" % (part, p["sample_status"], p["n_videos"], s["windows_evaluated"], f(s["window_weighted"]["ade_px"]["mean"]),
                                                             f(s["window_weighted"]["fde_px"]["mean"]), f(o["min_ade_px"]["mean"]), f(o["min_fde_px"]["mean"])))
    L += ["", "Oracle minADE@K / minFDE@K let the ground truth choose the mode; they are diagnostics, not the model's single-trajectory accuracy.", ""]
    for part, p in summary["per_partition"].items():
        if p["baselines_single_trajectory"]:
            L += ["## Classical baselines on %s (single trajectory, same windows)" % part, "", "| model | ADE mean | FDE mean |", "|---|---|---|"]
            for k, s in p["baselines_single_trajectory"].items():
                L.append("| %s | %s | %s |" % (k, f(s["window_weighted"]["ade_px"]["mean"]), f(s["window_weighted"]["fde_px"]["mean"])))
            L.append("")
    return "\n".join(L)


__all__ = ["evaluate_all", "render_evaluation_markdown", "future_isolation_check", "SAMPLE_STATUS", "STAGE"]
