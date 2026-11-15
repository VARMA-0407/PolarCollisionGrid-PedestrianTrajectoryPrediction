"""P5.7.5 neural evaluator: verified checkpoint -> history-only inference -> the UNCHANGED P5.4 evaluator -> comparison with classical baselines.

Before any inference the evaluator verifies (all failures are errors, nothing is repaired): the training run manifest and every data
artifact hash, the checkpoint file hash and canonical model-state hash, the train-only scaler statistics (hash, source identity,
split identity, equality with the scaler buffers stored in the checkpoint), the encoder / decoder / window configuration hashes
and the P5.3 window run (manifest + output hashes).

Inference is history-only: the model is called with the history ``EncoderInput``, the raw pixel anchor and the history-derived
target-validity mask; ground truth is read from the window dataset only afterwards, inside the evaluator.

Tensor contract handed to ``evaluate_predictions`` (P5.4, unchanged): predictions ``[N,K,F,2]`` (raw pixels), prediction masks
``[N,K,F]``, probabilities ``[N,K]``, ground truth ``[N,F,2]``, ground-truth masks ``[N,F]``.
PRIMARY metrics use the MOST-PROBABLE mode (argmax of the model's own probabilities); minADE@K / minFDE@K are ORACLE (ground truth
selects the mode) and are reported separately. The model was trained on these very windows: every number is IN-SAMPLE.
"""

import datetime
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from ...errors import ConfigError
from ...image_kinematics_run import sha256_file
from ...interaction.runner import peak_memory_bytes
from ...interaction.storage import write_json, write_npz, write_text
from ...output_dir import check_output_dir
from ..baselines import build_history_input, make_predictor
from ..errors import WindowContractError
from ..evaluation.experiment import eval_config_from, load_baseline_config, verify_window_run
from ..evaluation.metrics import EvalConfig, EvaluationResult, common_window_mask, evaluate_predictions, summarize
from ..training_data import TrainingWindowDataset
from ..training_scaler import TrainOnlyFeatureStatistics
from ..window_manifest import PILOT_LABEL, git_provenance
from .checkpoint import file_sha256, load_checkpoint
from .encoder import EncoderInput
from .forecaster import ForecastingModel
from .training_harness import CLAIMS as TRAIN_CLAIMS
from .training_harness import load_run_configs, resolve_device, scaler_source_identity, TrainingConfig

REPO = Path(__file__).resolve().parents[3]
MANIFEST_VERSION = "phase5-neural-evaluation-run-manifest-1.0.0"
IN_SAMPLE = "IN_SAMPLE_MODEL_TRAINED_ON_THESE_WINDOWS"
CLAIMS = dict(TRAIN_CLAIMS, baseline_superiority_established=False, note="In-sample development evaluation of a model trained on the same pilot windows; not independent evidence.")
LIMITATIONS = [
    "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT and IN-SAMPLE: the neural model was trained on every window evaluated here (one video, one scene group, overlapping windows, perceived-track pseudo-targets). "
    "The classical baselines have no trainable parameters. The comparison therefore cannot show that the neural model generalises or is better than any baseline.",
    "Primary metrics use the most-probable mode; minADE@5 / minFDE@5 are best-of-K oracle metrics (ground truth selects the mode) and are never used as the single-trajectory result.",
    "Mode probabilities are not calibrated and no calibration was assessed; no collision / conflict probability exists.",
    "Errors are image-plane pixel distances: depth-dependent, include camera ego-motion, not metric; no TTC / PET / physical interpretation.",
    "Miss rate is NOT_CONFIGURED (no approved pixel threshold). No confidence intervals: windows are correlated and only one video exists.",
    "CPU single-thread offline measurements only; no real-time claim.",
]


@dataclass(frozen=True)
class NeuralEvaluationConfig:
    config_version: str
    label: str
    baseline_evaluation_config_path: str
    p54_reference_results_path: Optional[str]
    model_id: str
    checkpoint: str
    batch_size: int
    device: str
    torch_threads: Optional[int]
    oracle_k: int
    include_baseline_comparison: bool
    sample_status: str

    def __post_init__(self):
        if self.label != PILOT_LABEL:
            raise ConfigError("label must be %s" % PILOT_LABEL)
        for n in ("config_version", "baseline_evaluation_config_path", "model_id", "checkpoint"):
            if not (isinstance(getattr(self, n), str) and getattr(self, n).strip()):
                raise ConfigError("%s must be a non-empty string" % n)
        if not (isinstance(self.batch_size, int) and not isinstance(self.batch_size, bool) and self.batch_size >= 1):
            raise ConfigError("batch_size must be a positive integer")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ConfigError("device must be cpu, cuda or auto")
        if self.torch_threads is not None and not (isinstance(self.torch_threads, int) and self.torch_threads >= 1):
            raise ConfigError("torch_threads must be null or a positive integer")
        if not (isinstance(self.oracle_k, int) and self.oracle_k >= 1):
            raise ConfigError("oracle_k must be a positive integer")
        if self.sample_status != IN_SAMPLE:
            raise ConfigError("sample_status must be %s (the pilot has no independent partition)" % IN_SAMPLE)
        if self.checkpoint != "final" and not self.checkpoint.startswith("checkpoints/"):
            raise ConfigError("checkpoint must be 'final' or a 'checkpoints/...' name recorded in the training manifest")

    @classmethod
    def load(cls, path) -> "NeuralEvaluationConfig":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError("cannot read %s: %s" % (path, exc))
        data = {k: v for k, v in d.items() if not k.startswith("_")}
        unknown, missing = set(data) - set(cls.__dataclass_fields__), set(cls.__dataclass_fields__) - set(data)
        if unknown or missing:
            raise ConfigError("unknown keys %s, missing keys %s" % (sorted(unknown), sorted(missing)))
        return cls(**data)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class LoadedModel:
    model: ForecastingModel
    training_manifest: Dict[str, Any]
    checkpoint_name: str
    checkpoint_record: Dict[str, Any]
    scaler_statistics: TrainOnlyFeatureStatistics
    dataset: TrainingWindowDataset
    verification: Dict[str, Any]


def _rel(p) -> str:
    p = Path(p).resolve()
    return p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else str(p)


def load_trained_model(train_dir, windows_run, *, checkpoint: str = "final", device: str = "cpu") -> LoadedModel:
    d = Path(train_dir)
    mp = d / "run_manifest.json"
    if not mp.is_file():
        raise WindowContractError("TRAIN_RUN_MISSING", "missing %s" % mp)
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m.get("status") != "completed" or m.get("label") != PILOT_LABEL:
        raise WindowContractError("TRAIN_RUN_STATUS", "training run must be completed and labelled %s" % PILOT_LABEL)
    bad = [n for n, r in m["data_artifacts"].items() if not (d / n).is_file() or sha256_file(d / n) != r["sha256"] or (d / n).stat().st_size != r["bytes"]]
    if bad:
        raise WindowContractError("TRAIN_ARTIFACT_HASH", "training artifacts missing or modified: %s" % bad)
    name = m["training"]["final_checkpoint"] if checkpoint == "final" else checkpoint
    if name not in m["data_artifacts"] or not name.startswith("checkpoints/"):
        raise WindowContractError("CHECKPOINT_NOT_IN_MANIFEST", "%s is not a checkpoint recorded in the training manifest" % name)
    rec = m["data_artifacts"][name]
    dataset = TrainingWindowDataset(windows_run)
    inp = m["inputs"]
    if dataset.provenance["run_manifest_sha256"] != inp["windows_run_manifest_sha256"] or dataset.config.sha256() != inp["window_config_sha256"]:
        raise WindowContractError("DATASET_PROVENANCE_MISMATCH", "the window run differs from the one the model was trained on")
    tcfg = TrainingConfig.load(REPO / inp["training_config"]["path"])
    enc_cfg, dec_cfg, enc_path, dec_path = load_run_configs(tcfg)
    resolved = json.loads((d / "config_resolved.json").read_text(encoding="utf-8"))
    if tcfg.sha256() != inp["training_config"]["sha256_canonical"] or resolved["training_config"] != tcfg.to_dict():
        raise WindowContractError("CONFIG_HASH_MISMATCH", "training configuration differs from the recorded one")
    if enc_cfg.sha256() != inp["encoder_config"]["sha256_canonical"] or dec_cfg.sha256() != inp["decoder_config"]["sha256_canonical"]:
        raise WindowContractError("CONFIG_HASH_MISMATCH", "encoder or decoder configuration differs from the recorded one")
    stats = TrainOnlyFeatureStatistics.load(d / "scaler_statistics.json", expected_source_identity=scaler_source_identity(dataset),
                                            expected_split_identity=inp["scaler"]["split_identity"])
    if stats.compute_hash() != inp["scaler"]["statistics_sha256"]:
        raise WindowContractError("SCALER_HASH_MISMATCH", "scaler statistics hash differs from the training manifest")
    model = ForecastingModel(enc_cfg, dec_cfg, dataset.config, scaler=stats.feature_scaler(), decoder_seed=tcfg.decoder_seed)
    payload = load_checkpoint(d / name, expected_file_sha256=rec["sha256"], expected_model_state_sha256=rec["model_state_sha256"], model=model)
    md = payload["metadata"]
    for k, v in (("scaler_statistics_sha256", stats.compute_hash()), ("encoder_config_sha256", enc_cfg.sha256()), ("decoder_config_sha256", dec_cfg.sha256()),
                 ("window_config_sha256", dataset.config.sha256()), ("windows_run_manifest_sha256", dataset.provenance["run_manifest_sha256"])):
        if md.get(k) != v:
            raise WindowContractError("CHECKPOINT_PROVENANCE_MISMATCH", "checkpoint metadata %s differs from the verified artifact" % k)
    ref = stats.feature_scaler()
    for buf in ("node_offset", "node_scale", "edge_offset", "edge_scale"):
        if not torch.equal(getattr(model.encoder.scaler, buf), getattr(ref, buf)):
            raise WindowContractError("SCALER_STATE_MISMATCH", "checkpoint scaler buffer %s differs from the verified statistics" % buf)
    dev = resolve_device(device)
    model.to(dev).eval()
    verification = {"training_artifacts_verified": sorted(m["data_artifacts"]), "checkpoint": name, "checkpoint_file_sha256": rec["sha256"],
                    "checkpoint_model_state_sha256": rec["model_state_sha256"], "scaler_statistics_sha256": stats.compute_hash(),
                    "scaler_split_identity": inp["scaler"]["split_identity"], "scaler_source_identity_verified": True, "scaler_buffers_equal_statistics": True,
                    "config_hashes_verified": True, "window_run_manifest_sha256": dataset.provenance["run_manifest_sha256"], "checkpoint_epoch": int(payload["epoch"])}
    return LoadedModel(model, m, name, rec, stats, dataset, verification)


def make_inference_inputs(batch: Mapping[str, Any], device):
    """History-only model inputs from a collated batch: no future array is even referenced."""
    enc = TrainingWindowDataset.encoder_input(batch).to(device)
    anchor = torch.as_tensor(np.asarray(batch["anchor_uv_px"]), dtype=torch.float32).to(device)
    valid = torch.as_tensor(np.asarray(batch["target_valid"]), dtype=torch.bool).to(device)
    return enc, anchor, valid


def predict_windows(model: ForecastingModel, dataset: TrainingWindowDataset, batch_size: int, indices=None) -> Dict[str, np.ndarray]:
    """K-mode predictions for the selected windows (all by default), in deterministic order, with the model in eval mode."""
    idx = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    device = next(model.parameters()).device
    model.eval()
    pos, valid, prob, anchors, ids = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            batch = dataset.collate([dataset[i] for i in idx[s:s + batch_size]])
            enc, anchor, tv = make_inference_inputs(batch, device)
            dec = model(enc, anchor, tv).decoded
            pos.append(dec.trajectories_uv_px.detach().cpu().numpy())
            valid.append(dec.prediction_valid.detach().cpu().numpy())
            prob.append(dec.mode_probabilities.detach().cpu().numpy())
            anchors.append(anchor.detach().cpu().numpy())
            ids.append(np.asarray(batch["metadata"]["window_id"], dtype=np.int64))
    return {"window_id": np.concatenate(ids), "positions_uv_px": np.concatenate(pos).astype(np.float32), "valid": np.concatenate(valid),
            "mode_probabilities": np.concatenate(prob).astype(np.float32), "anchor_uv_px": np.concatenate(anchors).astype(np.float32)}


def mode_diagnostics(pred: Mapping[str, np.ndarray], gt: np.ndarray, gt_valid: np.ndarray) -> Dict[str, Any]:
    """Plain-numpy diagnostics: how often the most-probable mode is the oracle-best mode, selected-mode histogram, mode spread (all in-sample)."""
    p, pr = pred["positions_uv_px"].astype(np.float64), pred["mode_probabilities"].astype(np.float64)
    ok = gt_valid.all(axis=1)
    d = np.linalg.norm(p - gt[:, None], axis=-1).mean(axis=2)                       # [N, K] per-mode ADE (px)
    sel, best = pr.argmax(axis=1), d.argmin(axis=1)
    K = p.shape[1]
    spread = np.linalg.norm(p[:, :, None] - p[:, None, :], axis=-1).mean(axis=-1).sum(axis=(1, 2)) / max(K * (K - 1), 1)
    return {"windows": int(ok.sum()), "selected_equals_oracle_best_fraction": float((sel[ok] == best[ok]).mean()),
            "selected_mode_histogram": np.bincount(sel[ok], minlength=K).tolist(), "oracle_best_mode_histogram": np.bincount(best[ok], minlength=K).tolist(),
            "mean_pairwise_mode_distance_px": float(spread[ok].mean()), "mean_max_probability": float(pr.max(axis=1)[ok].mean()),
            "mean_probability_entropy": float((-(pr * np.log(np.clip(pr, 1e-12, None))).sum(axis=1))[ok].mean()),
            "note": "diagnostics on the training windows (in-sample); probabilities are not calibrated"}


def _short(s: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in s.items() if k != "per_track"}


def evaluate_neural(loaded: LoadedModel, ncfg: NeuralEvaluationConfig, baseline_cfg: Mapping[str, Any], ecfg: EvalConfig):
    ds = loaded.dataset
    pred = predict_windows(loaded.model, ds, ncfg.batch_size)
    if not np.array_equal(pred["window_id"], ds.ws.window_id):
        raise WindowContractError("WINDOW_ORDER", "prediction order differs from the window dataset")
    wcfg = ds.config
    hz = wcfg.future_offsets / wcfg.source_fps
    gt, gv = ds.ws.target_future_uv_px.astype(np.float64), ds.ws.target_future_valid
    if pred["positions_uv_px"].shape[1] != ncfg.oracle_k:
        raise WindowContractError("ORACLE_K_MISMATCH", "model has %d modes but oracle_k=%d" % (pred["positions_uv_px"].shape[1], ncfg.oracle_k))
    neural = evaluate_predictions(gt, gv, pred["positions_uv_px"], pred["valid"], ds.ws.window_id, ds.ws.target_track_id, coordinate_system=wcfg.coordinate_system,
                                  unit=wcfg.coordinate_unit, model_id=ncfg.model_id, horizon_offsets_s=hz, cfg=ecfg, mode_probabilities=pred["mode_probabilities"])
    results = {ncfg.model_id: neural}
    comparison = None
    if ncfg.include_baseline_comparison:
        hist = build_history_input(ds.ws, ds.bank, ds.extra["history_target_bank_node"], wcfg)
        for spec in baseline_cfg["baselines"]:
            pr = make_predictor(spec).predict(hist)
            results[spec["baseline_id"]] = evaluate_predictions(gt, gv, pr.positions, pr.valid, ds.ws.window_id, ds.ws.target_track_id,
                                                                coordinate_system=wcfg.coordinate_system, unit=wcfg.coordinate_unit, model_id=spec["baseline_id"],
                                                                horizon_offsets_s=hz, cfg=ecfg)
        common = common_window_mask(list(results.values()))
        comparison = {"definition": "windows evaluated by the neural model and every baseline (same set for all)", "n_windows": int(common.sum()),
                      "n_tracks": int(np.unique(ds.ws.target_track_id[common]).shape[0]),
                      "status": IN_SAMPLE + "; the baselines have no trainable parameters",
                      "per_model": {k: _short(summarize(r, hz, ecfg, common)) for k, r in results.items()}}
    return pred, results, comparison, mode_diagnostics(pred, gt, gv), hz


def write_neural_predictions(path: Path, pred: Mapping[str, np.ndarray], ds: TrainingWindowDataset, model_id: str, hz: np.ndarray, wcfg) -> str:
    return write_npz(path, {"window_id": pred["window_id"], "target_track_index": ds.ws.target_track_index, "anchor_frame": ds.ws.anchor_frame,
                            "positions_uv_px": pred["positions_uv_px"], "valid": pred["valid"], "mode_probabilities": pred["mode_probabilities"],
                            "anchor_uv_px": pred["anchor_uv_px"], "horizon_offsets_s": hz, "coordinate_system": np.array(wcfg.coordinate_system),
                            "position_unit": np.array(wcfg.coordinate_unit), "model_id": np.array(model_id)})


def write_neural_metrics(path: Path, r: EvaluationResult) -> str:
    return write_npz(path, {"window_id": r.window_id, "track_id": r.track_id, "evaluated": r.evaluated, "ade_px": r.ade, "fde_px": r.fde, "step_error_px": r.step_error,
                            "min_ade_px": r.min_ade, "min_fde_px": r.min_fde, "oracle_evaluated": r.oracle_evaluated, "k": np.array(r.k)})


def render_report(res: Dict[str, Any]) -> str:
    f = lambda x: "—" if x is None else "%.2f" % x                                               # noqa: E731
    n = res["neural_model"]
    s = n["summary"]
    ww, tw = s["window_weighted"], s["track_weighted"]
    o = s["oracle_best_of_k"]
    L = ["# Phase 5 P5.7.5 neural evaluation — PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT, IN-SAMPLE", "",
         "**The neural model was trained on every window evaluated here** (one PIE video, one scene group, overlapping windows, perceived-track pseudo-targets). "
         "These numbers are in-sample development diagnostics. They do not measure generalisation and do not show that the model is better than any baseline.", "",
         "Model `%s` (checkpoint `%s`, epoch %d), K = %d modes. Windows evaluated: %d of %d; tracks: %d. Miss rate: %s." % (
             n["model_id"], n["checkpoint"], n["checkpoint_epoch"], n["num_modes"], s["windows_evaluated"], s["windows_considered"], s["unique_target_tracks_evaluated"],
             res["policies"]["miss_rate"]), "",
         "## Primary (most-probable mode) — pixels", "", "| aggregation | ADE mean | ADE median | ADE std | FDE mean | FDE median | FDE std |", "|---|---|---|---|---|---|---|",
         "| window-weighted | %s | %s | %s | %s | %s | %s |" % (f(ww["ade_px"]["mean"]), f(ww["ade_px"]["median"]), f(ww["ade_px"]["std"]), f(ww["fde_px"]["mean"]), f(ww["fde_px"]["median"]), f(ww["fde_px"]["std"])),
         "| track-weighted (%d tracks) | %s | %s | — | %s | %s | — |" % (tw["n_tracks"], f(tw["ade_px_mean_of_track_means"]), f(tw["ade_px_median_of_track_means"]),
                                                                       f(tw["fde_px_mean_of_track_means"]), f(tw["fde_px_median_of_track_means"])), "",
         "Error by horizon (window-weighted mean, px): " + ", ".join("%s: %s" % (k, f(v)) for k, v in s["per_horizon_mean_displacement_error_px"]["at_report_horizons"].items()), "",
         "## Best-of-K ORACLE (ground truth selects the mode) — reported separately, not a single-trajectory result", "",
         "| metric | mean | median |", "|---|---|---|", "| minADE@%d | %s | %s |" % (o["k"], f(o["min_ade_px"]["mean"]), f(o["min_ade_px"]["median"])),
         "| minFDE@%d | %s | %s |" % (o["k"], f(o["min_fde_px"]["mean"]), f(o["min_fde_px"]["median"])), "",
         "Mode diagnostics (in-sample): selected mode equals the oracle-best mode in %.1f%% of windows; selected-mode histogram %s; mean pairwise mode distance %.1f px; "
         "mean max-probability %.3f (uncalibrated)." % (100 * res["mode_diagnostics"]["selected_equals_oracle_best_fraction"], res["mode_diagnostics"]["selected_mode_histogram"],
                                                      res["mode_diagnostics"]["mean_pairwise_mode_distance_px"], res["mode_diagnostics"]["mean_max_probability"]), ""]
    c = res.get("baseline_comparison")
    if c:
        L += ["## Comparison on common windows (%d windows, %d tracks) — %s" % (c["n_windows"], c["n_tracks"], c["status"]), "",
              "| model | ADE mean | FDE mean | ADE median | FDE median | err @0.5 s | @1.0 s | @1.5 s |", "|---|---|---|---|---|---|---|---|"]
        for k, e in c["per_model"].items():
            at = e["per_horizon_mean_displacement_error_px"]["at_report_horizons"]
            L.append("| %s (%s) | %s | %s | %s | %s | %s | %s | %s |" % (k, "most-probable mode" if k == n["model_id"] else "single trajectory", f(e["window_weighted"]["ade_px"]["mean"]),
                                                                    f(e["window_weighted"]["fde_px"]["mean"]), f(e["window_weighted"]["ade_px"]["median"]),
                                                                    f(e["window_weighted"]["fde_px"]["median"]), f(at.get("0.5s")), f(at.get("1.0s")), f(at.get("1.5s"))))
        L += ["", "This table is not evidence of superiority: the neural row is in-sample on one video.", ""]
    L += ["Units: pixels of `captured_1920x1080`. No metric distance, TTC, PET or collision probability is computed.", ""]
    return "\n".join(L)


def run_neural_evaluation(train_dir, windows_run, config_path, out_dir, *, command: Optional[str] = None) -> Dict[str, Any]:
    out = check_output_dir(out_dir)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    git = git_provenance()
    ncfg = NeuralEvaluationConfig.load(config_path)
    if ncfg.torch_threads is not None:
        torch.set_num_threads(ncfg.torch_threads)
    bcfg_path = REPO / ncfg.baseline_evaluation_config_path
    bcfg, bsha = load_baseline_config(bcfg_path)
    ecfg = eval_config_from(bcfg)
    verify_window_run(windows_run, bcfg)                                   # P5.4 check: window run + window configuration reference
    loaded = load_trained_model(train_dir, windows_run, checkpoint=ncfg.checkpoint, device=ncfg.device)
    t1 = time.perf_counter()
    pred, results, comparison, diag, hz = evaluate_neural(loaded, ncfg, bcfg, ecfg)
    t2 = time.perf_counter()
    ds, wcfg = loaded.dataset, loaded.dataset.config
    neural = results[ncfg.model_id]
    out.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Dict[str, Any]] = {}

    def rec(name, sha):
        files[name] = {"sha256": sha, "bytes": (out / name).stat().st_size}
    rec("predictions_neural.npz", write_neural_predictions(out / "predictions_neural.npz", pred, ds, ncfg.model_id, hz, wcfg))
    rec("evaluation_neural.npz", write_neural_metrics(out / "evaluation_neural.npz", neural))
    p54_check = None
    if ncfg.p54_reference_results_path and (REPO / ncfg.p54_reference_results_path).is_file() and comparison:
        ref = json.loads((REPO / ncfg.p54_reference_results_path).read_text(encoding="utf-8"))["baselines"]
        p54_check = {b: {"p54_ade_mean_px": ref[b]["summary"]["window_weighted"]["ade_px"]["mean"], "recomputed_ade_mean_px": results[b].summary["window_weighted"]["ade_px"]["mean"],
                         "equal": abs(ref[b]["summary"]["window_weighted"]["ade_px"]["mean"] - results[b].summary["window_weighted"]["ade_px"]["mean"]) < 1e-9}
                     for b in ref if b in results}
    res = {"label": PILOT_LABEL, "sample_status": ncfg.sample_status, "results_version": "phase5-neural-evaluation-results-1.0.0",
           "neural_model": {"model_id": ncfg.model_id, "checkpoint": loaded.checkpoint_name, "checkpoint_epoch": loaded.verification["checkpoint_epoch"],
                            "num_modes": int(pred["positions_uv_px"].shape[1]), "primary_selection": "most probable mode (argmax of the model's own mode probabilities)",
                            "summary": neural.summary, "parameter_counts": loaded.model.parameter_counts()},
           "oracle_note": "minADE@K / minFDE@K inside neural_model.summary.oracle_best_of_k are best-of-K oracle metrics and are separate from the primary metrics",
           "mode_diagnostics": diag, "baseline_comparison": comparison, "p54_reference_check": p54_check,
           "verification": loaded.verification,
           "policies": {"partial_target_policy": ecfg.partial_target_policy, "invalid_prediction_policy": ecfg.invalid_prediction_policy,
                        "miss_rate": "NOT_CONFIGURED" if ecfg.miss_rate_threshold_px is None else "%.3f px" % ecfg.miss_rate_threshold_px, "aggregation": bcfg["aggregation"]},
           "coordinates": {"coordinate_system": wcfg.coordinate_system, "unit": wcfg.coordinate_unit, "origin": "top-left", "world_coordinates": "unavailable"},
           "horizon_offsets_s": [round(float(x), 9) for x in hz], "claims": CLAIMS, "limitations": LIMITATIONS}
    rec("evaluation_results.json", write_json(out / "evaluation_results.json", res))
    rec("evaluation_report.md", write_text(out / "evaluation_report.md", render_report(res)))
    finished = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "manifest_version": MANIFEST_VERSION, "task": "phase5_p5.7.5_neural_evaluation", "status": "completed", "label": PILOT_LABEL, "sample_status": ncfg.sample_status,
        "data_artifacts": dict(sorted(files.items())), "data_artifacts_note": "deterministic content: identical checkpoint, scaler, windows and configuration give identical hashes",
        "inputs": {"training_run": _rel(train_dir), "training_run_manifest_sha256": sha256_file(Path(train_dir) / "run_manifest.json"), "checkpoint": loaded.checkpoint_name,
                   "checkpoint_file_sha256": loaded.verification["checkpoint_file_sha256"], "checkpoint_model_state_sha256": loaded.verification["checkpoint_model_state_sha256"],
                   "scaler_statistics_sha256": loaded.verification["scaler_statistics_sha256"], "windows_run": _rel(windows_run),
                   "windows_run_manifest_sha256": loaded.verification["window_run_manifest_sha256"],
                   "neural_evaluation_config": {"path": _rel(config_path), "sha256_canonical": ncfg.sha256(), "sha256_file": sha256_file(Path(config_path))},
                   "baseline_evaluation_config": {"path": ncfg.baseline_evaluation_config_path, "sha256_canonical": bsha},
                   "future_arrays_in_model_input": False, "offline_kinematics_used": False},
        "counts": {"windows": len(ds), "unique_target_tracks": int(np.unique(ds.ws.target_track_id).shape[0]), "windows_evaluated": neural.summary["windows_evaluated"],
                   "num_modes": int(pred["positions_uv_px"].shape[1]), "common_windows_with_baselines": comparison["n_windows"] if comparison else None},
        "coordinates": res["coordinates"],
        "run_metadata": {"git": git, "started_utc": started, "finished_utc": finished, "command": command,
                         "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(),
                                      "torch_threads": torch.get_num_threads(), "device": str(next(loaded.model.parameters()).device)},
                         "runtime_s": {"verify_and_load": t1 - t0, "inference_and_evaluation": t2 - t1, "total": time.perf_counter() - t0,
                                       "note": "offline CPU measurement; not a real-time claim"}, "peak_memory_bytes": peak_memory_bytes()},
        "claims": CLAIMS, "limitations": LIMITATIONS}
    write_json(out / "run_manifest.json", manifest)
    return manifest


__all__ = ["NeuralEvaluationConfig", "LoadedModel", "load_trained_model", "predict_windows", "make_inference_inputs", "evaluate_neural", "mode_diagnostics",
           "run_neural_evaluation", "IN_SAMPLE", "CLAIMS", "LIMITATIONS"]
