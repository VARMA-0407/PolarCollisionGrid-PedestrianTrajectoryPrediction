"""Multi-video Phase 5 training on the TRAIN partition of the frozen recording-level split.

Why this module exists: ``pedsafety.phase5.models.training_harness.run_training`` accepts exactly one P5.3 window run (one video) and its
``split_policy`` admits only the single-video pilot policy. Training on all train-partition videos therefore needs a wrapper. This module
reuses, unchanged, every scientific component of the harness: ``TrainingConfig`` validation and hyperparameters, ``ForecastingModel``
(heterogeneous GATv2 + causal TCN + multimodal decoder), ``build_optimizer`` (AdamW, decay on weight matrices only), ``build_scheduler``
(cosine per optimizer step), ``make_train_batch`` / ``train_step`` / ``eval_step`` (masked winner-takes-all loss), ``fit_feature_statistics``
(train-only scaler), ``save_checkpoint`` / ``load_checkpoint`` (hash-verified, never overwritten).

Documented deviations from the single-video harness (recorded in every manifest):
* ``split_policy = RECORDING_LEVEL_FROZEN_SPLIT_TRAIN_PARTITION_ONLY`` (the scaler and the model see train-partition windows only);
* ``batch_policy = VIDEO_HOMOGENEOUS_BATCHES``: each mini-batch holds windows of one video, because a window's graph lives in its video's
  graph bank; the order of batches across videos is shuffled per epoch with ``default_rng([seed, epoch])``;
* a per-epoch validation MONITOR (loss and selected/oracle-mode ADE on validation-partition windows, pseudo-targets) is logged; it is
  never used to select a checkpoint (``checkpoint_selection = FINAL_EPOCH_ONLY`` is kept);
* GPU execution is allowed (the harness was only run on CPU); CUDA kernels are not bitwise deterministic, which is recorded.

Resume: checkpoints are written every ``checkpoint_every_epochs``; a restarted run continues from the newest checkpoint after verifying
its configuration, scaler and dataset identity hashes, and the log is truncated to that epoch.
"""

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..interaction.runner import peak_memory_bytes
from ..phase5.models.checkpoint import load_checkpoint, save_checkpoint
from ..phase5.models.forecaster import ForecastingModel
from ..phase5.models.training_harness import (TrainingConfig, build_optimizer, build_scheduler, load_run_configs, resolve_device,
                                               torch)       # torch is reached through the Phase 5 models package (dependency policy: torch only under phase5/models)
from ..phase5.models.training_step import eval_step, make_train_batch, train_step
from ..phase5.training_data import TrainingWindowDataset
from ..phase5.training_scaler import TrainOnlyFeatureStatistics, fit_feature_statistics
from . import MULTIVIDEO_LABEL, TARGET_SOURCE
from .common import StageError, canonical_sha256, move_aside, read_json, sha256_file, utc_now, write_json

MANIFEST_VERSION = "multivideo-phase5-training-run-manifest-1.0.0"
SPLIT_POLICY = "RECORDING_LEVEL_FROZEN_SPLIT_TRAIN_PARTITION_ONLY"
BATCH_POLICY = "VIDEO_HOMOGENEOUS_BATCHES"
OVERRIDABLE = ("epochs", "batch_size", "learning_rate", "device", "torch_threads", "checkpoint_every_epochs", "log_every_steps", "scaler_batch_size")
CLAIMS = {"coordinate_space": "image (captured raster pixels)", "metric_world_coordinates": False, "ttc_pet_or_collision_risk": False,
          "annotation_ground_truth_targets": False, "independent_generalization_established": False, "best_checkpoint_selected": False,
          "calibrated_probabilities": False, "real_time_claim": False}
LIMITATIONS = [
    "Targets are perceived-track pseudo-targets (YOLO + ByteTrack), not PIE annotation ground truth; ANNOTATED_TARGET is not implemented.",
    "Image-plane pixels of a moving camera; no ego-motion compensation, no metric or physical interpretation.",
    "Hyperparameters are the repository defaults (a-priori engineering values chosen for the single-video pilot); no search was performed.",
    "Checkpoint selection is FINAL_EPOCH_ONLY; the validation monitor is logged but not used for selection.",
    "CUDA training is not bitwise reproducible; CPU with torch_threads=1 is.",
]


def _count_windows(run_dir: Path) -> int:
    m = read_json(Path(run_dir) / "run_manifest.json")
    return int(m["counts"]["accepted_windows"])


class MultiVideoWindows:
    """Ordered collection of per-video ``TrainingWindowDataset`` views (all must share one window configuration)."""

    def __init__(self, window_dirs: Dict[str, Any], *, groups: Dict[str, str], include_optional: bool = False):
        self.datasets: Dict[str, TrainingWindowDataset] = {}
        self.empty: List[str] = []
        cfg_sha = set()
        for k in sorted(window_dirs):
            d = Path(window_dirs[k])
            if _count_windows(d) == 0:
                self.empty.append(k)
                continue
            ds = TrainingWindowDataset(d, include_optional=include_optional)
            vids = set(map(str, ds.extra["video_id"].tolist()))
            gids = set(map(str, ds.extra["group_id"].tolist()))
            if vids != {k} or gids != {groups[k]}:
                raise StageError("WINDOW_IDENTITY", "%s holds video ids %s / group ids %s, expected %s / %s" % (d, vids, gids, k, groups[k]))
            cfg_sha.add(ds.config.sha256())
            self.datasets[k] = ds
        if len(cfg_sha) > 1:
            raise StageError("WINDOW_CONFIG_MISMATCH", "window runs use different window configurations: %s" % sorted(cfg_sha))
        if not self.datasets:
            raise StageError("NO_WINDOWS", "no video of this partition has accepted windows")
        self.keys = sorted(self.datasets)
        self.config = self.datasets[self.keys[0]].config
        self.identity = [{"video": k, "group": groups[k], "windows": len(self.datasets[k]), "run_manifest_sha256": self.datasets[k].provenance["run_manifest_sha256"]}
                         for k in self.keys]
        self.identity_sha256 = canonical_sha256(self.identity)

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets.values())

    def steps_per_epoch(self, batch_size: int) -> int:
        return sum(-(-len(self.datasets[k]) // batch_size) for k in self.keys)

    def epoch_batches(self, batch_size: int, seed: int, epoch: int, shuffle: bool) -> List[Tuple[str, np.ndarray]]:
        rng = np.random.default_rng([seed, epoch])
        out: List[Tuple[str, np.ndarray]] = []
        for k in self.keys:
            n = len(self.datasets[k])
            idx = rng.permutation(n) if shuffle else np.arange(n)
            out += [(k, idx[s:s + batch_size]) for s in range(0, n, batch_size)]
        order = rng.permutation(len(out)) if shuffle else np.arange(len(out))
        return [out[i] for i in order]

    def collate(self, key: str, idx: Sequence[int]) -> Dict[str, Any]:
        ds = self.datasets[key]
        return ds.collate([ds[int(i)] for i in idx])

    def sequential_batches(self, batch_size: int, max_windows: Optional[int] = None):
        """Deterministic evenly spaced subset (at most ``max_windows``) in fixed order, for the scaler (None = all) and the validation monitor."""
        total = len(self)
        step = 1 if not max_windows or total <= max_windows else int(np.ceil(total / max_windows))
        for k in self.keys:
            idx = np.arange(0, len(self.datasets[k]), step)
            for s in range(0, len(idx), batch_size):
                yield k, idx[s:s + batch_size]


def _resolved(tcfg: TrainingConfig, overrides: Dict[str, Any], mv: MultiVideoWindows, split: Dict[str, Any], monitor_cfg: Dict[str, Any]) -> Dict[str, Any]:
    enc, dec, _, _ = load_run_configs(tcfg)
    return {"training_config": tcfg.to_dict(), "overrides_applied": dict(sorted(overrides.items())), "encoder_config": enc.to_dict(), "decoder_config": dec.to_dict(),
            "encoder_config_sha256": enc.sha256(), "decoder_config_sha256": dec.sha256(), "window_config": mv.config.to_dict(), "window_config_sha256": mv.config.sha256(),
            "split_id": split["split_id"], "split_content_sha256": split["content_sha256"], "train_identity": mv.identity, "train_identity_sha256": mv.identity_sha256,
            "split_policy": SPLIT_POLICY, "batch_policy": BATCH_POLICY, "checkpoint_selection": tcfg.checkpoint_selection, "monitor": monitor_cfg,
            "steps_per_epoch": mv.steps_per_epoch(tcfg.batch_size), "total_steps": mv.steps_per_epoch(tcfg.batch_size) * tcfg.epochs, "label": MULTIVIDEO_LABEL,
            "target_source": TARGET_SOURCE,
            "ignored_base_fields": {"split_policy": tcfg.split_policy, "scaler_split_identity": tcfg.scaler_split_identity,
                                    "note": "single-video pilot fields kept only so the unchanged TrainingConfig validator accepts the configuration; the multi-video run uses split_policy/split_id above"}}


def _clean(v):
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return _clean(v.item())
    return v


def _monitor(model, mv: Optional[MultiVideoWindows], dec_cfg, tcfg, device, max_windows: int) -> Optional[Dict[str, Any]]:
    if mv is None:
        return None
    sums = {"loss": 0.0, "selected_mode_ade_px": 0.0, "oracle_mode_ade_px": 0.0}
    n = 0
    for k, idx in mv.sequential_batches(tcfg.batch_size, max_windows):
        r = eval_step(model, make_train_batch(mv.collate(k, idx), device), classification_weight=dec_cfg.classification_weight, smooth_l1_beta=tcfg.smooth_l1_beta)
        w = r.usable_windows
        n += w
        for key in sums:
            v = getattr(r, key)
            sums[key] += (v if np.isfinite(v) else 0.0) * w
    return {"windows": n, **{"mean_" + k if not k.startswith("mean_") else k: (sums[k] / n if n else None) for k in sums}}


def train_multivideo(train_dirs: Dict[str, Any], split: Dict[str, Any], base_training_config_path, out_dir, *, overrides: Optional[Dict[str, Any]] = None,
                     val_dirs: Optional[Dict[str, Any]] = None, monitor_max_windows: int = 4096, quarantine_root=None, git: Optional[Dict[str, Any]] = None,
                     progress=print, verify_only: bool = False) -> Dict[str, Any]:
    out = Path(out_dir)
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    bad = sorted(set(overrides) - set(OVERRIDABLE))
    if bad:
        raise StageError("OVERRIDE", "only %s may be overridden, got %s" % (OVERRIDABLE, bad))
    parts = {v: g["partition"] for g in split["groups"].values() for v in g["videos"]}
    groups = {v: s for s, g in split["groups"].items() for v in g["videos"]}
    leak = sorted(k for k in train_dirs if parts.get(k) != "train")
    if leak:
        raise StageError("TRAIN_PARTITION_LEAK", "videos outside the train partition were passed for training: %s" % leak)
    vleak = sorted(k for k in (val_dirs or {}) if parts.get(k) != "validation")
    if vleak:
        raise StageError("MONITOR_PARTITION", "the validation monitor accepts validation-partition videos only: %s" % vleak)
    tcfg = dataclasses.replace(TrainingConfig.load(base_training_config_path), **overrides)
    enc_cfg, dec_cfg, _, _ = load_run_configs(tcfg)
    if tcfg.torch_threads is not None:
        torch.set_num_threads(tcfg.torch_threads)
    device = resolve_device(tcfg.device)
    mv = MultiVideoWindows(train_dirs, groups=groups, include_optional=tcfg.include_optional_context)
    mval = MultiVideoWindows(val_dirs, groups=groups, include_optional=tcfg.include_optional_context) if val_dirs else None
    if mval is not None and mval.config.sha256() != mv.config.sha256():
        raise StageError("WINDOW_CONFIG_MISMATCH", "validation windows use a different window configuration")
    monitor_cfg = {"partition": "validation", "videos": mval.keys if mval else [], "max_windows": monitor_max_windows, "used_for_selection": False}
    resolved = _resolved(tcfg, overrides, mv, split, monitor_cfg)
    res_sha = canonical_sha256(resolved)
    total_steps, spe = resolved["total_steps"], resolved["steps_per_epoch"]
    # ---- resume / skip / fresh
    if (out / "run_manifest.json").is_file():
        m = read_json(out / "run_manifest.json")
        if m.get("status") == "completed" and m.get("inputs", {}).get("resolved_config_sha256") == res_sha:
            progress("training already completed in %s (resolved configuration identical): skipped" % out)
            return dict(m, skipped_already_complete=True)
        raise StageError("STALE_TRAINING_RUN", "%s holds a training run of a different configuration or dataset; use a new experiment" % out)
    if verify_only:
        raise StageError("NOT_TRAINED_VERIFY_ONLY", "training is disabled and %s holds no completed run of the current configuration" % out)
    ckpts = sorted((out / "checkpoints").glob("checkpoint_epoch_*.pt")) if (out / "checkpoints").is_dir() else []
    if not ckpts and out.exists() and any(out.iterdir()):
        if quarantine_root is None:
            raise StageError("TRAIN_DIR_NOT_EMPTY", "%s is not empty and has no checkpoint" % out)
        move_aside(out, quarantine_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    t_start = time.perf_counter()
    state_path = out / "training_state.json"
    state = read_json(state_path) if state_path.is_file() and ckpts else {"started_utc": utc_now(), "sessions": [], "epoch_wall_s": {}}
    state["sessions"].append({"utc": utc_now(), "device": str(device), "git": git})
    scaler_path = out / "scaler_statistics.json"
    source_identity = {"multivideo_train_identity_sha256": mv.identity_sha256, "videos": mv.identity, "split_id": split["split_id"],
                       "split_content_sha256": split["content_sha256"], "label": MULTIVIDEO_LABEL}
    t_scaler = 0.0
    if ckpts:
        if read_json(out / "config_resolved.json") != json.loads(json.dumps(resolved, default=str)):
            raise StageError("RESUME_CONFIG_MISMATCH", "the resolved configuration differs from the interrupted run in %s" % out)
        stats = TrainOnlyFeatureStatistics.load(scaler_path, expected_source_identity=source_identity, expected_split_identity=split["split_id"])
    else:
        write_json(out / "config_resolved.json", resolved)
        t0 = time.perf_counter()
        progress("fitting train-only scaler statistics on %d windows of %d train videos ..." % (len(mv), len(mv.keys)))
        stats = fit_feature_statistics((mv.collate(k, idx) for k, idx in mv.sequential_batches(tcfg.scaler_batch_size)), fit_partition="train",
                                       split_identity=split["split_id"], source_identity=source_identity,
                                       configuration={"fit_windows": len(mv), "split_policy": SPLIT_POLICY, "fit_videos": mv.keys})
        stats.save(scaler_path)
        t_scaler = time.perf_counter() - t0
    stats_hash = stats.compute_hash()
    model = ForecastingModel(enc_cfg, dec_cfg, mv.config, scaler=stats.feature_scaler(), decoder_seed=tcfg.decoder_seed).to(device)
    optimizer = build_optimizer(model, tcfg)
    scheduler = build_scheduler(optimizer, tcfg, total_steps)
    metadata = {"label": MULTIVIDEO_LABEL, "resolved_config_sha256": res_sha, "training_config_sha256": tcfg.sha256(), "encoder_config_sha256": enc_cfg.sha256(),
                "decoder_config_sha256": dec_cfg.sha256(), "window_config_sha256": mv.config.sha256(), "train_identity_sha256": mv.identity_sha256,
                "split_content_sha256": split["content_sha256"], "scaler_statistics_sha256": stats_hash, "batch_size": tcfg.batch_size, "seed": tcfg.seed}
    start_epoch, global_step = 0, 0
    log_path = out / "train_log.jsonl"
    if ckpts:
        payload = load_checkpoint(ckpts[-1], model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=True)
        md = payload["metadata"]
        diff = [k for k in metadata if md.get(k) != metadata[k]]
        if diff:
            raise StageError("RESUME_PROVENANCE_MISMATCH", "checkpoint %s differs in %s" % (ckpts[-1].name, diff))
        start_epoch, global_step = int(payload["epoch"]), int(payload["global_step"])
        keep = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("epoch", 0) <= start_epoch] if log_path.is_file() else []
        log_path.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
        progress("resuming from %s (epoch %d, step %d)" % (ckpts[-1].name, start_epoch, global_step))
    write_json(state_path, state)
    K = dec_cfg.num_modes
    peak_gpu = None
    t_collate = t_compute = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with open(log_path, "a", encoding="utf-8", newline="\n") as log:
        for epoch in range(start_epoch + 1, tcfg.epochs + 1):
            te = time.perf_counter()
            sums = {k: 0.0 for k in ("loss", "regression_loss", "classification_loss", "selected_mode_ade_px", "oracle_mode_ade_px", "mean_pairwise_mode_distance_px",
                                     "mean_mode_entropy", "max_mode_probability_mean", "grad_norm_total")}
            usable, clipped, winners = 0, 0, np.zeros(K, np.int64)
            for k, idx in mv.epoch_batches(tcfg.batch_size, tcfg.seed, epoch, tcfg.shuffle):
                t0 = time.perf_counter()
                tb = make_train_batch(mv.collate(k, idx), device)
                t1 = time.perf_counter()
                r = train_step(model, optimizer, tb, classification_weight=dec_cfg.classification_weight, grad_clip_norm=tcfg.grad_clip_norm, scheduler=scheduler,
                               smooth_l1_beta=tcfg.smooth_l1_beta)
                t_collate += t1 - t0
                t_compute += time.perf_counter() - t1
                global_step += 1
                w = r.usable_windows
                usable += w
                for key in sums:
                    v = getattr(r, key)
                    sums[key] += (0.0 if (isinstance(v, float) and not np.isfinite(v)) else v) * w
                clipped += int(bool(r.grad_clipped))
                winners += np.asarray(r.winner_counts, np.int64)
                if global_step % tcfg.log_every_steps == 0:
                    log.write(json.dumps({"type": "step", "epoch": epoch, "step": global_step, "video": k, "windows": int(len(idx)),
                                          **{a: _clean(b) for a, b in r.to_dict().items()}}, allow_nan=False, sort_keys=True) + "\n")
            summ = {"type": "epoch", "epoch": epoch, "step": global_step, "windows": int(usable), "steps": spe, "clipped_steps": clipped, "lr_end": optimizer.param_groups[0]["lr"],
                    "winner_counts": winners.tolist(), **{("mean_" + k if not k.startswith("mean_") else k): sums[k] / max(usable, 1) for k in sums}}
            log.write(json.dumps({a: _clean(b) for a, b in summ.items()}, allow_nan=False, sort_keys=True) + "\n")
            mon = _monitor(model, mval, dec_cfg, tcfg, device, monitor_max_windows)
            if mon is not None:
                log.write(json.dumps({"type": "validation_monitor", "epoch": epoch, "used_for_selection": False, **{a: _clean(b) for a, b in mon.items()}},
                                     allow_nan=False, sort_keys=True) + "\n")
            log.flush()
            if epoch % tcfg.checkpoint_every_epochs == 0 or epoch == tcfg.epochs:
                save_checkpoint(out / "checkpoints" / ("checkpoint_epoch_%04d.pt" % epoch), model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
                                global_step=global_step, metadata=dict(metadata, epoch=epoch))
            state["epoch_wall_s"][str(epoch)] = time.perf_counter() - te
            write_json(state_path, state)
            progress("epoch %d/%d  loss %.3f  selected-mode ADE %.1f px  oracle ADE %.1f px%s  (%.0f s)" % (
                epoch, tcfg.epochs, summ["mean_loss"], summ["mean_selected_mode_ade_px"], summ["mean_oracle_mode_ade_px"],
                ("  | val-monitor loss %.3f ADE %.1f px" % (mon["mean_loss"], mon["mean_selected_mode_ade_px"])) if mon and mon["windows"] else "", state["epoch_wall_s"][str(epoch)]))
    if device.type == "cuda":
        peak_gpu = int(torch.cuda.max_memory_allocated())
    final_name = "checkpoints/checkpoint_epoch_%04d.pt" % tcfg.epochs
    files = {}
    for n in ("config_resolved.json", "scaler_statistics.json", "train_log.jsonl"):
        files[n] = {"sha256": sha256_file(out / n), "bytes": (out / n).stat().st_size}
    for p in sorted((out / "checkpoints").glob("checkpoint_epoch_*.pt")):
        pl = load_checkpoint(p)
        files["checkpoints/" + p.name] = {"sha256": pl["file_sha256"], "model_state_sha256": pl["model_state_sha256"], "bytes": p.stat().st_size, "epoch": int(pl["epoch"]),
                                         "global_step": int(pl["global_step"])}
    if final_name not in files:
        raise StageError("FINAL_CHECKPOINT_MISSING", "%s was not written" % final_name)
    env = {"python": __import__("platform").python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
           "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch_threads": torch.get_num_threads()}
    manifest = {"manifest_version": MANIFEST_VERSION, "task": "multivideo_phase5_training", "status": "completed", "label": MULTIVIDEO_LABEL, "data_artifacts": files,
                "inputs": {"resolved_config_sha256": res_sha, "split_id": split["split_id"], "split_content_sha256": split["content_sha256"], "train_videos": mv.identity,
                           "train_videos_without_windows": mv.empty, "train_windows": len(mv), "validation_monitor_videos": monitor_cfg["videos"],
                           "window_config_sha256": mv.config.sha256(), "base_training_config": str(base_training_config_path), "overrides": overrides,
                           "scaler": {"statistics_sha256": stats_hash, "split_identity": split["split_id"], "fit_partition": "train", "fit_windows": len(mv),
                                      "source_identity": source_identity, "split_policy": SPLIT_POLICY},
                           "future_targets_in_model_input": False, "target_source": TARGET_SOURCE},
                "training": {"epochs": tcfg.epochs, "batch_size": tcfg.batch_size, "learning_rate": tcfg.learning_rate, "steps_per_epoch": spe, "total_steps": total_steps,
                             "final_global_step": global_step, "final_checkpoint": final_name, "checkpoint_selection": tcfg.checkpoint_selection, "device": str(device),
                             "parameter_counts": model.parameter_counts(), "split_policy": SPLIT_POLICY, "batch_policy": BATCH_POLICY, "resumed_sessions": len(state["sessions"])},
                "run_metadata": {"finished_utc": utc_now(), "git": git, "environment": env, "runtime_s_this_session": {"total": time.perf_counter() - t_start, "scaler_fit": t_scaler,
                                 "collate": t_collate, "forward_backward": t_compute}, "epoch_wall_s": state["epoch_wall_s"], "peak_memory_bytes": peak_memory_bytes(),
                                 "peak_gpu_memory_bytes": peak_gpu, "determinism": "CPU+torch_threads=1 is bitwise reproducible; CUDA is not"},
                "deviations_from_single_video_harness": ["split_policy " + SPLIT_POLICY, "batch_policy " + BATCH_POLICY,
                                                         "validation monitor logged, never used for selection", "device %s" % device],
                "claims": CLAIMS, "limitations": LIMITATIONS}
    write_json(out / "run_manifest.json", manifest)
    return manifest


def load_multivideo_model(train_dir, *, device: str = "cpu", checkpoint: str = "final"):
    """Verify a completed multi-video training run (artifact hashes, configuration hashes, scaler, checkpoint) and return
    (model, manifest, checkpoint_name, checkpoint_record, stats, verification, window_config)."""
    from ..phase5.window_config import WindowConfig
    d = Path(train_dir)
    m = read_json(d / "run_manifest.json")
    if m.get("status") != "completed" or m.get("label") != MULTIVIDEO_LABEL or m.get("manifest_version") != MANIFEST_VERSION:
        raise StageError("TRAIN_RUN_STATUS", "%s is not a completed multi-video training run" % d)
    bad = [n for n, r in m["data_artifacts"].items() if not (d / n).is_file() or (d / n).stat().st_size != r["bytes"] or sha256_file(d / n) != r["sha256"]]
    if bad:
        raise StageError("TRAIN_ARTIFACT_HASH", "training artifacts missing or modified: %s" % bad)
    resolved = read_json(d / "config_resolved.json")
    if canonical_sha256(resolved) != m["inputs"]["resolved_config_sha256"]:
        raise StageError("CONFIG_HASH_MISMATCH", "config_resolved.json differs from the recorded hash")
    tcfg = TrainingConfig.from_dict(resolved["training_config"])
    enc, dec, _, _ = load_run_configs(tcfg)
    if enc.sha256() != resolved["encoder_config_sha256"] or dec.sha256() != resolved["decoder_config_sha256"]:
        raise StageError("CONFIG_HASH_MISMATCH", "encoder/decoder configuration files changed since training")
    wcfg = WindowConfig.from_dict(resolved["window_config"])
    if wcfg.sha256() != resolved["window_config_sha256"]:
        raise StageError("CONFIG_HASH_MISMATCH", "window configuration hash mismatch")
    sc = m["inputs"]["scaler"]
    stats = TrainOnlyFeatureStatistics.load(d / "scaler_statistics.json", expected_source_identity=sc["source_identity"], expected_split_identity=sc["split_identity"])
    if stats.compute_hash() != sc["statistics_sha256"]:
        raise StageError("SCALER_HASH_MISMATCH", "scaler statistics hash differs from the manifest")
    name = m["training"]["final_checkpoint"] if checkpoint == "final" else checkpoint
    rec = m["data_artifacts"].get(name)
    if rec is None or not name.startswith("checkpoints/"):
        raise StageError("CHECKPOINT_NOT_IN_MANIFEST", name)
    model = ForecastingModel(enc, dec, wcfg, scaler=stats.feature_scaler(), decoder_seed=tcfg.decoder_seed)
    payload = load_checkpoint(d / name, expected_file_sha256=rec["sha256"], expected_model_state_sha256=rec["model_state_sha256"], model=model)
    md = payload["metadata"]
    for k, v in (("scaler_statistics_sha256", stats.compute_hash()), ("resolved_config_sha256", m["inputs"]["resolved_config_sha256"]),
                 ("window_config_sha256", wcfg.sha256()), ("split_content_sha256", m["inputs"]["split_content_sha256"])):
        if md.get(k) != v:
            raise StageError("CHECKPOINT_PROVENANCE_MISMATCH", "checkpoint metadata %s differs" % k)
    ref = stats.feature_scaler()
    for buf in ("node_offset", "node_scale", "edge_offset", "edge_scale"):
        if not torch.equal(getattr(model.encoder.scaler, buf), getattr(ref, buf)):
            raise StageError("SCALER_STATE_MISMATCH", "checkpoint scaler buffer %s differs from the verified statistics" % buf)
    model.to(resolve_device(device)).eval()
    verification = {"training_run": str(d), "training_run_manifest_sha256": sha256_file(d / "run_manifest.json"), "checkpoint": name,
                    "checkpoint_file_sha256": rec["sha256"], "checkpoint_model_state_sha256": rec["model_state_sha256"], "scaler_statistics_sha256": stats.compute_hash(),
                    "scaler_split_identity": sc["split_identity"], "config_hashes_verified": True, "scaler_buffers_equal_statistics": True,
                    "checkpoint_epoch": int(payload["epoch"]), "split_content_sha256": m["inputs"]["split_content_sha256"]}
    return model, m, name, rec, stats, verification, wcfg


__all__ = ["train_multivideo", "load_multivideo_model", "MultiVideoWindows", "SPLIT_POLICY", "BATCH_POLICY", "OVERRIDABLE", "MANIFEST_VERSION"]
