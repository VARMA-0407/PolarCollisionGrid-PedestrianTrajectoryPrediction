"""P5.7.4 pilot training harness (development only): AdamW + cosine schedule, gradient clipping, seeded and resumable.

Scientific scope (recorded in every manifest): the P5.3 pilot is ONE video / ONE scene group with strongly overlapping windows.
There is therefore no independent train/validation/test split. ``split_policy = PILOT_ALL_WINDOWS_AS_TRAIN_DEVELOPMENT_ONLY`` fits
the train-only scaler statistics on all pilot windows (the only "train" partition that exists) and trains on them; the
resulting losses are IN-SAMPLE. ``checkpoint_selection = FINAL_EPOCH_ONLY``: no best-checkpoint selection is made, because the only
available "validation" windows would be correlated neighbours of the training windows and would make a selection indefensible.
Periodic checkpoints exist for resumption and inspection only. Nothing here measures generalisation.

Reproducibility: all randomness is derived from ``seed`` (model initialisation uses separate fixed seeds; the window order of epoch e
is ``default_rng([seed, e])`` so a resumed run continues exactly where the original would). Wall-clock quantities live only in
``run_manifest.json -> run_metadata``; ``train_log.jsonl`` contains deterministic values only.
"""

import datetime
import hashlib
import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from ...errors import ConfigError
from ...image_kinematics_run import sha256_file
from ...interaction.runner import peak_memory_bytes
from ...interaction.storage import write_json
from ...output_dir import check_output_dir
from ..errors import WindowContractError
from ..training_data import TrainingWindowDataset
from ..training_scaler import TrainOnlyFeatureStatistics
from ..window_manifest import PILOT_LABEL, git_provenance
from .checkpoint import load_checkpoint, save_checkpoint
from .config import EncoderConfig
from .decoder import DecoderConfig
from .forecaster import ForecastingModel
from .training_step import StepResult, make_train_batch, train_step

REPO = Path(__file__).resolve().parents[3]
MANIFEST_VERSION = "phase5-training-run-manifest-1.0.0"
SPLIT_POLICIES = ("PILOT_ALL_WINDOWS_AS_TRAIN_DEVELOPMENT_ONLY",)
CHECKPOINT_SELECTIONS = ("FINAL_EPOCH_ONLY",)
SCHEDULERS = ("cosine", "none")
CLAIMS = {"coordinate_space": "image (captured raster pixels)", "metric_world_coordinates": False, "physical_features": False, "ttc_pet_or_collision_risk": False,
          "independent_validation": False, "independent_generalization": False, "forecasting_accuracy_established": False, "best_checkpoint_selected": False,
          "calibrated_probabilities": False, "real_time_claim": False,
          "note": "Pilot training on one video / one scene; all losses are in-sample development diagnostics."}
LIMITATIONS = [
    "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT: one PIE video, one scene group, 2,327 heavily overlapping windows from 84 pedestrian tracks; targets are perceived-track pseudo-targets.",
    "No independent train/validation/test split exists; the model is trained on every pilot window, so every loss / metric computed on these windows is in-sample and says nothing about generalisation.",
    "Checkpoint selection is FINAL_EPOCH_ONLY: best-checkpoint selection on correlated windows of the same video would be indefensible, so none is performed.",
    "Train-only scaler statistics are fitted on all pilot windows because they are the only training partition; they are 'train-only' in procedure, not evidence of out-of-sample behaviour.",
    "Hyperparameters are a-priori engineering defaults for a smoke-scale run; they were not tuned for accuracy and no hyperparameter search was performed.",
    "WTA training on one scene may collapse onto few modes; mode diagnostics are logged but mode quality is not validated.",
    "Image-space pixels only; no metric, physical, TTC/PET, collision-probability or deployment interpretation.",
]


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


@dataclass(frozen=True)
class TrainingConfig:
    config_version: str
    label: str
    encoder_config_path: str
    decoder_config_path: str
    seed: int
    decoder_seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    grad_clip_norm: Optional[float]
    smooth_l1_beta: float
    scheduler: str
    min_lr_ratio: float
    device: str
    torch_threads: Optional[int]
    log_every_steps: int
    checkpoint_every_epochs: int
    shuffle: bool
    scaler_batch_size: int
    scaler_split_identity: str
    split_policy: str
    checkpoint_selection: str
    include_optional_context: bool

    def __post_init__(self):
        for n in ("config_version", "encoder_config_path", "decoder_config_path", "scaler_split_identity"):
            if not (isinstance(getattr(self, n), str) and getattr(self, n).strip()):
                raise ConfigError("%s must be a non-empty string" % n)
        if self.label != PILOT_LABEL:
            raise ConfigError("label must be %s" % PILOT_LABEL)
        for n in ("epochs", "batch_size", "log_every_steps", "checkpoint_every_epochs", "scaler_batch_size"):
            if not _int(getattr(self, n)) or getattr(self, n) < 1:
                raise ConfigError("%s must be a positive integer" % n)
        for n in ("seed", "decoder_seed"):
            if not _int(getattr(self, n)) or getattr(self, n) < 0:
                raise ConfigError("%s must be a non-negative integer" % n)
        if not (_num(self.learning_rate) and self.learning_rate > 0):
            raise ConfigError("learning_rate must be positive")
        if not (_num(self.weight_decay) and self.weight_decay >= 0):
            raise ConfigError("weight_decay must be >= 0")
        if not (_num(self.adam_beta1) and 0 <= self.adam_beta1 < 1 and _num(self.adam_beta2) and 0 <= self.adam_beta2 < 1):
            raise ConfigError("Adam betas must be in [0, 1)")
        if not (_num(self.adam_eps) and self.adam_eps > 0):
            raise ConfigError("adam_eps must be positive")
        if self.grad_clip_norm is not None and not (_num(self.grad_clip_norm) and self.grad_clip_norm > 0):
            raise ConfigError("grad_clip_norm must be null or positive")
        if not (_num(self.smooth_l1_beta) and self.smooth_l1_beta > 0):
            raise ConfigError("smooth_l1_beta must be positive")
        if self.scheduler not in SCHEDULERS:
            raise ConfigError("scheduler must be one of %s" % (SCHEDULERS,))
        if not (_num(self.min_lr_ratio) and 0 <= self.min_lr_ratio <= 1):
            raise ConfigError("min_lr_ratio must be in [0, 1]")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ConfigError("device must be cpu, cuda or auto")
        if self.torch_threads is not None and (not _int(self.torch_threads) or self.torch_threads < 1):
            raise ConfigError("torch_threads must be null or a positive integer")
        if not isinstance(self.shuffle, bool) or not isinstance(self.include_optional_context, bool):
            raise ConfigError("shuffle and include_optional_context must be booleans")
        if self.split_policy not in SPLIT_POLICIES:
            raise ConfigError("split_policy must be one of %s (no independent split exists for the pilot)" % (SPLIT_POLICIES,))
        if self.checkpoint_selection not in CHECKPOINT_SELECTIONS:
            raise ConfigError("checkpoint_selection must be one of %s" % (CHECKPOINT_SELECTIONS,))
        if self.include_optional_context:
            raise ConfigError("bicyclist context is semantically uncertain and not supported by the default encoder configuration")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrainingConfig":
        if not isinstance(d, Mapping):
            raise ConfigError("training configuration must be a JSON object")
        data = {k: v for k, v in d.items() if not k.startswith("_")}
        unknown, missing = set(data) - set(cls.__dataclass_fields__), set(cls.__dataclass_fields__) - set(data)
        if unknown or missing:
            raise ConfigError("unknown keys %s, missing keys %s" % (sorted(unknown), sorted(missing)))
        return cls(**data)

    @classmethod
    def load(cls, path) -> "TrainingConfig":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError("cannot read training configuration %s: %s" % (path, exc))
        return cls.from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def sha256_without_epochs(self) -> str:
        d = self.to_dict()
        d.pop("epochs")
        return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise WindowContractError("CUDA_UNAVAILABLE", "device 'cuda' was requested but no CUDA device is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model: torch.nn.Module, cfg: TrainingConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on weight matrices only (parameters with ndim >= 2); biases and normalisation parameters are not decayed."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=cfg.learning_rate,
                             betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps)


def build_scheduler(optimizer, cfg: TrainingConfig, total_steps: int):
    if cfg.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=cfg.learning_rate * cfg.min_lr_ratio)


def epoch_order(seed: int, epoch: int, n: int, shuffle: bool) -> np.ndarray:
    """Window order of ``epoch`` (1-based): depends only on (seed, epoch), so a resumed run reproduces the original order."""
    return np.random.default_rng([int(seed), int(epoch)]).permutation(n) if shuffle else np.arange(n)


def _clean(v):
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _environment(device: torch.device) -> Dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version
    pk = {}
    for n in ("numpy", "torch", "jsonschema", "pytest"):
        try:
            pk[n] = version(n)
        except PackageNotFoundError:
            pk[n] = None
    env = {"python": platform.python_version(), "implementation": platform.python_implementation(), "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor(), "packages": pk, "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
           "cuda_available": bool(torch.cuda.is_available()), "device": str(device)}
    if torch.cuda.is_available():
        env["cuda_device_name"] = torch.cuda.get_device_name(0)
        env["cuda_version"] = torch.version.cuda
    return env


def load_run_configs(cfg: TrainingConfig):
    enc_path, dec_path = REPO / cfg.encoder_config_path, REPO / cfg.decoder_config_path
    return EncoderConfig.load(enc_path), DecoderConfig.load(dec_path), enc_path, dec_path


def scaler_source_identity(dataset: TrainingWindowDataset) -> Dict[str, Any]:
    """Exactly the identity ``TrainingWindowDataset.fit_scaler_statistics`` stores in the statistics (used to verify provenance)."""
    from copy import deepcopy
    return {"run_manifest_sha256": dataset.provenance["run_manifest_sha256"], "sources": deepcopy(dataset.provenance["sources"]), "label": dataset.provenance["label"]}


def run_training(windows_run, training_config_path, out_dir, *, resume_checkpoint=None, command: Optional[str] = None) -> Dict[str, Any]:
    out = check_output_dir(out_dir)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_start = time.perf_counter()
    git = git_provenance()
    cfg = TrainingConfig.load(training_config_path)
    enc_cfg, dec_cfg, enc_path, dec_path = load_run_configs(cfg)
    if cfg.torch_threads is not None:
        torch.set_num_threads(cfg.torch_threads)
    device = resolve_device(cfg.device)
    dataset = TrainingWindowDataset(windows_run, include_optional=cfg.include_optional_context)
    wcfg = dataset.config
    n = len(dataset)
    steps_per_epoch = -(-n // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    resume_payload = None
    start_epoch = 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir()
    # ---- scaler statistics (train partition = all pilot windows; see module docstring) ----
    t_scaler = time.perf_counter()
    scaler_path = out / "scaler_statistics.json"
    if resume_checkpoint is None:
        stats = dataset.fit_scaler_statistics(list(range(n)), fit_partition="train", split_identity=cfg.scaler_split_identity, batch_size=cfg.scaler_batch_size,
                                              configuration={"fit_windows": n, "split_policy": cfg.split_policy})
        stats_hash = stats.save(scaler_path)
    else:
        ck = Path(resume_checkpoint)
        src_stats = ck.parent.parent / "scaler_statistics.json"
        stats = TrainOnlyFeatureStatistics.load(src_stats, expected_source_identity=scaler_source_identity(dataset), expected_split_identity=cfg.scaler_split_identity)
        shutil.copyfile(src_stats, scaler_path)
        stats_hash = stats.compute_hash()
    t_scaler = time.perf_counter() - t_scaler

    model = ForecastingModel(enc_cfg, dec_cfg, wcfg, scaler=stats.feature_scaler(), decoder_seed=cfg.decoder_seed).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    metadata = {"label": PILOT_LABEL, "training_config_sha256": cfg.sha256(), "training_config_core_sha256": cfg.sha256_without_epochs(),
                "encoder_config_sha256": enc_cfg.sha256(), "decoder_config_sha256": dec_cfg.sha256(), "window_config_sha256": wcfg.sha256(),
                "windows_run_manifest_sha256": dataset.provenance["run_manifest_sha256"], "scaler_statistics_sha256": stats_hash,
                "scaler_split_identity": cfg.scaler_split_identity, "split_policy": cfg.split_policy, "checkpoint_selection": cfg.checkpoint_selection,
                "batch_size": cfg.batch_size, "seed": cfg.seed}
    if resume_checkpoint is not None:
        resume_payload = load_checkpoint(resume_checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=True)
        md = resume_payload["metadata"]
        for k in ("training_config_core_sha256", "encoder_config_sha256", "decoder_config_sha256", "window_config_sha256", "windows_run_manifest_sha256",
                  "scaler_statistics_sha256", "scaler_split_identity", "split_policy", "batch_size", "seed"):
            if md.get(k) != metadata[k]:
                raise WindowContractError("RESUME_PROVENANCE_MISMATCH", "checkpoint metadata %s=%r differs from the current run (%r)" % (k, md.get(k), metadata[k]))
        if scheduler is not None and getattr(scheduler, "T_max", None) != total_steps:
            raise WindowContractError("RESUME_SCHEDULE_MISMATCH", "the checkpoint's schedule horizon (T_max=%r) differs from this run's total steps (%d): resume with the same "
                                      "epochs as the original run" % (getattr(scheduler, "T_max", None), total_steps))
        start_epoch = int(resume_payload["epoch"])
        if start_epoch >= cfg.epochs:
            raise WindowContractError("RESUME_NOTHING_TO_DO", "checkpoint is at epoch %d but epochs=%d" % (start_epoch, cfg.epochs))

    log_path = out / "train_log.jsonl"
    log = open(log_path, "w", encoding="utf-8", newline="\n")
    global_step = int(resume_payload["global_step"]) if resume_payload else 0
    ckpt_records: Dict[str, Dict[str, Any]] = {}
    epoch_summaries: List[Dict[str, Any]] = []
    t_collate = t_compute = 0.0
    K = dec_cfg.num_modes
    peak_gpu = None
    try:
        for epoch in range(start_epoch + 1, cfg.epochs + 1):
            order = epoch_order(cfg.seed, epoch, n, cfg.shuffle)
            sums = {"loss": 0.0, "regression_loss": 0.0, "classification_loss": 0.0, "selected_mode_ade_px": 0.0, "oracle_mode_ade_px": 0.0,
                    "mean_pairwise_mode_distance_px": 0.0, "mean_mode_entropy": 0.0, "max_mode_probability_mean": 0.0, "grad_norm_total": 0.0}
            usable_total, clipped_steps, winner_total = 0, 0, np.zeros(K, np.int64)
            for b in range(steps_per_epoch):
                t0 = time.perf_counter()
                idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                batch = dataset.collate([dataset[int(i)] for i in idx])
                tb = make_train_batch(batch, device)
                t1 = time.perf_counter()
                r: StepResult = train_step(model, optimizer, tb, classification_weight=dec_cfg.classification_weight, grad_clip_norm=cfg.grad_clip_norm,
                                           scheduler=scheduler, smooth_l1_beta=cfg.smooth_l1_beta)
                t_collate += t1 - t0
                t_compute += time.perf_counter() - t1
                global_step += 1
                w = r.usable_windows
                usable_total += w
                for k in sums:
                    v = getattr(r, k)
                    sums[k] += (0.0 if (isinstance(v, float) and not np.isfinite(v)) else v) * w
                clipped_steps += int(bool(r.grad_clipped))
                winner_total += np.asarray(r.winner_counts, np.int64)
                if global_step % cfg.log_every_steps == 0:
                    rec = {"type": "step", "epoch": epoch, "step": global_step, "windows": int(len(idx)), **{k: _clean(v) for k, v in r.to_dict().items()}}
                    log.write(json.dumps(rec, allow_nan=False, sort_keys=True) + "\n")
            summary = {"type": "epoch", "epoch": epoch, "step": global_step, "windows": int(usable_total), "steps": steps_per_epoch, "clipped_steps": clipped_steps,
                       "lr_end": optimizer.param_groups[0]["lr"], "winner_counts": winner_total.tolist(),
                       **{("mean_" + k if not k.startswith("mean_") else k): sums[k] / max(usable_total, 1) for k in sums}}
            summary = {k: _clean(v) for k, v in summary.items()}
            epoch_summaries.append(summary)
            log.write(json.dumps(summary, allow_nan=False, sort_keys=True) + "\n")
            log.flush()
            if epoch % cfg.checkpoint_every_epochs == 0 or epoch == cfg.epochs:
                name = "checkpoint_epoch_%04d.pt" % epoch
                rec = save_checkpoint(out / "checkpoints" / name, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, global_step=global_step,
                                      metadata=dict(metadata, epoch=epoch))
                rec["bytes"] = (out / "checkpoints" / name).stat().st_size
                rec["epoch"], rec["global_step"] = epoch, global_step
                ckpt_records["checkpoints/" + name] = rec
        if device.type == "cuda":
            peak_gpu = int(torch.cuda.max_memory_allocated())
    finally:
        log.close()
    finished = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    final_name = "checkpoints/checkpoint_epoch_%04d.pt" % cfg.epochs
    resolved = {"training_config": cfg.to_dict(), "encoder_config": enc_cfg.to_dict(), "decoder_config": dec_cfg.to_dict(), "window_config_sha256": wcfg.sha256(),
                "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "optimizer": {"name": "AdamW", "param_groups": ["weights (ndim>=2): decayed", "biases/norms: not decayed"]},
                "scheduler": cfg.scheduler, "decoder_classification_weight": dec_cfg.classification_weight}
    write_json(out / "config_resolved.json", resolved)
    write_json(out / "environment.json", _environment(device))
    files = {}
    for name in ("config_resolved.json", "environment.json", "scaler_statistics.json", "train_log.jsonl"):
        files[name] = {"sha256": sha256_file(out / name), "bytes": (out / name).stat().st_size}
    # environment.json holds run-specific values (machine, threads): keep it out of the deterministic artifact set
    env_record = files.pop("environment.json")
    counts = model.parameter_counts()
    manifest = {
        "manifest_version": MANIFEST_VERSION, "task": "phase5_p5.7.4_pilot_training", "status": "completed", "label": PILOT_LABEL,
        "data_artifacts": {**dict(sorted(files.items())), **{k: {"sha256": v["file_sha256"], "model_state_sha256": v["model_state_sha256"], "bytes": v["bytes"],
                                                                 "epoch": v["epoch"], "global_step": v["global_step"]} for k, v in sorted(ckpt_records.items())}},
        "data_artifacts_note": "deterministic content (same inputs, configuration, seeds, platform and thread count -> same hashes); environment.json and run_metadata are run-specific",
        "environment_json": env_record,
        "inputs": {"windows_run": str(Path(windows_run).resolve().relative_to(REPO).as_posix()) if Path(windows_run).resolve().is_relative_to(REPO) else str(windows_run),
                   "windows_run_manifest_sha256": dataset.provenance["run_manifest_sha256"], "window_config_sha256": wcfg.sha256(), "windows": n,
                   "windows_run_label": dataset.provenance["label"], "dataset_artifact_hashes_verified": dataset.hashes_verified,
                   "encoder_config": {"path": cfg.encoder_config_path, "sha256_canonical": enc_cfg.sha256(), "sha256_file": sha256_file(enc_path)},
                   "decoder_config": {"path": cfg.decoder_config_path, "sha256_canonical": dec_cfg.sha256(), "sha256_file": sha256_file(dec_path)},
                   "training_config": {"path": str(Path(training_config_path).resolve().relative_to(REPO).as_posix()) if Path(training_config_path).resolve().is_relative_to(REPO) else str(training_config_path),
                                       "sha256_canonical": cfg.sha256(), "sha256_file": sha256_file(Path(training_config_path))},
                   "scaler": {"statistics_sha256": stats_hash, "split_identity": cfg.scaler_split_identity, "fit_partition": "train", "fit_windows": n,
                              "split_policy": cfg.split_policy, "valid_counts_min": int(min(stats.node_valid_count + stats.edge_valid_count))},
                   "future_targets_in_model_input": False, "offline_kinematics_used": False},
        "training": {"epochs": cfg.epochs, "epochs_run_in_this_directory": cfg.epochs - start_epoch, "resumed_from": (str(resume_checkpoint) if resume_checkpoint else None),
                     "batch_size": cfg.batch_size, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "final_global_step": global_step,
                     "optimizer": "AdamW", "learning_rate": cfg.learning_rate, "weight_decay": cfg.weight_decay, "grad_clip_norm": cfg.grad_clip_norm, "scheduler": cfg.scheduler,
                     "seed": cfg.seed, "decoder_seed": cfg.decoder_seed, "device": str(device), "parameter_counts": counts,
                     "final_checkpoint": final_name, "checkpoint_selection": cfg.checkpoint_selection,
                     "checkpoint_selection_limitation": "final-epoch checkpoint only; no best-checkpoint selection (no independent validation partition exists)",
                     "first_epoch_summary": epoch_summaries[0], "final_epoch_summary": epoch_summaries[-1], "epoch_summaries_in_sample": True},
        "run_metadata": {"git": git, "started_utc": started, "finished_utc": finished, "command": command, "environment": _environment(device),
                         "runtime_s": {"total": time.perf_counter() - t_start, "scaler_fit": t_scaler, "collate_and_batch_prep": t_collate, "forward_backward_step": t_compute,
                                       "note": "offline measurement on one machine; not a real-time claim"},
                         "peak_memory_bytes": peak_memory_bytes(), "peak_gpu_memory_bytes": peak_gpu},
        "claims": CLAIMS, "limitations": LIMITATIONS}
    write_json(out / "run_manifest.json", manifest)
    return manifest


__all__ = ["TrainingConfig", "run_training", "epoch_order", "build_optimizer", "build_scheduler", "resolve_device", "scaler_source_identity", "load_run_configs",
           "CLAIMS", "LIMITATIONS", "SPLIT_POLICIES", "CHECKPOINT_SELECTIONS"]
