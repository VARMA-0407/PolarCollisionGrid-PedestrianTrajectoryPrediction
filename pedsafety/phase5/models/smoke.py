"""P5.5 engineering smoke test: forward pass of the UNTRAINED encoder over the P5.3 pilot windows (no training, no accuracy).

Verifies the P5.3 window run (manifest, hashes, configuration), builds history-only encoder inputs with the P5.3 adapter,
runs ``SpatioTemporalEncoder`` in eval mode under ``torch.no_grad`` and records numerical validity, shapes, parameter
count, runtime, peak memory and SHA-256 hashes of the output tensors (the embeddings themselves are not stored).
Randomly initialised weights make the values meaningless as features; nothing here is a forecast or a model comparison.
"""

import datetime
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ...image_kinematics_run import sha256_file
from ...interaction.runner import peak_memory_bytes
from ...interaction.storage import write_json
from ...output_dir import check_output_dir
from ..errors import WindowContractError
from ..graph_window_adapter import collate_windows
from ..window_config import WindowConfig
from ..window_manifest import PILOT_LABEL, git_provenance
from ..window_storage import read_dataset
from .config import EncoderConfig
from .encoder import EncoderInput, SpatioTemporalEncoder

REPO = Path(__file__).resolve().parents[3]
MANIFEST_VERSION = "phase5-encoder-smoke-run-manifest-1.0.0"
CLAIMS = {"coordinate_space": "image (captured raster pixels)", "metric_world_coordinates": False, "physical_features": False,
          "ttc_pet_or_collision_risk": False, "model_trained": False, "forecasting_accuracy": False, "independent_generalization": False,
          "baseline_comparison": False, "architecture_validated_for_forecasting": False, "real_time_claim": False,
          "note": "Engineering forward-pass smoke test of an untrained, randomly initialised encoder on the pilot windows."}
LIMITATIONS = [
    "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT: one PIE video, one scene group, overlapping windows; nothing is evaluated or generalised.",
    "Weights are randomly initialised (seed in the configuration) and untrained: outputs only demonstrate shapes, masking, finiteness and determinism.",
    "Feature scaling uses nominal, non-fitted development constants; train-only statistics belong to the training workflow (P5.7).",
    "Runtime and memory are offline CPU measurements on one machine, not real-time claims; CUDA was not exercised.",
    "Image-space inputs only; no metric or physical interpretation.",
]


def _rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def verify_p53_run(run_dir):
    d = Path(run_dir)
    m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    if m.get("status") != "completed" or m.get("label") != PILOT_LABEL or m.get("validation", {}).get("ok") is not True:
        raise WindowContractError("WINDOW_RUN_STATUS", "P5.3 run must be completed, validated and labelled %s" % PILOT_LABEL)
    hashes = {}
    for name, rec in m["outputs"].items():
        p = d / name
        if not p.is_file() or p.stat().st_size != rec["bytes"] or sha256_file(p) != rec["sha256"]:
            raise WindowContractError("WINDOW_RUN_HASH", "P5.3 output %s is missing or does not match its manifest" % name)
        hashes[name] = rec["sha256"]
    wcfg = WindowConfig.from_dict(m["config"]["content"])
    if wcfg.sha256() != m["config"]["sha256_resolved"]:
        raise WindowContractError("WINDOW_CONFIG_MISMATCH", "window configuration hash differs from the P5.3 manifest")
    hashes["run_manifest.json"] = sha256_file(d / "run_manifest.json")
    return m, wcfg, hashes


class _Hasher:
    def __init__(self):
        self.h = hashlib.sha256()
        self.n = 0

    def add(self, t: torch.Tensor):
        a = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4"))
        self.h.update(a.tobytes())
        self.n += a.size

    def hexdigest(self) -> str:
        return self.h.hexdigest()


def forward_pass(ws, bank, wcfg: WindowConfig, ecfg: EncoderConfig, batch_size: int, limit: Optional[int] = None, include_optional: bool = False) -> Dict[str, Any]:
    """Deterministic forward pass; returns validity statistics and tensor hashes (deterministic content only)."""
    n = ws.n_windows if limit is None else min(limit, ws.n_windows)
    model = SpatioTemporalEncoder(ecfg, history_length=wcfg.history_length)
    model.eval()
    h_seq, h_tgt, h_sp = _Hasher(), _Hasher(), _Hasher()
    stats = {"windows": 0, "batches": 0, "max_agent_slots": 0, "present_node_steps": 0, "nonfinite_values": 0, "absent_nonzero_values": 0,
             "target_slot_mismatch": 0, "target_absent_steps": 0, "edges_by_relation": {}, "sequence_shape_first": None, "sequence_shape_last": None,
             "target_summary_shape": None, "batch_latency_s": []}
    with torch.no_grad():
        for i in range(0, n, batch_size):
            idx = range(i, min(i + batch_size, n))
            t0 = time.perf_counter()
            batch = collate_windows(bank, ws, idx, wcfg, include_optional=include_optional)
            inp = EncoderInput.from_collated_batch(batch)
            out = model(inp)
            stats["batch_latency_s"].append(time.perf_counter() - t0)
            B, A = out.sequence_embeddings.shape[:2]
            stats["windows"] += B
            stats["batches"] += 1
            stats["max_agent_slots"] = max(stats["max_agent_slots"], A)
            stats["present_node_steps"] += int(out.presence.sum())
            stats["nonfinite_values"] += int((~torch.isfinite(out.sequence_embeddings)).sum() + (~torch.isfinite(out.target_summary)).sum()
                                             + (~torch.isfinite(out.spatial_embeddings)).sum())
            stats["absent_nonzero_values"] += int((out.sequence_embeddings[~out.presence] != 0).sum())
            stats["target_slot_mismatch"] += int((out.track_index[:, 0] != torch.as_tensor(ws.target_track_index[list(idx)].astype(np.int64))).sum())
            stats["target_absent_steps"] += int((~out.presence[:, 0]).sum())
            for r, e in batch["edges"].items():
                stats["edges_by_relation"][r] = stats["edges_by_relation"].get(r, 0) + int(e["bhsd"].shape[0])
            stats["sequence_shape_first"] = stats["sequence_shape_first"] or list(out.sequence_embeddings.shape)
            stats["sequence_shape_last"] = list(out.sequence_embeddings.shape)
            stats["target_summary_shape"] = list(out.target_summary.shape)
            h_seq.add(out.sequence_embeddings)
            h_tgt.add(out.target_summary)
            h_sp.add(out.spatial_embeddings)
    lat = stats.pop("batch_latency_s")
    return {"model": model, "stats": stats, "latency_s": lat,
            "hashes": {"sequence_embeddings_sha256": h_seq.hexdigest(), "target_summary_sha256": h_tgt.hexdigest(), "spatial_embeddings_sha256": h_sp.hexdigest(),
                       "hashed_value_counts": {"sequence_embeddings": h_seq.n, "target_summary": h_tgt.n, "spatial_embeddings": h_sp.n},
                       "encoding": "little-endian float32 bytes of the per-batch tensors, concatenated in batch order (batches are padded to the batch's own maximum agent count)"}}


def bicycle_encoder_config(ecfg: EncoderConfig) -> EncoderConfig:
    d = ecfg.to_dict()
    d.update(node_types=["pedestrian", "vehicle", "bicyclist"], bicyclist_context_enabled=True,
             relations=["pedestrian__pedestrian", "vehicle__pedestrian", "pedestrian__vehicle", "bicyclist__pedestrian", "pedestrian__bicyclist",
                        "bicyclist__vehicle", "vehicle__bicyclist"])
    return EncoderConfig.from_dict(d)


def run_smoke(windows_run, encoder_config_path, out_dir, *, batch_size: int = 64, optional_bicycle_windows: int = 128, command: Optional[str] = None) -> Dict[str, Any]:
    out = check_output_dir(out_dir)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    git = git_provenance()
    ecfg = EncoderConfig.load(encoder_config_path)
    wman, wcfg, in_hashes = verify_p53_run(windows_run)
    ws, bank, _ = read_dataset(windows_run)
    t1 = time.perf_counter()
    main = forward_pass(ws, bank, wcfg, ecfg, batch_size)
    t2 = time.perf_counter()
    model = main["model"]
    bike_cfg = bicycle_encoder_config(ecfg)
    bike = forward_pass(ws, bank, wcfg, bike_cfg, batch_size, limit=optional_bicycle_windows, include_optional=True)
    t3 = time.perf_counter()
    s = main["stats"]
    valid = (s["nonfinite_values"] == 0 and s["absent_nonzero_values"] == 0 and s["target_slot_mismatch"] == 0 and s["target_absent_steps"] == 0
             and bike["stats"]["nonfinite_values"] == 0 and bike["stats"]["absent_nonzero_values"] == 0)
    results = {
        "label": PILOT_LABEL, "results_version": "phase5-encoder-smoke-results-1.0.0",
        "purpose": "engineering forward-pass smoke test of the UNTRAINED encoder; no training, no accuracy, no baseline comparison",
        "encoder_config": {"path": _rel(encoder_config_path), "sha256_canonical": ecfg.sha256(), "content": ecfg.to_dict(), "receptive_field": model.receptive_field,
                           "history_length_from_window_config": wcfg.history_length},
        "model": {"parameter_count": model.parameter_count(), "parameter_bytes_float32": 4 * model.parameter_count(), "trainable": True, "trained": False,
                  "initialisation": "torch.manual_seed(init_seed) default PyTorch initialisation", "mode": "eval, torch.no_grad",
                  "feature_scaling": {"source": model.scaler.source, "provenance": model.scaler.provenance}},
        "windows": {"windows_run": _rel(windows_run), "windows": ws.n_windows, "unique_target_tracks": int(np.unique(ws.target_track_id).shape[0]),
                    "history_length": wcfg.history_length, "target_source_mode": wcfg.target_source_mode},
        "default_context_forward": {"context": "pedestrian + vehicle (P5.3 default), relations " + ", ".join(ecfg.relations), "batch_size": batch_size, **s,
                                    "output_shapes": {"sequence_embeddings": "[B, A, %d, %d]" % (wcfg.history_length, ecfg.hidden_dim), "target_summary": "[B, %d]" % ecfg.hidden_dim,
                                                      "spatial_embeddings": "[B, %d, A, %d]" % (wcfg.history_length, ecfg.hidden_dim)},
                                    "hashes": main["hashes"]},
        "optional_bicycle_context_forward": {"context": "pedestrian + vehicle + bicyclist (semantically uncertain; optional) on the first %d windows" % bike["stats"]["windows"],
                                             "parameter_count": bike["model"].parameter_count(), **bike["stats"], "hashes": bike["hashes"]},
        "numerical_validity": {"all_finite": s["nonfinite_values"] == 0 and bike["stats"]["nonfinite_values"] == 0, "absent_steps_exactly_zero": s["absent_nonzero_values"] == 0,
                               "slot0_is_window_target_track": s["target_slot_mismatch"] == 0, "target_present_all_steps": s["target_absent_steps"] == 0, "overall": bool(valid)},
        "device": "cpu", "cuda_available": bool(torch.cuda.is_available()), "claims": CLAIMS, "limitations": LIMITATIONS}
    out.mkdir(parents=True, exist_ok=True)
    rh = write_json(out / "smoke_results.json", results)
    finished = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lat = main["latency_s"]
    manifest = {
        "manifest_version": MANIFEST_VERSION, "task": "phase5_p5.5_encoder_smoke", "status": "completed", "label": PILOT_LABEL,
        "data_artifacts": {"smoke_results.json": {"sha256": rh, "bytes": (out / "smoke_results.json").stat().st_size}},
        "data_artifacts_note": "deterministic content: identical inputs, configuration and seed give identical hashes (including the output-tensor hashes inside)",
        "inputs": {"windows_run": _rel(windows_run), "windows_run_files": in_hashes, "window_config_sha256": wcfg.sha256(),
                   "encoder_config": {"path": _rel(encoder_config_path), "sha256_canonical": ecfg.sha256(), "sha256_file": sha256_file(Path(encoder_config_path))},
                   "targets_or_future_arrays_read": False, "offline_kinematics_used": False},
        "counts": {"windows": ws.n_windows, "parameter_count": model.parameter_count(), "valid": bool(valid)},
        "run_metadata": {"git": git, "started_utc": started, "finished_utc": finished, "command": command,
                         "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(),
                                      "torch_threads": torch.get_num_threads(), "cuda_available": bool(torch.cuda.is_available())},
                         "runtime_s": {"verify_and_load": t1 - t0, "default_forward_total": t2 - t1, "optional_bicycle_forward_total": t3 - t2,
                                       "default_batch_latency_mean": float(np.mean(lat)), "default_batch_latency_max": float(np.max(lat)), "batches": len(lat),
                                       "note": "includes P5.3 collation and encoder forward; offline CPU; not a real-time claim"},
                         "peak_memory_bytes": peak_memory_bytes(), "peak_memory_note": "process peak working set at the end of the run (Python, numpy, torch, P5.3 dataset)"},
        "claims": CLAIMS, "limitations": LIMITATIONS}
    write_json(out / "run_manifest.json", manifest)
    return manifest


__all__ = ["run_smoke", "forward_pass", "verify_p53_run", "bicycle_encoder_config", "CLAIMS", "LIMITATIONS"]
