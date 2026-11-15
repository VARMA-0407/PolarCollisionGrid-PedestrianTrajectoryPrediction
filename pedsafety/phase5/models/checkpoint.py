"""Checkpoint saving / loading with hash verification (plain tensors and primitives only; loaded with ``weights_only=True``).

A checkpoint file holds: model state, optimizer state, scheduler state, epoch / global step, CPU RNG state and a JSON-able
``metadata`` mapping (configuration hashes, scaler statistics hash, dataset identity). Two hashes identify it:
``file_sha256`` (the bytes on disk) and ``model_state_sha256`` (canonical hash of the model tensors, independent of pickling).
The scaler buffers are part of the model state, so a checkpoint always carries the exact feature scaling it was trained with.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from ..errors import WindowContractError

CHECKPOINT_FORMAT = "phase5-forecaster-checkpoint-1.0.0"


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Canonical hash of a tensor mapping: sorted keys, dtype, shape and little-endian bytes of every tensor."""
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8") + b"\0" + str(t.dtype).encode("utf-8") + b"\0" + str(tuple(t.shape)).encode("utf-8") + b"\0")
        h.update(t.numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def _plain(value: Any) -> Any:
    """Metadata must be primitives so that ``weights_only`` loading is possible."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    raise WindowContractError("CHECKPOINT_METADATA", "checkpoint metadata must be plain JSON-able values, got %s" % type(value).__name__)


def save_checkpoint(path, *, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[Any], epoch: int, global_step: int,
                    metadata: Mapping[str, Any]) -> Dict[str, str]:
    p = Path(path)
    if p.exists():
        raise WindowContractError("CHECKPOINT_EXISTS", "refusing to overwrite %s" % p)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload = {"format": CHECKPOINT_FORMAT, "epoch": int(epoch), "global_step": int(global_step), "model_state": state,
               "model_state_sha256": state_dict_sha256(state),
               "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
               "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
               "rng_state_cpu": torch.get_rng_state(), "metadata": _plain(metadata)}
    tmp = p.with_name(p.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, p)
    return {"file_sha256": file_sha256(p), "model_state_sha256": payload["model_state_sha256"]}


def load_checkpoint(path, *, expected_file_sha256: Optional[str] = None, expected_model_state_sha256: Optional[str] = None, model: Optional[torch.nn.Module] = None,
                    optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None, restore_rng: bool = False) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise WindowContractError("CHECKPOINT_MISSING", "missing checkpoint %s" % p)
    actual = file_sha256(p)
    if expected_file_sha256 is not None and actual != expected_file_sha256:
        raise WindowContractError("CHECKPOINT_HASH_MISMATCH", "checkpoint file hash %s != expected %s" % (actual, expected_file_sha256))
    try:
        payload = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as exc:                                                    # noqa: BLE001
        raise WindowContractError("CHECKPOINT_UNREADABLE", "cannot load %s safely: %s" % (p, exc))
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise WindowContractError("CHECKPOINT_FORMAT", "unsupported checkpoint format %r" % payload.get("format"))
    recomputed = state_dict_sha256(payload["model_state"])
    if recomputed != payload["model_state_sha256"] or (expected_model_state_sha256 is not None and recomputed != expected_model_state_sha256):
        raise WindowContractError("CHECKPOINT_STATE_HASH_MISMATCH", "model state hash %s does not match the recorded/expected value" % recomputed)
    if model is not None:
        model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        if payload["optimizer_state"] is None:
            raise WindowContractError("CHECKPOINT_NO_OPTIMIZER", "checkpoint holds no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if payload["scheduler_state"] is None:
            raise WindowContractError("CHECKPOINT_NO_SCHEDULER", "checkpoint holds no scheduler state")
        scheduler.load_state_dict(payload["scheduler_state"])
    if restore_rng:
        torch.set_rng_state(payload["rng_state_cpu"])
    payload["file_sha256"] = actual
    return payload


__all__ = ["save_checkpoint", "load_checkpoint", "state_dict_sha256", "file_sha256", "CHECKPOINT_FORMAT"]
