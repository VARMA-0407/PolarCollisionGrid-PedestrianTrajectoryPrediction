"""P5.7.3 reusable training / evaluation step: dataset batch -> history-only model call -> masked multimodal loss -> gradients.

    collated batch --(TrainingWindowDataset.encoder_input: history keys only)--> EncoderInput ----+
    raw pixel anchor (last observed position) + target_valid (history-derived) -----------------+--> ForecastingModel --> K trajectories
    future targets / masks  ------------------------------- used ONLY below, after the prediction exists --> masked_wta_loss

Raw pixels: anchors, offsets, trajectories and targets are never scaled; the train-only feature scaler acts on encoder INPUT
features inside the encoder. A step raises (never silently continues) on a non-finite loss or gradient.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from ...errors import PedsafetyError
from ..errors import WindowContractError
from ..training_data import TrainingWindowDataset
from .encoder import EncoderInput
from .forecaster import ForecastingModel, ForecastOutput
from .losses import MultimodalLossOutput, masked_wta_loss


class TrainingNumericsError(PedsafetyError):
    """A non-finite loss or gradient was detected; ``code`` is stable."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__("[%s] %s" % (code, message))


@dataclass(frozen=True)
class TrainBatch:
    encoder_input: EncoderInput            # history only
    anchor_uv_px: torch.Tensor             # [B, 2] raw pixels (last observed target position)
    target_valid: torch.Tensor             # [B] bool
    target_future_uv_px: torch.Tensor      # [B, F, 2] raw pixels: SUPERVISION ONLY (loss / evaluator)
    target_future_valid: torch.Tensor      # [B, F] bool
    window_id: np.ndarray                  # [B] int64 (traceability)


def make_train_batch(batch: Mapping[str, Any], device="cpu") -> TrainBatch:
    for key in ("anchor_uv_px", "target_valid", "target_future_uv_px", "target_future_valid", "metadata"):
        if key not in batch:
            raise WindowContractError("BATCH_MISSING_KEY", "dataset batch lacks %r" % key)
    enc = TrainingWindowDataset.encoder_input(batch).to(device)
    t = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt).to(device)      # noqa: E731
    return TrainBatch(enc, t(batch["anchor_uv_px"], torch.float32), t(batch["target_valid"], torch.bool),
                      t(batch["target_future_uv_px"], torch.float32), t(batch["target_future_valid"], torch.bool),
                      np.asarray(batch["metadata"]["window_id"], dtype=np.int64))


@dataclass
class StepResult:
    loss: float
    regression_loss: float
    classification_loss: float
    usable_windows: int
    excluded_windows: int
    grad_norm_total: Optional[float] = None          # before clipping
    grad_norm_encoder: Optional[float] = None
    grad_norm_decoder: Optional[float] = None
    grad_clipped: Optional[bool] = None
    parameters_without_grad: List[str] = field(default_factory=list)
    lr: Optional[float] = None
    winner_counts: List[int] = field(default_factory=list)              # windows per mode index chosen as WTA winner
    selected_mode_ade_px: float = float("nan")                          # ADE of the most-probable mode (training-batch, in-sample monitor)
    oracle_mode_ade_px: float = float("nan")                            # min over modes (oracle, in-sample monitor)
    mean_pairwise_mode_distance_px: float = float("nan")
    mean_mode_entropy: float = float("nan")
    max_mode_probability_mean: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def forward_loss(model: ForecastingModel, tb: TrainBatch, classification_weight: float, smooth_l1_beta: float = 1.0):
    """History-only forward pass, then the masked loss with the supervision tensors. Returns (ForecastOutput, MultimodalLossOutput)."""
    out: ForecastOutput = model(tb.encoder_input, tb.anchor_uv_px, tb.target_valid)               # <- no future argument exists
    loss = masked_wta_loss(out.decoded.trajectories_uv_px, out.decoded.mode_logits, tb.target_future_uv_px, tb.target_future_valid, tb.target_valid,
                           classification_weight=classification_weight, smooth_l1_beta=smooth_l1_beta)
    return out, loss


def _diagnostics(out: ForecastOutput, loss: MultimodalLossOutput) -> Dict[str, Any]:
    with torch.no_grad():
        u = loss.valid_window_mask
        K = out.decoded.num_modes
        traj, probs = out.decoded.trajectories_uv_px[u], out.decoded.mode_probabilities[u]
        winners = loss.winner_index[u]
        counts = torch.bincount(winners, minlength=K).tolist()
        sel = probs.argmax(dim=1)
        ade = loss.mode_ade_px[u]
        pair = float("nan")
        if K > 1:
            d = torch.linalg.vector_norm(traj[:, :, None] - traj[:, None, :], dim=-1).mean(dim=-1)        # [B, K, K] mean over time (px)
            pair = float(d.sum(dim=(1, 2)).div(K * (K - 1)).mean())
        ent = float(-(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean())
        return {"winner_counts": counts, "selected_mode_ade_px": float(ade[torch.arange(ade.shape[0]), sel].mean()),
                "oracle_mode_ade_px": float(ade.min(dim=1).values.mean()), "mean_pairwise_mode_distance_px": pair, "mean_mode_entropy": ent,
                "max_mode_probability_mean": float(probs.max(dim=1).values.mean())}


def _grad_norms(model: ForecastingModel):
    def norm(module):
        gs = [p.grad for p in module.parameters() if p.grad is not None]
        return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in gs])) if gs else torch.zeros(())
    return norm(model.encoder), norm(model.decoder)


def train_step(model: ForecastingModel, optimizer: torch.optim.Optimizer, tb: TrainBatch, *, classification_weight: float, grad_clip_norm: Optional[float] = None,
               scheduler: Optional[Any] = None, smooth_l1_beta: float = 1.0) -> StepResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out, loss = forward_loss(model, tb, classification_weight, smooth_l1_beta)
    if not torch.isfinite(loss.total_loss):
        raise TrainingNumericsError("NONFINITE_LOSS", "total loss is %s" % float(loss.total_loss))
    loss.total_loss.backward()
    enc_n, dec_n = _grad_norms(model)
    total = torch.sqrt(enc_n.pow(2) + dec_n.pow(2))
    if not (torch.isfinite(enc_n) and torch.isfinite(dec_n) and torch.isfinite(total)):
        optimizer.zero_grad(set_to_none=True)
        raise TrainingNumericsError("NONFINITE_GRADIENT", "gradient norm is not finite (encoder %s, decoder %s)" % (float(enc_n), float(dec_n)))
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    clipped = False
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        clipped = bool(total > grad_clip_norm)
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    d = _diagnostics(out, loss)
    return StepResult(float(loss.total_loss), float(loss.regression_loss), float(loss.classification_loss), loss.valid_window_count,
                      loss.diagnostics["excluded_windows"], float(total), float(enc_n), float(dec_n), clipped, no_grad, float(lr), **d)


def eval_step(model: ForecastingModel, tb: TrainBatch, *, classification_weight: float, smooth_l1_beta: float = 1.0) -> StepResult:
    """Loss on a batch without gradients or parameter updates (eval mode)."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out, loss = forward_loss(model, tb, classification_weight, smooth_l1_beta)
            d = _diagnostics(out, loss)
    finally:
        model.train(was_training)
    return StepResult(float(loss.total_loss), float(loss.regression_loss), float(loss.classification_loss), loss.valid_window_count,
                      loss.diagnostics["excluded_windows"], **d)


__all__ = ["TrainBatch", "StepResult", "TrainingNumericsError", "make_train_batch", "forward_loss", "train_step", "eval_step"]
