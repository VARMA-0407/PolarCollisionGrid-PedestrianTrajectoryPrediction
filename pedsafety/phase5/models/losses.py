"""Masked multimodal trajectory losses for P5.6."""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from ..errors import WindowContractError


@dataclass
class MultimodalLossOutput:
    total_loss: torch.Tensor
    regression_loss: torch.Tensor
    classification_loss: torch.Tensor
    winner_index: torch.Tensor
    mode_ade_px: torch.Tensor
    valid_window_mask: torch.Tensor
    valid_window_count: int
    diagnostics: Dict[str, int]


def masked_wta_loss(predicted_trajectories: torch.Tensor, mode_logits: torch.Tensor,
                    target_future_uv_px: torch.Tensor, target_future_valid: torch.Tensor,
                    target_valid: torch.Tensor, classification_weight: float = 0.1,
                    smooth_l1_beta: float = 1.0) -> MultimodalLossOutput:
    """Compute masked ADE winner-takes-all SmoothL1 plus mode cross-entropy.

    The regression reduction is the sum of SmoothL1 errors over valid
    coordinates divided by ``2 * valid_timestep_count`` for each usable
    window, then averaged over usable windows. Invalid windows are excluded.
    """
    if predicted_trajectories.ndim != 4 or predicted_trajectories.shape[-1] != 2:
        raise WindowContractError("LOSS_SHAPE", "predicted_trajectories must be [B,K,F,2]")
    batch, modes, future, _ = predicted_trajectories.shape
    if mode_logits.shape != (batch, modes):
        raise WindowContractError("LOSS_SHAPE", "mode_logits must be [B,K]")
    if target_future_uv_px.shape != (batch, future, 2):
        raise WindowContractError("LOSS_SHAPE", "target_future_uv_px must be [B,F,2]")
    if target_future_valid.shape != (batch, future) or target_future_valid.dtype != torch.bool:
        raise WindowContractError("LOSS_MASK", "target_future_valid must be boolean [B,F]")
    if target_valid.shape != (batch,) or target_valid.dtype != torch.bool:
        raise WindowContractError("LOSS_MASK", "target_valid must be boolean [B]")
    if not isinstance(classification_weight, (int, float)) or isinstance(classification_weight, bool):
        raise WindowContractError("LOSS_CONFIG", "classification_weight must be finite and non-negative")
    if not (0 <= classification_weight and torch.isfinite(torch.as_tensor(classification_weight))):
        raise WindowContractError("LOSS_CONFIG", "classification_weight must be finite and non-negative")
    if smooth_l1_beta <= 0:
        raise WindowContractError("LOSS_CONFIG", "smooth_l1_beta must be positive")

    usable = target_valid & target_future_valid.any(dim=1)
    if not usable.any():
        raise WindowContractError("NO_VALID_TARGETS", "loss batch contains no target with at least one valid future timestep")
    valid_values = target_future_valid & target_valid[:, None]
    target_check = target_future_uv_px[valid_values]
    if target_check.numel() and not torch.isfinite(target_check).all():
        raise WindowContractError("NONFINITE_VALID_VALUE", "target_future_uv_px contains a non-finite valid value")
    pred_check = predicted_trajectories[valid_values[:, None, :].expand(batch, modes, future)]
    if pred_check.numel() and not torch.isfinite(pred_check).all():
        raise WindowContractError("NONFINITE_VALID_VALUE", "predicted_trajectories contains a non-finite valid value")
    if not torch.isfinite(mode_logits[usable]).all():
        raise WindowContractError("NONFINITE_VALID_VALUE", "mode_logits contains a non-finite usable value")

    mask = valid_values[:, None, :, None]
    safe_target = torch.where(valid_values[:, :, None], target_future_uv_px, torch.zeros_like(target_future_uv_px))
    safe_target = safe_target[:, None, :, :]
    safe_pred = torch.where(mask, predicted_trajectories, torch.zeros_like(predicted_trajectories))
    distance = torch.linalg.vector_norm(safe_pred - safe_target, dim=-1)
    counts = valid_values.sum(dim=1).to(distance.dtype)
    mode_ade = distance.sum(dim=2) / counts.clamp_min(1)[:, None]
    winner = torch.full((batch,), -1, dtype=torch.long, device=predicted_trajectories.device)
    winner[usable] = mode_ade[usable].argmin(dim=1)

    rows = torch.arange(batch, device=predicted_trajectories.device)
    winner_pred = safe_pred[rows, winner.clamp_min(0)]
    smooth = F.smooth_l1_loss(winner_pred, safe_target[:, 0], beta=smooth_l1_beta, reduction="none")
    regression_per_window = (smooth * valid_values[:, :, None].to(smooth.dtype)).sum(dim=(1, 2)) / (2.0 * counts)
    regression_loss = regression_per_window[usable].mean()
    if modes == 1:
        classification_loss = regression_loss.new_zeros(())
    else:
        classification_loss = F.cross_entropy(mode_logits[usable], winner[usable])
    total = regression_loss + float(classification_weight) * classification_loss
    diagnostics = {
        "batch_size": int(batch),
        "usable_windows": int(usable.sum()),
        "excluded_windows": int((~usable).sum()),
        "num_modes": int(modes),
        "future_steps": int(future),
    }
    return MultimodalLossOutput(total, regression_loss, classification_loss, winner, mode_ade, usable,
                                int(usable.sum()), diagnostics)


__all__ = ["MultimodalLossOutput", "masked_wta_loss"]
