"""Direct multimodal trajectory decoder for the P5.5 target embedding.

The decoder consumes only the target's history-derived summary and its final
observed image-space position. It predicts raw pixel displacement offsets; no
normalisation or future information is applied here.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ...errors import ConfigError
from ..errors import WindowContractError


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class DecoderConfig:
    config_version: str
    num_modes: int
    history_steps: int
    future_steps: int
    sampling_hz: float
    coordinate_space: str
    coordinate_unit: str
    output_representation: str
    input_embedding_dim: int
    decoder_hidden_dim: int
    dropout: float
    regression_loss: str
    classification_loss: str
    classification_weight: float
    winner_selection: str
    autoregressive: bool

    def __post_init__(self):
        if not isinstance(self.config_version, str) or not self.config_version.strip():
            raise ConfigError("config_version must be a non-empty string")
        for name in ("num_modes", "history_steps", "future_steps", "input_embedding_dim", "decoder_hidden_dim"):
            value = getattr(self, name)
            if not _int(value) or value < 1:
                raise ConfigError("%s must be a positive integer" % name)
        if not isinstance(self.sampling_hz, (int, float)) or isinstance(self.sampling_hz, bool) or self.sampling_hz <= 0:
            raise ConfigError("sampling_hz must be positive")
        if self.coordinate_space != "image":
            raise ConfigError("coordinate_space must be 'image'")
        if self.coordinate_unit != "pixels":
            raise ConfigError("coordinate_unit must be 'pixels'")
        if self.output_representation != "raw_pixel_offsets":
            raise ConfigError("output_representation must be 'raw_pixel_offsets'")
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool) or not 0 <= self.dropout < 1:
            raise ConfigError("dropout must be in [0, 1)")
        if self.regression_loss != "smooth_l1":
            raise ConfigError("regression_loss must be 'smooth_l1'")
        if self.classification_loss != "cross_entropy":
            raise ConfigError("classification_loss must be 'cross_entropy'")
        if not isinstance(self.classification_weight, (int, float)) or isinstance(self.classification_weight, bool):
            raise ConfigError("classification_weight must be a finite non-negative number")
        if not np.isfinite(self.classification_weight) or self.classification_weight < 0:
            raise ConfigError("classification_weight must be a finite non-negative number")
        if self.winner_selection != "minimum_masked_ADE":
            raise ConfigError("winner_selection must be 'minimum_masked_ADE'")
        if self.autoregressive is not False:
            raise ConfigError("P5.6 decoder is direct and non-autoregressive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecoderConfig":
        if not isinstance(data, Mapping):
            raise ConfigError("decoder configuration must be a JSON object")
        values = {key: value for key, value in data.items() if not key.startswith("_")}
        fields = set(cls.__dataclass_fields__)
        unknown = set(values) - fields
        missing = fields - set(values)
        if unknown or missing:
            raise ConfigError("unknown keys %s, missing keys %s" % (sorted(unknown), sorted(missing)))
        return cls(**values)

    @classmethod
    def load(cls, path) -> "DecoderConfig":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError("cannot read decoder configuration %s: %s" % (path, exc))
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class DecoderOutput:
    trajectory_offsets_px: torch.Tensor
    trajectories_uv_px: torch.Tensor
    mode_logits: torch.Tensor
    mode_probabilities: torch.Tensor
    prediction_valid: torch.Tensor

    @property
    def num_modes(self) -> int:
        return int(self.mode_logits.shape[1])

    @property
    def future_steps(self) -> int:
        return int(self.trajectory_offsets_px.shape[2])


class MultimodalTrajectoryDecoder(nn.Module):
    """Direct K-mode decoder producing image-space pixel trajectories."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.input_projection = nn.Linear(cfg.input_embedding_dim, cfg.decoder_hidden_dim)
        self.norm = nn.LayerNorm(cfg.decoder_hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.trajectory_head = nn.Linear(cfg.decoder_hidden_dim, cfg.num_modes * cfg.future_steps * 2)
        self.mode_head = nn.Linear(cfg.decoder_hidden_dim, cfg.num_modes)

    def forward(self, target_summary: torch.Tensor, anchor_uv_px: torch.Tensor,
                target_valid: torch.Tensor) -> DecoderOutput:
        if target_summary.ndim != 2 or target_summary.shape[1] != self.cfg.input_embedding_dim:
            raise WindowContractError(
                "DECODER_INPUT_SHAPE",
                "target_summary must be [B,%d], got %s" % (self.cfg.input_embedding_dim, tuple(target_summary.shape)),
            )
        batch = target_summary.shape[0]
        if anchor_uv_px.shape != (batch, 2):
            raise WindowContractError("DECODER_INPUT_SHAPE", "anchor_uv_px must be [B,2], got %s" % (tuple(anchor_uv_px.shape),))
        if target_valid.shape != (batch,) or target_valid.dtype != torch.bool:
            raise WindowContractError("DECODER_INPUT_MASK", "target_valid must be boolean [B]")
        if not target_summary.is_floating_point() or not anchor_uv_px.is_floating_point():
            raise WindowContractError("DECODER_INPUT_DTYPE", "decoder inputs must be floating point")
        valid_summary = target_summary[target_valid]
        valid_anchor = anchor_uv_px[target_valid]
        if valid_summary.numel() and not torch.isfinite(valid_summary).all():
            raise WindowContractError("NONFINITE_VALID_VALUE", "target_summary contains a non-finite valid value")
        if valid_anchor.numel() and not torch.isfinite(valid_anchor).all():
            raise WindowContractError("NONFINITE_VALID_VALUE", "anchor_uv_px contains a non-finite valid value")

        dtype = self.input_projection.weight.dtype
        safe_summary = torch.where(target_valid[:, None], target_summary, torch.zeros_like(target_summary))
        safe_anchor = torch.where(target_valid[:, None], anchor_uv_px, torch.zeros_like(anchor_uv_px))
        hidden = self.input_projection(safe_summary.to(dtype))
        hidden = self.dropout(F.gelu(self.norm(hidden)))
        offsets = self.trajectory_head(hidden).reshape(batch, self.cfg.num_modes, self.cfg.future_steps, 2)
        logits = self.mode_head(hidden)
        probabilities = torch.softmax(logits, dim=1)
        anchor = safe_anchor.to(offsets.dtype)[:, None, None, :]
        trajectories = anchor + offsets
        prediction_valid = target_valid[:, None, None].expand(batch, self.cfg.num_modes, self.cfg.future_steps)
        return DecoderOutput(offsets, trajectories, logits, probabilities, prediction_valid)


def evaluator_arrays(output: DecoderOutput) -> Dict[str, np.ndarray]:
    """Convert decoder predictions to the P5.4 evaluator's multimodal arrays."""
    return {
        "positions_uv_px": output.trajectories_uv_px.detach().cpu().numpy(),
        "valid": output.prediction_valid.detach().cpu().numpy(),
        "mode_probabilities": output.mode_probabilities.detach().cpu().numpy(),
    }


__all__ = ["DecoderConfig", "DecoderOutput", "MultimodalTrajectoryDecoder", "evaluator_arrays"]
