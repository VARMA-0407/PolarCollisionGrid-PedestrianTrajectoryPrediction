"""P5.7.3 integrated forecasting model: P5.5 encoder + P5.6 decoder (established interfaces, unchanged).

Data flow (history only):
    EncoderInput (history graphs)  --P5.5-->  target_summary [B, D]
    target_summary + raw pixel anchor [B, 2] + target_valid [B]  --P5.6-->  K trajectories [B, K, F, 2] (raw pixels) + mode probabilities

The forward pass accepts NO future information: its arguments are the history-only ``EncoderInput``, the raw image-space anchor
(the target's last observed position, a history value) and the history-derived target-validity mask. Future targets are used
only by the loss / evaluator after the prediction exists. Feature scaling (train-only statistics) is applied inside the encoder to
input features only; anchors, offsets and trajectories stay raw ``captured_1920x1080`` pixels.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ...errors import ConfigError
from ..window_config import WindowConfig
from .config import EncoderConfig
from .decoder import DecoderConfig, DecoderOutput, MultimodalTrajectoryDecoder
from .encoder import EncoderInput, EncoderOutput, SpatioTemporalEncoder
from .scaling import FeatureScaler


@dataclass
class ForecastOutput:
    encoded: EncoderOutput
    decoded: DecoderOutput


def check_compatibility(encoder_cfg: EncoderConfig, decoder_cfg: DecoderConfig, window_cfg: Optional[WindowConfig] = None) -> None:
    """The three configurations must describe the same tensor contract (no silent reshaping)."""
    if decoder_cfg.input_embedding_dim != encoder_cfg.hidden_dim:
        raise ConfigError("decoder input_embedding_dim %d != encoder hidden_dim %d" % (decoder_cfg.input_embedding_dim, encoder_cfg.hidden_dim))
    if window_cfg is not None:
        if decoder_cfg.history_steps != window_cfg.history_length or decoder_cfg.future_steps != window_cfg.prediction_length:
            raise ConfigError("decoder history/future steps (%d/%d) differ from the window configuration (%d/%d)"
                              % (decoder_cfg.history_steps, decoder_cfg.future_steps, window_cfg.history_length, window_cfg.prediction_length))
        if abs(decoder_cfg.sampling_hz - window_cfg.target_sampling_hz) > 1e-9:
            raise ConfigError("decoder sampling_hz differs from the window configuration")
        encoder_cfg.check_history(window_cfg.history_length)


class ForecastingModel(nn.Module):
    """Encoder + decoder with explicit, deterministic initialisation (``decoder_seed`` is separate from the encoder's ``init_seed``)."""

    def __init__(self, encoder_cfg: EncoderConfig, decoder_cfg: DecoderConfig, window_cfg: WindowConfig, scaler: Optional[FeatureScaler] = None,
                 decoder_seed: int = 0):
        super().__init__()
        check_compatibility(encoder_cfg, decoder_cfg, window_cfg)
        self.encoder_cfg, self.decoder_cfg = encoder_cfg, decoder_cfg
        self.encoder = SpatioTemporalEncoder(encoder_cfg, history_length=window_cfg.history_length, scaler=scaler)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(decoder_seed))
            self.decoder = MultimodalTrajectoryDecoder(decoder_cfg)

    def forward(self, encoder_input: EncoderInput, anchor_uv_px: torch.Tensor, target_valid: torch.Tensor) -> ForecastOutput:
        encoded = self.encoder(encoder_input)
        decoded = self.decoder(encoded.target_summary, anchor_uv_px, target_valid)
        return ForecastOutput(encoded, decoded)

    def parameter_counts(self):
        count = lambda m: sum(p.numel() for p in m.parameters())      # noqa: E731
        return {"encoder": count(self.encoder), "decoder": count(self.decoder), "total": count(self)}


__all__ = ["ForecastingModel", "ForecastOutput", "check_compatibility"]
