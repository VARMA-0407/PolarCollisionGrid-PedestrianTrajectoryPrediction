"""Explicit feature scaling for the encoder inputs.

Two sources, declared in ``EncoderConfig.feature_scaling``:

``NOMINAL_DEVELOPMENT_DEFAULT_NOT_FITTED`` (default, development only)
    Fixed nominal constants chosen from the image geometry and sampling (1920x1080 raster, 30 fps), NOT computed from any
    data: position/offset features / 1000 px, velocities / 500 px/s, box height / 1080 px, frames_since_prev_obs / 30.
    They only bring inputs to O(1); they are not statistics of the pilot.

``TRAIN_PARTITION_STATISTICS`` (future training workflow)
    Per-feature offset and scale fitted on the TRAINING partition only. The mapping must declare
    ``fit_partition == "train"`` and carry ``statistics_sha256``; anything else is refused (``SCALER_NOT_TRAIN_ONLY``).
    Nothing in this module fits statistics.

Scaling is ``(x - offset) / scale`` applied to values whose validity mask is True; masked values are then set to exactly 0
(always together with the mask channels that the encoder concatenates). Scales must be finite and positive.
"""

from typing import Any, Dict, Mapping, Sequence

import torch
from torch import nn

from ...errors import ConfigError
from ..errors import WindowContractError
from ..task3_source import EDGE_FEATURES
from .config import NODE_INPUT_FEATURES


def _vec(m: Mapping[str, Any], names: Sequence[str], what: str, default: float, positive: bool) -> torch.Tensor:
    unknown = set(m) - set(names)
    if unknown:
        raise ConfigError("%s: unknown feature names %s" % (what, sorted(unknown)))
    out = []
    for n in names:
        v = m.get(n, default)
        if not (isinstance(v, (int, float)) and not isinstance(v, bool) and float("-inf") < v < float("inf")):
            raise ConfigError("%s.%s must be a finite number" % (what, n))
        if positive and v <= 0:
            raise ConfigError("%s.%s must be positive" % (what, n))
        out.append(float(v))
    return torch.tensor(out, dtype=torch.float32)


class FeatureScaler(nn.Module):
    """Holds node/edge offsets and scales as buffers (saved with the model, never trained)."""

    def __init__(self, node_offset, node_scale, edge_offset, edge_scale, source: str, provenance: Dict[str, Any]):
        super().__init__()
        for name, t, n in (("node_offset", node_offset, len(NODE_INPUT_FEATURES)), ("node_scale", node_scale, len(NODE_INPUT_FEATURES)),
                           ("edge_offset", edge_offset, len(EDGE_FEATURES)), ("edge_scale", edge_scale, len(EDGE_FEATURES))):
            if t.shape != (n,) or not torch.isfinite(t).all():
                raise ConfigError("%s must be a finite vector of length %d" % (name, n))
        if (node_scale <= 0).any() or (edge_scale <= 0).any():
            raise ConfigError("scales must be positive")
        self.register_buffer("node_offset", node_offset.clone())
        self.register_buffer("node_scale", node_scale.clone())
        self.register_buffer("edge_offset", edge_offset.clone())
        self.register_buffer("edge_scale", edge_scale.clone())
        self.source = source
        self.provenance = dict(provenance)

    @classmethod
    def from_arrays(cls, node_offset, node_scale, edge_offset, edge_scale, source: str, provenance: Dict[str, Any]) -> "FeatureScaler":
        """Build a scaler from plain sequences (float32 buffers). Used by the P5.7.2 train-only statistics so that the code outside
        ``pedsafety/phase5/models`` never imports torch (dependency policy: requirements-phase5.txt applies to ``models`` only)."""
        return cls(torch.tensor(list(node_offset), dtype=torch.float32), torch.tensor(list(node_scale), dtype=torch.float32),
                   torch.tensor(list(edge_offset), dtype=torch.float32), torch.tensor(list(edge_scale), dtype=torch.float32), source, provenance)

    @classmethod
    def from_config(cls, fs: Mapping[str, Any]) -> "FeatureScaler":
        src = fs["source"]
        if src == "NOMINAL_DEVELOPMENT_DEFAULT_NOT_FITTED":
            return cls(_vec(fs.get("node_offset", {}), NODE_INPUT_FEATURES, "node_offset", 0.0, False),
                       _vec(fs["node"], NODE_INPUT_FEATURES, "node", 1.0, True),
                       _vec(fs.get("edge_offset", {}), EDGE_FEATURES, "edge_offset", 0.0, False),
                       _vec(fs["edge"], EDGE_FEATURES, "edge", 1.0, True), src,
                       {"fitted": False, "note": "fixed nominal constants; not computed from any dataset"})
        if fs.get("fit_partition") != "train" or not fs.get("statistics_sha256"):
            raise WindowContractError("SCALER_NOT_TRAIN_ONLY", "scaling statistics must be fitted on the training partition only and carry "
                                      "statistics_sha256 (got fit_partition=%r)" % (fs.get("fit_partition"),))
        return cls(_vec(fs["node_offset"], NODE_INPUT_FEATURES, "node_offset", 0.0, False), _vec(fs["node_scale"], NODE_INPUT_FEATURES, "node_scale", 1.0, True),
                   _vec(fs["edge_offset"], EDGE_FEATURES, "edge_offset", 0.0, False), _vec(fs["edge_scale"], EDGE_FEATURES, "edge_scale", 1.0, True), src,
                   {"fitted": True, "fit_partition": "train", "statistics_sha256": fs["statistics_sha256"]})

    def scale_node(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [..., 9] raw values (NaN allowed where invalid), valid [..., 9] bool -> scaled, masked values exactly 0."""
        return torch.where(valid, (x - self.node_offset) / self.node_scale, torch.zeros_like(x))

    def scale_edge(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return torch.where(valid, (x - self.edge_offset) / self.edge_scale, torch.zeros_like(x))


__all__ = ["FeatureScaler"]
