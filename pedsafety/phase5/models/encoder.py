"""Integrated spatial-temporal encoder: per-frame heterogeneous GATv2 -> per-agent sequences -> causal dilated TCN.

INPUT CONTRACT (``EncoderInput``, built from P5.3 ``collate_windows`` output by :meth:`EncoderInput.from_collated_batch`)
    History only. The dataclass has no field for future positions, future frames, future graphs or targets, and
    ``from_collated_batch`` copies only the listed history keys, so ``target_future_*`` / ``future_frames`` of a collated
    batch can never reach the model. The forward pass accepts only an ``EncoderInput``.

    track_index [B, A] long   sequence-scoped Task 3 track index per agent SLOT (-1 = padding). Slot 0 is the target pedestrian.
                              Slots are assigned by track identity in the P5.3 adapter, so the same slot index is the same track
                              at every history step: that is how per-frame embeddings are associated across time. The encoder
                              checks that valid slots of one window hold distinct tracks and never joins tracks across windows.
    node_type   [B, A] long   Task 3 codes (0 pedestrian, 1 vehicle, 2 bicyclist), -1 padding; constant per slot
    slot_valid  [B, A] bool
    present     [B, H, A] bool   the agent has an observed Task 3 node at that history sample (absent = masked, never filled)
    node_features [B, H, A, 10] float   Task 3 node columns u_px, v_px, vel_u_px_s, vel_v_px_s, speed_px_s, heading_sin,
                              heading_cos, box_h_px, frames_since_prev_obs, track_age_frames (NaN allowed where masked)
    node_masks  [B, H, A, 4] bool        position_valid, velocity_valid, heading_valid, frames_since_prev_obs_valid
    edges       {relation: EdgeBatch}    COO ``bhsd`` [E, 4] = (window, history step, source slot, destination slot), ``attr`` [E, 14],
                              ``mask`` [E, 14] (and optional ``flags`` [E, 6])
    history_frames [B, H], anchor_frame [B], window_id [B]: identifiers / provenance, passed through

DERIVED MODEL INPUTS (causal; history only)
    node: target-centred position (u, v minus the target's position at the LAST history sample), velocity (2), speed, heading
    sin/cos, box height, frames_since_prev_obs (Task 3: number of MISSING native frames since the same track's previous
    accepted observation; 0 = consecutive; masked for a track's first observation), their validity bits, and an is-target bit.
    ``track_age_frames`` is not used (unbounded past-only counter; see input contract section 10).
    edge: the 14 Task 3 edge attributes (scaled, masked entries 0) and their 14 validity bits (+ 6 flags if configured).
    Scaling constants come from ``EncoderConfig.feature_scaling`` (nominal, not fitted) or, later, train-only statistics.

OUTPUT (``EncoderOutput``): per-agent per-step embeddings ``sequence_embeddings`` [B, A, H, D] with ``presence`` [B, A, H]
    (zero and False where absent), ``target_summary`` [B, D] (target at the last step), ``last_step_embeddings`` [B, A, D]
    with ``last_step_mask`` [B, A], and the identifiers a decoder needs (``track_index``, ``node_type``, ``slot_valid``,
    ``is_target``, ``window_id``, ``anchor_frame``, ``history_frames``). No trajectory, probability or risk is produced.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn

from ..errors import WindowContractError
from ..task3_source import EDGE_FEATURES, NODE_FEATURES, NODE_MASKS, NODE_TYPE_NAMES
from .config import NODE_INPUT_FEATURES, EncoderConfig
from .scaling import FeatureScaler
from .spatial import HeteroGATv2Encoder
from .temporal import CausalDilatedTCN

N_RAW_NODE = len(NODE_FEATURES)            # 10
N_EDGE = len(EDGE_FEATURES)                # 14
N_FLAGS = 6
NODE_IN_DIM = len(NODE_INPUT_FEATURES) + 3 + 1       # 9 scaled + 3 validity bits + is-target
HISTORY_KEYS = ("window_id", "anchor_frame", "history_frames", "track_index", "node_type", "slot_valid", "present", "node_features", "node_masks", "edges")
_IDX = {n: i for i, n in enumerate(NODE_FEATURES)}


def _t(a, dtype=None) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(a) if not isinstance(a, torch.Tensor) else a)
    return t if dtype is None else t.to(dtype)


@dataclass(frozen=True)
class EdgeBatch:
    bhsd: torch.Tensor                       # [E, 4] long
    attr: torch.Tensor                       # [E, 14] float
    mask: torch.Tensor                       # [E, 14] bool
    flags: Optional[torch.Tensor] = None     # [E, 6] bool


@dataclass(frozen=True)
class EncoderInput:
    window_id: torch.Tensor
    anchor_frame: torch.Tensor
    history_frames: torch.Tensor
    track_index: torch.Tensor
    node_type: torch.Tensor
    slot_valid: torch.Tensor
    present: torch.Tensor
    node_features: torch.Tensor
    node_masks: torch.Tensor
    edges: Mapping[str, EdgeBatch] = field(default_factory=dict)

    def __post_init__(self):
        B, H, A = self.present.shape
        if self.node_features.shape != (B, H, A, N_RAW_NODE) or self.node_masks.shape != (B, H, A, len(NODE_MASKS)):
            raise WindowContractError("INPUT_SHAPE", "node_features must be [B,H,A,%d] and node_masks [B,H,A,%d]; got %s / %s"
                                      % (N_RAW_NODE, len(NODE_MASKS), tuple(self.node_features.shape), tuple(self.node_masks.shape)))
        for name, shape in (("track_index", (B, A)), ("node_type", (B, A)), ("slot_valid", (B, A)), ("window_id", (B,)), ("anchor_frame", (B,)),
                            ("history_frames", (B, H))):
            if getattr(self, name).shape != shape:
                raise WindowContractError("INPUT_SHAPE", "%s must have shape %s, got %s" % (name, shape, tuple(getattr(self, name).shape)))
        if self.present.dtype != torch.bool or self.slot_valid.dtype != torch.bool or self.node_masks.dtype != torch.bool:
            raise WindowContractError("INPUT_DTYPE", "present, slot_valid and node_masks must be boolean")
        if B and (not self.slot_valid[:, 0].all() or not self.present[:, -1, 0].all()):
            raise WindowContractError("TARGET_MISSING", "slot 0 (the target pedestrian) must be valid and present at the last history sample")
        if (self.present & ~self.slot_valid.unsqueeze(1)).any():
            raise WindowContractError("PRESENT_WITHOUT_SLOT", "an agent is present at a step but its slot is not valid")
        if B and (self.node_type[:, 0] != NODE_TYPE_NAMES.index("pedestrian")).any():
            raise WindowContractError("TARGET_NOT_PEDESTRIAN", "slot 0 must be a pedestrian")
        for b in range(B):
            tr = self.track_index[b][self.slot_valid[b]]
            if torch.unique(tr).numel() != tr.numel() or (tr < 0).any():
                raise WindowContractError("DUPLICATE_TRACK_SLOT", "window %d: valid slots must hold distinct, non-negative track indices" % b)
        if H > 1 and (not (self.history_frames[:, 1:] > self.history_frames[:, :-1]).all() or not torch.equal(self.history_frames[:, -1], self.anchor_frame)):
            raise WindowContractError("HISTORY_ORDER", "history_frames must increase strictly and end at anchor_frame")
        pos_ok = self.node_masks[..., 0] | ~self.present
        if not pos_ok.all():
            raise WindowContractError("POSITION_INVALID", "a present node has position_valid = false")
        pv = self.present.unsqueeze(-1)
        for c in (0, 1):
            if not torch.isfinite(self.node_features[..., c][self.present]).all():
                raise WindowContractError("NONFINITE_VALID_VALUE", "present node has non-finite %s" % NODE_FEATURES[c])
        for col, mcol in ((2, 1), (3, 1), (4, 1), (5, 2), (6, 2), (8, 3)):
            sel = self.present & self.node_masks[..., mcol]
            if not torch.isfinite(self.node_features[..., col][sel]).all():
                raise WindowContractError("NONFINITE_VALID_VALUE", "valid %s contains NaN or inf" % NODE_FEATURES[col])
        if not torch.isfinite(self.node_features[..., _IDX["box_h_px"]][self.present]).all():
            raise WindowContractError("NONFINITE_VALID_VALUE", "present node has non-finite box height")
        for r, e in self.edges.items():
            if e.bhsd.ndim != 2 or e.bhsd.shape[1] != 4 or e.attr.shape != (e.bhsd.shape[0], N_EDGE) or e.mask.shape != e.attr.shape:
                raise WindowContractError("INPUT_SHAPE", "edge batch of relation %s has inconsistent shapes" % r)
            if e.bhsd.numel():
                if (e.bhsd < 0).any() or (e.bhsd[:, 0] >= B).any() or (e.bhsd[:, 1] >= H).any() or (e.bhsd[:, 2:] >= A).any():
                    raise WindowContractError("EDGE_ENDPOINT", "relation %s has an index outside [B, H, A]" % r)
            if not torch.isfinite(e.attr[e.mask]).all():
                raise WindowContractError("NONFINITE_VALID_VALUE", "relation %s has a non-finite edge attribute whose mask is true" % r)

    def to(self, device) -> "EncoderInput":
        """Copy of this input on ``device`` (the encoder itself never moves data between devices)."""
        mv = lambda t: t.to(device)                                                                  # noqa: E731
        return EncoderInput(**{k: (mv(getattr(self, k)) if k != "edges" else {r: EdgeBatch(mv(e.bhsd), mv(e.attr), mv(e.mask), None if e.flags is None else mv(e.flags))
                                                                              for r, e in self.edges.items()}) for k in self.__dataclass_fields__})

    @classmethod
    def from_collated_batch(cls, batch: Mapping[str, Any]) -> "EncoderInput":
        """Build an input from a P5.3 ``collate_windows`` dict, copying ONLY the history keys in ``HISTORY_KEYS``."""
        missing = [k for k in HISTORY_KEYS if k not in batch]
        if missing:
            raise WindowContractError("INPUT_MISSING", "collated batch lacks %s" % missing)
        edges = {}
        for r, e in batch["edges"].items():
            edges[r] = EdgeBatch(_t(e["bhsd"], torch.long).reshape(-1, 4), _t(e["edge_attr"], torch.float32).reshape(-1, N_EDGE),
                                 _t(e["edge_mask"], torch.bool).reshape(-1, N_EDGE),
                                 _t(e["flags"], torch.bool).reshape(-1, N_FLAGS) if "flags" in e else None)
        return cls(window_id=_t(batch["window_id"], torch.long), anchor_frame=_t(batch["anchor_frame"], torch.long),
                   history_frames=_t(batch["history_frames"], torch.long), track_index=_t(batch["track_index"], torch.long),
                   node_type=_t(batch["node_type"], torch.long), slot_valid=_t(batch["slot_valid"], torch.bool),
                   present=_t(batch["present"], torch.bool), node_features=_t(batch["node_features"], torch.float32),
                   node_masks=_t(batch["node_masks"], torch.bool), edges=edges)


@dataclass
class EncoderOutput:
    sequence_embeddings: torch.Tensor       # [B, A, H, D]
    presence: torch.Tensor                  # [B, A, H]
    spatial_embeddings: torch.Tensor        # [B, H, A, D] (per-frame GATv2 output, before the TCN)
    target_summary: torch.Tensor            # [B, D]
    last_step_embeddings: torch.Tensor      # [B, A, D]
    last_step_mask: torch.Tensor            # [B, A]
    slot_valid: torch.Tensor
    is_target: torch.Tensor                 # [B, A]
    node_type: torch.Tensor
    track_index: torch.Tensor
    window_id: torch.Tensor
    anchor_frame: torch.Tensor
    history_frames: torch.Tensor
    receptive_field: int


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, history_length: Optional[int] = None, scaler: Optional[FeatureScaler] = None):
        super().__init__()
        if history_length is not None:
            cfg.check_history(history_length)
        self.cfg = cfg
        self.scaler = scaler if scaler is not None else FeatureScaler.from_config(cfg.feature_scaling)
        self.edge_in_dim = 2 * N_EDGE + (N_FLAGS if cfg.use_edge_flags else 0)
        with torch.random.fork_rng(devices=[]):                                # deterministic init without touching the caller's global RNG state
            torch.manual_seed(cfg.init_seed)
            self.spatial = HeteroGATv2Encoder(cfg, NODE_IN_DIM, self.edge_in_dim)
            self.temporal = CausalDilatedTCN(cfg.hidden_dim, cfg.hidden_dim, cfg.tcn_kernel_size, cfg.tcn_dilations, cfg.dropout)
        lut = torch.full((len(NODE_TYPE_NAMES) + 1,), -1, dtype=torch.long)       # index code+1 (so padding -1 -> slot 0 -> -1)
        for i, t in enumerate(cfg.node_types):
            lut[NODE_TYPE_NAMES.index(t) + 1] = i
        self.register_buffer("type_lut", lut)

    @property
    def receptive_field(self) -> int:
        return self.temporal.receptive_field

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- derived, causal inputs ----
    def _node_inputs(self, inp: EncoderInput) -> torch.Tensor:
        f, m, pres = inp.node_features, inp.node_masks, inp.present
        ref = f[:, -1, 0, 0:2]                                                # target position at the LAST history sample
        rel = f[..., 0:2] - ref[:, None, None, :]
        # columns 2..8 of the Task 3 node features are vel_u, vel_v, speed, heading_sin, heading_cos, box_h, frames_since_prev_obs
        # (track_age_frames, column 9, is deliberately not used)
        raw = torch.cat([rel, f[..., 2:9]], dim=-1)                          # [B,H,A,9] = NODE_INPUT_FEATURES order
        p = pres.unsqueeze(-1)
        valid = torch.cat([p, p, m[..., 1:2], m[..., 1:2], m[..., 1:2], m[..., 2:3], m[..., 2:3], p, m[..., 3:4]], dim=-1) & p
        x = self.scaler.scale_node(raw, valid)
        tgt = torch.zeros_like(pres, dtype=x.dtype)
        tgt[:, :, 0] = 1.0
        bits = torch.cat([(m[..., 1:2] & p).to(x.dtype), (m[..., 2:3] & p).to(x.dtype), (m[..., 3:4] & p).to(x.dtype), (tgt * pres.to(x.dtype)).unsqueeze(-1)], dim=-1)
        return torch.cat([x, bits], dim=-1)                                  # [B,H,A,13]

    def _edge_inputs(self, inp: EncoderInput) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, H, A = inp.present.shape
        out = {}
        for r, e in inp.edges.items():
            if r not in self.cfg.relations:
                raise WindowContractError("UNSUPPORTED_RELATION", "relation %s is not enabled in this encoder (enabled: %s)" % (r, list(self.cfg.relations)))
            x = self.scaler.scale_edge(e.attr, e.mask)
            parts = [x, e.mask.to(x.dtype)]
            if self.cfg.use_edge_flags:
                if e.flags is None:
                    raise WindowContractError("INPUT_MISSING", "use_edge_flags is enabled but relation %s carries no flags" % r)
                parts.append(e.flags.to(x.dtype))
            frame = e.bhsd[:, 0] * H + e.bhsd[:, 1]
            out[r] = (frame * A + e.bhsd[:, 2], frame * A + e.bhsd[:, 3], torch.cat(parts, dim=-1))
        return out

    def forward(self, inp: EncoderInput) -> EncoderOutput:
        if not isinstance(inp, EncoderInput):
            raise WindowContractError("ENCODER_INPUT_TYPE", "the encoder accepts only an EncoderInput (history-only); got %s" % type(inp).__name__)
        B, H, A = inp.present.shape
        if self.cfg.require_full_history_coverage:
            self.cfg.check_history(H)
        x = self._node_inputs(inp).reshape(B * H * A, NODE_IN_DIM)
        local = self.type_lut[(inp.node_type + 1)]                             # [B, A] local type index or -1
        valid_types = local.unsqueeze(1).expand(B, H, A)
        present = inp.present
        if (present & (valid_types < 0)).any():
            raise WindowContractError("UNSUPPORTED_NODE_TYPE", "a present node has a type that is not enabled in this encoder (node_types %s)" % (self.cfg.node_types,))
        s = self.spatial(x, valid_types.reshape(-1), present.reshape(-1), self._edge_inputs(inp))
        s = s.reshape(B, H, A, -1)
        seq = s.permute(0, 2, 1, 3).reshape(B * A, H, -1)
        pres_seq = present.permute(0, 2, 1).reshape(B * A, H)
        y = self.temporal(seq, pres_seq).reshape(B, A, H, -1)
        presence = present.permute(0, 2, 1)
        is_target = torch.zeros(B, A, dtype=torch.bool, device=present.device)
        is_target[:, 0] = True
        return EncoderOutput(y, presence, s, y[:, 0, -1], y[:, :, -1], presence[:, :, -1], inp.slot_valid, is_target, inp.node_type, inp.track_index,
                             inp.window_id, inp.anchor_frame, inp.history_frames, self.temporal.receptive_field)


__all__ = ["SpatioTemporalEncoder", "EncoderInput", "EncoderOutput", "EdgeBatch", "HISTORY_KEYS", "NODE_IN_DIM"]
