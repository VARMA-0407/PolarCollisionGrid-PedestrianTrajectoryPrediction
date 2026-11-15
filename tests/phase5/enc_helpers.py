"""Synthetic heterogeneous multi-frame graphs for the P5.5 encoder tests (no PIE data, no model weights from data)."""

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from pedsafety.phase5.models import EdgeBatch, EncoderConfig, EncoderInput

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "phase5" / "encoder_default.json"
CODE = {"pedestrian": 0, "vehicle": 1, "bicyclist": 2}
NAME = {v: k for k, v in CODE.items()}
ALL_RELATIONS = ("pedestrian__pedestrian", "vehicle__pedestrian", "pedestrian__vehicle", "bicyclist__pedestrian", "pedestrian__bicyclist",
                 "bicyclist__vehicle", "vehicle__bicyclist")


def cfg(**over) -> EncoderConfig:
    d = json.loads(CONFIG.read_text(encoding="utf-8"))
    d.update(over)
    return EncoderConfig.from_dict(d)


def cfg_bike(**over) -> EncoderConfig:
    d = dict(node_types=["pedestrian", "vehicle", "bicyclist"], bicyclist_context_enabled=True, relations=list(ALL_RELATIONS))
    d.update(over)
    return cfg(**d)


def make_input(B=2, H=6, A=6, seed=0, types=(0, 0, 1, 1, 0), absent=None, relations=None, edge_prob=1.0, n_pad=None) -> EncoderInput:
    """Slot 0 = target pedestrian; ``types`` gives the Task 3 codes of slots 0..len-1; remaining slots up to A are padding.
    ``absent`` is a list of (b, h, slot) triples that are not observed. Complete directed graph among present nodes of one frame."""
    g = np.random.default_rng(seed)
    n = len(types)
    assert A >= n and types[0] == 0
    node_type = np.full((B, A), -1, np.int64)
    node_type[:, :n] = np.array(types)[None]
    slot_valid = node_type >= 0
    track_index = np.where(slot_valid, np.arange(A)[None] + 100, -1).astype(np.int64)
    present = np.repeat(slot_valid[:, None, :], H, axis=1).copy()
    for b, h, s in (absent or []):
        present[b, h, s] = False
    f = np.zeros((B, H, A, 10), np.float32)
    f[..., 0] = g.uniform(0, 1900, (B, H, A))
    f[..., 1] = g.uniform(500, 1050, (B, H, A))
    f[..., 2:4] = g.normal(0, 60, (B, H, A, 2))
    f[..., 4] = np.hypot(f[..., 2], f[..., 3])
    ang = g.uniform(-np.pi, np.pi, (B, H, A))
    f[..., 5], f[..., 6] = np.sin(ang), np.cos(ang)
    f[..., 7] = g.uniform(40, 250, (B, H, A))
    f[..., 8] = g.integers(0, 3, (B, H, A))
    f[..., 9] = g.integers(0, 500, (B, H, A))
    masks = np.ones((B, H, A, 4), bool)
    masks[..., 1] = g.random((B, H, A)) > 0.1            # some velocities missing
    masks[..., 2] = masks[..., 1]
    masks[..., 3] = g.random((B, H, A)) > 0.1
    for col, mc in ((2, 1), (3, 1), (4, 1), (5, 2), (6, 2), (8, 3)):
        f[..., col] = np.where(masks[..., mc], f[..., col], np.nan)
    f[~present] = np.nan
    masks[~present] = False
    rel_names = relations or ("pedestrian__pedestrian", "vehicle__pedestrian", "pedestrian__vehicle")
    rows = {r: [] for r in rel_names}
    for b in range(B):
        for h in range(H):
            for s, d in itertools.permutations(range(n), 2):
                if present[b, h, s] and present[b, h, d] and g.random() <= edge_prob:
                    r = "%s__%s" % (NAME[types[s]], NAME[types[d]])
                    if r in rows:
                        rows[r].append((b, h, s, d))
    edges = {}
    for r, rr in rows.items():
        E = len(rr)
        attr = g.normal(0, 30, (E, 14)).astype(np.float32)
        mask = g.random((E, 14)) > 0.15
        mask[:, [0, 1, 2, 7, 8, 9, 13]] = True
        attr = np.where(mask, attr, np.nan).astype(np.float32)
        edges[r] = EdgeBatch(torch.tensor(np.array(rr, np.int64).reshape(-1, 4)), torch.tensor(attr), torch.tensor(mask),
                             torch.tensor(g.random((E, 6)) > 0.5))
    hist = (30 + 3 * torch.arange(H)).repeat(B, 1)
    return EncoderInput(window_id=torch.arange(B), anchor_frame=hist[:, -1].clone(), history_frames=hist,
                        track_index=torch.tensor(track_index), node_type=torch.tensor(node_type), slot_valid=torch.tensor(slot_valid),
                        present=torch.tensor(present), node_features=torch.tensor(f), node_masks=torch.tensor(masks), edges=edges)


def replace(inp: EncoderInput, **kw) -> EncoderInput:
    d = {k: getattr(inp, k) for k in inp.__dataclass_fields__}
    d.update(kw)
    return EncoderInput(**d)


def pad_slots(inp: EncoderInput, extra: int) -> EncoderInput:
    """Append ``extra`` padding slots (invalid, absent, NaN features, False masks) after the existing slots."""
    B, H, A = inp.present.shape
    nan = float("nan")
    return replace(inp, track_index=torch.cat([inp.track_index, torch.full((B, extra), -1, dtype=torch.long)], 1),
                   node_type=torch.cat([inp.node_type, torch.full((B, extra), -1, dtype=torch.long)], 1),
                   slot_valid=torch.cat([inp.slot_valid, torch.zeros(B, extra, dtype=torch.bool)], 1),
                   present=torch.cat([inp.present, torch.zeros(B, H, extra, dtype=torch.bool)], 2),
                   node_features=torch.cat([inp.node_features, torch.full((B, H, extra, 10), nan)], 2),
                   node_masks=torch.cat([inp.node_masks, torch.zeros(B, H, extra, 4, dtype=torch.bool)], 2))


def permute_slots(inp: EncoderInput, p) -> EncoderInput:
    """new slot i holds old slot p[i] (p[0] must be 0: the target stays in slot 0); edge endpoints are remapped."""
    p = np.asarray(p)
    assert p[0] == 0
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p))
    inv_t = torch.tensor(inv)
    edges = {}
    for r, e in inp.edges.items():
        b = e.bhsd.clone()
        b[:, 2] = inv_t[e.bhsd[:, 2]]
        b[:, 3] = inv_t[e.bhsd[:, 3]]
        edges[r] = EdgeBatch(b, e.attr, e.mask, e.flags)
    pt = torch.tensor(p)
    return replace(inp, track_index=inp.track_index[:, pt], node_type=inp.node_type[:, pt], slot_valid=inp.slot_valid[:, pt],
                   present=inp.present[:, :, pt], node_features=inp.node_features[:, :, pt], node_masks=inp.node_masks[:, :, pt], edges=edges)
