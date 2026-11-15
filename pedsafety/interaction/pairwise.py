"""T3.3-T3.4: typed candidate pairs, image-space relational features, masks and sensitivity flags.

Edge direction (P1): ``j -> i`` = source j (neighbour) to destination i (target); displacement ``p_j - p_i``.
All pairs j != i of one frame whose (type_j, type_i) is an enabled relation are stored (complete candidate graph, Q5).
Radius and kNN flags are diagnostics and never remove an edge (P4, P5).

Features are computed in float64 from the STORED float32 node values (so ``nodes.npz`` and the edge files are mutually
consistent) and stored as float32. Unavailable values are NaN with a False mask (P6). Every feature is image space and
``physical: false``; ``closing_rate_px_s`` is the rate of change of PIXEL separation, not a road-relative closing speed.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import NODE_TYPES, InteractionConfig
from .nodes import NODE_FLOAT_FEATURES, NODE_MASKS, NodeTable

EDGE_FEATURES = ("du_px", "dv_px", "sep_px", "bearing_sin", "bearing_cos", "rel_bearing_sin", "rel_bearing_cos", "sep_over_h",
                 "log_h_ratio", "dv_row_px", "dvel_u_px_s", "dvel_v_px_s", "closing_rate_px_s", "overlap_ios")
_F = {n: i for i, n in enumerate(NODE_FLOAT_FEATURES)}
_M = {n: i for i, n in enumerate(NODE_MASKS)}


def flag_names(cfg: InteractionConfig) -> Tuple[str, ...]:
    return (("scale_truncation_affected",) + tuple("sep_over_h_le_%g" % r for r in cfg.radius_thresholds_sep_over_h)
            + tuple("knn_%d" % k for k in cfg.knn_k))


def relation_name(src: str, dst: str) -> str:
    return "%s__%s" % (src, dst)


@dataclass
class EdgeSet:
    relation: Tuple[str, str]
    frame: np.ndarray           # int64 (E,)
    edge_index: np.ndarray      # int64 (2, E): row 0 = source node index j, row 1 = destination node index i
    edge_attr: np.ndarray       # float32 (E, len(EDGE_FEATURES))
    edge_mask: np.ndarray       # bool (E, len(EDGE_FEATURES))
    flags: np.ndarray           # bool (E, len(flag_names))


def _frame_pairs(nt: NodeTable, a: int, b: int, cfg: InteractionConfig, rel_code: Dict[Tuple[int, int], int], not_represented: Counter):
    n = b - a
    if n < 2:
        return None
    J, I = np.meshgrid(np.arange(a, b), np.arange(a, b), indexing="xy")      # J varies along columns (source), I along rows
    J, I = J.ravel(), I.ravel()
    keep = J != I
    J, I = J[keep], I[keep]
    tj, ti = nt.node_type[J].astype(np.int64), nt.node_type[I].astype(np.int64)
    code = np.full(J.shape, -1, np.int64)
    for (sj, si), c in rel_code.items():
        code[(tj == sj) & (ti == si)] = c
    for (sj, si), cnt in Counter(zip(tj[code < 0].tolist(), ti[code < 0].tolist())).items():
        not_represented[(NODE_TYPES[sj], NODE_TYPES[si])] += cnt
    sel = code >= 0
    J, I, code = J[sel], I[sel], code[sel]
    if J.size == 0:
        return None
    X = nt.features.astype(np.float64)
    B = nt.boxes.astype(np.float64)
    Mk = nt.masks
    uj, vj, ui, vi = X[J, _F["u_px"]], X[J, _F["v_px"]], X[I, _F["u_px"]], X[I, _F["v_px"]]
    du, dv = uj - ui, vj - vi
    sep = np.hypot(du, dv)
    pos_sep = sep > 0.0
    safe = np.where(pos_sep, sep, 1.0)
    bs = np.where(pos_sep, dv / safe, np.nan)
    bc = np.where(pos_sep, du / safe, np.nan)
    head_ok = Mk[I, _M["heading_valid"]] & pos_sep
    hs, hc = X[I, _F["heading_sin"]], X[I, _F["heading_cos"]]
    rs = np.where(head_ok, bs * hc - bc * hs, np.nan)                  # sin(bearing - heading_i)
    rc = np.where(head_ok, bc * hc + bs * hs, np.nan)                  # cos(bearing - heading_i)
    hj, hi = X[J, _F["box_h_px"]], X[I, _F["box_h_px"]]
    sep_over_h = sep / np.maximum(hj, hi)
    log_h = np.log(hj / hi)
    vel_ok = Mk[J, _M["velocity_valid"]] & Mk[I, _M["velocity_valid"]]
    dvu = np.where(vel_ok, X[J, _F["vel_u_px_s"]] - X[I, _F["vel_u_px_s"]], np.nan)
    dvv = np.where(vel_ok, X[J, _F["vel_v_px_s"]] - X[I, _F["vel_v_px_s"]], np.nan)
    close_ok = vel_ok & pos_sep
    closing = np.where(close_ok, -(du * np.where(vel_ok, dvu, 0.0) + dv * np.where(vel_ok, dvv, 0.0)) / safe, np.nan)
    iw = np.minimum(B[J, 2], B[I, 2]) - np.maximum(B[J, 0], B[I, 0])
    ih = np.minimum(B[J, 3], B[I, 3]) - np.maximum(B[J, 1], B[I, 1])
    inter = np.clip(iw, 0.0, None) * np.clip(ih, 0.0, None)
    area_j = (B[J, 2] - B[J, 0]) * (B[J, 3] - B[J, 1])
    area_i = (B[I, 2] - B[I, 0]) * (B[I, 3] - B[I, 1])
    ios = inter / np.minimum(area_j, area_i)
    ones = np.ones(J.shape, bool)
    attr = np.stack([du, dv, sep, bs, bc, rs, rc, sep_over_h, log_h, dv, dvu, dvv, closing, ios], axis=1)
    mask = np.stack([ones, ones, ones, pos_sep, pos_sep, head_ok, head_ok, ones, ones, ones, vel_ok, vel_ok, close_ok, ones], axis=1)
    # sensitivity flags (never filters)
    trunc = nt.border[J] | nt.border[I]
    radius = [sep_over_h <= r for r in cfg.radius_thresholds_sep_over_h]
    # P4: per destination, rank ALL enabled incoming candidates by (sep_over_h, sep_px, source canonical order = node index)
    order = np.lexsort((J, sep, sep_over_h, I))                          # last key (I) is the primary grouping key
    rank = np.empty(J.shape, np.int64)
    Is = I[order]
    starts = np.r_[0, np.flatnonzero(np.diff(Is)) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, Is.size]))
    rank[order] = np.arange(Is.size) - group_start
    knn = [rank < k for k in cfg.knn_k]
    flags = np.stack([trunc] + radius + knn, axis=1)
    return J, I, code, attr, mask, flags


def build_edges(nt: NodeTable, cfg: InteractionConfig) -> Tuple[Dict[str, EdgeSet], Dict[str, int]]:
    rels = cfg.relations
    rel_code = {(NODE_TYPES.index(s), NODE_TYPES.index(d)): c for c, (s, d) in enumerate(rels)}
    parts: Dict[int, List[tuple]] = {c: [] for c in range(len(rels))}
    not_represented: Counter = Counter()
    nf = len(flag_names(cfg))
    for fr in sorted(nt.frame_starts):
        a, b = nt.frame_starts[fr]
        out = _frame_pairs(nt, a, b, cfg, rel_code, not_represented)
        if out is None:
            continue
        J, I, code, attr, mask, flags = out
        for c in np.unique(code).tolist():
            s = code == c
            o = np.lexsort((J[s], I[s]))                                   # canonical order within a frame: (destination, source)
            parts[c].append((np.full(int(s.sum()), fr, np.int64), J[s][o], I[s][o], attr[s][o], mask[s][o], flags[s][o]))
    edges: Dict[str, EdgeSet] = {}
    for c, (sname, dname) in enumerate(rels):
        p = parts[c]
        if p:
            frame = np.concatenate([x[0] for x in p])
            ei = np.stack([np.concatenate([x[1] for x in p]), np.concatenate([x[2] for x in p])]).astype(np.int64)
            attr = np.concatenate([x[3] for x in p]).astype(np.float32)
            mask = np.concatenate([x[4] for x in p])
            flags = np.concatenate([x[5] for x in p])
        else:
            frame, ei = np.empty(0, np.int64), np.empty((2, 0), np.int64)
            attr, mask, flags = np.empty((0, len(EDGE_FEATURES)), np.float32), np.empty((0, len(EDGE_FEATURES)), bool), np.empty((0, nf), bool)
        edges[relation_name(sname, dname)] = EdgeSet((sname, dname), frame, ei, attr, mask, flags)
    return edges, {relation_name(*k): v for k, v in sorted(not_represented.items())}


__all__ = ["EDGE_FEATURES", "flag_names", "relation_name", "EdgeSet", "build_edges"]
