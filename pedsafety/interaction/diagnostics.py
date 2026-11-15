"""T3.6: interaction diagnostics (descriptive, image space; nothing here is a physical or safety measure)."""

import math
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from .config import NODE_TYPES, InteractionConfig
from .nodes import NODE_FLOAT_FEATURES, NODE_MASKS, NodeTable
from .pairwise import EDGE_FEATURES, EdgeSet, flag_names

_E = {n: i for i, n in enumerate(EDGE_FEATURES)}


def pct(values) -> Optional[Dict[str, float]]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    q = np.percentile(a, [0, 5, 25, 50, 75, 95, 99, 100])
    return {"n": int(a.size), "min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "p50": float(q[3]), "p75": float(q[4]),
            "p95": float(q[5]), "p99": float(q[6]), "max": float(q[7]), "mean": float(a.mean()), "std": float(a.std())}


def int_hist(values) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(int(x) for x in values).items())}


def theoretical_edge_counts(nt: NodeTable) -> Dict[str, int]:
    out: Dict[str, int] = {"%s__%s" % (s, d): 0 for s in NODE_TYPES for d in NODE_TYPES}
    for a, b in nt.frame_starts.values():
        c = np.bincount(nt.node_type[a:b], minlength=len(NODE_TYPES))
        for si, s in enumerate(NODE_TYPES):
            for di, d in enumerate(NODE_TYPES):
                out["%s__%s" % (s, d)] += int(c[si]) * (int(c[di]) - (1 if si == di else 0))
    return out


def build_diagnostics(nt: NodeTable, edges: Dict[str, EdgeSet], not_represented: Dict[str, int], cfg: InteractionConfig,
                      adapter_counts: Dict[str, Any]) -> Dict[str, Any]:
    n = nt.n
    per_frame_nodes = [b - a for a, b in nt.frame_starts.values()]
    fnames = flag_names(cfg)
    theo = theoretical_edge_counts(nt)
    in_deg = np.zeros(n, np.int64)
    out_deg = np.zeros(n, np.int64)
    edges_per_frame: Counter = Counter()
    rel: Dict[str, Any] = {}
    for name, e in edges.items():
        E = int(e.frame.size)
        if E:
            np.add.at(out_deg, e.edge_index[0], 1)
            np.add.at(in_deg, e.edge_index[1], 1)
            edges_per_frame.update(e.frame.tolist())
        a, m = e.edge_attr, e.edge_mask
        dvel_u = a[m[:, _E["dvel_u_px_s"]], _E["dvel_u_px_s"]]
        dvel_v = a[m[:, _E["dvel_v_px_s"]], _E["dvel_v_px_s"]]
        rel[name] = {
            "edges": E, "theoretical_complete_graph_edges": theo[name], "equals_theoretical": E == theo[name],
            "feature_missing_fraction": {f: (float(1.0 - m[:, i].mean()) if E else None) for i, f in enumerate(EDGE_FEATURES)},
            "flag_counts": {f: int(e.flags[:, i].sum()) for i, f in enumerate(fnames)},
            "flag_fractions": {f: (float(e.flags[:, i].mean()) if E else None) for i, f in enumerate(fnames)},
            "sep_px": pct(a[:, _E["sep_px"]]), "sep_over_h": pct(a[:, _E["sep_over_h"]]), "log_h_ratio": pct(a[:, _E["log_h_ratio"]]),
            "relative_image_speed_px_s": pct(np.hypot(dvel_u, dvel_v)),
            "closing_rate_px_s": pct(a[m[:, _E["closing_rate_px_s"]], _E["closing_rate_px_s"]]),
            "overlap_ios": pct(a[:, _E["overlap_ios"]]), "overlap_ios_positive_fraction": (float((a[:, _E["overlap_ios"]] > 0).mean()) if E else None),
        }
    # ---- P17 uncertainty diagnostics (exploratory; no feature; not a threshold) ----
    sigma_agent = cfg.noise_reference_per_agent_velocity_px_s
    ref = math.sqrt(2.0) * sigma_agent
    unc: Dict[str, Any] = {
        "label": "EXPLORATORY DIAGNOSTIC — not calibrated uncertainty, not a threshold, not physical",
        "per_agent_velocity_noise_px_s_per_axis": sigma_agent,
        "component_noise_reference_px_s": ref,
        "reference_definition": ("approximate component-wise standard deviation of relative image velocity (dvel_u or dvel_v) under the assumption "
                                 "that the two agents' velocity errors are independent with ~%g px/s per axis each (sqrt(2) x %g = %.1f). The per-agent "
                                 "value is the Task 2 single-stride jitter estimate; Task 2's stride test shows the independence/white-noise "
                                 "assumption is imperfect." % (sigma_agent, sigma_agent, ref)),
        "per_relation": {},
    }
    for name, e in edges.items():
        a, m = e.edge_attr, e.edge_mask
        ok = m[:, _E["dvel_u_px_s"]] & m[:, _E["dvel_v_px_s"]]
        du, dv = a[ok, _E["dvel_u_px_s"]], a[ok, _E["dvel_v_px_s"]]

        def robust(x):
            return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else None
        unc["per_relation"][name] = {
            "edges_with_valid_relative_velocity": int(ok.sum()),
            "dvel_u_px_s": pct(du), "dvel_v_px_s": pct(dv),
            "dvel_u_robust_sd_px_s": robust(du), "dvel_v_robust_sd_px_s": robust(dv),
            "exploratory_both_components_within_one_reference": (float(((np.abs(du) <= ref) & (np.abs(dv) <= ref)).mean()) if du.size else None),
            "exploratory_both_components_within_two_references": (float(((np.abs(du) <= 2 * ref) & (np.abs(dv) <= 2 * ref)).mean()) if du.size else None),
        }
    nf = nt.features
    node_types = {t: int((nt.node_type == i).sum()) for i, t in enumerate(NODE_TYPES)}
    return {
        "data": "image space only; descriptive statistics, not physical or safety measures",
        "inputs": adapter_counts,
        "frames_with_nodes": len(nt.frame_starts), "nodes": n, "tracks": len(nt.track_ids),
        "nodes_by_type": node_types,
        "nodes_per_frame": pct(per_frame_nodes), "nodes_per_frame_max": max(per_frame_nodes) if per_frame_nodes else 0,
        "edges_total": int(sum(r["edges"] for r in rel.values())),
        "edges_per_frame": pct([edges_per_frame.get(f, 0) for f in nt.frame_starts]),
        "theoretical_complete_graph_edges_by_type_pair": theo,
        "candidate_pairs_not_represented_by_enabled_relations": not_represented,
        "relations": rel,
        "degree": {"in_degree": pct(in_deg), "out_degree": pct(out_deg), "in_degree_hist": int_hist(in_deg), "out_degree_hist": int_hist(out_deg),
                   "isolated_nodes": int(((in_deg == 0) & (out_deg == 0)).sum())},
        "node_feature_missing_fraction": {f: (float(np.isnan(nf[:, i]).mean()) if n else None) for i, f in enumerate(NODE_FLOAT_FEATURES)},
        "node_mask_valid_fraction": {mname: (float(nt.masks[:, i].mean()) if n else None) for i, mname in enumerate(NODE_MASKS)},
        "border_touching_nodes": int(nt.border.sum()), "border_touching_node_fraction": (float(nt.border.mean()) if n else None),
        "track_age_frames": pct(nf[:, NODE_FLOAT_FEATURES.index("track_age_frames")]),
        "frames_since_prev_obs_hist": int_hist(nf[nt.masks[:, NODE_MASKS.index("frames_since_prev_obs_valid")], NODE_FLOAT_FEATURES.index("frames_since_prev_obs")]),
        "box_h_px": pct(nf[:, NODE_FLOAT_FEATURES.index("box_h_px")]),
        "uncertainty": unc,
    }


__all__ = ["build_diagnostics", "theoretical_edge_counts", "pct"]
