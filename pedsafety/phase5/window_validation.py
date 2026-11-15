"""Independent validation of a window dataset (in memory or re-read from disk).

Every check returns a named result; :func:`validate_window_dataset` returns ``{"ok": bool, "checks": {...}}`` and
never repairs anything. When the source Task 3 run is supplied, every stored target value and every bank node/edge is
compared with the Task 3 arrays (proof that nothing was interpolated, filled or merged).
"""

from typing import Any, Dict, Optional

import numpy as np

from ..interaction.config import DEFAULT_RELATIONS, OPTIONAL_VV
from .graph_window_adapter import GraphBank
from .task3_source import NODE_FEATURE_MASK, NODE_TYPE_NAMES, Task3Run
from .window_builder import REJECTION_REASONS, WindowSet
from .window_config import WindowConfig


def _eq_nan(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(np.nan_to_num(a), np.nan_to_num(b)))
    return bool(np.array_equal(a, b))


def validate_window_dataset(ws: WindowSet, bank: GraphBank, target_bank_node: np.ndarray, cfg: WindowConfig,
                            run: Optional[Task3Run] = None) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}

    def chk(name: str, ok: bool, detail: Any = None):
        checks[name] = {"pass": bool(ok), "detail": detail}

    W, H, F = ws.n_windows, cfg.history_length, cfg.prediction_length
    tol = cfg.timestamp_tolerance_s
    hoff, foff = cfg.history_offsets, cfg.future_offsets
    chk("shapes", ws.history_frames.shape == (W, H) and ws.future_frames.shape == (W, F) and ws.history_timestamps.shape == (W, H)
        and ws.future_timestamps.shape == (W, F) and ws.target_future_uv_px.shape == (W, F, 2) and target_bank_node.shape == (W, H),
        {"windows": W, "H": H, "F": F})
    chk("anchor_is_last_history_frame", bool(np.array_equal(ws.history_frames[:, -1], ws.anchor_frame)))
    chk("anchors_on_grid", bool(np.all(ws.anchor_frame % cfg.anchor_stride_frames == cfg.anchor_phase_frames)))
    chk("history_frame_offsets", bool(np.array_equal(ws.history_frames - ws.anchor_frame[:, None], np.broadcast_to(hoff, (W, H)))),
        hoff.tolist())
    chk("future_frame_offsets", bool(np.array_equal(ws.future_frames - ws.anchor_frame[:, None], np.broadcast_to(foff, (W, F)))),
        foff.tolist())
    allf = np.concatenate([ws.history_frames, ws.future_frames], axis=1)
    chk("frame_stride_equals_fps_over_hz", bool(np.all(np.diff(allf, axis=1) == cfg.sampling_stride_frames)),
        {"stride": cfg.sampling_stride_frames, "source_fps": cfg.source_fps, "target_hz": cfg.target_sampling_hz})
    chk("history_timestamps_strictly_increasing", bool(np.all(np.diff(ws.history_timestamps, axis=1) > 0)))
    chk("future_timestamps_strictly_increasing", bool(np.all(np.diff(ws.future_timestamps, axis=1) > 0)))
    chk("future_after_last_history", bool(np.all(ws.future_timestamps[:, 0] > ws.history_timestamps[:, -1])) if W else True)
    chk("history_future_frames_disjoint", bool(np.all(ws.history_frames.max(axis=1) < ws.future_frames.min(axis=1))) if W else True)
    allt = np.concatenate([ws.history_timestamps, ws.future_timestamps], axis=1)
    dt = np.diff(allt, axis=1)
    chk("sampling_interval_10hz_within_tolerance", bool(np.all(np.abs(dt - cfg.sampling_interval_s) <= tol)),
        {"expected_s": cfg.sampling_interval_s, "tolerance_s": tol, "max_abs_dev_s": float(np.abs(dt - cfg.sampling_interval_s).max()) if W else 0.0})
    chk("timestamps_equal_frame_over_fps", bool(np.all(np.abs(allt - allf / cfg.source_fps) <= tol)))
    chk("history_span_s", bool(np.all(np.abs(ws.history_timestamps[:, -1] - ws.history_timestamps[:, 0] - cfg.history_span_s) <= tol)),
        cfg.history_span_s)
    chk("future_horizon_s", bool(np.all(np.abs(ws.future_timestamps[:, -1] - ws.history_timestamps[:, -1] - cfg.future_horizon_s) <= tol)),
        cfg.future_horizon_s)
    chk("strict_target_future_complete", bool(ws.target_future_valid.all() and np.isfinite(ws.target_future_uv_px).all()
                                               and np.all(ws.target_future_task3_node >= 0)))
    chk("strict_target_history_complete", bool(np.all(ws.history_target_task3_node >= 0) and np.isfinite(ws.history_target_uv_px).all()))
    keys = set(zip(ws.target_track_index.tolist(), ws.anchor_frame.tolist()))
    rkeys = set(zip(ws.rejected_track_index.tolist(), ws.rejected_anchor_frame.tolist()))
    chk("candidates_unique_and_disjoint", len(keys) == W and len(rkeys) == ws.n_rejected and not (keys & rkeys))
    chk("rejection_codes_known", bool(np.all((ws.rejected_reason_code >= 0) & (ws.rejected_reason_code < len(REJECTION_REASONS)))))

    # ----- graph bank -----
    bn = bank.nodes
    K = bank.frames.shape[0]
    chk("bank_frames_equal_history_frames", bool(np.array_equal(bank.frames, np.unique(ws.history_frames))) if W else K == 0,
        {"bank_frames": int(K)})
    nf_ok = all(bool(np.all(bn["frame"][bank.node_ptr[k]:bank.node_ptr[k + 1]] == bank.frames[k])) for k in range(K))
    chk("bank_node_frames_consistent", nf_ok and int(bank.node_ptr[-1]) == bank.n_nodes)
    ef_ok = True
    for name, e in bank.edges.items():
        for k in range(K):
            if not np.all(e.frame[e.frame_ptr[k]:e.frame_ptr[k + 1]] == bank.frames[k]):
                ef_ok = False
        if e.edge_index.size and not (np.array_equal(bn["frame"][e.edge_index[0]], e.frame) and np.array_equal(bn["frame"][e.edge_index[1]], e.frame)):
            ef_ok = False
        if e.edge_index.size and not (np.all(bn["node_type"][e.edge_index[0]] == NODE_TYPE_NAMES.index(e.relation[0]))
                                      and np.all(bn["node_type"][e.edge_index[1]] == NODE_TYPE_NAMES.index(e.relation[1]))):
            ef_ok = False
    chk("bank_edge_frames_and_types_consistent", ef_ok)
    mask_ok = all(np.array_equal(np.isfinite(bn["features"][:, c]), bn["masks"][:, m]) if m is not None else bool(np.isfinite(bn["features"][:, c]).all())
                  for c, m in enumerate(NODE_FEATURE_MASK)) and bool(bn["masks"][:, 0].all())
    mask_ok = mask_ok and all(np.array_equal(np.isfinite(e.edge_attr), e.edge_mask) for e in bank.edges.values())
    chk("feature_validity_masks", mask_ok, "node NaN exactly where the governing mask is False; edge NaN exactly where edge_mask is False")
    unc = np.isin(bn["node_type"], [NODE_TYPE_NAMES.index(t) for t in cfg.semantically_uncertain_node_types])
    chk("semantic_uncertainty_flags", bool(np.array_equal(unc, bn["semantically_uncertain"])))
    approved = set(DEFAULT_RELATIONS) | {OPTIONAL_VV}
    rel_ok = all(tuple(r) in approved for r in bank.relations) and ("bicyclist", "bicyclist") not in bank.relations
    chk("approved_relations_only", rel_ok, ["%s__%s" % r for r in bank.relations])
    if W:
        tb = target_bank_node
        chk("target_bank_nodes_are_target_track", bool(np.all(bn["track_index"][tb] == ws.target_track_index[:, None])))
        chk("target_bank_nodes_at_history_frames", bool(np.array_equal(bn["frame"][tb], ws.history_frames)))
        chk("targets_are_pedestrians", bool(np.all(bn["node_type"][tb] == NODE_TYPE_NAMES.index("pedestrian"))))
        chk("history_inputs_not_after_anchor", bool(np.all(bn["frame"][tb] <= ws.anchor_frame[:, None])))
        chk("history_target_uv_equals_bank", bool(np.array_equal(bn["features"][tb][:, :, 0:2], ws.history_target_uv_px)))
        chk("history_target_task3_node_equals_bank", bool(np.array_equal(bn["task3_node_index"][tb], ws.history_target_task3_node)))

    # ----- cross-check against the Task 3 source -----
    if run is not None:
        t3 = bn["task3_node_index"]
        same = all(_eq_nan(bn[k], getattr(run, k)[t3]) for k in ("frame", "timestamp", "track_index", "node_type", "source_line", "features",
                                                               "boxes", "border", "masks"))
        chk("bank_nodes_equal_task3", same)
        complete = all(int(bank.node_ptr[k + 1] - bank.node_ptr[k]) == run.frame_slices[int(f)][1] - run.frame_slices[int(f)][0]
                       for k, f in enumerate(bank.frames.tolist()))
        chk("bank_contains_every_task3_node_of_its_frames", complete)
        eok = True
        ecount = {}
        for name, e in bank.edges.items():
            s = run.edges[name]
            r = e.task3_edge_index
            eok &= _eq_nan(e.edge_attr, s.edge_attr[r]) and _eq_nan(e.edge_mask, s.edge_mask[r]) and _eq_nan(e.flags, s.flags[r]) \
                and _eq_nan(e.frame, s.frame[r]) and bool(np.array_equal(t3[e.edge_index], s.edge_index[:, r]))
            n_src = int(np.isin(s.frame, bank.frames).sum())
            eok &= n_src == e.frame.shape[0]
            ecount[name] = int(e.frame.shape[0])
        chk("bank_edges_equal_task3_and_complete", eok, ecount)
        chk("relations_equal_task3", set(bank.edges) == set(run.edges), sorted(run.edges))
        if W:
            hn, fn = ws.history_target_task3_node, ws.target_future_task3_node
            chk("no_interpolation_history_targets_are_task3_nodes",
                bool(np.all(run.track_index[hn] == ws.target_track_index[:, None]) and np.array_equal(run.frame[hn], ws.history_frames)
                     and np.array_equal(run.features[hn][:, :, 0:2], ws.history_target_uv_px)
                     and np.array_equal(run.source_line[hn], ws.history_target_source_line)))
            chk("no_interpolation_future_targets_are_task3_nodes",
                bool(np.all(run.track_index[fn] == ws.target_track_index[:, None]) and np.array_equal(run.frame[fn], ws.future_frames)
                     and np.array_equal(run.features[fn][:, :, 0:2], ws.target_future_uv_px)
                     and np.array_equal(run.source_line[fn], ws.target_future_source_line)
                     and np.array_equal(run.features[fn][:, :, 7], ws.target_future_box_h_px)
                     and np.array_equal(run.border[fn], ws.target_future_border)))
            chk("timestamps_equal_task3", bool(np.array_equal(run.timestamp[hn], ws.history_timestamps) and np.array_equal(run.timestamp[fn], ws.future_timestamps)))
            ids = np.array([run.track_id(int(i)) for i in ws.target_track_index], dtype=str)
            chk("target_track_ids_equal_task3", bool(np.array_equal(ids, ws.target_track_id)))
    return {"ok": all(c["pass"] for c in checks.values()), "n_checks": len(checks), "checks": checks}


__all__ = ["validate_window_dataset"]
