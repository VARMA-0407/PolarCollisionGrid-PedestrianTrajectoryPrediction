"""Pairwise conflict inputs: Phase 5 pedestrian predictions + history-only context-agent motion -> ``ConflictPairBatch``.

Pedestrian side: the Phase 5 multimodal predictions of each window's target ([K,F,2] raw pixels, mode weights, validity), unchanged.
Context side: every agent of an enabled type that has an OBSERVED node at the window's anchor frame (vehicles by default; context
pedestrians optionally). Its future is a classical extrapolation of its own OBSERVED HISTORY (the P5.4 predictors B0/B1/B2, applied through
the unchanged ``HistoryInput`` interface; there is no learned vehicle forecaster). Nothing after the anchor is read: only the graph-bank nodes
of the window's history frames are accessed (asserted). Missing history samples stay masked (no interpolation / forward fill); a context agent whose
motion predictor is invalid yields a row with an all-False validity mask, counted in the exclusion statistics, not silently dropped.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..phase5.baselines import HistoryInput, make_predictor
from ..phase5.graph_window_adapter import GraphBank
from ..phase5.task3_source import NODE_TYPE_NAMES
from ..phase5.window_builder import WindowSet
from .config import PAIR_TYPE_CODES, ConflictConfig
from .errors import ConflictContractError
from .features import compute_pair_features

NODE_TYPE_CODE = {n: i for i, n in enumerate(NODE_TYPE_NAMES)}          # pedestrian 0, vehicle 1, bicyclist 2 (Task 3 codes)


@dataclass
class Phase5Predictions:
    """The Phase 5 neural prediction artifact (``predictions_neural.npz``) as plain arrays."""
    window_id: np.ndarray            # [N] int64
    target_track_index: np.ndarray   # [N] int32
    anchor_frame: np.ndarray         # [N] int64
    positions_uv_px: np.ndarray      # [N,K,F,2] float32 raw pixels
    valid: np.ndarray                # [N,K,F] bool
    mode_probabilities: np.ndarray   # [N,K] float32 (uncalibrated softmax)
    anchor_uv_px: np.ndarray         # [N,2] float32 raw pixels
    horizon_offsets_s: np.ndarray    # [F] float64
    coordinate_system: str
    position_unit: str


@dataclass
class ConflictPairBatch:
    window_id: np.ndarray
    ped_track_index: np.ndarray
    ped_track_id: np.ndarray
    other_track_index: np.ndarray
    other_track_id: np.ndarray
    other_type: np.ndarray            # int8 Task 3 codes
    pair_type: np.ndarray             # int8 PAIR_TYPE_CODES
    anchor_frame: np.ndarray
    mode_probabilities: np.ndarray    # [P,K]
    horizon_s: np.ndarray             # [F]
    features: Dict[str, np.ndarray]
    masks: Dict[str, np.ndarray]
    context_prediction_valid: np.ndarray   # [P] the context agent's motion prediction was valid
    window_index: np.ndarray               # [P] row of the pair's window in the window dataset
    geometry: Dict[str, np.ndarray]        # history-derived anchors and last observed box sizes (raw pixels)
    coordinate_system: str
    coordinate_unit: str
    config_sha256: str
    exclusions: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return int(self.window_id.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.mode_probabilities.shape[1])


def check_predictions(pred: Phase5Predictions, ws: WindowSet, window_cfg, cfg: ConflictConfig) -> None:
    if pred.coordinate_system != cfg.coordinate_system or pred.position_unit != cfg.coordinate_unit:
        raise ConflictContractError("COORDINATE_SYSTEM", "predictions are %s / %s, expected %s / %s" % (pred.coordinate_system, pred.position_unit, cfg.coordinate_system, cfg.coordinate_unit))
    N = ws.n_windows
    K, F = pred.positions_uv_px.shape[1], pred.positions_uv_px.shape[2]
    if pred.positions_uv_px.shape != (N, K, F, 2) or pred.valid.shape != (N, K, F) or pred.mode_probabilities.shape != (N, K) or pred.anchor_uv_px.shape != (N, 2):
        raise ConflictContractError("SHAPE", "prediction arrays are inconsistent with %d windows" % N)
    if not (np.array_equal(pred.window_id, ws.window_id) and np.array_equal(pred.anchor_frame, ws.anchor_frame) and np.array_equal(pred.target_track_index, ws.target_track_index)):
        raise ConflictContractError("IDENTITY", "prediction identities (window id / anchor frame / target track) differ from the window dataset")
    expected = window_cfg.future_offsets / window_cfg.source_fps
    if pred.horizon_offsets_s.shape != expected.shape or not np.allclose(pred.horizon_offsets_s, expected, atol=1e-9):
        raise ConflictContractError("TEMPORAL_ALIGNMENT", "prediction horizon %s differs from the window configuration %s" % (pred.horizon_offsets_s.tolist(), expected.tolist()))
    if not np.array_equal(pred.anchor_uv_px, ws.history_target_uv_px[:, -1, :].astype(np.float32)):
        raise ConflictContractError("ANCHOR", "prediction anchors are not the last observed target footpoints")
    if not np.isfinite(pred.positions_uv_px[pred.valid]).all():
        raise ConflictContractError("NONFINITE_VALID_VALUE", "a valid predicted position is not finite")


def _wh(boxes: np.ndarray) -> np.ndarray:
    return np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=-1).astype(np.float64)


def build_conflict_pairs(pred: Phase5Predictions, ws: WindowSet, bank: GraphBank, target_bank_node: np.ndarray, window_cfg, cfg: ConflictConfig,
                         baseline_cfg: Mapping[str, Any], track_ids: Optional[List[str]] = None, window_indices: Optional[np.ndarray] = None,
                         provenance: Optional[Mapping[str, Any]] = None) -> ConflictPairBatch:
    check_predictions(pred, ws, window_cfg, cfg)
    sel = np.arange(ws.n_windows) if window_indices is None else np.asarray(window_indices, np.int64)
    type_codes = [NODE_TYPE_CODE[t] for t in cfg.context_agent_types]
    nodes = bank.nodes
    lookup = {(int(t), int(f)): i for i, (t, f) in enumerate(zip(nodes["track_index"].tolist(), nodes["frame"].tolist()))}
    H = ws.history_frames.shape[1]
    rows_w, rows_anchor_node, rows_hist = [], [], []
    excl = {"windows": int(sel.shape[0]), "windows_without_context_agent": 0, "context_agents_total": 0}
    for w in sel.tolist():
        anchor = int(ws.anchor_frame[w])
        k = bank.frame_pos(anchor)
        a, b = int(bank.node_ptr[k]), int(bank.node_ptr[k + 1])
        cand = [i for i in range(a, b) if int(nodes["node_type"][i]) in type_codes and int(nodes["track_index"][i]) != int(ws.target_track_index[w])]
        if not cand:
            excl["windows_without_context_agent"] += 1
        for i in cand:
            tr = int(nodes["track_index"][i])
            hist = []
            for f in ws.history_frames[w].tolist():
                if f > anchor:
                    raise ConflictContractError("FUTURE_FRAME_ACCESS", "window %d requests frame %d after its anchor %d" % (w, f, anchor))
                hist.append(lookup.get((tr, int(f)), -1))
            rows_w.append(w)
            rows_anchor_node.append(i)
            rows_hist.append(hist)
    R = len(rows_w)
    excl["context_agents_total"] = R
    F = pred.positions_uv_px.shape[2]
    K = pred.positions_uv_px.shape[1]
    hz = pred.horizon_offsets_s.astype(np.float64)
    if R == 0:
        raise ConflictContractError("NO_PAIRS", "no context agent was observed at any selected anchor; nothing to build")
    rw = np.asarray(rows_w, np.int64)
    an = np.asarray(rows_anchor_node, np.int64)
    hn = np.asarray(rows_hist, np.int64)                                                                # [R,H], -1 = not observed
    present = hn >= 0
    safe = np.where(present, hn, 0)
    pos_valid = present & nodes["masks"][safe, 0]
    vel_valid = present & nodes["masks"][safe, 1]
    hist_uv = np.where(pos_valid[..., None], nodes["features"][safe][..., 0:2].astype(np.float64), 0.0)
    hist_vel = np.where(vel_valid[..., None], nodes["features"][safe][..., 2:4].astype(np.float64), 0.0)
    ts = ws.history_timestamps[rw]
    hist_in = HistoryInput(window_id=np.arange(R), target_track_index=nodes["track_index"][an].astype(np.int32), target_track_id=[str(i) for i in nodes["track_index"][an]],
                           anchor_frame=ws.anchor_frame[rw], anchor_timestamp=ts[:, -1], history_timestamps=ts, history_uv_px=hist_uv, history_position_valid=pos_valid,
                           history_velocity_px_s=hist_vel, history_velocity_valid=vel_valid, horizon_offsets_s=hz, coordinate_system=cfg.coordinate_system,
                           position_unit=cfg.coordinate_unit, velocity_source="task2_causal_backward_difference_via_p53_graph_bank")
    spec = next((s for s in baseline_cfg["baselines"] if s["baseline_id"] == cfg.context_motion_model), None)
    if spec is None:
        raise ConflictContractError("CONFIG", "context_motion_model %s is not defined in the baseline configuration" % cfg.context_motion_model)
    motion = make_predictor(spec).predict(hist_in)
    ctx_valid = motion.valid.copy()                                                                      # [R,F]
    ctx_pos = np.where(ctx_valid[..., None], motion.positions, 0.0)
    excl["context_motion_invalid_rows"] = int((~motion.window_valid).sum())
    excl["context_motion_invalid_reasons"] = motion.invalid_counts()
    ped_pos = pred.positions_uv_px[rw].astype(np.float64)
    ped_valid = pred.valid[rw]
    probs = pred.mode_probabilities[rw].astype(np.float64)
    ped_node_anchor = target_bank_node[rw, -1]
    ped_wh, oth_wh = _wh(nodes["boxes"][ped_node_anchor]), _wh(nodes["boxes"][an])
    oth_anchor_uv = nodes["features"][an][..., 0:2].astype(np.float64)
    prev_dt = float(ws.history_timestamps[0, -1] - ws.history_timestamps[0, -2])
    res = compute_pair_features(ped_pos, ped_valid, probs, ctx_pos, ctx_valid, pred.anchor_uv_px[rw].astype(np.float64), oth_anchor_uv, ped_wh, oth_wh, hz,
                                min_valid_steps=cfg.min_valid_steps, kappa=cfg.score_kappa_sep_over_h, sens_thresholds=cfg.sensitivity_sep_over_h, eps=cfg.norm_epsilon,
                                ped_prev=ws.history_target_uv_px[rw, -2, :].astype(np.float64), oth_prev=hist_uv[:, -2, :], prev_valid=pos_valid[:, -2], prev_dt=prev_dt)
    other_tr = nodes["track_index"][an].astype(np.int32)
    other_type = nodes["node_type"][an].astype(np.int8)
    names = {v: k for k, v in NODE_TYPE_CODE.items()}
    pair_name = lambda t: "pedestrian_" + names[int(t)]                                                  # noqa: E731
    ptype = np.array([PAIR_TYPE_CODES[pair_name(t)] for t in other_type], np.int8)
    tid = (lambda i: track_ids[i]) if track_ids is not None else (lambda i: str(i))
    excl["pairs"] = R
    excl["pairs_by_type"] = {n: int((ptype == c).sum()) for n, c in PAIR_TYPE_CODES.items()}
    excl["pairs_valid"] = int(res["masks"]["pair_valid"].sum())
    excl["pairs_invalid"] = int((~res["masks"]["pair_valid"]).sum())
    prov = dict(provenance or {})
    prov.update({"coordinate_system": cfg.coordinate_system, "coordinate_unit": cfg.coordinate_unit, "context_motion_model": cfg.context_motion_model,
                 "context_motion_is_learned": False, "context_motion_inputs": "observed history only (graph-bank nodes of the window's history frames)",
                 "pedestrian_predictions": "Phase 5 neural multimodal predictions (uncalibrated mode weights)", "box_assumption": cfg.box_assumption,
                 "ego_motion_compensated": False, "physical_quantities": False, "labels_used": False, "future_observations_used": False})
    return ConflictPairBatch(window_id=ws.window_id[rw], ped_track_index=ws.target_track_index[rw], ped_track_id=ws.target_track_id[rw].astype(str), other_track_index=other_tr,
                             other_track_id=np.array([tid(int(i)) for i in other_tr], dtype=str), other_type=other_type, pair_type=ptype, anchor_frame=ws.anchor_frame[rw],
                             mode_probabilities=probs, horizon_s=hz, features=res["features"], masks=res["masks"], context_prediction_valid=motion.window_valid[:],
                             window_index=rw, geometry={"ped_anchor_uv_px": pred.anchor_uv_px[rw].astype(np.float64), "other_anchor_uv_px": oth_anchor_uv,
                                                        "ped_box_wh_px": ped_wh, "other_box_wh_px": oth_wh},
                             coordinate_system=cfg.coordinate_system, coordinate_unit=cfg.coordinate_unit, config_sha256=cfg.sha256(), exclusions=excl, provenance=prov)


__all__ = ["Phase5Predictions", "ConflictPairBatch", "build_conflict_pairs", "check_predictions"]
