"""Build the history-only predictor input from a stored P5.3 window dataset.

Sources (all history, verified in P5.3 against Task 3 / Task 2 / Phase 3):
    positions     windows.npz ``history_target_uv_px`` [W, 6, 2] (== graph-bank node features u_px, v_px)
    timestamps    windows.npz ``history_timestamps`` [W, 6] (frame / 30)
    position mask graph-bank ``masks[:, 0]`` (position_valid) at ``history_target_bank_node``
    velocity      graph-bank ``features[:, 2:4]`` (vel_u_px_s, vel_v_px_s) = Task 2 CAUSAL one-frame backward difference
    velocity mask graph-bank ``masks[:, 1]`` (velocity_valid)
No ``target_future_*`` / ``future_*`` array is read here. Horizon offsets come from the window configuration.
"""

import numpy as np

from ..errors import WindowContractError
from ..graph_window_adapter import GraphBank
from ..task3_source import NODE_FEATURES, NODE_MASKS
from ..window_builder import WindowSet
from ..window_config import WindowConfig
from .base import HistoryInput

VELOCITY_SOURCE = "task2_causal_backward_difference_via_p53_graph_bank"


def build_history_input(ws: WindowSet, bank: GraphBank, target_bank_node: np.ndarray, cfg: WindowConfig) -> HistoryInput:
    tbn = np.asarray(target_bank_node, np.int64)
    if tbn.shape != ws.history_frames.shape:
        raise WindowContractError("HISTORY_SHAPE", "history_target_bank_node must match history_frames")
    bn = bank.nodes
    if not (np.array_equal(bn["track_index"][tbn], np.broadcast_to(ws.target_track_index[:, None], tbn.shape))
            and np.array_equal(bn["frame"][tbn], ws.history_frames)):
        raise WindowContractError("HISTORY_REFERENCE", "graph-bank target references do not match the windows")
    if np.any(bn["frame"][tbn] > ws.anchor_frame[:, None]):
        raise WindowContractError("FUTURE_FRAME_IN_INPUT", "a history reference lies after its anchor")
    iu, iv = NODE_FEATURES.index("u_px"), NODE_FEATURES.index("v_px")
    ivu, ivv = NODE_FEATURES.index("vel_u_px_s"), NODE_FEATURES.index("vel_v_px_s")
    feats = bn["features"][tbn]
    masks = bn["masks"][tbn]
    uv = ws.history_target_uv_px.astype(np.float64)
    if not np.array_equal(feats[:, :, [iu, iv]], ws.history_target_uv_px):
        raise WindowContractError("HISTORY_REFERENCE", "windows history positions differ from graph-bank node positions")
    return HistoryInput(
        window_id=ws.window_id, target_track_index=ws.target_track_index, target_track_id=ws.target_track_id, anchor_frame=ws.anchor_frame,
        anchor_timestamp=ws.history_timestamps[:, -1], history_timestamps=ws.history_timestamps, history_uv_px=uv,
        history_position_valid=masks[:, :, NODE_MASKS.index("position_valid")],
        history_velocity_px_s=feats[:, :, [ivu, ivv]].astype(np.float64),
        history_velocity_valid=masks[:, :, NODE_MASKS.index("velocity_valid")],
        horizon_offsets_s=cfg.future_offsets / cfg.source_fps, coordinate_system=cfg.coordinate_system, position_unit=cfg.coordinate_unit,
        velocity_source=VELOCITY_SOURCE)


__all__ = ["build_history_input", "VELOCITY_SOURCE"]
