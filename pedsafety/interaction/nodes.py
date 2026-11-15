"""T3.2: frame index and typed node table (image space; decision record P6-P9, P11).

One node per causal row with observation_type == 'observed' and position_state == 'ok' (P9). Nodes are ordered by
(frame_index, canonical track order). Unavailable values are NaN with an explicit boolean mask (P6). Acceleration is not
a node feature (A1). The border flag is inherited from Task 2 ``bbox_touches_border`` (P11), never recomputed.
"""

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import NODE_TYPES
from .input_adapter import JoinedRow

NODE_TYPE_CODE = {t: i for i, t in enumerate(NODE_TYPES)}          # pedestrian 0, vehicle 1, bicyclist 2

# float32 node feature columns, in storage order
NODE_FLOAT_FEATURES = ("u_px", "v_px", "vel_u_px_s", "vel_v_px_s", "speed_px_s", "heading_sin", "heading_cos", "box_h_px",
                       "frames_since_prev_obs", "track_age_frames")
NODE_BOX_COLUMNS = ("box_x_min_px", "box_y_min_px", "box_x_max_px", "box_y_max_px")
NODE_MASKS = ("position_valid", "velocity_valid", "heading_valid", "frames_since_prev_obs_valid")


def canonical_track_key(track_id: str) -> Tuple:
    """Deterministic natural order: split on ':' and on digit runs, comparing numbers numerically."""
    parts: List[Tuple[int, object]] = []
    for seg in track_id.split(":"):
        for tok in re.findall(r"\d+|\D+", seg):
            parts.append((0, int(tok)) if tok.isdigit() else (1, tok))
        parts.append((2, ""))
    return tuple(parts) + ((3, track_id),)


@dataclass
class NodeTable:
    frame: np.ndarray                 # int64 (N,)
    timestamp: np.ndarray             # float64 (N,)
    track_index: np.ndarray           # int32 (N,)
    node_type: np.ndarray             # int8 (N,)
    source_line: np.ndarray           # int64 (N,)
    features: np.ndarray              # float32 (N, len(NODE_FLOAT_FEATURES)), NaN where unavailable
    boxes: np.ndarray                 # float32 (N, 4)
    border: np.ndarray                # bool (N,)  inherited bbox_touches_border
    masks: np.ndarray                 # bool (N, len(NODE_MASKS))
    track_ids: List[str]              # track_index -> track_id
    track_source_labels: List[Dict[str, int]]
    frame_starts: Dict[int, Tuple[int, int]]   # frame -> [start, stop) node slice

    @property
    def n(self) -> int:
        return int(self.frame.shape[0])


def build_node_table(rows: Sequence[JoinedRow]) -> NodeTable:
    eligible = [r for r in rows if r.observation_type == "observed" and r.position_state == "ok"]
    tracks = sorted({r.track_id for r in eligible}, key=canonical_track_key)
    tindex = {t: i for i, t in enumerate(tracks)}
    first_frame: Dict[str, int] = {}
    for r in eligible:
        first_frame[r.track_id] = min(first_frame.get(r.track_id, r.frame_index), r.frame_index)
    eligible.sort(key=lambda r: (r.frame_index, tindex[r.track_id]))
    n = len(eligible)
    frame = np.empty(n, np.int64)
    ts = np.empty(n, np.float64)
    tix = np.empty(n, np.int32)
    ntype = np.empty(n, np.int8)
    sline = np.empty(n, np.int64)
    feats = np.full((n, len(NODE_FLOAT_FEATURES)), np.nan, np.float32)
    boxes = np.empty((n, 4), np.float32)
    border = np.empty(n, bool)
    masks = np.zeros((n, len(NODE_MASKS)), bool)
    labels: List[Dict[str, int]] = [dict() for _ in tracks]
    for k, r in enumerate(eligible):
        frame[k], ts[k], tix[k], ntype[k], sline[k] = r.frame_index, r.timestamp, tindex[r.track_id], NODE_TYPE_CODE[r.agent_class], r.source_line
        f = [r.position_x, r.position_y, np.nan, np.nan, np.nan, np.nan, np.nan, r.bbox_xyxy[3] - r.bbox_xyxy[1], np.nan,
             r.frame_index - first_frame[r.track_id]]
        vel_ok = r.velocity_state == "valid"
        if vel_ok:
            f[2], f[3], f[4] = r.velocity_x, r.velocity_y, r.speed
        head_ok = r.heading_state == "valid"
        if head_ok:
            f[5], f[6] = math.sin(r.heading), math.cos(r.heading)
        gap_ok = r.frame_gap_before is not None
        if gap_ok:
            f[8] = r.frame_gap_before
        feats[k] = f
        boxes[k] = r.bbox_xyxy
        border[k] = r.bbox_touches_border
        masks[k] = (True, vel_ok, head_ok, gap_ok)
        lab = labels[tindex[r.track_id]]
        lab[str(r.source_label)] = lab.get(str(r.source_label), 0) + 1
    starts: Dict[int, Tuple[int, int]] = {}
    if n:
        change = np.flatnonzero(np.diff(frame)) + 1
        bounds = np.concatenate(([0], change, [n]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            starts[int(frame[a])] = (int(a), int(b))
    return NodeTable(frame, ts, tix, ntype, sline, feats, boxes, border, masks, tracks, labels, starts)


__all__ = ["NODE_TYPE_CODE", "NODE_FLOAT_FEATURES", "NODE_BOX_COLUMNS", "NODE_MASKS", "canonical_track_key", "NodeTable", "build_node_table"]
