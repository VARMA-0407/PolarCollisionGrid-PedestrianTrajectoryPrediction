"""Phase 4 Task 2: OFFLINE / NON-CAUSAL image-space kinematics. NOT for real-time inference.

Central differences use the NEXT observation, so every row produced here is labelled ``OFFLINE_NON_CAUSAL`` and carries
``velocity_method = central_difference_actual_dt``. The causal module (``image_kinematics``) does not import this module,
and ``image_kinematics.require_causal_rows`` refuses these rows.

For accepted observations i-1, i, i+1 of one track that are on consecutive frames (no missing frame on either side),
have available positions and the same identity (raster and agent class):

    velocity      v_i = (p_{i+1} - p_{i-1}) / (t_{i+1} - t_{i-1})
    acceleration  a_i = 2 * [ (p_{i+1} - p_i)/(t_{i+1} - t_i) - (p_i - p_{i-1})/(t_i - t_{i-1}) ] / (t_{i+1} - t_{i-1})
    heading       atan2(v_y, v_x) wrapped to [-pi, pi), only if speed > 0 and speed >= min_heading_speed

``a_i`` is exact for quadratic motion on any spacing; ``v_i`` is exact for quadratic motion only on equal spacing (on
unequal spacing it is the secant slope). No interpolation is performed: an absent frame stays absent and its neighbours
get ``gap_adjacent``. Actual timestamps are used throughout; nominal FPS is never substituted.
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .image_kinematics import (
    CausalBatchResult, ImageKinematicsConfig, RecordCheck, VALID, base_row, position_info,
)
from .trajectory import VIEW_OFFLINE, wrap_angle

OFFLINE_VELOCITY_METHOD = "central_difference_actual_dt"
OFFLINE_ACCELERATION_METHOD = "nonuniform_second_central_difference_actual_dt"

S_FIRST_OR_LAST = "insufficient_neighbours"
S_GAP_ADJACENT = "gap_adjacent"
S_NEIGHBOUR_UNAVAILABLE = "neighbour_position_unavailable"
S_IDENTITY = "identity_changed"
S_DT = "nonpositive_or_non_finite_dt"
S_DT_TOO_LARGE = "dt_too_large"
S_BELOW_MIN_SPEED = "below_min_speed"
S_POSITION_UNAVAILABLE = "position_unavailable"
S_NON_FINITE = "non_finite_result"


def _rows_for_track(idxs: List[int], records, checks: Sequence[RecordCheck], source_lines, cfg: ImageKinematicsConfig) -> List[Dict[str, Any]]:
    out = []
    n = len(idxs)
    for k, i in enumerate(idxs):
        r, c = records[i], checks[i]
        row = base_row(r, source_lines[i], VIEW_OFFLINE)
        ok = c.position is not None
        row.update({"position_x": c.position[0] if ok else None, "position_y": c.position[1] if ok else None, "position_state": c.reason,
                    "observation_type": "observed" if ok else "invalid", "bbox_touches_border": c.touches_border,
                    "velocity_method": OFFLINE_VELOCITY_METHOD, "acceleration_method": OFFLINE_ACCELERATION_METHOD, "filter_kind": "none",
                    "dt_s": None, "frame_gap_before": None if k == 0 else (r["frame_index"] - records[idxs[k - 1]]["frame_index"]) // cfg.frame_step - 1})
        for key in ("filtered_position_x", "filtered_position_y", "filtered_velocity_x", "filtered_velocity_y", "filtered_speed",
                    "filtered_velocity_state", "filtered_acceleration_x", "filtered_acceleration_y", "filtered_acceleration_state",
                    "filtered_heading", "filtered_heading_state"):
            row[key] = None
        state = _support_state(k, idxs, records, checks, cfg)
        vx = vy = sp = ax = ay = hd = None
        hstate = state
        if state == VALID:
            im, ip = idxs[k - 1], idxs[k + 1]
            (x0, y0), (x1, y1), (x2, y2) = checks[im].position, c.position, checks[ip].position
            t0, t1, t2 = float(records[im]["timestamp"]), float(r["timestamp"]), float(records[ip]["timestamp"])
            h1, h2, span = t1 - t0, t2 - t1, t2 - t0
            vx, vy = (x2 - x0) / span, (y2 - y0) / span
            ax = 2.0 * ((x2 - x1) / h2 - (x1 - x0) / h1) / span
            ay = 2.0 * ((y2 - y1) / h2 - (y1 - y0) / h1) / span
            sp = math.hypot(vx, vy)
            row["dt_s"] = span
            if not all(math.isfinite(v) for v in (vx, vy, ax, ay, sp)):
                state = hstate = S_NON_FINITE
                vx = vy = sp = ax = ay = None
            elif sp > 0.0 and sp >= cfg.min_heading_speed_px_per_s:
                hd, hstate = wrap_angle(math.atan2(vy, vx)), VALID
            else:
                hstate = S_BELOW_MIN_SPEED
        row.update({"velocity_x": vx, "velocity_y": vy, "speed": sp, "velocity_state": state,
                    "acceleration_x": ax, "acceleration_y": ay, "acceleration_state": state, "heading": hd, "heading_state": hstate})
        out.append(row)
    return out


def _support_state(k: int, idxs: List[int], records, checks: Sequence[RecordCheck], cfg: ImageKinematicsConfig) -> str:
    i = idxs[k]
    if checks[i].position is None:
        return S_POSITION_UNAVAILABLE
    if k == 0 or k == len(idxs) - 1:
        return S_FIRST_OR_LAST
    im, ip = idxs[k - 1], idxs[k + 1]
    f = records[i]["frame_index"]
    if f - records[im]["frame_index"] != cfg.frame_step or records[ip]["frame_index"] - f != cfg.frame_step:
        return S_GAP_ADJACENT
    if checks[im].position is None or checks[ip].position is None:
        return S_NEIGHBOUR_UNAVAILABLE
    ident = position_info(records[i], checks[i].position).identity
    if position_info(records[im], checks[im].position).identity != ident or position_info(records[ip], checks[ip].position).identity != ident:
        return S_IDENTITY
    t0, t1, t2 = float(records[im]["timestamp"]), float(records[i]["timestamp"]), float(records[ip]["timestamp"])
    h1, h2 = t1 - t0, t2 - t1
    if not (math.isfinite(h1) and math.isfinite(h2)) or h1 <= 0 or h2 <= 0:
        return S_DT
    if cfg.max_dt_seconds is not None and (h1 > cfg.max_dt_seconds or h2 > cfg.max_dt_seconds):
        return S_DT_TOO_LARGE
    return VALID


def process_offline_records(records: Sequence[Mapping[str, Any]], source_lines: Sequence[int], causal: CausalBatchResult,
                            cfg: ImageKinematicsConfig) -> List[Dict[str, Any]]:
    """Offline rows for exactly the observations the causal pass accepted (same validation and grouping), in input order.
    The causal result is only read for its accepted indices and record checks; its kinematic values are not used."""
    rows_by_index: Dict[int, Dict[str, Any]] = {}
    for tid, idxs in causal.accepted.items():
        idxs = [i for i in idxs if i in causal.steps]                  # stream-rejected rows are never support
        for i, row in zip(idxs, _rows_for_track(idxs, records, causal.checks, source_lines, cfg)):
            rows_by_index[i] = row
    out = [rows_by_index[i] for i in sorted(rows_by_index)]
    for row in out:
        assert row["view"] == VIEW_OFFLINE and row["velocity_method"] == OFFLINE_VELOCITY_METHOD
    return out


__all__ = ["OFFLINE_VELOCITY_METHOD", "OFFLINE_ACCELERATION_METHOD", "process_offline_records"]
