"""Feature dictionary: formula, unit, coordinate space, mask rule and missingness reason for every stored array column."""

from typing import Any, Dict, List

from . import INTERACTION_VERSION
from .config import NODE_TYPES, InteractionConfig
from .nodes import NODE_BOX_COLUMNS, NODE_FLOAT_FEATURES, NODE_MASKS
from .pairwise import EDGE_FEATURES, flag_names

DICTIONARY_VERSION = "interaction-feature-dictionary-1.0.0"
SPACE = "image (captured raster pixels; origin top-left, +u right, +v down)"

_NODE = {
    "u_px": ("footpoint u (bbox bottom-centre x) from Task 2 position_x", "px", "always valid (node exists only with a valid position)", None),
    "v_px": ("footpoint v (bbox bottom y) from Task 2 position_y", "px", "always valid", None),
    "vel_u_px_s": ("Task 2 causal backward-difference velocity u", "px/s", "velocity_valid", "Task 2 velocity_state != valid (track start, gap, ...)"),
    "vel_v_px_s": ("Task 2 causal backward-difference velocity v", "px/s", "velocity_valid", "Task 2 velocity_state != valid"),
    "speed_px_s": ("Task 2 causal image speed |vel|", "px/s", "velocity_valid", "Task 2 velocity_state != valid"),
    "heading_sin": ("sin(Task 2 image heading); heading = atan2(vel_v, vel_u), from +u toward +v (clockwise on screen)", "1", "heading_valid", "Task 2 heading_state != valid"),
    "heading_cos": ("cos(Task 2 image heading)", "1", "heading_valid", "Task 2 heading_state != valid"),
    "box_h_px": ("y_max - y_min of the Phase 3 bbox joined through source_line", "px", "always valid (non-positive height is a contract failure)", None),
    "frames_since_prev_obs": ("Task 2 frame_gap_before: missing frames since the same track's previous accepted row (0 = consecutive)", "frames", "frames_since_prev_obs_valid", "first accepted row of the track"),
    "track_age_frames": ("frame_index - first observed frame of the track (0 at first observation; gap frames count)", "frames", "always valid", None),
}
_NODE_BOX = {c: ("Phase 3 bbox_xyxy column joined through source_line", "px") for c in NODE_BOX_COLUMNS}

_EDGE = {
    "du_px": ("u_j - u_i (edge j -> i)", "px", "antisymmetric", "always valid", None),
    "dv_px": ("v_j - v_i", "px", "antisymmetric", "always valid", None),
    "sep_px": ("sqrt(du^2 + dv^2): image-plane pixel separation, NOT a physical distance", "px", "symmetric", "always valid", None),
    "bearing_sin": ("dv / sep_px = sin(atan2(dv, du))", "1", "antisymmetric (reversal adds pi)", "sep_px > 0", "zero separation"),
    "bearing_cos": ("du / sep_px = cos(atan2(dv, du))", "1", "antisymmetric (reversal adds pi)", "sep_px > 0", "zero separation"),
    "rel_bearing_sin": ("sin(bearing - heading_i) using the DESTINATION heading", "1", "none", "sep_px > 0 and heading_valid(i)", "zero separation or destination heading unavailable"),
    "rel_bearing_cos": ("cos(bearing - heading_i)", "1", "none", "sep_px > 0 and heading_valid(i)", "zero separation or destination heading unavailable"),
    "sep_over_h": ("sep_px / max(box_h_j, box_h_i): heuristic depth-normalised proximity, not calibrated", "1", "symmetric", "always valid; see flag scale_truncation_affected", None),
    "log_h_ratio": ("ln(box_h_j / box_h_i): heuristic relative-scale proxy, not calibrated depth", "1", "antisymmetric", "always valid; see flag scale_truncation_affected", None),
    "dv_row_px": ("v_j - v_i (footpoint row difference; depth-order heuristic only under a flat road below the horizon). Numerically identical to dv_px by definition; kept as specified in the design", "px", "antisymmetric", "always valid", None),
    "dvel_u_px_s": ("vel_u_j - vel_u_i: relative IMAGE velocity (includes perspective and ego-motion parallax), not physical relative velocity", "px/s", "antisymmetric", "velocity_valid(j) and velocity_valid(i)", "either endpoint velocity unavailable"),
    "dvel_v_px_s": ("vel_v_j - vel_v_i", "px/s", "antisymmetric", "velocity_valid(j) and velocity_valid(i)", "either endpoint velocity unavailable"),
    "closing_rate_px_s": ("-(du*dvel_u + dv*dvel_v) / sep_px = -d(sep_px)/dt; positive = image-plane separation shrinking. Rate of change of PIXEL separation, not road-relative closing speed", "px/s", "symmetric", "velocity_valid(j) and velocity_valid(i) and sep_px > 0", "either velocity unavailable or zero separation"),
    "overlap_ios": ("intersection area / smaller box area (boxes j and i); occlusion indicator", "1", "symmetric", "always valid", None),
}


def build_feature_dictionary(cfg: InteractionConfig) -> Dict[str, Any]:
    def node_entry(i, name):
        formula, unit, mask, missing = _NODE[name]
        return {"name": name, "index": i, "dtype": "float32", "formula": formula, "unit": unit, "coordinate_space": SPACE,
                "mask": mask, "missing_reason": missing, "missing_value": "NaN" if missing else None, "physical": False, "causal": True}

    def edge_entry(i, name):
        formula, unit, sym, mask, missing = _EDGE[name]
        return {"name": name, "index": i, "dtype": "float32", "formula": formula, "unit": unit, "coordinate_space": SPACE,
                "reversal_property": sym, "mask": mask, "missing_reason": missing, "missing_value": "NaN" if missing else None,
                "physical": False, "causal": True}

    flags: List[Dict[str, Any]] = [{"name": "scale_truncation_affected", "index": 0, "dtype": "bool", "physical": False,
                                    "definition": "either endpoint inherits Task 2 bbox_touches_border = true (box within 1 px of the image border): "
                                                  "POSSIBLE truncation affecting box-height-based features; not confirmed truncation"}]
    for k, r in enumerate(cfg.radius_thresholds_sep_over_h, start=1):
        flags.append({"name": "sep_over_h_le_%g" % r, "index": k, "dtype": "bool", "physical": False,
                      "definition": "sep_over_h <= %g; exploratory sensitivity flag, not a validated threshold; never removes an edge" % r})
    base = 1 + len(cfg.radius_thresholds_sep_over_h)
    for k, kk in enumerate(cfg.knn_k):
        flags.append({"name": "knn_%d" % kk, "index": base + k, "dtype": "bool", "physical": False,
                      "definition": "source is among the %d nearest incoming candidates of the destination across all enabled relations, ranked by "
                                    "(sep_over_h, sep_px, source canonical track order); exploratory; never removes an edge" % kk})
    assert [f["name"] for f in flags] == list(flag_names(cfg))
    return {
        "dictionary_version": DICTIONARY_VERSION, "producer": INTERACTION_VERSION, "coordinate_system": cfg.expected_coordinate_system,
        "coordinate_space": SPACE, "physical": False, "causal_inputs_only": True,
        "conventions": {"edge_direction": "j -> i: source j (neighbour) to destination i (target); edge_index[0] = source node, edge_index[1] = destination node",
                        "displacement": "p_j - p_i", "heading": "image-plane radians from +u toward +v (clockwise on screen), range [-pi, pi)",
                        "relative_bearing": "relative to the destination agent's heading",
                        "closing_rate": "positive when image-plane separation shrinks; not road-relative closing speed",
                        "missing_values": "NaN in float arrays, with a False entry in the parallel boolean mask array",
                        "edge_order": "per relation: frame ascending, then destination node index, then source node index",
                        "node_order": "frame ascending, then canonical (natural) track order"},
        "node_types": list(NODE_TYPES), "node_type_codes": {t: i for i, t in enumerate(NODE_TYPES)},
        "relations": ["%s__%s" % r for r in cfg.relations],
        "node_features": [node_entry(i, n) for i, n in enumerate(NODE_FLOAT_FEATURES)],
        "node_boxes": [{"name": c, "index": i, "dtype": "float32", "unit": "px", "formula": _NODE_BOX[c][0], "coordinate_space": SPACE, "physical": False}
                       for i, c in enumerate(NODE_BOX_COLUMNS)],
        "node_masks": [{"name": m, "index": i, "dtype": "bool"} for i, m in enumerate(NODE_MASKS)],
        "edge_features": [edge_entry(i, n) for i, n in enumerate(EDGE_FEATURES)],
        "edge_flags": flags,
        "excluded": {"acceleration": "excluded from default node features (decision A1: frame-to-frame raw acceleration dominated by non-smooth components)",
                     "same_track_family": "reserved in the design table and always false in Task 3 (fragments are never linked); not stored because it carries no information",
                     "uncertainty_features": "none in the default representation (Q8); see diagnostics.uncertainty",
                     "world_coordinates": "never produced; world_x/world_y null and world_position_valid false upstream"},
    }


__all__ = ["DICTIONARY_VERSION", "build_feature_dictionary"]
