"""Pairwise image-space conflict features (pure numpy, no learning, no labels).

Everything is computed in the image plane of ``captured_1920x1080`` (pixels, origin top-left, +u right, +v down). Nothing here is a
physical distance, speed, TTC or PET: pixel separations are depth dependent and the camera moves (no ego-motion compensation).

Notation. A pair is (pedestrian target i, context agent j). Pedestrian i has K predicted modes with probabilities pi_k (Phase 5; NOT calibrated)
and positions p_i^k(t); agent j has one motion hypothesis q_j(t) (classical extrapolation of its observed history). t = 1..F indexes the
horizon grid tau_t = horizon_s[t-1] (seconds after the anchor); tau_0 = 0 is the anchor (last observed sample). a_i, a_j are the observed
anchor footpoints. h_i is the pedestrian's last observed box height (px), (w, h) are last observed box sizes.

Step level (shape [P,K,F]), valid where pedestrian mode k and agent j are valid at step t and the scale reference is valid:
    rel_u, rel_v      p_i^k(t) - q_j(t)                                   [px]
    sep_px            || p_i^k(t) - q_j(t) ||_2                           [px]
    sep_over_h        sep_px / h_i                                        [dimensionless heuristic depth proxy, not a distance]
    closing_rate      (sep(t-1) - sep(t)) / (tau_t - tau_{t-1}), sep(0) = ||a_i - a_j||   [px/s, image plane; >0 = separation shrinking]
                      valid when steps t-1 (or the observed anchor for t = 1) and t are valid
    box_overlap_ios   |A_i(t) ∩ A_j(t)| / min(|A_i|, |A_j|)  in [0,1], A = axis-aligned box of the last OBSERVED size placed with its
                      bottom-centre at the predicted footpoint (assumption CONSTANT_ANCHOR_BOX_SIZE...: depth change is ignored)
Mode level (shape [P,K]), valid when n_valid_steps >= min_valid_steps:
    min_sep_px, min_sep_over_h, t_min_sep_s (tau at the first minimum of sep_px; closest predicted APPROACH in the image plane, NOT a TTC),
    max_closing_rate, approach_fraction (share of valid closing steps with closing_rate > 0), max_box_overlap_ios,
    first_box_overlap_time_s (first tau with ios > 0; defined only where max_box_overlap_ios > 0),
    heuristic_proximity_score = max_t clip(1 - sep_over_h(t) / kappa, 0, 1)   [0..1; UNCALIBRATED heuristic, not a probability],
    sens_sep_over_h_le_x = (min_sep_over_h <= x) for each exploratory x                    [bool; unvalidated sensitivity flags],
    los_convergence_cos = -(d_0 · v_rel) / (|d_0| |v_rel|), d_0 = a_i - a_j, v_rel = (d_last - d_first)/(tau_last - tau_first) over the
                      first / last valid step; in [-1,1], +1 = the pair converges along the observed line of sight; needs >= 2 valid steps.
Pair level (shape [P]):
    valid_mode_mass = sum_k pi_k [mode valid]; mode_weighted_* = sum_k pi_k [valid] x_k / valid_mode_mass (an average over the model's
    UNCALIBRATED mode weights, independence between the pedestrian modes and the single agent hypothesis is implicit);
    most_probable_mode and its features; worst_case_* = min / max over valid modes (needs no ground truth);
    observed_sep_anchor_px, observed_sep_anchor_over_h, observed_closing_rate_px_s (from the last two OBSERVED history samples),
    log_scale_ratio = ln(h_j / h_i) (large |value| suggests different depths: image proximity is then less meaningful).
All invalid values are NaN and every quantity has an explicit boolean mask in ``masks``.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .errors import ConflictContractError

# name -> (level, unit, definition, validity rule, mask key)
FEATURE_SPECS = {
    "rel_u": ("step", "px", "pedestrian footpoint u minus agent footpoint u at the predicted step", "step_valid", "step_valid"),
    "rel_v": ("step", "px", "pedestrian footpoint v minus agent footpoint v at the predicted step", "step_valid", "step_valid"),
    "sep_px": ("step", "px", "Euclidean image-plane distance between predicted footpoints", "step_valid", "step_valid"),
    "sep_over_h": ("step", "1", "sep_px divided by the pedestrian's last observed box height (heuristic depth proxy)", "step_valid", "step_valid"),
    "closing_rate_px_s": ("step", "px/s", "(sep(t-1) - sep(t)) / dt with sep(0) the observed anchor separation; positive = converging in the image", "closing_valid", "closing_valid"),
    "box_overlap_ios": ("step", "1", "intersection over the smaller box area of anchor-size boxes at predicted footpoints", "step_valid", "step_valid"),
    "n_valid_steps": ("mode", "steps", "number of valid horizon steps", "always defined", "all"),
    "min_sep_px": ("mode", "px", "minimum predicted image-plane separation over the horizon", "mode_valid", "mode_valid"),
    "min_sep_over_h": ("mode", "1", "minimum of sep_over_h over the horizon", "mode_valid", "mode_valid"),
    "t_min_sep_s": ("mode", "s", "horizon time of the first minimum separation (closest predicted approach; not a TTC)", "mode_valid", "mode_valid"),
    "max_closing_rate_px_s": ("mode", "px/s", "maximum image-plane closing rate over valid closing steps", "closing_summary_valid", "closing_summary_valid"),
    "approach_fraction": ("mode", "1", "share of valid closing steps with positive closing rate", "closing_summary_valid", "closing_summary_valid"),
    "max_box_overlap_ios": ("mode", "1", "maximum predicted anchor-size box overlap over the horizon", "mode_valid", "mode_valid"),
    "first_box_overlap_time_s": ("mode", "s", "first horizon time with positive box overlap", "overlap_defined", "overlap_defined"),
    "heuristic_proximity_score": ("mode", "1", "max_t clip(1 - sep_over_h/kappa, 0, 1); uncalibrated heuristic, NOT a probability", "mode_valid", "mode_valid"),
    "los_convergence_cos": ("mode", "1", "cosine between the observed line of sight (reversed) and the predicted relative displacement direction", "los_valid", "los_valid"),
    "valid_mode_mass": ("pair", "1", "sum of the pedestrian mode weights over valid modes (uncalibrated weights)", "always defined", "all"),
    "mode_weighted_score": ("pair", "1", "mode-weighted mean of heuristic_proximity_score", "pair_valid", "pair_valid"),
    "mode_weighted_box_overlap_mass": ("pair", "1", "mode-weighted share of modes whose boxes overlap in the image (uncalibrated)", "pair_valid", "pair_valid"),
    "mode_weighted_sens_mass": ("pair", "1", "mode-weighted share of modes below each exploratory sep_over_h threshold (uncalibrated)", "pair_valid", "pair_valid"),
    "mp_min_sep_over_h": ("pair", "1", "min_sep_over_h of the most probable pedestrian mode", "mp_valid", "mp_valid"),
    "mp_score": ("pair", "1", "heuristic_proximity_score of the most probable pedestrian mode", "mp_valid", "mp_valid"),
    "worst_case_min_sep_over_h": ("pair", "1", "minimum min_sep_over_h over valid modes (no ground truth used)", "pair_valid", "pair_valid"),
    "worst_case_max_box_overlap_ios": ("pair", "1", "maximum max_box_overlap_ios over valid modes", "pair_valid", "pair_valid"),
    "observed_sep_anchor_px": ("pair", "px", "observed image-plane separation at the anchor", "scale_ok", "scale_ok"),
    "observed_sep_anchor_over_h": ("pair", "1", "observed anchor separation divided by the pedestrian box height", "scale_ok", "scale_ok"),
    "observed_closing_rate_px_s": ("pair", "px/s", "observed closing rate from the last two history samples", "observed_closing_valid", "observed_closing_valid"),
    "log_scale_ratio": ("pair", "1", "ln(agent box height / pedestrian box height) at the anchor", "scale_ok", "scale_ok"),
}


def _check(cond: bool, code: str, msg: str):
    if not cond:
        raise ConflictContractError(code, msg)


def _finite_where(a: np.ndarray, mask: np.ndarray, what: str):
    if mask.any() and not np.isfinite(a[mask]).all():
        raise ConflictContractError("NONFINITE_VALID_VALUE", "%s contains a non-finite value under a valid mask" % what)


def box_overlap_ios(pos_a, wh_a, pos_b, wh_b):
    """Overlap of two axis-aligned boxes (bottom-centre at ``pos``, size ``wh`` = (w, h)); broadcasting over leading axes. In [0, 1]."""
    ax1, ax2 = pos_a[..., 0] - wh_a[..., 0] / 2.0, pos_a[..., 0] + wh_a[..., 0] / 2.0
    ay1, ay2 = pos_a[..., 1] - wh_a[..., 1], pos_a[..., 1]
    bx1, bx2 = pos_b[..., 0] - wh_b[..., 0] / 2.0, pos_b[..., 0] + wh_b[..., 0] / 2.0
    by1, by2 = pos_b[..., 1] - wh_b[..., 1], pos_b[..., 1]
    iw = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    area = np.minimum(wh_a[..., 0] * wh_a[..., 1], wh_b[..., 0] * wh_b[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(area > 0, iw * ih / area, np.nan)


def compute_pair_features(ped_pos, ped_valid, ped_probs, oth_pos, oth_valid, ped_anchor, oth_anchor, ped_wh, oth_wh, horizon_s, *, min_valid_steps: int,
                          kappa: float, sens_thresholds: Sequence[float], eps: float, ped_prev: Optional[np.ndarray] = None,
                          oth_prev: Optional[np.ndarray] = None, prev_valid: Optional[np.ndarray] = None, prev_dt: Optional[float] = None) -> Dict[str, Any]:
    """Compute all pair features. Shapes: ped_pos [P,K,F,2], ped_valid [P,K,F], ped_probs [P,K], oth_pos [P,F,2], oth_valid [P,F], anchors [P,2],
    ped_wh / oth_wh [P,2] (w, h), horizon_s [F]. Returns ``{"features": {...}, "masks": {...}}`` (all float64 / bool numpy)."""
    ped_pos, ped_valid = np.asarray(ped_pos, np.float64), np.asarray(ped_valid, bool)
    ped_probs, oth_pos, oth_valid = np.asarray(ped_probs, np.float64), np.asarray(oth_pos, np.float64), np.asarray(oth_valid, bool)
    ped_anchor, oth_anchor = np.asarray(ped_anchor, np.float64), np.asarray(oth_anchor, np.float64)
    ped_wh, oth_wh, hz = np.asarray(ped_wh, np.float64), np.asarray(oth_wh, np.float64), np.asarray(horizon_s, np.float64)
    _check(ped_pos.ndim == 4 and ped_pos.shape[-1] == 2, "SHAPE", "ped_pos must be [P,K,F,2], got %s" % (ped_pos.shape,))
    P, K, F, _ = ped_pos.shape
    _check(ped_valid.shape == (P, K, F) and ped_probs.shape == (P, K) and oth_pos.shape == (P, F, 2) and oth_valid.shape == (P, F)
           and ped_anchor.shape == (P, 2) and oth_anchor.shape == (P, 2) and ped_wh.shape == (P, 2) and oth_wh.shape == (P, 2) and hz.shape == (F,),
           "SHAPE", "inconsistent input shapes for P=%d K=%d F=%d" % (P, K, F))
    _check(F >= 1 and np.all(np.isfinite(hz)) and hz[0] > 0 and np.all(np.diff(hz) > 0), "HORIZON", "horizon_s must be finite, positive and strictly increasing")
    _check(np.isfinite(ped_probs).all() and (ped_probs >= 0).all(), "PROBABILITIES", "mode probabilities must be finite and non-negative")
    _check(P == 0 or np.allclose(ped_probs.sum(axis=1), 1.0, atol=1e-4), "PROBABILITIES", "mode probabilities must sum to 1 per pair")
    _check(isinstance(min_valid_steps, int) and 1 <= min_valid_steps <= F, "CONFIG", "min_valid_steps must be in [1, F]")
    _finite_where(ped_pos, np.repeat(ped_valid[..., None], 2, axis=3), "predicted pedestrian positions")
    _finite_where(oth_pos, np.repeat(oth_valid[..., None], 2, axis=2), "agent positions")
    anchors_ok = np.isfinite(ped_anchor).all(1) & np.isfinite(oth_anchor).all(1)
    scale_ok = anchors_ok & np.isfinite(ped_wh).all(1) & np.isfinite(oth_wh).all(1) & (ped_wh > 0).all(1) & (oth_wh > 0).all(1)
    S = len(sens_thresholds)
    nan = np.nan
    ped_h = ped_wh[:, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        step_valid = ped_valid & oth_valid[:, None, :] & scale_ok[:, None, None]
        d = ped_pos - oth_pos[:, None, :, :]
        rel_u, rel_v = np.where(step_valid, d[..., 0], nan), np.where(step_valid, d[..., 1], nan)
        sep = np.where(step_valid, np.hypot(d[..., 0], d[..., 1]), nan)
        sep_over_h = sep / ped_h[:, None, None]
        sep0 = np.where(scale_ok, np.hypot(*(ped_anchor - oth_anchor).T), nan)                       # [P]
        prev_sep = np.concatenate([np.repeat(sep0[:, None, None], K, axis=1), sep[:, :, :-1]], axis=2)
        prev_ok = np.concatenate([np.repeat(scale_ok[:, None, None], K, axis=1), step_valid[:, :, :-1]], axis=2)
        dt = np.diff(np.concatenate([[0.0], hz]))
        closing_valid = step_valid & prev_ok
        closing = np.where(closing_valid, (prev_sep - sep) / dt[None, None, :], nan)
        ped_boxpos, oth_boxpos = ped_pos, oth_pos[:, None, :, :]
        ios = np.where(step_valid, box_overlap_ios(ped_boxpos, ped_wh[:, None, None, :], oth_boxpos, oth_wh[:, None, None, :]), nan)
        n_valid = step_valid.sum(axis=2)
        mode_valid = n_valid >= min_valid_steps
        inf = np.inf
        min_sep = np.where(mode_valid, np.min(np.where(step_valid, sep, inf), axis=2), nan)
        min_soh = np.where(mode_valid, np.min(np.where(step_valid, sep_over_h, inf), axis=2), nan)
        arg_min = np.argmin(np.where(step_valid, sep, inf), axis=2)
        t_min = np.where(mode_valid, hz[arg_min], nan)
        closing_summary_valid = closing_valid.sum(axis=2) >= 1
        max_closing = np.where(closing_summary_valid, np.max(np.where(closing_valid, closing, -inf), axis=2), nan)
        approach = np.where(closing_summary_valid, (closing_valid & (closing > 0)).sum(axis=2) / np.maximum(closing_valid.sum(axis=2), 1), nan)
        max_ios = np.where(mode_valid, np.max(np.where(step_valid, ios, -inf), axis=2), nan)
        overlap_steps = step_valid & (ios > 0)
        overlap_defined = mode_valid & overlap_steps.any(axis=2)
        first_overlap = np.where(overlap_defined, hz[np.argmax(overlap_steps, axis=2)], nan)
        score_t = np.clip(1.0 - sep_over_h / kappa, 0.0, 1.0)
        score = np.where(mode_valid, np.max(np.where(step_valid, score_t, -inf), axis=2), nan)
        sens = np.stack([(min_soh <= th) & mode_valid for th in sens_thresholds], axis=2) if S else np.zeros((P, K, 0), bool)
        # line-of-sight convergence
        has2 = n_valid >= 2
        first_i = np.argmax(step_valid, axis=2)
        last_i = F - 1 - np.argmax(step_valid[:, :, ::-1], axis=2)
        gi = lambda arr, idx: np.take_along_axis(arr, idx[..., None], axis=2)[..., 0]                    # noqa: E731
        d_first = np.stack([gi(rel_u, first_i), gi(rel_v, first_i)], axis=-1)
        d_last = np.stack([gi(rel_u, last_i), gi(rel_v, last_i)], axis=-1)
        span = hz[last_i] - hz[first_i]
        v_rel = (d_last - d_first) / span[..., None]
        d0 = (ped_anchor - oth_anchor)[:, None, :]
        n0, nv = np.linalg.norm(d0, axis=-1), np.linalg.norm(v_rel, axis=-1)
        los_valid = has2 & scale_ok[:, None] & (n0 > eps) & (nv > eps)
        los = np.where(los_valid, -(d0 * v_rel).sum(-1) / (n0 * nv), nan)
        los = np.where(los_valid, np.clip(los, -1.0, 1.0), nan)
        # pair aggregates
        w = ped_probs * mode_valid
        mass = w.sum(axis=1)
        pair_valid = mass > 0
        wn = np.where(pair_valid[:, None], w / np.where(mass > 0, mass, 1.0)[:, None], 0.0)
        agg = lambda x: np.where(pair_valid, np.sum(wn * np.nan_to_num(x), axis=1), nan)                  # noqa: E731
        mw_score = agg(score)
        mw_overlap = agg((np.nan_to_num(max_ios) > 0).astype(float))
        mw_sens = np.stack([np.where(pair_valid, np.sum(wn * sens[:, :, s], axis=1), nan) for s in range(S)], axis=1) if S else np.zeros((P, 0))
        mp = np.argmax(ped_probs, axis=1)
        rows = np.arange(P)
        mp_valid = mode_valid[rows, mp]
        mp_soh, mp_score = np.where(mp_valid, min_soh[rows, mp], nan), np.where(mp_valid, score[rows, mp], nan)
        wc_soh = np.where(pair_valid, np.min(np.where(mode_valid, min_soh, inf), axis=1), nan)
        wc_ios = np.where(pair_valid, np.max(np.where(mode_valid, max_ios, -inf), axis=1), nan)
        # observed context
        obs_sep = sep0
        obs_soh = sep0 / ped_h
        if ped_prev is not None:
            pv = np.asarray(prev_valid, bool) & scale_ok & np.isfinite(np.asarray(ped_prev, float)).all(1) & np.isfinite(np.asarray(oth_prev, float)).all(1)
            _check(prev_dt is not None and prev_dt > 0, "CONFIG", "prev_dt must be positive when previous positions are supplied")
            sep_prev = np.hypot(*(np.asarray(ped_prev, float) - np.asarray(oth_prev, float)).T)
            obs_closing = np.where(pv, (sep_prev - sep0) / prev_dt, nan)
        else:
            pv = np.zeros(P, bool)
            obs_closing = np.full(P, nan)
        log_ratio = np.where(scale_ok, np.log(oth_wh[:, 1] / ped_h), nan)
    features = {"rel_u": rel_u, "rel_v": rel_v, "sep_px": sep, "sep_over_h": sep_over_h, "closing_rate_px_s": closing, "box_overlap_ios": ios,
                "n_valid_steps": n_valid.astype(np.int16), "min_sep_px": min_sep, "min_sep_over_h": min_soh, "t_min_sep_s": t_min, "max_closing_rate_px_s": max_closing,
                "approach_fraction": approach, "max_box_overlap_ios": max_ios, "first_box_overlap_time_s": first_overlap, "heuristic_proximity_score": score,
                "sens_flags": sens, "los_convergence_cos": los, "valid_mode_mass": mass, "mode_weighted_score": mw_score, "mode_weighted_box_overlap_mass": mw_overlap,
                "mode_weighted_sens_mass": mw_sens, "most_probable_mode": mp.astype(np.int16), "mp_min_sep_over_h": mp_soh, "mp_score": mp_score,
                "worst_case_min_sep_over_h": wc_soh, "worst_case_max_box_overlap_ios": wc_ios, "observed_sep_anchor_px": obs_sep, "observed_sep_anchor_over_h": obs_soh,
                "observed_closing_rate_px_s": obs_closing, "log_scale_ratio": log_ratio}
    masks = {"step_valid": step_valid, "closing_valid": closing_valid, "mode_valid": mode_valid, "closing_summary_valid": closing_summary_valid,
             "overlap_defined": overlap_defined, "los_valid": los_valid, "pair_valid": pair_valid, "mp_valid": mp_valid, "scale_ok": scale_ok,
             "observed_closing_valid": pv, "all": np.ones(P, bool)}
    return {"features": features, "masks": masks}


__all__ = ["compute_pair_features", "box_overlap_ios", "FEATURE_SPECS"]
