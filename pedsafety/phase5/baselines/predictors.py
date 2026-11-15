"""Classical single-trajectory baselines (K = 1). All operate on ``HistoryInput`` only; image space, pixels.

Notation: anchor time t = last history timestamp; p(t) = last observed footpoint (u, v) px; tau_m = horizon offset (s).

B0 zero velocity                 p_hat(t + tau) = p(t)
B1 last valid causal velocity    p_hat(t + tau) = p(t) + v(t_k*) * tau,  k* = latest history index with velocity_valid
                                 (Task 2 causal one-frame backward difference; searched backward within the H samples only);
                                 no valid velocity -> window INVALID (no zero-velocity fallback)
B2 history linear regression     least squares x(t) = a_x + b_x t, y(t) = a_y + b_y t over valid history samples
                                 (actual timestamps, centred at their mean for conditioning);
                                 p_hat(t + tau) = p(t) + [b_x, b_y] * tau  (anchored at the last observation, not the intercept);
                                 invalid if the last sample is not observed, fewer than ``min_valid_samples`` valid samples,
                                 time span < ``min_time_span_s`` or centred time variance <= ``min_centered_time_variance_s2``
"""

from typing import Any, Dict

import numpy as np

from ..errors import WindowContractError
from .base import HistoryInput, Predictor

# Absolute tolerance for the B2 time-span rule: spans are differences of absolute timestamps (frame / fps), so an exact
# 0.2 s span can evaluate to 0.19999999999999 s; without the tolerance such windows would be rejected spuriously.
SPAN_TOLERANCE_S = 1e-9


def _anchor(hist: HistoryInput):
    p = hist.history_uv_px[:, -1, :]
    ok = hist.history_position_valid[:, -1] & np.all(np.isfinite(p), axis=1)
    return p, ok


class ZeroVelocity(Predictor):
    BASELINE_ID = "B0_zero_velocity"
    INVALID_REASONS = ("anchor_position_unavailable",)

    def config(self) -> Dict[str, Any]:
        return {"baseline_id": self.BASELINE_ID, "type": "zero_velocity", "params": {}, "uses": ["history position at anchor"], "learned": False}

    def _predict(self, hist: HistoryInput):
        p, ok = _anchor(hist)
        pos = np.repeat(p[:, None, :], hist.f, axis=1)
        reason = np.where(ok, -1, 0)
        return pos, ok, reason, {}


class LastValidCausalVelocity(Predictor):
    BASELINE_ID = "B1_last_valid_causal_velocity"
    INVALID_REASONS = ("anchor_position_unavailable", "no_valid_causal_velocity_in_history")

    def __init__(self, max_lookback_samples: int):
        if not (isinstance(max_lookback_samples, int) and max_lookback_samples >= 0):
            raise WindowContractError("BASELINE_CONFIG", "max_lookback_samples must be an integer >= 0")
        self.max_lookback_samples = max_lookback_samples

    def config(self) -> Dict[str, Any]:
        return {"baseline_id": self.BASELINE_ID, "type": "constant_velocity_last_valid_causal",
                "params": {"max_lookback_samples": self.max_lookback_samples, "velocity_source": "Task 2 causal backward difference (P5.3 graph bank)",
                           "fallback": "none: window marked invalid"},
                "uses": ["history position at anchor", "history causal velocity + validity"], "learned": False}

    def _predict(self, hist: HistoryInput):
        p, ok = _anchor(hist)
        H = hist.h
        vv = hist.history_velocity_valid & np.all(np.isfinite(hist.history_velocity_px_s), axis=2)
        src = np.full(hist.n, -1, np.int64)
        for lag in range(min(self.max_lookback_samples, H - 1) + 1):      # lag 0 = anchor sample, then backward
            k = H - 1 - lag
            take = (src < 0) & vv[:, k]
            src[take] = k
        has_v = src >= 0
        v = np.zeros((hist.n, 2))
        v[has_v] = hist.history_velocity_px_s[np.flatnonzero(has_v), src[has_v]]
        pos = p[:, None, :] + v[:, None, :] * hist.horizon_offsets_s[None, :, None]
        wvalid = ok & has_v
        reason = np.where(~ok, 0, np.where(~has_v, 1, -1))
        return pos, wvalid, reason, {"velocity_source_index": src.astype(np.int8),
                                     "velocity_px_s": np.where(has_v[:, None], v, np.nan)}


class HistoryLinearRegression(Predictor):
    BASELINE_ID = "B2_history_linear_regression"
    INVALID_REASONS = ("anchor_position_unavailable", "too_few_valid_history_samples", "history_time_span_too_short",
                       "ill_conditioned_fit")

    def __init__(self, min_valid_samples: int, min_time_span_s: float, min_centered_time_variance_s2: float, center_timestamps: bool = True):
        if not (isinstance(min_valid_samples, int) and min_valid_samples >= 2):
            raise WindowContractError("BASELINE_CONFIG", "min_valid_samples must be an integer >= 2")
        if not (min_time_span_s > 0 and min_centered_time_variance_s2 > 0):
            raise WindowContractError("BASELINE_CONFIG", "time thresholds must be positive")
        if center_timestamps is not True:
            raise WindowContractError("BASELINE_CONFIG", "center_timestamps must be true (the only implemented, documented variant)")
        self.min_valid_samples = min_valid_samples
        self.min_time_span_s = float(min_time_span_s)
        self.min_centered_time_variance_s2 = float(min_centered_time_variance_s2)

    def config(self) -> Dict[str, Any]:
        return {"baseline_id": self.BASELINE_ID, "type": "least_squares_constant_velocity_anchored_at_last_observation",
                "params": {"min_valid_samples": self.min_valid_samples, "min_time_span_s": self.min_time_span_s,
                           "min_centered_time_variance_s2": self.min_centered_time_variance_s2, "center_timestamps": True,
                           "anchor": "last observed history position"},
                "uses": ["valid history positions", "actual history timestamps"], "learned": False}

    def _predict(self, hist: HistoryInput):
        p, ok = _anchor(hist)
        m = hist.history_position_valid & np.all(np.isfinite(hist.history_uv_px), axis=2)
        t = hist.history_timestamps
        cnt = m.sum(axis=1)
        w = m.astype(np.float64)
        safe = np.maximum(cnt, 1)
        tbar = (t * w).sum(axis=1) / safe
        tc = np.where(m, t - tbar[:, None], 0.0)                              # centred times; masked samples contribute 0
        stt = (tc * tc).sum(axis=1)
        tmin = np.where(m, t, np.inf).min(axis=1)
        tmax = np.where(m, t, -np.inf).max(axis=1)
        span = np.where(cnt > 0, tmax - tmin, 0.0)
        uv = np.where(m[:, :, None], hist.history_uv_px, 0.0)
        ubar = uv.sum(axis=1) / safe[:, None]
        suv = (tc[:, :, None] * np.where(m[:, :, None], uv - ubar[:, None, :], 0.0)).sum(axis=1)
        var_c = stt / safe
        good_var = var_c > self.min_centered_time_variance_s2
        slope = np.where(good_var[:, None], suv / np.where(stt > 0, stt, 1.0)[:, None], np.nan)
        reason = np.full(hist.n, -1, np.int64)
        reason[~good_var] = 3
        reason[span < self.min_time_span_s - SPAN_TOLERANCE_S] = 2
        reason[cnt < self.min_valid_samples] = 1
        reason[~ok] = 0
        wvalid = reason < 0
        pos = p[:, None, :] + np.where(wvalid[:, None], slope, 0.0)[:, None, :] * hist.horizon_offsets_s[None, :, None]
        return pos, wvalid, reason, {"valid_history_samples": cnt.astype(np.int8), "fitted_velocity_px_s": np.where(wvalid[:, None], slope, np.nan),
                                     "fit_time_span_s": span}


REGISTRY = {"B0_zero_velocity": ZeroVelocity, "B1_last_valid_causal_velocity": LastValidCausalVelocity,
            "B2_history_linear_regression": HistoryLinearRegression}


def make_predictor(spec: Dict[str, Any]) -> Predictor:
    bid = spec.get("baseline_id")
    if bid not in REGISTRY:
        raise WindowContractError("UNKNOWN_BASELINE", "unknown baseline_id %r" % bid)
    params = spec.get("params", {})
    try:
        return REGISTRY[bid](**params)
    except TypeError as exc:
        raise WindowContractError("BASELINE_CONFIG", "%s: %s" % (bid, exc))


__all__ = ["ZeroVelocity", "LastValidCausalVelocity", "HistoryLinearRegression", "REGISTRY", "make_predictor"]
