"""Model-agnostic trajectory evaluator (pixels, image space). Reusable for later neural / multimodal models.

Inputs (per call, N windows, F horizon steps, K hypotheses):
    gt_uv [N, F, 2], gt_valid [N, F]                         ground-truth (here: PILOT_SMOKE pseudo-targets)
    pred_uv [N, F, 2] or [N, K, F, 2], pred_valid [N, F] or [N, K, F]
    window_id [N], track_id [N], coordinate_system, unit, model_id, horizon_offsets_s [F]
    mode_probabilities [N, K] (optional; required to define a single-trajectory metric when K > 1)

Policies (configurable, recorded):
    partial_target_policy  STRICT_FULL_HORIZON: a window is evaluated only if all F target steps are valid; otherwise it is
                           excluded and counted (``target_incomplete``). Missing targets are never zero-filled.
    invalid_prediction     EXCLUDE_AND_COUNT: a hypothesis is usable only if all F steps are valid; a window whose primary
                           hypothesis is unusable is excluded and counted (``prediction_invalid``). Never zero-filled.
Metrics per window (Euclidean distance in pixels, e_t = ||p_hat_t - p_t||_2):
    ADE = mean_t e_t over the F steps;  FDE = e_F.
    Primary (single-trajectory) hypothesis: K = 1 -> the only one; K > 1 -> argmax mode probability (never the closest).
    minADE@K = min_k ADE_k, minFDE@K = min_k FDE_k over usable hypotheses: BEST-OF-K ORACLE (uses ground truth to select);
    for K = 1 they equal ADE/FDE.
    MissRate(delta) = fraction of evaluated windows with FDE > delta px, only if delta is configured; else NOT_CONFIGURED.
Aggregation: WINDOW-WEIGHTED (each evaluated window weight 1) and, separately labelled, TRACK-WEIGHTED (mean of per-track
means). Windows overlap heavily and are not independent samples; no confidence interval or significance test is reported.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..errors import WindowContractError

AUTHORIZED_COORDINATE_SYSTEM = "captured_1920x1080"
AUTHORIZED_UNIT = "pixel"
PARTIAL_TARGET_POLICIES = ("STRICT_FULL_HORIZON",)
INVALID_PREDICTION_POLICIES = ("EXCLUDE_AND_COUNT",)
WINDOW_INVALID_REASONS = ("target_incomplete", "prediction_invalid")


@dataclass(frozen=True)
class EvalConfig:
    partial_target_policy: str = "STRICT_FULL_HORIZON"
    invalid_prediction_policy: str = "EXCLUDE_AND_COUNT"
    miss_rate_threshold_px: Optional[float] = None
    std_ddof: int = 0
    report_horizons_s: Sequence[float] = (0.5, 1.0, 1.5)

    def __post_init__(self):
        if self.partial_target_policy not in PARTIAL_TARGET_POLICIES:
            raise WindowContractError("EVAL_CONFIG", "unsupported partial_target_policy %r" % self.partial_target_policy)
        if self.invalid_prediction_policy not in INVALID_PREDICTION_POLICIES:
            raise WindowContractError("EVAL_CONFIG", "unsupported invalid_prediction_policy %r" % self.invalid_prediction_policy)
        t = self.miss_rate_threshold_px
        if t is not None and not (isinstance(t, (int, float)) and not isinstance(t, bool) and np.isfinite(t) and t > 0):
            raise WindowContractError("EVAL_CONFIG", "miss_rate_threshold_px must be null or a positive number of pixels")
        if self.std_ddof not in (0, 1):
            raise WindowContractError("EVAL_CONFIG", "std_ddof must be 0 or 1")


@dataclass
class EvaluationResult:
    model_id: str
    window_id: np.ndarray        # int64 [N]
    track_id: np.ndarray         # str [N]
    evaluated: np.ndarray        # bool [N] primary single-trajectory metric defined
    invalid_reason: np.ndarray   # int8 [N] index into WINDOW_INVALID_REASONS, -1 if evaluated
    ade: np.ndarray              # float64 [N] NaN if not evaluated
    fde: np.ndarray              # float64 [N]
    step_error: np.ndarray       # float64 [N, F] primary hypothesis, NaN if not evaluated
    min_ade: np.ndarray          # float64 [N] oracle, NaN if undefined
    min_fde: np.ndarray          # float64 [N]
    oracle_evaluated: np.ndarray # bool [N]
    k: int
    summary: Dict[str, Any] = field(default_factory=dict)


def _stats(x: np.ndarray, ddof: int) -> Dict[str, Any]:
    n = int(x.shape[0])
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None, "p90": None}
    return {"n": n, "mean": float(np.mean(x)), "median": float(np.median(x)),
            "std": float(np.std(x, ddof=ddof)) if n > ddof else None, "min": float(np.min(x)), "max": float(np.max(x)),
            "p90": float(np.percentile(x, 90))}


def _check_finite_where_valid(a: np.ndarray, valid: np.ndarray, what: str):
    if not np.all(np.isfinite(a[valid])):
        raise WindowContractError("NONFINITE_VALID_VALUE", "%s: a value flagged valid is NaN or infinite" % what)


def evaluate_predictions(gt_uv, gt_valid, pred_uv, pred_valid, window_id, track_id, *, coordinate_system: str, unit: str,
                         model_id: str, horizon_offsets_s, cfg: EvalConfig, mode_probabilities=None) -> EvaluationResult:
    if coordinate_system != AUTHORIZED_COORDINATE_SYSTEM or unit != AUTHORIZED_UNIT:
        raise WindowContractError("COORDINATE_SYSTEM_MISMATCH", "evaluation requires %s / %s, got %s / %s"
                                  % (AUTHORIZED_COORDINATE_SYSTEM, AUTHORIZED_UNIT, coordinate_system, unit))
    gt = np.asarray(gt_uv, np.float64)
    gv = np.asarray(gt_valid, bool)
    p = np.asarray(pred_uv, np.float64)
    pv = np.asarray(pred_valid, bool)
    wid = np.asarray(window_id, np.int64)
    tid = np.asarray(track_id, dtype=str)
    hz = np.asarray(horizon_offsets_s, np.float64)
    if gt.ndim != 3 or gt.shape[2] != 2:
        raise WindowContractError("EVAL_SHAPE", "gt_uv must be [N, F, 2], got %s" % (gt.shape,))
    n, f = gt.shape[:2]
    if p.ndim == 3:
        p, pv = p[:, None], pv[:, None] if pv.ndim == 2 else pv
    if p.ndim != 4 or p.shape[0] != n or p.shape[2:] != (f, 2):
        raise WindowContractError("EVAL_SHAPE", "pred_uv must be [N, F, 2] or [N, K, F, 2] with N=%d, F=%d; got %s" % (n, f, (p.shape,)))
    k = p.shape[1]
    if pv.shape != (n, k, f) or gv.shape != (n, f) or wid.shape != (n,) or tid.shape != (n,) or hz.shape != (f,):
        raise WindowContractError("EVAL_SHAPE", "mask / identifier / horizon shapes inconsistent with N=%d, K=%d, F=%d" % (n, k, f))
    if np.unique(wid).shape[0] != n:
        raise WindowContractError("DUPLICATE_WINDOW_ID", "window ids must be unique")
    _check_finite_where_valid(gt, np.repeat(gv[:, :, None], 2, axis=2), "ground truth")
    _check_finite_where_valid(p, np.repeat(pv[..., None], 2, axis=3), "prediction")
    probs = None
    if mode_probabilities is not None:
        probs = np.asarray(mode_probabilities, np.float64)
        if probs.shape != (n, k) or not np.all(np.isfinite(probs)) or np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise WindowContractError("EVAL_MODE_PROBABILITIES", "mode probabilities must be finite, non-negative [N, K] rows summing to 1")
    if k > 1 and probs is None:
        primary = None
    else:
        primary = np.zeros(n, np.int64) if k == 1 else np.argmax(probs, axis=1)

    target_ok = gv.all(axis=1)
    mode_ok = pv.all(axis=2)                                        # [N, K]
    diff = np.where((gv[:, None, :, None] & pv[..., None]), p - gt[:, None], np.nan)
    err = np.sqrt(np.sum(diff * diff, axis=3))                      # [N, K, F], NaN where either side invalid
    usable = target_ok[:, None] & mode_ok
    # STRICT_FULL_HORIZON: a usable (window, hypothesis) has all F target and prediction steps valid, so the mean is over F
    ade_k = np.where(usable, np.where(usable[..., None], err, 0.0).sum(axis=2) / f, np.nan)
    fde_k = np.where(usable, err[:, :, -1], np.nan)

    reason = np.full(n, -1, np.int8)
    if primary is None:
        evaluated = np.zeros(n, bool)
        ade = fde = np.full(n, np.nan)
        step = np.full((n, f), np.nan)
        reason[:] = 1
    else:
        rows = np.arange(n)
        prim_ok = usable[rows, primary]
        evaluated = prim_ok
        ade = np.where(prim_ok, ade_k[rows, primary], np.nan)
        fde = np.where(prim_ok, fde_k[rows, primary], np.nan)
        step = np.where(prim_ok[:, None], err[rows, primary], np.nan)
        reason[~mode_ok[rows, primary]] = 1
    reason[~target_ok] = 0
    if primary is not None:
        reason[evaluated] = -1
    oracle_ok = usable.any(axis=1)
    min_ade = np.where(oracle_ok, np.min(np.where(usable, ade_k, np.inf), axis=1), np.nan)
    min_fde = np.where(oracle_ok, np.min(np.where(usable, fde_k, np.inf), axis=1), np.nan)
    res = EvaluationResult(model_id, wid, tid, evaluated, reason, ade, fde, step, min_ade, min_fde, oracle_ok, k)
    res.summary = summarize(res, hz, cfg)
    return res


def summarize(res: EvaluationResult, horizon_offsets_s: np.ndarray, cfg: EvalConfig, subset: Optional[np.ndarray] = None) -> Dict[str, Any]:
    sel = np.ones(res.window_id.shape[0], bool) if subset is None else np.asarray(subset, bool)
    ev = res.evaluated & sel
    ddof = cfg.std_ddof
    tracks = res.track_id[ev]
    per_track: List[Dict[str, Any]] = []
    for t in sorted(set(tracks.tolist())):
        m = ev & (res.track_id == t)
        per_track.append({"track_id": t, "n_windows": int(m.sum()), "ade_mean": float(res.ade[m].mean()), "ade_median": float(np.median(res.ade[m])),
                          "fde_mean": float(res.fde[m].mean()), "fde_median": float(np.median(res.fde[m]))})
    ta = np.array([r["ade_mean"] for r in per_track])
    tf = np.array([r["fde_mean"] for r in per_track])
    reasons = {r: int(((res.invalid_reason == i) & sel).sum()) for i, r in enumerate(WINDOW_INVALID_REASONS)}
    hz = np.asarray(horizon_offsets_s, np.float64)
    per_step = [float(np.mean(res.step_error[ev, j])) if ev.any() else None for j in range(hz.shape[0])]
    at = {}
    for h in cfg.report_horizons_s:
        j = int(np.argmin(np.abs(hz - h)))
        if abs(hz[j] - h) < 1e-6:
            at["%.1fs" % h] = per_step[j]
    oe = res.oracle_evaluated & sel
    mr: Dict[str, Any]
    if cfg.miss_rate_threshold_px is None:
        mr = {"status": "NOT_CONFIGURED", "threshold_px": None,
              "note": "no defensible pixel threshold has been approved; not computed"}
    else:
        mr = {"status": "CONFIGURED", "threshold_px": float(cfg.miss_rate_threshold_px),
              "miss_rate": float(np.mean(res.fde[ev] > cfg.miss_rate_threshold_px)) if ev.any() else None,
              "min_fde_miss_rate_oracle": float(np.mean(res.min_fde[oe] > cfg.miss_rate_threshold_px)) if oe.any() else None}
    return {
        "model_id": res.model_id, "unit": "pixel", "coordinate_system": AUTHORIZED_COORDINATE_SYSTEM,
        "windows_considered": int(sel.sum()), "windows_evaluated": int(ev.sum()), "windows_not_evaluated": int((sel & ~res.evaluated).sum()),
        "not_evaluated_reasons": reasons, "unique_target_tracks_evaluated": len(per_track),
        "window_weighted": {"ade_px": _stats(res.ade[ev], ddof), "fde_px": _stats(res.fde[ev], ddof),
                            "note": "each evaluated window has weight 1; overlapping windows are NOT independent samples"},
        "track_weighted": {"n_tracks": len(per_track), "ade_px_mean_of_track_means": float(ta.mean()) if len(ta) else None,
                           "ade_px_median_of_track_means": float(np.median(ta)) if len(ta) else None,
                           "fde_px_mean_of_track_means": float(tf.mean()) if len(tf) else None,
                           "fde_px_median_of_track_means": float(np.median(tf)) if len(tf) else None,
                           "note": "each track has weight 1 (mean over its windows first)"},
        "per_horizon_mean_displacement_error_px": {"offsets_s": [round(float(x), 9) for x in hz], "window_weighted_mean": per_step,
                                                   "at_report_horizons": at},
        "oracle_best_of_k": {"k": res.k, "label": "BEST-OF-K ORACLE (ground truth selects the hypothesis); not a single-trajectory metric",
                             "min_ade_px": _stats(res.min_ade[oe], ddof), "min_fde_px": _stats(res.min_fde[oe], ddof),
                             "identical_to_ade_fde": res.k == 1},
        "miss_rate": mr, "per_track": per_track,
        "std_ddof": ddof,
    }


def common_window_mask(results: Sequence[EvaluationResult]) -> np.ndarray:
    """Windows evaluated by every model (identical window_id order required)."""
    base = results[0].window_id
    for r in results[1:]:
        if not np.array_equal(r.window_id, base):
            raise WindowContractError("WINDOW_ID_MISMATCH", "results must share the same window_id order for a common-window comparison")
    m = np.ones(base.shape[0], bool)
    for r in results:
        m &= r.evaluated
    return m


__all__ = ["EvalConfig", "EvaluationResult", "evaluate_predictions", "summarize", "common_window_mask", "WINDOW_INVALID_REASONS",
           "AUTHORIZED_COORDINATE_SYSTEM", "AUTHORIZED_UNIT"]
