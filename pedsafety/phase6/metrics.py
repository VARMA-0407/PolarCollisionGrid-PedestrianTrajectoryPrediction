"""Evaluation: SUPERVISED metrics (verified labels only) and, separately, candidate-indicator AGREEMENT diagnostics (no labels).

``evaluate_supervised`` computes precision, recall, F1, PR-AUC (average precision), ROC-AUC, false-alarm rate and event-level lead time. It REFUSES
label sets that are not verified (``validate_label_set`` with ``require_supervision``): candidate labels derived from perceived tracks can never
produce these metrics. The decision threshold must be supplied by the caller (declared before looking at results); there is no default.

``candidate_indicator_agreement`` compares a heuristic indicator evaluated on PREDICTED trajectories with the same indicator evaluated on the REALISED
perceived future. It reports counts and an agreement rate only: it is a forecast-consistency diagnostic of a heuristic, not conflict-prediction
performance, and deliberately offers no precision / recall / F1 / AUC naming.
"""

from typing import Any, Dict, Mapping, Optional

import numpy as np

from .errors import ConflictContractError, ConflictLabelError
from .labels import ConflictLabelSet, validate_label_set
from .scoring import ConflictScores


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """PR-AUC as average precision: sum_k (R_k - R_{k-1}) P_k over scores sorted descending (ties handled by grouping equal scores)."""
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tp, fp = np.cumsum(y), np.cumsum(~y)
    last = np.r_[s[1:] != s[:-1], True]                                   # end of each tie group
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / max(int(y.sum()), 1)
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties replaced by their average rank."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic with average ranks for ties."""
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ConflictContractError("LABEL_CLASS_ABSENT", "ROC-AUC needs both classes")
    ranks = _average_ranks(np.asarray(s, float))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_supervised(scores: ConflictScores, labels: ConflictLabelSet, *, threshold: float, fps: Optional[float] = None, pair_keys=None) -> Dict[str, Any]:
    if threshold is None or not np.isfinite(threshold):
        raise ConflictContractError("THRESHOLD", "a decision threshold declared in advance is required (no default exists)")
    report = validate_label_set(labels, require_supervision=True, pair_keys=pair_keys)          # raises LABELS_NOT_VERIFIED for candidates
    if scores.window_id is None:
        raise ConflictContractError("IDENTITY", "scores must carry pair identities")
    index = {k: i for i, k in enumerate(zip(scores.window_id.tolist(), scores.ped_track_index.tolist(), scores.other_track_index.tolist()))}
    rows = np.array([index.get(k, -1) for k in labels.pair_keys()])
    if (rows < 0).any():
        raise ConflictLabelError("LABEL_IDENTITY_MISMATCH", "label pairs missing from the scores")
    use = labels.valid & scores.valid[rows]
    y, s = labels.positive[use], scores.scores[rows][use]
    if not (y.any() and (~y).any()):
        raise ConflictLabelError("LABEL_CLASS_ABSENT", "both classes are required among the evaluated pairs")
    pred = s >= threshold
    tp, fp, fn, tn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and (prec + rec) > 0 else float("nan")
    out = {"kind": scores.kind, "score_calibrated": False, "label_kind": labels.label_kind, "n_pairs": int(use.sum()), "threshold": float(threshold),
           "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "precision": prec, "recall": rec, "f1": f1, "false_alarm_rate": fp / (fp + tn) if fp + tn else float("nan"),
           "pr_auc_average_precision": average_precision(y, s), "roc_auc": roc_auc(y, s), "prevalence": float(y.mean()), "label_report": report}
    if labels.event_frame is not None and fps:
        det = use.copy()
        det[use] = pred & y
        lead = (labels.event_frame[det] - labels.anchor_frame[det]) / float(fps)
        out["event_level"] = {"positives": int(y.sum()), "detected": int(det.sum()), "detection_rate": float(det.sum() / y.sum()),
                              "lead_time_s": {"mean": float(lead.mean()) if lead.size else None, "median": float(np.median(lead)) if lead.size else None,
                                              "definition": "event frame minus the anchor at which the warning is evaluated, in seconds of video time"}}
    return out


def candidate_indicator_agreement(predicted_flag: np.ndarray, realised_flag: np.ndarray, valid: np.ndarray) -> Dict[str, Any]:
    """Counts / agreement of a heuristic indicator on predicted vs realised (perceived) trajectories. NOT conflict-prediction performance."""
    v = np.asarray(valid, bool)
    p, r = np.asarray(predicted_flag, bool)[v], np.asarray(realised_flag, bool)[v]
    tp, fp, fn, tn = int((p & r).sum()), int((p & ~r).sum()), int((~p & r).sum()), int((~p & ~r).sum())
    n = int(v.sum())
    return {"n_pairs": n, "both_flagged": tp, "predicted_only": fp, "realised_only": fn, "neither": tn, "agreement_rate": (tp + tn) / n if n else None,
            "predicted_flag_rate": float(p.mean()) if n else None, "realised_flag_rate": float(r.mean()) if n else None,
            "interpretation": "forecast consistency of a heuristic image-space indicator against the realised PERCEIVED future; not verified conflict ground truth"}


def spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return None
    ra, rb = _average_ranks(a), _average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


__all__ = ["evaluate_supervised", "candidate_indicator_agreement", "average_precision", "roc_auc", "spearman"]
