"""Probability calibration of Phase 6 conflict scores (numpy only), gated on verified labels and disjoint partitions.

A calibrator maps a raw score in [0,1] to P(verified conflict label = 1 | score) for ONE (score kind, source checkpoint, Phase 6 config, label source)
combination. It may be fitted only when:
  * the label set passes ``phase6.labels.validate_label_set`` (verified kind, documented independent source, label times after the anchor, >= 3 groups);
  * train / calibration / test group sets are pairwise disjoint and each non-empty (calibration never sees the test groups, nor the groups the
    scorer was trained on); rows of one group (video / scene) are never split across partitions;
  * the calibration rows have at least ``MIN_POS`` positives and ``MIN_NEG`` negatives.
No class re-weighting is applied (it would change the base rate the probability refers to). Methods: Platt scaling (2 parameters, tiny L2) by default;
isotonic regression (PAV) only with at least ``ISO_MIN_N`` rows and ``ISO_MIN_POS`` positives. Persisted artifacts carry a content hash that is verified on load
and a provenance block that is verified on apply.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ..phase6.errors import ConflictLabelError
from ..phase6.labels import ConflictLabelSet, validate_label_set
from .errors import CalibrationBlocked, ProvenanceMismatch, RiskContractError

SCHEMA = "phase7-calibration-artifact-1.0.0"
METHODS = ("platt", "isotonic")
MIN_POS, MIN_NEG = 20, 20
ISO_MIN_N, ISO_MIN_POS = 1000, 100
EPS = 1e-6


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, EPS, 1 - EPS)
    return np.log(s / (1 - s))


def fit_platt(s: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 100) -> Dict[str, float]:
    x, t = _logit(s), y.astype(float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * x + b)))
        g = np.array([((p - t) * x).sum() + l2 * (a - 1.0), (p - t).sum()])
        w = p * (1 - p) + 1e-12
        H = np.array([[(w * x * x).sum() + l2, (w * x).sum()], [(w * x).sum(), w.sum()]])
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return {"a": float(a), "b": float(b)}


def fit_isotonic(s: np.ndarray, y: np.ndarray) -> Dict[str, list]:
    """Pool-adjacent-violators on score-sorted data; returns step knots (upper score edge, value)."""
    o = np.argsort(s, kind="mergesort")
    xs, ys = s[o], y[o].astype(float)
    blocks = []                                     # [sum, count, max_score]
    for x, v in zip(xs, ys):
        blocks.append([v, 1, x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1] = [blocks[-1][0] + b[0], blocks[-1][1] + b[1], b[2]]
    return {"x": [float(b[2]) for b in blocks], "y": [float(b[0] / b[1]) for b in blocks]}


def apply_method(method: str, params: Mapping[str, Any], s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float)
    if method == "platt":
        p = 1 / (1 + np.exp(-(params["a"] * _logit(s) + params["b"])))
    elif method == "isotonic":
        xk, yk = np.asarray(params["x"]), np.asarray(params["y"])
        p = yk[np.minimum(np.searchsorted(xk, s, side="left"), len(yk) - 1)]
    else:
        raise RiskContractError("METHOD", "unknown calibration method %r" % method)
    return np.clip(p, EPS, 1 - EPS)


# ---- metrics (all require labels; computed only on a held-out partition by the callers) ----
def brier(p, y) -> float:
    return float(np.mean((p - y.astype(float)) ** 2))


def nll(p, y) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    y = y.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p, y, n_bins: int = 10) -> Dict[str, Any]:
    """Equal-width bins on [0,1]. ECE = sum_b (n_b/n) |mean(y_b) - mean(p_b)|."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n:
            conf, acc = float(p[m].mean()), float(y[m].mean())
            ece += n / len(p) * abs(acc - conf)
            bins.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n, "mean_confidence": conf, "observed_frequency": acc})
    return {"ece": float(ece), "n_bins": n_bins, "bins": bins, "n": int(len(p))}


def group_bootstrap_ci(p, y, groups, fn, n_boot: int = 500, seed: int = 0, min_groups: int = 5) -> Optional[Dict[str, Any]]:
    """Cluster (group-level) bootstrap 95% interval; rows of a group are resampled together because they are correlated. None if too few groups."""
    gs = sorted(set(map(str, groups)))
    if len(gs) < min_groups:
        return {"available": False, "reason": "%d group(s) < %d: a cluster bootstrap would be meaningless" % (len(gs), min_groups)}
    rng = np.random.default_rng(seed)
    gmap = {g: np.flatnonzero(np.asarray(groups).astype(str) == g) for g in gs}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(gs), len(gs))
        rows = np.concatenate([gmap[gs[i]] for i in pick])
        if y[rows].any() and (~y[rows]).any():
            vals.append(fn(p[rows], y[rows]))
    if len(vals) < 20:
        return {"available": False, "reason": "too few resamples with both classes"}
    return {"available": True, "lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975)), "n_groups": len(gs), "n_boot": len(vals)}


def calibration_report(p, y, groups) -> Dict[str, Any]:
    r = reliability(p, y)
    return {"n": int(len(p)), "n_positive": int(y.sum()), "prevalence": float(y.mean()), "brier": brier(p, y), "nll": nll(p, y), "ece": r["ece"],
            "reliability": r["bins"], "n_groups": len(set(map(str, groups))),
            "ci_brier": group_bootstrap_ci(p, y, groups, brier), "ci_ece": group_bootstrap_ci(p, y, groups, lambda a, b: reliability(a, b)["ece"])}


# ---- artifact ----
class CalibrationArtifact:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload

    @property
    def sha256(self) -> str:
        return self.payload["content_sha256"]

    @staticmethod
    def build(body: Dict[str, Any]) -> "CalibrationArtifact":
        return CalibrationArtifact({**body, "content_sha256": canonical_sha(body)})

    def verify(self) -> None:
        body = {k: v for k, v in self.payload.items() if k != "content_sha256"}
        if canonical_sha(body) != self.payload.get("content_sha256"):
            raise ProvenanceMismatch("ARTIFACT_HASH", "calibration artifact content hash does not match its content")
        if body.get("schema") != SCHEMA:
            raise ProvenanceMismatch("ARTIFACT_SCHEMA", "unknown calibration artifact schema %r" % body.get("schema"))

    def save(self, path) -> str:
        Path(path).write_text(json.dumps(self.payload, indent=1, sort_keys=True), encoding="utf-8")
        return self.sha256

    @classmethod
    def load(cls, path, expected_sha256: Optional[str] = None) -> "CalibrationArtifact":
        art = cls(json.loads(Path(path).read_text(encoding="utf-8")))
        art.verify()
        if expected_sha256 is not None and art.sha256 != expected_sha256:
            raise ProvenanceMismatch("ARTIFACT_HASH", "artifact hash %s differs from the expected %s" % (art.sha256, expected_sha256))
        return art

    def check_provenance(self, *, score_kind: str, checkpoint_file_sha256: str, phase6_config_sha256: str, coordinate_system: str) -> None:
        p = self.payload
        for key, got in (("score_kind", score_kind), ("source_checkpoint_file_sha256", checkpoint_file_sha256), ("phase6_config_sha256", phase6_config_sha256),
                         ("coordinate_system", coordinate_system)):
            if p[key] != got:
                raise ProvenanceMismatch("PROVENANCE_" + key.upper(), "calibration artifact %s=%s but the scores have %s" % (key, p[key], got))

    def apply(self, scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
        out = np.full(scores.shape, np.nan)
        v = np.asarray(valid, bool) & np.isfinite(scores)
        out[v] = apply_method(self.payload["method"], self.payload["params"], scores[v])
        return out


def fit_calibrator(scores: np.ndarray, valid: np.ndarray, labels: ConflictLabelSet, pair_keys, *, train_groups: Sequence[str], calibration_groups: Sequence[str],
                   test_groups: Sequence[str], score_kind: str, source: Mapping[str, Any], method: str = "platt") -> CalibrationArtifact:
    """Fit on the calibration groups only. ``source`` must hold checkpoint_file_sha256, model_state_sha256, phase6_config_sha256, coordinate_system, scorer_id."""
    if method not in METHODS:
        raise RiskContractError("METHOD", "method must be one of %s" % (METHODS,))
    for k in ("checkpoint_file_sha256", "model_state_sha256", "phase6_config_sha256", "coordinate_system", "scorer_id"):
        if not source.get(k):
            raise CalibrationBlocked("PROVENANCE_MISSING", "source provenance lacks %s" % k)
    try:
        report = validate_label_set(labels, require_supervision=True, pair_keys=pair_keys)
    except ConflictLabelError as exc:
        raise CalibrationBlocked("LABELS_NOT_VERIFIED", "calibration is blocked: %s" % exc)
    tr, ca, te = set(map(str, train_groups)), set(map(str, calibration_groups)), set(map(str, test_groups))
    if not (tr and ca and te):
        raise CalibrationBlocked("PARTITIONS", "train, calibration and test group sets must all be non-empty")
    if (tr & ca) or (tr & te) or (ca & te):
        raise CalibrationBlocked("PARTITION_OVERLAP", "train / calibration / test groups must be pairwise disjoint")
    index = {k: i for i, k in enumerate(pair_keys)}
    rows = np.array([index[k] for k in labels.pair_keys()])
    sel = labels.valid & valid[rows] & np.isin(labels.group_id.astype(str), list(ca)) & np.isfinite(scores[rows])
    y, s = labels.positive[sel], scores[rows][sel]
    if y.sum() < MIN_POS or (~y).sum() < MIN_NEG:
        raise CalibrationBlocked("CLASS_COUNTS", "calibration rows hold %d positives / %d negatives; at least %d / %d are required" % (y.sum(), (~y).sum(), MIN_POS, MIN_NEG))
    if method == "isotonic" and (len(y) < ISO_MIN_N or y.sum() < ISO_MIN_POS):
        raise CalibrationBlocked("ISOTONIC_DATA", "isotonic needs >= %d rows and >= %d positives (have %d / %d); use platt" % (ISO_MIN_N, ISO_MIN_POS, len(y), y.sum()))
    params = fit_platt(s, y) if method == "platt" else fit_isotonic(s, y)
    fit_p = apply_method(method, params, s)
    body = {"schema": SCHEMA, "method": method, "params": params, "score_kind": score_kind, "probability_semantics": "P(verified conflict label=1 | score) for this source only",
            "source_checkpoint_file_sha256": source["checkpoint_file_sha256"], "source_model_state_sha256": source["model_state_sha256"],
            "phase6_config_sha256": source["phase6_config_sha256"], "coordinate_system": source["coordinate_system"], "scorer_id": source["scorer_id"],
            "label_source": labels.source, "label_kind": labels.label_kind, "label_report": {k: v for k, v in report.items() if k != "groups"},
            "split": {"train_groups": sorted(tr), "calibration_groups": sorted(ca), "test_groups": sorted(te)}, "n_calibration_rows": int(len(y)),
            "n_calibration_positive": int(y.sum()), "calibration_fit_metrics_in_sample": {"brier": brier(fit_p, y), "nll": nll(fit_p, y)},
            "note": "in-sample fit metrics are NOT calibration evidence; evaluate on the test groups with evaluate_calibrated()"}
    return CalibrationArtifact.build(body)


def evaluate_calibrated(art: CalibrationArtifact, scores: np.ndarray, valid: np.ndarray, labels: ConflictLabelSet, pair_keys) -> Dict[str, Any]:
    """Evaluate on the artifact's TEST groups only (refuses rows of other groups)."""
    art.verify()
    validate_label_set(labels, require_supervision=True, pair_keys=pair_keys)
    index = {k: i for i, k in enumerate(pair_keys)}
    rows = np.array([index[k] for k in labels.pair_keys()])
    test = set(art.payload["split"]["test_groups"])
    sel = labels.valid & valid[rows] & np.isin(labels.group_id.astype(str), list(test)) & np.isfinite(scores[rows])
    if not sel.any() or not labels.positive[sel].any() or labels.positive[sel].all():
        raise CalibrationBlocked("TEST_CLASSES", "the test groups need both classes to evaluate calibration")
    p = art.apply(scores[rows], sel)[sel]
    return {"evaluated_on": "test_groups", "groups": sorted(test), "calibrated": calibration_report(p, labels.positive[sel], labels.group_id[sel])}


__all__ = ["CalibrationArtifact", "fit_calibrator", "evaluate_calibrated", "fit_platt", "fit_isotonic", "apply_method", "brier", "nll", "reliability",
           "calibration_report", "group_bootstrap_ci", "canonical_sha", "SCHEMA", "MIN_POS", "MIN_NEG"]
