"""Conflict score interface, the non-learned heuristic scorer and a gated, deterministic logistic head (numpy only).

* ``ConflictPredictor`` is the interface a downstream risk-calibration stage consumes: it returns ``ConflictScores`` that always declare their kind
  and that they are neither calibrated nor a probability of a physical event.
* ``HeuristicProximityScorer`` is NOT learned: it exposes the mode-weighted heuristic proximity score of ``features.py`` (an image-plane geometric
  heuristic in [0, 1]).
* ``LogisticConflictHead`` is the supervised predictor. Its public ``fit`` first validates the label set (``labels.validate_label_set``): candidate
  or unverified labels, fewer than three independent groups, label times not after the anchor, missing provenance or an identity mismatch raise
  ``ConflictTrainingBlocked``. No verified conflict label exists in the repository, so it has never been trained on real data; the fitting arithmetic is
  exercised only by synthetic unit tests through ``_fit_arrays``. Statistics (standardisation) are fitted on training groups only, class imbalance is
  handled by balanced class weights, invalid features are masked (value 0 plus a mask column), optimisation is plain full-batch gradient descent with
  fixed iterations (no randomness). Its output is ``UNCALIBRATED_CLASSIFIER_SCORE``.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .errors import ConflictContractError, ConflictLabelError, ConflictTrainingBlocked
from .labels import ConflictLabelSet, MIN_GROUPS, assert_trainable
from .pairs import ConflictPairBatch

KIND_HEURISTIC = "UNCALIBRATED_HEURISTIC_PROXIMITY_SCORE"
KIND_LEARNED = "UNCALIBRATED_CLASSIFIER_SCORE"
FEATURE_COLUMNS = ("mp_min_sep_over_h", "mp_score", "worst_case_min_sep_over_h", "worst_case_max_box_overlap_ios", "mode_weighted_score", "mode_weighted_box_overlap_mass",
                   "observed_sep_anchor_over_h", "observed_closing_rate_px_s", "log_scale_ratio")


@dataclass
class ConflictScores:
    scores: np.ndarray          # [P] float64 (NaN where invalid)
    valid: np.ndarray           # [P] bool
    kind: str
    calibrated: bool = False
    is_probability: bool = False
    window_id: Optional[np.ndarray] = None
    ped_track_index: Optional[np.ndarray] = None
    other_track_index: Optional[np.ndarray] = None
    notes: str = "image-space heuristic; not a physical TTC / PET / collision probability"

    def __post_init__(self):
        if self.calibrated or self.is_probability:
            raise ConflictContractError("SCORE_CLAIM", "scores of this module are never calibrated probabilities")
        if self.valid.any() and not np.isfinite(self.scores[self.valid]).all():
            raise ConflictContractError("NONFINITE_VALID_VALUE", "a valid score is not finite")


class ConflictPredictor:
    """Interface: ``predict(batch) -> ConflictScores`` over the pairs of a ``ConflictPairBatch``."""
    kind = "abstract"

    def predict(self, batch: ConflictPairBatch) -> ConflictScores:
        raise NotImplementedError


class HeuristicProximityScorer(ConflictPredictor):
    kind = KIND_HEURISTIC

    def predict(self, batch: ConflictPairBatch) -> ConflictScores:
        valid = batch.masks["pair_valid"].copy()
        s = np.where(valid, batch.features["mode_weighted_score"], np.nan)
        return ConflictScores(s, valid, self.kind, window_id=batch.window_id, ped_track_index=batch.ped_track_index, other_track_index=batch.other_track_index)


def group_split(group_ids: Sequence[str], fractions: Tuple[float, float, float] = (0.34, 0.33, 0.33), seed: int = 0, min_groups: int = MIN_GROUPS) -> Dict[str, Any]:
    """Whole-group train / validation / test assignment (deterministic hash order). Rows of one group never appear in two partitions."""
    groups = sorted(set(map(str, group_ids)))
    if len(groups) < min_groups:
        raise ConflictTrainingBlocked("GROUPS_INSUFFICIENT", "%d independent group(s); at least %d are required for a train / validation / test split" % (len(groups), min_groups))
    if abs(sum(fractions) - 1.0) > 1e-6 or min(fractions) <= 0:
        raise ConflictContractError("CONFIG", "fractions must be positive and sum to 1")
    order = sorted(groups, key=lambda g: hashlib.sha256(("%d|%s" % (seed, g)).encode()).hexdigest())
    n = len(order)
    n_tr, n_va = max(1, int(round(fractions[0] * n))), max(1, int(round(fractions[1] * n)))
    n_tr = min(n_tr, n - 2)
    n_va = min(n_va, n - n_tr - 1)
    part = {"train": order[:n_tr], "validation": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    return {"partitions": part, "assignment": {g: p for p, gs in part.items() for g in gs}, "seed": seed}


def design_matrix(batch: ConflictPairBatch, columns: Sequence[str] = FEATURE_COLUMNS) -> Tuple[np.ndarray, np.ndarray]:
    """[P,C] feature values (NaN where invalid) and [P,C] validity (finite)."""
    X = np.stack([np.asarray(batch.features[c], np.float64) for c in columns], axis=1)
    return X, np.isfinite(X)


class LogisticConflictHead(ConflictPredictor):
    kind = KIND_LEARNED

    def __init__(self, columns: Sequence[str] = FEATURE_COLUMNS, l2: float = 1e-2, lr: float = 0.1, iterations: int = 500):
        self.columns, self.l2, self.lr, self.iterations = tuple(columns), float(l2), float(lr), int(iterations)
        self.w: Optional[np.ndarray] = None
        self.b = 0.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.fit_report: Dict[str, Any] = {}

    # -- numerics (no labels, no gating) --
    def _transform(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        Z = np.where(M, (X - self.mean) / self.std, 0.0)
        return np.concatenate([Z, M.astype(np.float64)], axis=1)

    def _fit_arrays(self, X: np.ndarray, M: np.ndarray, y: np.ndarray, row_weight: Optional[np.ndarray] = None) -> None:
        """Fit on already-selected TRAINING rows. Test-only entry point for the arithmetic; the public path is ``fit``."""
        n = X.shape[0]
        if n == 0 or not (y.any() and (~y).any()):
            raise ConflictContractError("LABEL_CLASS_ABSENT", "both classes are required among the training rows")
        cnt = np.maximum(M.sum(axis=0), 1)
        self.mean = np.where(M.any(axis=0), np.where(M, X, 0.0).sum(axis=0) / cnt, 0.0)
        var = np.where(M, (X - self.mean) ** 2, 0.0).sum(axis=0) / cnt
        self.std = np.where(var > 0, np.sqrt(var), 1.0)
        Z = self._transform(X, M)
        yf = y.astype(np.float64)
        n_pos, n_neg = yf.sum(), n - yf.sum()
        cw = np.where(y, n / (2.0 * n_pos), n / (2.0 * n_neg))                    # balanced class weights
        rw = cw if row_weight is None else cw * row_weight
        w, b = np.zeros(Z.shape[1]), 0.0
        for _ in range(self.iterations):
            p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
            g = (p - yf) * rw / rw.sum()
            w -= self.lr * (Z.T @ g + self.l2 * w)
            b -= self.lr * g.sum()
        self.w, self.b = w, float(b)

    def _score(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ConflictTrainingBlocked("NOT_TRAINED", "the learned conflict predictor has not been trained: no verified conflict labels are available")
        return 1.0 / (1.0 + np.exp(-(self._transform(X, M) @ self.w + self.b)))

    # -- public, gated --
    def fit(self, batch: ConflictPairBatch, labels: ConflictLabelSet, train_groups: Sequence[str]) -> Dict[str, Any]:
        keys = list(zip(batch.window_id.tolist(), batch.ped_track_index.tolist(), batch.other_track_index.tolist()))
        anchors = {k: int(a) for k, a in zip(keys, batch.anchor_frame.tolist())}
        try:
            report = assert_trainable(labels, pair_keys=keys, anchor_frames=anchors)
        except ConflictLabelError as exc:
            raise ConflictTrainingBlocked(exc.code, "supervised conflict training is blocked: %s" % exc)
        index = {k: i for i, k in enumerate(keys)}
        rows = np.array([index[k] for k in labels.pair_keys()])
        train_groups = set(map(str, train_groups))
        all_groups = set(labels.group_id[labels.valid].tolist())
        if not train_groups or not train_groups < all_groups:
            raise ConflictTrainingBlocked("SPLIT", "train_groups must be a non-empty proper subset of the label groups so that held-out groups remain (got %s of %s)" % (sorted(train_groups), sorted(all_groups)))
        sel = labels.valid & np.isin(labels.group_id, list(train_groups))
        X, M = design_matrix(batch, self.columns)
        self._fit_arrays(X[rows[sel]], M[rows[sel]], labels.positive[sel])
        self.fit_report = {"label_report": report, "train_groups": sorted(train_groups), "n_train": int(sel.sum()), "uncalibrated": True}
        return self.fit_report

    def predict(self, batch: ConflictPairBatch) -> ConflictScores:
        X, M = design_matrix(batch, self.columns)
        valid = batch.masks["pair_valid"] & M.any(axis=1)
        s = np.where(valid, self._score(X, M), np.nan)
        return ConflictScores(s, valid, self.kind, window_id=batch.window_id, ped_track_index=batch.ped_track_index, other_track_index=batch.other_track_index,
                              notes="uncalibrated classifier output on verified labels; not a calibrated or physical probability")


def train_conflict_predictor(batch: ConflictPairBatch, labels: ConflictLabelSet, train_groups: Sequence[str], **kw) -> LogisticConflictHead:
    head = LogisticConflictHead(**kw)
    head.fit(batch, labels, train_groups)
    return head


__all__ = ["ConflictScores", "ConflictPredictor", "HeuristicProximityScorer", "LogisticConflictHead", "train_conflict_predictor", "group_split", "design_matrix",
           "KIND_HEURISTIC", "KIND_LEARNED", "FEATURE_COLUMNS"]
