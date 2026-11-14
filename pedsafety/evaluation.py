"""Non-oracle trajectory evaluation.

Prediction generation is intentionally outside this module.  Candidate
selection uses a model-provided score or deterministic first-candidate policy;
ground truth is never inspected during selection.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import ConfigError, PedsafetyError


class EvaluationError(PedsafetyError):
    pass


@dataclass(frozen=True)
class PredictionCandidate:
    values: np.ndarray
    model_score: Optional[float] = None


@dataclass(frozen=True)
class EvaluationCase:
    window_id: str
    prediction: Optional[np.ndarray]
    ground_truth: Optional[np.ndarray]
    ade: Optional[float]
    fde: Optional[float]
    valid_steps: int
    status: str
    reason: Optional[str] = None


@dataclass(frozen=True)
class EvaluationResult:
    selection_rule: str
    coordinate_space: str
    physical_units: bool
    cases: Tuple[EvaluationCase, ...]

    @property
    def valid_cases(self) -> Tuple[EvaluationCase, ...]:
        return tuple(c for c in self.cases if c.status == "ok")


def select_prediction(candidates: Sequence[PredictionCandidate], rule: str = "first") -> np.ndarray:
    if not candidates:
        raise EvaluationError("NO_PREDICTION", "at least one prediction candidate is required")
    if rule == "first":
        return np.asarray(candidates[0].values, dtype=float)
    if rule == "model_score":
        if any(c.model_score is None for c in candidates):
            raise EvaluationError("MISSING_MODEL_SCORE", "model_score selection requires a score for every candidate")
        return np.asarray(min(candidates, key=lambda c: (float(c.model_score),)).values, dtype=float)
    if rule.lower() in ("best_of_k", "oracle", "ground_truth"):
        raise ConfigError("ground-truth-dependent oracle selection is forbidden")
    raise ConfigError("selection rule must be 'first' or 'model_score'")


def evaluate_predictions(predictions: Mapping[str, Sequence[PredictionCandidate]], ground_truth: Mapping[str, np.ndarray],
                         *, validity_masks: Optional[Mapping[str, np.ndarray]] = None, selection_rule: str = "first",
                         coordinate_space: str = "unspecified", physical_units: bool = False) -> EvaluationResult:
    """Calculate ADE/FDE after non-oracle selection.

    ``physical_units=True`` is accepted only for an explicitly metric-labelled
    coordinate space.  Missing, malformed, or entirely invalid cases are
    retained in the result rather than silently dropped.
    """
    if physical_units and coordinate_space not in ("metric", "metres", "meters"):
        raise ConfigError("physical_units requires a validated metric coordinate space")
    masks = validity_masks or {}
    cases: List[EvaluationCase] = []
    for window_id in sorted(set(predictions) | set(ground_truth)):
        if window_id not in predictions:
            cases.append(EvaluationCase(window_id, None, ground_truth.get(window_id), None, None, 0, "invalid", "missing_prediction"))
            continue
        if window_id not in ground_truth:
            cases.append(EvaluationCase(window_id, None, None, None, None, 0, "invalid", "missing_ground_truth"))
            continue
        try:
            pred = select_prediction(predictions[window_id], selection_rule)
            truth = np.asarray(ground_truth[window_id], dtype=float)
            if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != 2:
                raise ValueError("prediction and ground truth must both have shape (steps, 2)")
            mask = np.ones(pred.shape[0], dtype=bool) if window_id not in masks else np.asarray(masks[window_id], dtype=bool)
            if mask.shape != (pred.shape[0],):
                raise ValueError("validity mask must have one entry per target step")
            mask &= np.isfinite(pred).all(axis=1) & np.isfinite(truth).all(axis=1)
            if not mask.any():
                raise ValueError("no valid target steps")
            distance = np.linalg.norm(pred[mask] - truth[mask], axis=1)
            ade = float(np.mean(distance))
            fde = float(distance[-1]) if mask[-1] else None
            cases.append(EvaluationCase(window_id, pred, truth, ade, fde, int(mask.sum()), "ok", None if fde is not None else "last_step_invalid"))
        except (EvaluationError, ValueError, TypeError) as exc:
            cases.append(EvaluationCase(window_id, None, np.asarray(ground_truth[window_id], dtype=float), None, None, 0,
                                        "invalid", str(exc)))
    return EvaluationResult(selection_rule, coordinate_space, physical_units, tuple(cases))

