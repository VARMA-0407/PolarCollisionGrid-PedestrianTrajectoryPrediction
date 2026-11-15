"""P5.4 model-agnostic trajectory evaluation (ADE/FDE, per-horizon error, best-of-K oracle, optional miss rate)."""

from .metrics import (WINDOW_INVALID_REASONS, EvalConfig, EvaluationResult, common_window_mask, evaluate_predictions,
                      summarize)

__all__ = ["EvalConfig", "EvaluationResult", "evaluate_predictions", "summarize", "common_window_mask", "WINDOW_INVALID_REASONS"]
