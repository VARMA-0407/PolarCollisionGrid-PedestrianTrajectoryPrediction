"""P5.4 classical baselines (B0 zero velocity, B1 last valid causal velocity, B2 history linear regression).

Predictors receive only :class:`HistoryInput` (history fields; no future arrays) and return a :class:`Prediction`
with positions ``[N, F, 2]`` in pixels of ``captured_1920x1080``. Not learned models; K = 1.
"""

from .base import HistoryInput, Prediction, Predictor
from .history_input import build_history_input
from .predictors import REGISTRY, HistoryLinearRegression, LastValidCausalVelocity, ZeroVelocity, make_predictor

__all__ = ["HistoryInput", "Prediction", "Predictor", "build_history_input", "REGISTRY", "make_predictor", "ZeroVelocity",
           "LastValidCausalVelocity", "HistoryLinearRegression"]
