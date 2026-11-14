"""pedsafety - dataset-independent agent-state foundation and data pipeline (Phase 2, Task 7).

Stages 1-6 provide schema validation, dataset adapters, agent-state construction,
calibration-aware coordinate handling, causal/offline trajectory features, leakage-aware
splitting and windows, a config-driven CLI (``python -m pedsafety``), figures and a read-only
HBS smoke test. Stage 6.1 hardens output handling and declares dependencies.
Nothing here depends on PIE, inD, TITAN or the original PCG code.
"""

# PEP 440 development version: Stage 6.1 of the Phase 2 Task 7 pipeline (not a release).
__version__ = "0.6.1.dev0"
