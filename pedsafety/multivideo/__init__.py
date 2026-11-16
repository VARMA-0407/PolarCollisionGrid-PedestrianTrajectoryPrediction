"""Multi-video (full PIE) orchestration layer for the Google Colab notebook.

This package ADDS orchestration around the existing Phase 3-9 public APIs so that every downloaded PIE video can be processed with
video-level resume, a frozen recording-level split, multi-video Phase 5 training on the train partition only, and per-video
Phase 6-8 outputs. No earlier-phase module is modified and no scientific definition is changed:

* coordinates stay image-space pixels (``captured_1920x1080``); no metric, TTC/PET or calibrated probability is produced;
* forecasting targets are the existing perceived-track pseudo-targets (``target_source_mode = PILOT_SMOKE``); the
  ``ANNOTATED_TARGET`` mode is still refused by the window generator, so no annotation-ground-truth trajectory metric exists;
* the Phase 9 independent-evaluation gate is called unchanged and stays BLOCKED while annotations / identity mapping /
  timestamp alignment are unverified; held-out-recording metrics are therefore labelled pseudo-target diagnostics.

Modules: ``common`` (hashing, atomic JSON, stage markers, publish-with-verification), ``experiment`` (experiment identity,
configuration lock, stage tracker), ``inventory`` (PIE video / annotation inventory), ``split`` (official split discovery and
frozen recording-level split manifest), ``perception_batch`` (Phase 3 per video), ``preprocess`` (Phase 4 Task 2/3 and P5.3 per
video), ``training`` (multi-video Phase 5 training), ``evaluation`` (per-video / per-partition Phase 5 evaluation),
``downstream`` (Phase 6 / 7 / 8 per video), ``report`` (experiment report and final validation).
"""

MULTIVIDEO_LABEL = "MULTI_VIDEO_RECORDING_SPLIT_PSEUDO_TARGET_DEVELOPMENT"
TARGET_SOURCE = "PERCEIVED_TRACK_PSEUDO_TARGETS (window target_source_mode PILOT_SMOKE; not PIE annotation ground truth)"
COORDINATE_SYSTEM = "captured_1920x1080"
COORDINATE_UNIT = "pixel"

__all__ = ["MULTIVIDEO_LABEL", "TARGET_SOURCE", "COORDINATE_SYSTEM", "COORDINATE_UNIT"]
