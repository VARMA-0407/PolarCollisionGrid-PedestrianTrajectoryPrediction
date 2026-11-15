# Phase 9 — Independent evaluation protocol and gates

Current status of independent scientific evaluation: **`BLOCKED_INSUFFICIENT_INDEPENDENT_SCENES`** (one scene group, `set01/video_0004`; no annotation files; no frozen split; no verified events).
Nothing in this phase is an independent result. In-sample pilot numbers (Phase 5 ADE/FDE, Phase 7/8 category and alert counts) are labelled `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`.

## Gate (implemented in `pedsafety/phase9/evaluation.py`, tested)
1. `assess_independent_evaluation` — at least 3 independent scene groups processed, then a frozen split manifest that passes `validate_split_manifest`: `manifest_version`, `frozen` with timestamp,
   non-empty train / validation / test (calibration required for calibrated metrics), every group in exactly one partition, no video in two groups, per-group verified annotation provenance
   (name, type in the accepted independent source types, version, 64-hex sha256, protocol reference, verified), verified identity mapping and timestamp alignment, licence and acquisition record, `purge_gap_frames`.
2. `require_independent` — evaluated groups must be whole groups of the requested partition (`test` by default) and disjoint from the groups the model was trained on, which must lie in `train`.
3. Calibration evaluation additionally requires an artifact fitted on the `calibration` partition only. Random window splits are never accepted; groups, not windows, are the unit.

## Metric families (computed only after the gate passes; otherwise `EvaluationBlocked`)
| Family | Metrics | Evidence required |
|---|---|---|
| Trajectory | ADE, FDE (most-probable mode, primary), horizon-specific error, minADE@K / minFDE@K (oracle, separate), by scene and agent category | independent held-out groups with verified trajectory ground truth |
| Conflict | precision, recall, F1, PR-AUC, ROC-AUC, false-alarm rate (Phase 6 evaluator) | verified conflict labels on independent groups |
| Calibration | Brier, ECE, NLL, reliability (Phase 7 evaluator) | verified labels, disjoint calibration and test groups |
| Alerts | precision, recall, false alerts per hour, missed-event rate, duplicate-alert rate, warning coverage, lead-time distribution, time-to-clearance (`alert_level_metrics`) | verified events with timestamps; matching horizon declared by the caller (no default) |
Times stay distinct: observation time (frame / fps), warning-generation time (stream time of the deciding anchor), predicted conflict time (not exposed, not physical) and verified event time (blocked).
The metric arithmetic is unit-tested on synthetic arrays only; those tests are formula checks, not results.

## What is and is not claimed
Not claimed: independent generalisation, calibrated probabilities, verified conflicts or collisions, physical TTC/PET/distance/severity, validated early-warning time, real-time operation, deployment readiness.
Needed to unblock: ≥3 (preferably ≥5) independent annotated scenes with licence records, verified identity mapping and timestamp alignment, verified conflict/event annotations, a frozen split before training, then
retraining / calibration / threshold selection on the designated partitions only and one locked test evaluation.
