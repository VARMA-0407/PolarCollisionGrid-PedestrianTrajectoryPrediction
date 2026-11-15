# Phase 7 — Risk calibration and pedestrian safety risk assessment

Label: `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`. Base: branch `feature/phase7-risk-assessment` from Phase 6 commit `d6ac175` (Phase 5: `e383ea4`, `f3d44bf`).

## Status matrix
| Capability | Status |
|---|---|
| Heuristic raw conflict score → risk score → provisional category, structured output with provenance | **Implemented** (descriptive only, no accuracy claim) |
| Calibration interface, artifact schema, hash/provenance verification, partition isolation, calibration metrics | **Implemented, tested on synthetic labels only** |
| Fitted calibration on real data / calibrated probabilities | **BLOCKED** (no verified labels, one scene group) |
| Empirically derived category thresholds | **BLOCKED** (categories are PROVISIONAL_ENGINEERING) |
| Severity, physical TTC/PET, deployment readiness | **Not supported / not claimed** |

## 1. Pipeline and the five distinct concepts
Phase 5 predictions → Phase 6 pair features/scores → [calibration, gated] → risk score → category. (1) heuristic indicators (Phase 6 features, pixels);
(2) raw conflict score (`ConflictScores`, kind declared, never a probability); (3) calibrated probability (only with a verified artifact, else NaN +
`NOT_CALIBRATED`); (4) risk score (unitless engineering ranking value, not a probability); (5) risk category (provisional bin of the risk score).

## 2. Input schema (Phase 6 → Phase 7)
Required (row invalid / `INSUFFICIENT_EVIDENCE` if missing): `ConflictPairBatch` identities (window id, pedestrian and vehicle track ids, anchor frame), `ConflictScores`
(score in [0,1], validity), masks `pair_valid`, `context_prediction_valid`, `mode_valid`, `scale_ok`, `mp_valid`, features `t_min_sep_s`, `log_scale_ratio`,
`mode_weighted_score`, `mp_score`, mode probabilities, horizon grid, `coordinate_system = captured_1920x1080` / `pixel`, Phase 6 config hash.
Optional: provenance dict, calibration artifact (+ source checkpoint hash and Phase 6 config hash, both mandatory when an artifact is applied).
Suitable for calibration: scores from a scorer with a recorded checkpoint/config and verified labels. Heuristic-only: everything when no verified labels exist.

## 3. Calibration methodology
Method choice is driven by data: Platt scaling (2 parameters, tiny L2, Newton) when ≥20 positives and ≥20 negatives exist in the calibration groups;
isotonic (PAV) only with ≥1000 rows and ≥100 positives. Temperature scaling is not offered: scores are bounded heuristic/classifier outputs without logits.
No class re-weighting (it would shift the base rate the probability refers to); imbalance is handled by minimum class counts and reported prevalence.
Gates (violations raise `CalibrationBlocked`): labels pass the Phase 6 verification (verified kind, independent source with SHA-256, label times after the
anchor, ≥3 groups); train / calibration / test **groups** pairwise disjoint and non-empty (whole videos/scenes, never windows, because windows of one track
overlap); source provenance (checkpoint file hash, model state hash, Phase 6 config hash, coordinate system, scorer id) present.
Artifact `phase7-calibration-artifact-1.0.0`: method, parameters, score kind, source hashes, label source and report, split identity, counts, content SHA-256
(verified on load; mismatched checkpoint / config / score kind / coordinate system raise `ProvenanceMismatch`). In-sample fit metrics are stored but labelled
as non-evidence. Evaluation (`evaluate_calibrated`) uses the artifact's **test groups only**: Brier, NLL, ECE (10 equal-width bins), reliability table, and
group-cluster bootstrap 95% intervals only when ≥5 groups exist (otherwise `available: false` with the reason).
**Evidence on real data: none.** Required to unblock: verified conflict annotations from an independent source; ≥3 (preferably ≥5 for intervals) independent
videos/scenes split into disjoint train/calibration/test groups; ≥20 positives and negatives per class in the calibration groups; labels aligned to pair anchors
and strictly in the future of them; realised context-agent futures or event annotations.

## 4. Risk scoring specification
| Quantity | Definition | Unit | Validity | Source |
|---|---|---|---|---|
| base | calibrated probability if available, else raw score | 1 | conflict valid | Phase 6 / calibrator |
| temporal urgency u | `1 − t̄/T`, `t̄` = mode-weighted time of closest predicted image-plane approach, `T` = last horizon time | 1 | ≥1 valid mode | Phase 6 `t_min_sep_s`, Phase 5 mode weights (uncalibrated) |
| risk score | `base · (w + (1−w)·u)`, `w = urgency_weight_floor` | 1, in [0,1] | base and u finite | this module |
| severity | not supported (no metric geometry, speed, mass, impact data) | – | always NaN | – |
`u` is not a time-to-collision. The risk score is a ranking value for engineering inspection, not a probability or a physical hazard measure; `risk_basis` states what it was built from.

## 5. Categorisation specification
Config `configs/phase7/risk_default.json` (versioned, SHA-256 persisted in every output): thresholds `0.15 / 0.4 / 0.7` on the risk score → LOW / MODERATE / HIGH / CRITICAL.
`category_status = PROVISIONAL_ENGINEERING`, `threshold_evidence = NONE_UNVALIDATED_ENGINEERING_DEFAULT`: no empirical derivation exists, so these are not
validated safety levels (`EMPIRICALLY_DERIVED` is refused unless evidence is cited as `artifact:…`). `INSUFFICIENT_EVIDENCE` is returned for invalid/missing
conflict predictions, invalid context motion, non-finite risk, valid mode mass below 0.5, and out-of-distribution depth mismatch (`|ln(h_vehicle/h_pedestrian)| > 1.2`
or invalid scale). If mode-weighted and most-probable-mode scores differ by >0.4 the category is kept with status `…_CONFLICTING_INDICATORS`.

## 6. Output schema `phase7-risk-output-1.0.0`
Per pair: pedestrian_id, vehicle_id, anchor_frame, window_id, conflict_score, conflict_score_kind, calibrated_probability (null unless `CALIBRATED`),
calibration_status, temporal_urgency, severity (null) + severity_status, risk_score, risk_basis, risk_category, category_status, reason, masks
(`conflict_valid`, `category_assigned`, `out_of_distribution`). Run-level provenance: risk config hash/version, Phase 6 config hash and provenance, prediction /
window / graph-bank hashes, source checkpoint hash (`40aff17f…ab71b44`), calibration artifact hash (null), coordinates, `future_observations_used=false`, `labels_used=false`.

## 7. Evaluation
Only descriptive output on real data (`experiments/phase7_risk_pie_video_0004/`): 8233 pairs: LOW 6080, MODERATE 513, HIGH 297, CRITICAL 112, INSUFFICIENT_EVIDENCE 1231
(143 invalid prediction, 1088 depth mismatch). These are counts of a heuristic on an in-sample pilot, not performance. Precision/recall per category, false-alarm
rate, category confusion matrix, per-scene results, Brier/ECE/NLL are **not reported**: no verified labels or independent partitions exist. Heuristic evaluation,
supervised probability evaluation and calibrated-risk evaluation remain separate code paths (Phase 6 `evaluate_supervised`, Phase 7 `evaluate_calibrated`).

## 8. Limitations
Image-plane pixels only, moving camera, no ego-motion compensation; vehicle future is a classical extrapolation; Phase 5 predictions are in-sample; mode weights are
uncalibrated; thresholds are arbitrary engineering defaults; one video means no independence; calibrated risk, validated categories and deployment readiness are not claimed.

## 9. Tests
`tests/phase7/test_risk.py` (14): config validation, uncalibrated semantics, risk formula, thresholds and provisional status, INSUFFICIENT_EVIDENCE and mask propagation,
determinism and provenance, coordinate/identity enforcement, Platt fit and test-only evaluation, isotonic gating/monotonicity, partition isolation (perturbing non-calibration
groups does not change parameters), blocking without verified labels, artifact serialization and tamper detection, checkpoint/config/score-kind mismatch, metric
correctness, real-artifact integration (determinism, claims, checkpoint hash). Regression: Phase 4+5+6+7 = 428 passed, 2 skipped.
