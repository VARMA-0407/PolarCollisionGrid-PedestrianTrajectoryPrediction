# Phase 6 — Interaction-aware pedestrian–agent conflict representation

Status: implemented as a **heuristic image-space representation**; supervised conflict prediction is **BLOCKED** (no verified labels, one scene group).
Label: `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT` (in-sample Phase 5 model, one PIE video, `set01/video_0004`).

## 1. Phase 5 gate (verified, not assumed)
Branch `feature/phase6-conflict-prediction`, HEAD `f3d44bf` (Phase 5 feature commit `e383ea4`). Consumed artifacts, hashed in the report:
`predictions_neural.npz` sha256 `c070134f…99748` (2327 windows, K=5, F=15 at 10 Hz, raw pixels), `windows.npz`, `graph_bank.npz`, baseline config.
Phase 4/5 regression: 399 passed, 2 skipped. All P5 interfaces are consumed unchanged (no P4/P5 file modified).

## 2. Concepts kept separate
| Concept | Where | Status |
|---|---|---|
| Trajectory prediction | Phase 5 (pedestrian, multimodal); P5.4 classical extrapolation of the *observed history* for context agents (not learned) | upstream |
| Geometric conflict indicators | `features.py` (image plane, pixels) | implemented, heuristic |
| Conflict classification / probability | `scoring.py` `LogisticConflictHead` | implemented, **gated**; never trained on real data |
| Verified conflict ground truth | `labels.py` specification | none available |

## 3. Interfaces
Input (`Phase5Predictions` + window dataset + graph bank): pedestrian predicted `[K,F,2]` px, mode probabilities `[K]` (uncalibrated softmax), validity `[K,F]`,
window/track/anchor identities, horizon offsets (checked against the window config), coordinate system + unit (must be `captured_1920x1080` / `pixel`),
artifact SHA-256s. Context agent: vehicles (configurable) observed at the anchor; future = B0/B1/B2 extrapolation of its own history only.
Output (`ConflictPairBatch`): pair identities (window, ped track, other track, type), step/mode/pair features, boolean masks, history anchors and box sizes,
exclusion statistics, provenance (`future_observations_used=false`, `labels_used=false`, `physical_quantities=false`, `ego_motion_compensated=false`).
`ConflictScores` always declares `calibrated=False, is_probability=False` (enforced; constructing otherwise raises).

## 4. Mathematical definitions
Full definitions, units and validity rules are in the `features.py` module docstring and machine-readably in `FEATURE_SPECS`. Summary: pedestrian mode k position
`p_k(t)`, agent `q(t)`, `t` on the horizon grid; `sep = ‖p−q‖` [px]; `sep_over_h = sep / h_ped` (dimensionless heuristic depth proxy, not a distance);
`closing_rate = (sep(t−1)−sep(t))/Δτ` [px/s, image plane]; `box_overlap_ios` of constant-size boxes at predicted footpoints; `t_min_sep` is the time of
closest predicted image-plane approach (**not TTC**); `heuristic_proximity_score = max_t clip(1 − sep_over_h/κ, 0, 1)`; mode-weighted aggregates use the
uncalibrated Phase 5 weights; worst-case aggregates use no weights. Invalid values are NaN with an explicit mask. Thresholds (κ, sensitivity levels) are
exploratory, never a collision boundary.

## 5. Label specification and dataset feasibility
Supervisable kinds: `VERIFIED_COLLISION`, `VERIFIED_NEAR_CONFLICT`, `ANNOTATED_INTERACTION` from a documented independent source (name, type, version, SHA-256,
protocol), label interval strictly after the anchor, ≥3 independent groups, both classes, identities matching the pair batch. `CANDIDATE_HEURISTIC_*`
labels can never be supervision (`LABELS_NOT_VERIFIED`). Feasibility in this repository: only one PIE video and no annotation XML → no collision,
near-conflict, interaction or event-time annotation; the graph bank keeps only history nodes of context agents, so even a realised-future candidate label for a
vehicle cannot be derived. No labels were generated.

## 6. Evaluation
`evaluate_supervised` (precision, recall, F1, PR-AUC, ROC-AUC, false-alarm rate, event detection and lead time; caller-declared threshold, no default) refuses
unverified labels. `candidate_indicator_agreement` is a separate, label-free consistency diagnostic with no precision/recall naming. Results on real data:
**none reported** (blocked). Only descriptive heuristic-score statistics exist (`experiments/phase6_conflict_pie_video_0004/conflict_report.json`): 8233
pedestrian–vehicle pairs in 2298/2327 windows, 8090 valid, 143 invalid (no valid causal velocity), 29 windows without a vehicle.

## 7. Blocked / limitations
- Supervised training and evaluation: blocked (no verified labels; one group; ≥3 required).
- No physical distance, speed, TTC or PET: coordinates are image pixels, camera moves, no ego-motion compensation, no calibration.
- Pedestrian–pedestrian / bicyclist pair types are implemented but not enabled in the default config (vehicle only).
- Vehicle motion is a classical constant-velocity style extrapolation, not a learned forecast; Phase 5 predictions are in-sample.
- Mode weights are uncalibrated; scores are not probabilities; heuristic scores are not validated against any event.

## 8. Tests
`tests/phase6/test_conflict.py` (15): feature maths, masks, missing data, invalid numbers, mode pairs, horizon alignment, config/coordinate enforcement, label
validation and leakage, training gate, metric correctness, score claims, real-artifact integration (determinism, identity, corruption of non-history nodes does
not change features, coordinate/anchor mismatch rejected).
