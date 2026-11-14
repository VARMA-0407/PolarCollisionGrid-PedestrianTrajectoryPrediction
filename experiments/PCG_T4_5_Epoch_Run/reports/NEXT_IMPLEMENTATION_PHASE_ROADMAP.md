# Next Implementation Phase Roadmap

## Scope and status

This is a proposed roadmap only. It is **PLANNED — NOT IMPLEMENTED**.
It does not authorize changes to the authoritative `reproduction` branch,
model training, dataset replacement, or Phase 3 execution.

## Phase A — Dataset and coordinate readiness

| Field | Plan |
|---|---|
| Objective | Establish lawful provenance, coordinate semantics, units, and calibration requirements. |
| Prerequisites | Official HBS documentation, terms of use, source paper, and available metadata. |
| Tasks | Obtain provenance; confirm units/axes; locate calibration/homography; decide whether HBS supports metric safety claims. |
| Deliverables | Provenance record, coordinate specification, calibration decision, dataset suitability decision. |
| Acceptance criteria | Every coordinate and velocity field has an evidence-backed interpretation, or is explicitly declared uncalibrated. |
| Dependencies | None; must precede metric conflict claims. |

## Phase B — Corrected baseline evaluation

| Field | Plan |
|---|---|
| Objective | Create a deployment-feasible, non-oracle evaluation without altering reproduction. |
| Prerequisites | Task 6 review, coordinate decision, and split policy. |
| Tasks | Isolate corrected evaluator; hide future annotations; define actor birth/death; apply purge split; report deterministic and separately labelled oracle metrics. |
| Deliverables | Isolated evaluator, hidden-future tests, split artifacts, corrected metrics report. |
| Acceptance criteria | No future annotation reaches rollout or mode selection; all tests and provenance checks pass. |
| Dependencies | Phase A for coordinate-dependent metrics. |

## Phase C — Perception frontend

| Field | Plan |
|---|---|
| Objective | Define video-to-track inputs for real-time operation. |
| Prerequisites | Dataset/video availability and coordinate decision. |
| Tasks | Video ingestion, YOLO detection, ByteTrack IDs, confidence handling, association, missing detections, FPS policy. |
| Deliverables | Reproducible perception pipeline and track-quality evaluation. |
| Acceptance criteria | Stable track IDs and documented latency under declared conditions. |
| Dependencies | Phase A and corrected input contract. |

## Phase D — Geometric and kinematic processing

| Field | Plan |
|---|---|
| Objective | Convert detections into aligned states without assuming unverified metres. |
| Prerequisites | Calibration or an explicit uncalibrated operating mode. |
| Tasks | Ground-contact point, coordinate transform, velocity/acceleration estimation, smoothing, alignment, filtering, missing observations. |
| Deliverables | State-processing module and unit tests. |
| Acceptance criteria | Transform and derivative behaviour are verified on known fixtures. |
| Dependencies | Phase A and Phase C. |

## Phase E — Interaction-aware trajectory model

| Field | Plan |
|---|---|
| Objective | Implement the proposed heterogeneous interaction model and multimodal decoder. |
| Prerequisites | Leakage-free data splits, observed-only contract, and state representation. |
| Tasks | Heterogeneous graph, GATv2, dilated TCN, multimodal decoder, losses, ablations. |
| Deliverables | Isolated model implementation, training configuration, ablation reports. |
| Acceptance criteria | Reproducible training/evaluation with no future-input leakage and preserved baselines. |
| Dependencies | Phases B–D. |

## Phase F — Conflict forecasting and risk calibration

| Field | Plan |
|---|---|
| Objective | Evaluate TTC/PET/separation surrogates and calibrate conflict probabilities. |
| Prerequisites | Coordinate semantics and explicit conflict definitions. |
| Tasks | TTC, PET, minimum separation, probability calibration, ECE/Brier, alert thresholds, warning lead time, false-alert/missed-conflict analysis. |
| Deliverables | Conflict-label specification, calibration report, threshold analysis. |
| Acceptance criteria | Indicators are reproducible and labelled as surrogates unless real-world validation exists. |
| Dependencies | Phases A, B, and E. |

## Phase G — Real-time deployment evaluation

| Field | Plan |
|---|---|
| Objective | Measure end-to-end feasibility. |
| Prerequisites | Working perception, model, and conflict pipeline. |
| Tasks | Latency, FPS, memory, hardware, tracker overhead, alert scheduling. |
| Deliverables | Benchmark protocol and hardware-qualified performance report. |
| Acceptance criteria | Measurements include hardware, batch, resolution, and timing methodology. |
| Dependencies | Phases C–F. |

## Recommended immediate next task

**Resolve HBS provenance, license, coordinate semantics, and calibration
availability before implementing the corrected evaluator.** This must come
first because metric conflict claims, coordinate transformations, and
evaluation acceptance criteria depend on facts not currently established.

## Critical blockers and risks

- Missing official provenance and license may restrict redistribution or
  publication.
- Missing calibration prevents defensible metric-world safety claims.
- The one-scene dataset limits scene-independent generalization.
- Existing oracle/future-information leakage makes historical metrics
  unsuitable as deployment claims.
- Temporal window overlap may inflate comparability.
- Excluding bikes limits class coverage.
- The occupancy TTC-mask defect remains in the baseline.

## Compute and dataset requirements

The roadmap requires the existing Python environment for audit tooling, a
separate environment or branch for corrected evaluation, and sufficient CPU
for data reconstruction. GPU requirements should be measured from the chosen
model and are not assigned here without an implementation. Additional
dataset requirements are official HBS metadata, source documentation,
licensing terms, and calibration/video evidence if metric coordinates are
required.

## Validation strategy

Use source-anchored unit tests, hidden-future-input tests, split and purge
audits, deterministic metric fixtures, calibration checks, and separately
stored execution logs. Preserve all historical reproduction evidence and
compare corrected results without overwriting it.

## Claims prohibited before validation

Do not claim leakage-free performance, deployment-ready oracle metrics,
metric-world TTC/PET, calibrated conflict probabilities, cross-scene
generalization, real-time FPS, or completed Phase 3 implementation until
the corresponding evidence exists.

