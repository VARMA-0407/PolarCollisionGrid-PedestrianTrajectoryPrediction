# Real-Time AI-Based Pedestrian Safety Prediction

## Project identity

This project studies interaction-aware multimodal pedestrian trajectory and
conflict forecasting from video for shared pedestrian/vehicle spaces. The
repository contains the Polar Collision Grid (PCG) implementation from the
published ITSC 2023 work and its baseline models. The research objective is
to develop a future, deployment-feasible pipeline that combines perception,
tracking, geometric state estimation, interaction-aware forecasting, and
calibrated early warnings.

Expected contributions are limited to claims supported by verified evidence.
Metric-world safety interpretation, deployment performance, and generalized
cross-scene performance remain outside the current evidence.

## Proposed end-to-end architecture

The proposed architecture is:

Video ingestion → YOLO perception → ByteTrack tracking → geometric and
kinematic preprocessing → interaction graph construction → heterogeneous
GATv2 → dilated TCN → multimodal trajectory forecasting → conflict
prediction → risk calibration → early-warning output.

This is a proposed architecture, not a statement that every component is
implemented in this repository. The current repository contains the PCG
trajectory model and supporting baselines, not the complete proposed
perception-to-warning system.

## Baseline reproduction

The software baseline is the PCG repository described by `README.md`, with
the authoritative audit branch `reproduction`. The audited environment is
Windows, Python 3.9.25, pytest 8.4.2, using the `pcg-baseline` environment.
The historical PCG method-4 context uses six observed frames, six predicted
frames, and a five-sample historical evaluation configuration. The preserved
authoritative artifacts include:

- `experiments/PCG_T4_5_Epoch_Run/results/test_results_epoch4.pkl`
- `experiments/PCG_T4_5_Epoch_Run/results/ade_fde_reconstruction.json`
- `experiments/PCG_T4_5_Epoch_Run/manifest.json`
- `Store_Results/` checkpoints and logs where present

The independent ADE/FDE reconstruction differs from the recorded values by
less than `1e-5`, which validates reconstruction of the historical reported
values, not their deployment validity.

The baseline Task 8 suite collected 12 tests and produced 11 passed, 1
failed, and 0 skipped. The failure is
`test_ttc_occupancy_mask_collapses_agent_dimension`, caused by
`getInteractionGridMask()` returning `(frame_mask, TTC_mask)` while its
occupancy branch leaves `TTC_mask` uninitialized. The corrected result of
12 passed is isolated `audit-fixes` evidence only; it is not part of
`reproduction`.

## Dataset and preprocessing

The repository uses `Data/HBS/hbs.csv`. The audited CSV has 43,459 rows, nine
columns, 1,475 agents, 3,620 frames, and one scene. The inspected historical
class distribution is 28,123 pedestrian observations/1,115 agents, 14,414
car observations/331 agents, and 922 bike observations/29 agents.

`utils/DataLoader.py` reads pedestrian labels `pedestrian` and `ped`, vehicle
labels `car`, `cart`, and `veh`, and does not include bikes in the modeled
agent classes. It assumes `downsample_step = 1`, `fps = 2`, and a 0.5-second
timestamp interval. The CSV timestamps support the 0.5-second sampling
interval, but do not establish original video FPS.

The meanings and units of position and velocity are unresolved. No
calibration, homography, video, or dataset metadata file was found in the
repository. Dataset licensing and official provenance remain unresolved.

## Experimental methodology

The historical loader forms 12-frame windows from six observed and six
prediction frames. Task 5 audits found 66 cross-split overlapping window
pairs and 11 affected test windows in the authoritative reconstruction.
These results are retained as historical limitations.

The historical test path performs best-of-five oracle selection using future
ground-truth ADE and uses future pedestrian/vehicle information during
rollout interaction construction. These historical metrics must not be
presented as deployment metrics. The corrected protocol requires observed or
predicted states only, deterministic or predeclared multimodal aggregation,
and a split purge before window generation.

ADE and FDE reconstruction is independently validated against the preserved
prediction result. Conflict metrics remain coordinate-dependent and are not
yet physically validated.

## Phase 2 task status

| Task | Objective | Evidence | Status | Limitations |
|---|---|---|---|---|
| 1–3 | Earlier project/reproduction activities | Repository artifacts and historical reports | Historical evidence retained | Detailed task reports are outside the current consolidation evidence |
| 4 | Oracle and future-information audit | `reports/task4_oracle_future_information_audit.json` | Audit complete | Historical evaluation is not deployment-valid |
| 5 | Window-overlap audit | `reports/task5_window_overlap_authoritative.json` | Audit complete | Single scene and temporal overlap remain |
| 6 | Corrected validation protocol | `reports/task6_corrected_validation_protocol.json`, `reports/TASK6_VALIDATION_PROTOCOL_REVIEW.md` | Revision required | Protocol is not implemented or hidden-future tested |
| 7 | HBS coordinate/provenance audit | `reports/task7_hbs_coordinate_provenance_audit.json`, `reports/TASK7_HBS_COORDINATE_PROVENANCE_FINAL_AUDIT.md` | Partial — documented limitations | Units, calibration, provenance, and license unresolved |
| 8 | Boundary/unit tests | `reports/task8_unit_test_report.json`, `logs/task8_pytest_output.txt` | Audit complete — baseline defect documented | Reproduction remains 11/12 passing |
| 9 | Reproducibility and manifest integrity | `manifest_v2.json`, `logs/task9_manifest_verification.txt` | Audit complete | 20/20 final manifest scope |

## Task 10 consolidation

This document is the first consolidated project master because no previous
master file was found in the repository or recent Git history. The
consolidation report is
`experiments/PCG_T4_5_Epoch_Run/reports/TASK10_MASTER_FILE_CONSOLIDATION_REPORT.md`.
The Task 6 review, Task 7 final audit, and next-phase roadmap are retained as
separate supporting reports.

## Known defects and limitations

- Occupancy TTC-mask defect in the original implementation.
- Oracle best-of-N selection uses future ground truth.
- Future pedestrian and vehicle information enters the historical rollout.
- Cross-split temporal window overlap remains.
- Position and velocity units are unverified.
- Camera calibration and homography are unavailable.
- Dataset provenance and licensing are unresolved.
- Bikes are present in HBS but excluded by the baseline loader.
- Corrected protocol and deployment pipeline are not implemented.

## Relevant repository structure

```text
README.md
PROJECT_MASTER.md
Data/HBS/hbs.csv
utils/DataLoader.py
utils/Interaction.py
test.py
train.py
tests/phase2/
scripts/verify_phase2_manifest.py
experiments/PCG_T4_5_Epoch_Run/
  manifest_v2.json
  reports/
  logs/
  results/
```

## Current project status

The original PCG reproduction artifacts and historical ADE/FDE values have
been preserved and independently reconstructed. Phase 2 Tasks 8 and 9 are
audit-complete with the baseline defect documented and a final 20/20 manifest
verification. Task 6 is revision-required and Task 7 is partial. The
perception-to-warning architecture and corrected evaluation are proposed
only. No Phase 3 implementation has started.

## Pending work

1. Resolve HBS provenance, license, coordinate semantics, and calibration.
2. Implement and test a separate observed-only, non-oracle evaluator.
3. Materialize leakage-free splits and purge policy.
4. Build and validate the perception frontend.
5. Implement geometric/kinematic processing with verified semantics.
6. Implement and ablate the interaction-aware multimodal model.
7. Define and calibrate conflict indicators.
8. Measure end-to-end real-time performance.

## Next implementation phase

The ordered roadmap is
`experiments/PCG_T4_5_Epoch_Run/reports/NEXT_IMPLEMENTATION_PHASE_ROADMAP.md`.
It is planned work awaiting explicit user approval and is not an
implementation report.

## Revision history

| Date | Revision |
|---|---|
| 2026-11-11 | Established the first consolidated master from repository and Phase 2 audit evidence; no original master file was found. |

