# Phase 5 Kickoff Report

**Date:** 15 November 2026. **Status:** `PHASE5_PLANNING_COMPLETE — IMPLEMENTATION_NOT_STARTED`. No model, training or result exists.

## 1. Verified starting Git state
| Item | Value |
|---|---|
| Branch | `feature/phase4-task3-interaction-representation` |
| HEAD | `5d58ab7146ea4611733393769071b02a6bf9d187` — "feat(phase4): implement image-space interaction representation (Task 3)" (exists; verified) |
| Parent | `2306309c786fa5608da96062123633cbb18acf7e` (Task 2 commit) |
| Working tree at start | clean (no staged, unstaged or untracked files; no stash) |
| `docs/phase5/` at start | did not exist |

## 2. Phase 4 readiness (evidence)
| Task | Status | Evidence |
|---|---|---|
| 1 Calibration audit | `EVIDENCE_COLLECTION_PENDING` | `docs/phase4/PHASE4_TASK1_CAMERA_CALIBRATION_AUDIT.md`; no validated extrinsics/road plane |
| 2 Image-space kinematics | `COMPLETED_WITH_LIMITATIONS` | commit `2306309`; `docs/phase4/PHASE4_TASK2_IMAGE_SPACE_KINEMATICS.md` |
| 3 Interaction representation | `COMPLETED_WITH_LIMITATIONS`, committed | commit `5d58ab7`; 75 Task 3 tests; pilot manifests (three byte-identical runs) |
| 4 Image-to-world | `BLOCKED_PENDING_CALIBRATION_EVIDENCE` | Task 1 |
| 5 Geometry validation | `BLOCKED_PENDING_INDEPENDENT_GEOMETRY_REFERENCES` | Task 1 |

Overall: `IMAGE_SPACE_PIPELINE_COMPLETED_WITH_LIMITATIONS`. Phase 5 therefore runs in image space only.

**Master-file currency.** `PROJECT_MASTER.md` still states "No Phase 3 implementation has started" and the canonical handoff file (`Pedestrian_Safety_Prediction_Master_Project_File_UPDATED_11_Nov_2026_Phase2_Handoff(2).md`) predates Phases 3–4; both are stale relative to the repository and Git history. Their Phase 5–7 roadmap (GATv2, dilated TCN, multimodal head, comparison with PCG and ablations; conflict prediction in Phase 6) is consistent with this plan. Neither file was modified.

## 3. Available Phase 5 inputs
Pilot Task 3 run `experiments/phase4_task3_pie_video_0004/20261114T214634Z_pie_video_0004/primary/`: 65,388 nodes (pedestrian 23,661; vehicle 34,678; bicyclist 7,049) in 9,326 frames, 1,500 tracks, 404,710 edges in 7 relations, feature dictionary `interaction-feature-dictionary-1.0.0`. Arrays are local (gitignored); manifests, dictionary and diagnostics are committed. Full interface: `PHASE5_INPUT_CONTRACT.md`.
Provenance note: those manifests record commit `2306309` (Task 3 code was uncommitted at run time). Recommended: regenerate from `5d58ab7` and confirm identical hashes before Phase 5 consumes them.

## 4. Dataset readiness (summary of `PHASE5_DATASET_READINESS.md`)
Only **one PIE video** (`set01/video_0004`, ≈ 328 s) is processed; full PIE, inD and TITAN are not acquired; HBS is in the repository but in unresolved, non-image units. There are **no independent groups**, so no valid train/validation/test split exists yet. The pilot supports development windows but not generalization claims. *(Revised for P5.3: the kickoff estimate "2,482 pedestrian windows from 88 tracks" used the provisional H = 5 and is superseded; the recalculated H = 6 estimate is 2,380 windows / 83 tracks (per-track anchor phase) or 2,327 / 84 (global grid), see `PHASE5_DATASET_READINESS.md` §3; actual P5.3 counts are in `PHASE5_TASK3_WINDOW_GENERATION.md`.)* Phase 3 detection/tracking quality on PIE has not been measured (Task 4 evaluator not yet run on the real XML).

## 5. Technical decisions proposed (for review)
1. Image space only; targets and errors in pixels; scale-aware FDE/box-height metric as a labelled heuristic.
2. Windows: provisional `s = 3` (10 Hz), `H = 5`, `F = 15`; finalised by a training-data window study. *(Revised for P5.3: approved `H = 6` — six samples t−0.5 … t — with `s = 3`, `F = 15`; dependency interval `[t0 − 16, t0 + 45]`, purge gap 62 frames.)*
3. Strict continuity: target present at every history and target frame; no interpolation; neighbours only when observed.
4. Dependency interval `[t0 − (H−1)s − 1, t0 + F·s]` drives split isolation and purges.
5. Split by independent group (proposal: PIE set) before window generation; pilot only as a labelled smoke split.
6. Non-oracle primary metrics (most-likely mode); best-of-K labelled as oracle; brier-minFDE and NLL as probability-aware metrics; cluster bootstrap over tracks.
7. Baselines B0–B3; PCG retained as a non-comparable historical reference.
8. GATv2 in plain PyTorch (masked dense, relation-specific) to avoid a new dependency; causal dilated TCN; K-mode decoder with WTA + classification loss.
9. New code under `pedsafety/forecasting/`, tests under `tests/phase5/`, configs under `configs/phase5/`.

## 6. Open questions (your decision)
- **Q1** Approve the provisional window parameters and the 10 Hz sampling, or prefer native 30 Hz?
- **Q2** Target source: perceived tracks (Task 3 future nodes) or PIE ground-truth annotations (needs an adapter)?
- **Q3** Which PIE subset/sets to acquire and process, and confirmation of set independence semantics?
- **Q4** Forecast targets: pedestrians only (proposed) or also vehicles/bicyclists?
- **Q5** Plain-PyTorch GATv2 (proposed) or install PyTorch Geometric (compatibility check first)?
- **Q6** Regenerate the pilot Task 3 output from commit `5d58ab7` for clean provenance (proposed)?
- **Q7** Bicyclist→bicyclist relation (Task 3 open item) — needed for Phase 5?

## 7. Blockers
1. **Scene-diverse data** (P5.2): only one processed video; licence archive, subset selection and Colab perception runs required. Blocks final splits, training, ablations and test evaluation.
2. **Perception quality unknown** on PIE (Phase 3 Task 4 not run on real data): affects target reliability if perceived tracks are targets.
3. **No metric geometry** (Phase 4 Tasks 4/5): metric metrics, TTC/PET and safety claims remain out of scope (not a blocker for image-space forecasting).

## 8. Can Phase 5 proceed?
Yes, to **P5.3 window generation and P5.4 baselines on the pilot**, as development work with a labelled smoke split. No final training, model selection on real splits, or reported results until P5.2 provides independent groups.

## 9. Recommended next task
**P5.3 — window generation and split manifests (pilot, smoke split)**, after your decisions on Q1, Q2, Q4 and Q6. In parallel (outside this repository's code), start P5.2: archive PIE licence terms, choose the PIE subset, and schedule Colab perception runs.
