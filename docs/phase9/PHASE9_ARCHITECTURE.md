# Phase 9 — End-to-end integration and system evaluation (RESEARCH PROTOTYPE)

Label `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`. Branch `feature/phase9-end-to-end-validation`, cut from Phase 8 commit `cee4575` (parent `209b738`). Execution mode of the recorded run:
**`PRECOMPUTED_PHASE3_ARTIFACTS`** — Phases 4–8 were executed on the Phase 3 outputs produced on Colab; Phase 3 raw-video inference was **not** executed (no raw video, no OpenCV /
Ultralytics / `psutil`, no GPU in this environment, and none was installed). This is not a raw-video-to-alert execution.

## 1. Pipeline and what Phase 9 adds
Video → Phase 3 perception (precomputed) → Phase 4 Task 2 image kinematics → Phase 4 Task 3 interaction representation → Phase 5 P5.3 windows → Phase 5 forecasting → Phase 6 conflict
representation → Phase 7 risk → Phase 8 warning events → S9 evaluation gates. Phase 9 only adds the orchestration layer (`pedsafety/phase9/`); every stage calls the existing public API of its phase
(`run_image_kinematics`, `run_interaction_representation`, `generate_windows`, `load_trained_model` + `predict_windows`, `build_conflict_pairs`, `assess_risk`, `run_alerts`). No earlier-phase file
was modified.

| Module | Role |
|---|---|
| `contracts.py` | versioned stage contracts and validators: Phase 3 rows (fields, coordinate system, boxes, timestamps = frame/fps, unique/ordered tracks), manifest hash chain, history-only windows, predictions, Phase 7 record identity, alert outputs |
| `orchestrator.py` | `PipelineConfig` (strict JSON), `Pipeline` (stage runner with dependency handling), `run_pipeline` (manifest, benchmark, failure register) |
| `benchmark.py` | stage clock (wall, CPU, process peak working set), warm-up/steady-state repetition, statistics, hardware description, real-time assessment rule |
| `verification.py` | input hashing, exact comparison of a run with the committed Phase 5/7/8 artifacts |
| `reporting.py` | execution-mode statement and per-component time attribution derived from a manifest |
| `evaluation.py` | gated independent evaluation (frozen group split, verified annotations, partition isolation) and alert-level metrics |
| `__main__.py` | CLI: `run`, `evaluation-status`, `verify-run`, `attribute` |

## 2. Stage handling rules
A stage that is selected runs only after its selected upstream stages COMPLETED; otherwise it is `NOT_RUN` (`UPSTREAM_NOT_COMPLETED`). A stage that is not selected takes its input from a validated
reference run or fails with `MissingInputError`. Any exception is recorded (`FAILED`, stable code) and the run exits non-zero; nothing is interpolated, merged, repaired or substituted. The manifest
reports `execution_mode`, `raw_video_executed`, `phase4_to_phase8_chain_complete`, `precomputed_artifact_integration_success` and `raw_video_to_alert_pipeline_success`; the last is true only when
Phase 3 was executed on a video, so no precomputed or partial run can be mistaken for it.

## 3. Findings that shape the architecture
1. **Model provenance is bound to the exact window-run manifest hash.** A regenerated window run has a different manifest (timestamps, git), so the Phase 5 model rejects it
   (`DATASET_PROVENANCE_MISMATCH`, tested). Phase 9 therefore re-executes Phases 4–5 data stages and proves their data files equal the reference (sha256), but runs inference on the reference window run.
2. **The window builder needs the target's future.** A window is accepted only if the pedestrian also has its 15 future samples, so which windows (and alerts) exist depends on future track existence.
   Model inputs themselves are history-only (future-target scrambling changes no output, end to end), but the system is an offline replay, not a causal online pipeline. This is the main structural obstacle to any real-time claim.
3. The Phase 8 events recorded for 25 of 28 alerts end by `EVIDENCE_LOST`: the pair stops being observed (consequence of finding 2 and of vehicles leaving), not because risk subsided.

## 4. Coordinate and unit policy
Only `captured_1920x1080` image pixels exist. Every validator rejects other coordinate systems; no stage emits or accepts metres, speeds, TTC, PET or severity; the camera moves and no ego-motion compensation exists.
