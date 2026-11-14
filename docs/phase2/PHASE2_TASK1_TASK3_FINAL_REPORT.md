# Phase 2 Tasks 1-3 Final Report

**Date:** 11 November 2026  
**Branch:** `reproduction`  
**Overall status:** COMPLETED_WITH_LIMITATIONS  
**Approval state:** PENDING_USER_APPROVAL

## Task status

| Task | Status | Result |
|---|---|---|
| Task 1 - Dataset research and selection | COMPLETED_WITH_LIMITATIONS | Evidence-backed comparison of PIE, JAAD, TITAN, inD, rounD, and BDD100K. |
| Task 2 - Feasibility and acquisition audit | COMPLETED_WITH_LIMITATIONS | Conditional acquisition plan; no downloads initiated. |
| Task 3 - Common agent-state schema | COMPLETED_AND_VALIDATED | Revised Markdown specification, syntactically valid Draft 2020-12 JSON Schema, and positive/negative validation tests. |

## Files created

- `docs/phase2/task1_dataset_research.md`
- `docs/phase2/task2_dataset_feasibility.md`
- `docs/phase2/task3_agent_state_schema.md`
- `docs/phase2/schemas/agent_state.schema.json`
- `docs/phase2/PHASE2_TASK1_TASK3_FINAL_REPORT.md`

## Main verified findings

- PIE is the strongest provisional video-primary candidate.
- PIE's official materials describe onboard video at 30 FPS, pedestrian and
  vehicle annotations, and camera-parameter resources.
- TITAN is a strong heterogeneous alternative with 10 Hz annotations and
  object track IDs, but requires university access and is non-commercial.
- inD and rounD are valuable drone trajectory/map resources; they must not be
  presented as interchangeable raw roadside-video datasets.
- BDD100K is a large perception resource with an approximately 1.8 TB video
  download reported in its official documentation; it is not suitable for
  the first local end-to-end acquisition.
- Ego GPS/IMU or odometry does not establish world coordinates for every
  pedestrian and vehicle.
- The schema explicitly separates pixel and world coordinates and represents
  unavailable values as null with validity flags.
- Kinematic coordinate space, position unit, and time basis are explicit;
  heading and interpolation semantics are documented.

## Recommendation requiring approval

Approve PIE as the provisional primary dataset. Approve either:

1. inD as a trajectory/map supplement, or
2. TITAN as a heterogeneous video alternative after access approval.

Do not begin acquisition until license terms, storage capacity, calibration
semantics, and the first pilot subset are approved.

## Validation results

- Markdown documents were written with headings, tables, code blocks, and
  source links.
- The JSON Schema was authored as JSON Schema Draft 2020-12 and checked for
  valid JSON syntax.
- Positive records cover image-space, valid world-space, unavailable-world,
  and nullable optional fields.
- Negative records cover invalid world coordinates, array lengths,
  confidence bounds, undeclared properties, and missing identifiers.
- Exact schema test result: `22 passed in 0.84s`.
- Exact complete Phase 2 result: `33 passed, 1 failed, 13 warnings in
  2.82s`; the failure is the pre-existing occupancy `TTC_mask` defect in
  untouched `utils/Interaction.py`.
- External source URLs are included in the Task 1 and Task 2 documents.
- Published sizes are distinguished from estimates or unknown values.
- Image-space and world-space coordinates are explicitly separated.
- Missing world coordinates are represented as `null` and guarded by
  `world_position_valid`.
- No dataset download, license acceptance, application submission, training,
  model implementation, or baseline-source modification was performed.

## Important unresolved questions

1. What exact license applies to each PIE asset used for publication and
   redistribution?
2. Does the selected inD release provide any raw video access for this
   project, or only trajectories, metadata, background images, and maps?
3. What calibration and coordinate transforms are officially supported for
   PIE and TITAN?
4. What exact annotation frequency and identity semantics should be used for
   the selected PIE release?
5. Which supplementary dataset is approved, and should datasets be trained
   jointly or evaluated separately?

## Repository preservation

- Current branch is `reproduction`.
- Original PCG source, datasets, checkpoints, prediction artifacts, and
  historical Phase 2 audit evidence were not modified.
- No commits or pushes were made.
- The new work is documentation/schema work under `docs/phase2`.
- `jsonschema` was installed in `pcg-baseline` only after the environment
  reported it missing during schema-test collection; no project dependency
  manifest was changed.

## Required user decisions

- Approve or reject PIE as the provisional primary dataset.
- Select inD or TITAN as the first supplementary dataset, or reject both.
- Approve the conditional acquisition plan and storage location.
- Approve the schema as the input contract for later implementation.

Task 4, camera geometry implementation, preprocessing implementation,
splitting implementation, foundation-pipeline work, model training, and
dataset acquisition are intentionally not started.
