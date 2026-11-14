# Phase 2 Task 6 - Data Splitting and Evaluation Foundation

**Status:** DESIGN_COMPLETE_PENDING_REVIEW  
**Date:** 11 November 2026  
**Scope:** Protocol design only. No dataset splits, windows, model training,
or evaluation were generated.

## 1. Executive summary

This protocol defines leakage-resistant partitioning, window generation,
preprocessing fitting, and evaluation for trajectory prediction and
interaction-aware safety forecasting. The primary recommendation is to split
by the highest independent recording unit available—preferably scene/camera
or recording—before generating sliding windows. If multiple independent scenes
are unavailable, use chronological groups with an explicit purge/embargo
boundary covering the complete observation-plus-prediction dependency.

The final test set is consulted once after model, hyperparameter, threshold,
and calibration decisions are frozen. Training-only statistics are fitted
only on training data. Future trajectories, actor lists, TTC/PET values, and
best-of-K oracle choices are targets or evaluation inputs, never model inputs
at the prediction timestamp.

Historical PCG evidence remains separate: the original result used
oracle-selected best-of-five ADE/FDE, future actor information, and
overlapping windows. Those results are not corrected deployment metrics.

## 2. Scope and dependencies

Dependencies:

- [Task 3 schema](schemas/agent_state.schema.json)
- [Task 3 semantics](task3_agent_state_schema.md)
- [Task 4 geometry audit](task4_coordinate_systems_camera_geometry.md)
- [Task 5 preprocessing design](task5_trajectory_kinematic_preprocessing.md)
- [Task 6 validation review](../../experiments/PCG_T4_5_Epoch_Run/reports/TASK6_VALIDATION_PROTOCOL_REVIEW.md)
- [Corrected protocol evidence](../../experiments/PCG_T4_5_Epoch_Run/reports/task6_corrected_validation_protocol.json)

No prior document is silently changed by this protocol.

## 3. Dataset grouping hierarchy

Use the following hierarchy, with identifiers scoped by dataset release:

```text
dataset/release
  -> recording or camera session
    -> scene/location
      -> sequence/clip
        -> agent track
          -> observation/prediction window
```

The split unit must be the highest unit that is independent enough for the
intended claim. Track IDs are not globally unique unless the source
guarantees it. A track, recording, camera, map frame, or source frame must not
be represented in multiple partitions.

Different datasets remain separate evaluation domains unless coordinate,
label, identity, sampling, and provenance compatibility is demonstrated.

## 4. Split-strategy comparison

| Strategy | Leakage risk | Generalization claim | Recommendation |
|---|---|---|---|
| Random observation | Very high: same track and adjacent frames cross partitions. | Weak. | Reject. |
| Random trajectory | Same scene/camera and temporal context may remain shared. | Limited. | Use only for explicitly independent tracks with evidence. |
| Scene-level | Prevents same-scene visual and geometric leakage. | Unseen-scene generalization. | Preferred when enough scenes exist. |
| Sequence-level | Better than observation-level, but sequences may share recording context. | Unseen-sequence generalization. | Accept when scene grouping is unavailable. |
| Chronological | Tests later time, but adjacent windows require purge. | Temporal forward generalization. | Fallback or complementary protocol. |
| Grouped scene/sequence | Combines independence and reproducibility. | Strongest available claim. | Primary recommendation. |

For PIE, inD, or TITAN, the actual release grouping must be inspected after
approval. The provisional dataset recommendation does not establish that
scenes, routes, cameras, or identities are independent.

## 5. Recommended split protocol

1. Verify release, license, source checksum, and provenance.
2. Resolve dataset, recording, scene, sequence, and track identifiers.
3. Select the split unit before window generation.
4. Assign whole groups to train, validation, and test.
5. Apply chronological ordering and purge/embargo where relevant.
6. Verify no shared source frame, agent track, recording, camera, or map frame
   crosses partitions.
7. Generate windows independently inside each partition.
8. Fit normalization and learned preprocessing on training data only.
9. Freeze transformations for validation and test.
10. Select models, hyperparameters, checkpoints, thresholds, and calibration
    using validation only.
11. Evaluate the frozen selection once on the final test partition.
12. Report per-scene and aggregate results with the manifest.

## 6. Scene, sequence, and identity isolation

For each candidate split, assert:

- `dataset_release` is compatible across the run.
- `recording_id`, `camera_id`, and `scene_id` do not cross partitions.
- `sequence_id` is unique within its source scope and does not cross splits.
- `(sequence_id, track_id)` is assigned to one partition only.
- Source frame keys do not overlap.
- Map/calibration versions are compatible with the assigned group.
- Re-linked fragments are treated as one underlying identity for isolation.

If a dataset has repeated routes, cameras, or near-duplicate clips, those
groups should be held together. A unique track ID alone is insufficient proof
of independence.

## 7. Sliding-window generation

Let `Delta_t` be the target sampling interval, `H` the observation history,
`F` the prediction horizon, and `S` the stride. A window beginning at `t_0`
uses:

```text
input_times  = {t_0, ..., t_0 + (H-1) Delta_t}
target_times = {t_0 + H Delta_t, ..., t_0 + (H+F-1) Delta_t}
```

Eligibility requires the configured minimum valid observations, permitted
missingness, coordinate compatibility, and a valid target where the metric
requires one. Interpolated values must remain masked and labelled according
to Task 5.

Splitting must occur before window generation. For example, with `H=6`,
`F=6`, and stride one, windows beginning at frames 0 and 1 share 11 source
frames. Assigning individual windows randomly can put nearly identical
histories in train and test. Assigning a recording first prevents this.

## 8. Purge and temporal embargo strategy

Exact overlap prevention removes shared source observations. Dependence
reduction may require an additional embargo between chronological groups.
These are different guarantees.

For a complete window span:

```text
window_span = (H + F - 1) Delta_t
```

The minimum boundary purge must be derived from the actual window span,
timestamp tolerance, resampling policy, and any feature lookback. It must
not be an arbitrary universal duration. If a centred offline filter or
interpolation can read beyond the nominal input, its support must be added to
the purge.

Pseudocode:

```text
assign_groups_before_windows()
for partition in [train, validation, test]:
    generate_windows_only_within_partition(partition)
assert no_source_frame_intersection(partitions)
assert no_track_or_recording_intersection(partitions)
if chronological:
    assert boundary_gap >= required_purge_span
```

The 12-frame historical PCG horizon is a comparison point, not a universal
new-project setting.

## 9. Train, validation, and test responsibilities

| Partition | Permitted use |
|---|---|
| Train | Parameter learning, training-only augmentation, normalization, and feature statistics. |
| Validation | Hyperparameters, early stopping, checkpoint selection, calibration fitting, and threshold selection. |
| Test | Final frozen performance report only. No tuning, selection, threshold fitting, or preprocessing-statistic fitting. |

The test set must be versioned and access-controlled as far as practical.
Repeated test inspection changes its role from final evaluation to a
development resource.

## 10. Normalization and preprocessing leakage prevention

Fit all learned statistics on training data only:

- position, velocity, acceleration, and time-delta normalization;
- imputation or clipping parameters;
- class weights and balancing rates;
- feature selection;
- augmentation distributions;
- calibration and probability transforms.

Apply the frozen transform to validation and test. Keep statistics separate
for image, relative/projective, metric-world, dataset, and incompatible
coordinate frames. Do not estimate scale from test trajectories, future
observations, camera height, GPS, or arbitrary homography values.

## 11. Future-information boundaries

At prediction time, inputs may contain only observations available at or
before the current timestamp: tracked states, causal filters, and
deployment-feasible actor birth/death decisions. Future trajectories,
future actor presence, future visibility, future interaction graphs, and
future TTC/PET values are targets or evaluation annotations only.

The historical PCG audit found:

1. test windows overlapped training windows;
2. future pedestrian presence/states entered rollout interaction construction;
3. future vehicle information entered method-4 interaction construction;
4. best-of-five ADE selection used future ground truth;
5. the original process lacked a genuine validation loop.

The corrected protocol removes these paths. Oracle best-of-K remains allowed
only as a separately labelled upper-bound analysis, never as the deployment
metric.

## 12. Trajectory prediction metrics

For valid agents and forecast steps:

```text
ADE = mean_t ||p_hat_t - p_t||
FDE = ||p_hat_F - p_F||
```

Report coverage and valid-prediction rate with ADE/FDE. Break down by
forecast horizon, agent class, scene, and coordinate space. Pixel-space
errors are pixels, not metres. Relative/projective errors must retain their
declared representation. Metric distances require validated metric-world
coordinates.

Miss rate requires a predeclared distance threshold, valid ground truth, and
coordinate units. It must not be retrofitted after inspecting test results.

## 13. Multimodal prediction evaluation

Report separately:

- most-likely-mode ADE/FDE using the mode selected without future ground truth;
- probability-weighted or distributional scoring;
- mode-probability calibration and negative log-likelihood where supported;
- oracle best-of-K ADE/FDE only as a labelled upper bound;
- coverage and miss rate for a predeclared threshold.

Never select a checkpoint or mode using the final target trajectory. A
probability-weighted score should account for all modes and their declared
probabilities, rather than hiding poor mode probabilities behind an oracle.

## 14. Conflict and safety evaluation

Report pedestrian-pedestrian and pedestrian-vehicle results separately.
Possible outputs include TTC-related measures, PET, minimum predicted
separation, conflict precision/recall/F1, Brier score, ECE, reliability
diagrams, alert threshold performance, warning lead time, false-alert rate,
and missed-conflict rate.

Each metric record must state:

- ground-truth source and label definition;
- coordinate space and unit;
- aggregation level;
- valid-mask and coverage policy;
- whether metric world coordinates are required.

Pixel distance and pixel-derived TTC are image-space surrogates, not
physically valid safety measures. Metric TTC/PET/separation require validated
metric geometry and synchronized timestamps.

## 15. Evaluation aggregation

Report at multiple levels:

- agent;
- trajectory/window;
- scene/recording;
- dataset;
- agent class;
- interaction type;
- forecast horizon.

Use macro averages across scenes in addition to micro averages. Provide
sample counts, valid coverage, uncertainty intervals or bootstrap units, and
the number of correlated windows. Do not let a long recording or dense scene
dominate the only headline number.

## 16. Reproducibility and provenance

The JSON template at
[`configs/phase2/split_protocol.template.json`](../../configs/phase2/split_protocol.template.json)
defines the required manifest shape. It must record dataset release,
license/provenance reference, grouping and assignments when approved,
strategy, seed, window parameters, purge, preprocessing, normalization
reference, calibration version, metric configuration, software/model
references, and hashes.

No actual assignments are present in the template. Every generated result
must identify the manifest version and exact input/configuration hashes.

## 17. Historical PCG baseline lessons

Historical ADE `0.5587965606209528` and FDE `1.1381949186325073` are retained
as oracle-assisted historical results. HBS coordinate units remain
unverified; these values must not be labelled metres. Historical collision
percentages are threshold-based counts, not calibrated probabilities.

The corrected protocol is a design, not an executed evaluator. It does not
claim corrected metrics, deployment FPS, or calibrated conflict probabilities.

## 18. Recommended end-to-end evaluation workflow

```text
verify release/provenance
  -> resolve recording/scene/sequence hierarchy
  -> assign whole groups to partitions
  -> apply chronological purge/embargo
  -> verify source-frame/track isolation
  -> generate windows within partitions
  -> fit train-only transforms
  -> train on train
  -> select with validation
  -> freeze model, thresholds, and calibration
  -> evaluate final test once
  -> report per-scene, macro, micro, and horizon results
```

Later implementation should include explicit isolation assertions and a
fixture that withholds future annotations from rollout code.

## 19. Configuration parameters

All of the following remain unapproved parameters:

- grouping and split strategy;
- proportions or explicit group assignments;
- seed and chronological ordering;
- history, horizon, interval, and stride;
- purge/embargo calculation;
- minimum track/observation requirements;
- interpolation, filtering, and quality masks;
- normalization and augmentation;
- coordinate/calibration reference;
- metrics, thresholds, coverage, and aggregation;
- bootstrap and uncertainty policy;
- software/model/preprocessing versions.

## 20. Failure cases and mitigation

| Failure | Mitigation |
|---|---|
| Random window split | Assign groups before window generation. |
| Shared recording or track | Enforce group and identity intersection checks. |
| Boundary overlap | Derive purge from complete feature/target span and filter support. |
| Future actor list in rollout | Use observed/tracker/predicted states only. |
| Oracle mode selection | Report separately; use predeclared non-oracle selection. |
| Test tuning | Freeze decisions before test access. |
| Mixed coordinates | Partition by coordinate system and calibration version. |
| Invalid metric safety claim | Report image/relative surrogate or omit metric. |
| Correlated-scene domination | Add scene-level macro results and counts. |

## 21. Unresolved research questions

1. Which dataset release and independent grouping structure will be approved?
2. Are selected scenes/cameras independent enough for the intended claim?
3. What history, horizon, stride, and purge follow from the safety warning
   objective and source sampling?
4. What calibration and coordinate version will authorize metric safety
   metrics?
5. Which conflict labels and thresholds are independently defensible?
6. How will mode probabilities be calibrated without test leakage?
7. What bootstrap unit best reflects scene and track correlation?
8. Which corrected evaluator and hidden-future tests will be implemented?

## 22. Verified findings, assumptions, and recommendations

### Verified findings

- Task 3 requires explicit coordinate space and units for numeric kinematics.
- Task 4 does not establish project-wide metric world coordinates.
- Task 5 separates offline future-dependent operations from causal inference.
- Historical Task 6 evidence records oracle selection, future actor information,
  overlapping windows, and no genuine validation loop.

### Assumptions

- Scene/recording grouping will be available in the approved release.
- A purge can be computed from source timestamps and preprocessing support.

### Recommendations

- Prefer scene/camera-independent grouped splits.
- Use chronological purging as a fallback or complementary stress test.
- Keep historical and corrected evidence in separate reports.
- Implement hidden-future and partition-isolation tests before claiming
  corrected evaluation.

## 23. References and files inspected

- `Pedestrian_Safety_Prediction_Master_Project_File_UPDATED_11_Nov_2026_Phase2_Handoff(2).md`
- `docs/phase2/task1_dataset_research.md`
- `docs/phase2/task2_dataset_feasibility.md`
- `docs/phase2/task3_agent_state_schema.md`
- `docs/phase2/schemas/agent_state.schema.json`
- `docs/phase2/task4_coordinate_systems_camera_geometry.md`
- `docs/phase2/task5_trajectory_kinematic_preprocessing.md`
- `experiments/PCG_T4_5_Epoch_Run/reports/TASK6_VALIDATION_PROTOCOL_REVIEW.md`
- `experiments/PCG_T4_5_Epoch_Run/reports/task6_corrected_validation_protocol.json`
- `experiments/PCG_T4_5_Epoch_Run/reports/task5_window_overlap_authoritative.json`

## 24. Acceptance checklist

- [x] Grouped scene/recording split is preferred over random windows.
- [x] Dataset, recording, scene, sequence, track, and window hierarchy is defined.
- [x] Windows are generated only after partition assignment.
- [x] Purge derives from the complete feature/target span and preprocessing support.
- [x] Train-only fitting and final-test isolation are explicit.
- [x] Future actor state, visibility, graphs, TTC/PET, and oracle selection are prohibited from inputs.
- [x] ADE/FDE, multimodal, conflict, calibration, coverage, and warning metrics are defined.
- [x] Macro and micro aggregation levels are required.
- [x] Reproducibility manifest fields are defined.
- [x] Historical PCG leakage findings are addressed without rewriting history.
- [x] Metric safety claims require validated metric coordinates.
- [x] No actual split, window, training, or evaluation artifact was generated.

**Task 6 recommendation:** Ready for review as a protocol design. It is not
an implemented corrected evaluator and must not be reported as leakage-free
validation until the required code, fixtures, and split artifacts exist.
