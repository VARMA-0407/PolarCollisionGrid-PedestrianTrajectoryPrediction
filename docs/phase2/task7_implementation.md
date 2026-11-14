# Phase 2 Task 7 - Implementation Notes

**Status:** IN PROGRESS - Stages 1-5 accepted; Stage 6 implemented and Stage 6.1 hardening applied, awaiting review (see "Stage log", "Stage 6" and "Stage 6.1"). Phase 2 is NOT declared complete; that needs a separate exit review.
**Package:** `pedsafety/` (independent of the original PCG code).
**Tests:** `tests/task7/`.
**Schema:** `docs/phase2/schemas/agent_state.schema.json` is read, never modified.

## Stage log

| Stage | Scope | State |
|---:|---|---|
| 1 | Schema validation, adapter interface, CSV adapter, synthetic generator, raw-row parsing | Implemented; tested (see below) |
| 2 | Agent-state construction and validation | Implemented; tested; awaiting review (see "Stage 2") |
| 3 | Coordinate transformations | Implemented; tested; awaiting review (see "Stage 3") |
| 4 | Preprocessing and kinematics | Implemented; tested; awaiting review (see "Stage 4") |
| 5 | Leakage-aware splitting and windows | Implemented; tested; awaiting review (see "Stage 5") |
| 6 | Integration pipeline, CLI, visualisation, read-only HBS smoke test | Implemented; tested; awaiting review (see "Stage 6") |
| 6.1 | Output-directory policy, dependency declarations, version | Implemented; tested; awaiting review (see "Stage 6.1") |

## Stage 1 components

| Module | Purpose |
|---|---|
| `pedsafety/schema.py` | Load the approved schema (Draft 2020-12), report its SHA-256, validate records, list *all* violations with field paths, reject NaN/inf (which JSON Schema `number` accepts). |
| `pedsafety/errors.py` | `SchemaLoadError`, `SchemaViolation`, `ConfigError`, `ParseError`. |
| `pedsafety/io/base.py` | `AnnotationAdapter` ABC, `RawObservation`, `ParseIssue`/`IssueCode`, `ParseResult`. |
| `pedsafety/io/csv_adapter.py` | `CsvAdapterConfig` + `CsvAnnotationAdapter`. |
| `pedsafety/io/synthetic.py` | Deterministic synthetic scenes, CSV writer, matching adapter config. |

## Units, coordinates and assumptions (Stage 1)

- `RawObservation.timestamp` is **seconds**. It comes from a `timestamp` column
  (`timestamp_unit` is `s` or `ms`) or from `frame_index / fps`; `fps` is never
  guessed.
- `pos_x`/`pos_y` carry **no assumed unit**. If position or box columns are
  mapped, the config must declare `position_semantics` (for example
  `image_pixel`) and may declare `position_unit`; otherwise construction fails.
- Position declaration (checked when the config is built):

  | `position_semantics` | `position_unit` | `position_coordinate_system` |
  |---|---|---|
  | `image_pixel` | must be `pixel` | optional |
  | `world_coordinate` | must be `m`, `cm`, `mm` or `ft` | required (names the frame) |
  | `unverified_planar` | must be **None** | optional |

  Any other semantics string, a mismatched pair (for example `image_pixel` with
  `m`, `world_coordinate` with `pixel`, `unverified_planar` with any unit), or a
  declaration with no position/bbox columns mapped raises `ConfigError`. These
  are declarations by the configurer. They do not make a world position valid:
  `world_position_valid` stays a Stage 3 calibration-gate decision.
- Velocity columns (`vel_x`, `vel_y`) are passed through **without** a declared
  unit or space. Stage 2 must not emit them as schema kinematics unless the
  space/unit contract can be established.
- No two canonical fields may map to the same CSV header; the `ConfigError`
  lists the header and the conflicting fields. Header names must be non-empty
  strings.
- The synthetic generator rejects non-finite or non-positive `fps`, non-finite
  or negative noise, a confidence outside [0, 1], non-numeric/bool values, and a
  non-integer `n_frames`, naming the argument.
- `bbox_xyxy` is passed through unchanged. Ordering (`x_min <= x_max`, ...),
  duplicate or non-monotonic timestamps, class mapping and schema conversion
  are **Stage 2**; the parser neither repairs nor reorders nor deduplicates.
- Bad rows never abort a non-strict parse: they become `ParseIssue`s with line,
  column, code and raw value. `strict=True` raises `ParseError` on the first bad
  row. File/config problems (missing mapped column, duplicate header, bad UTF-8,
  empty file) raise `ConfigError`.
- Synthetic data are **image pixels** (pixel, pixel/s). Footpoint is the
  bottom-centre of the box. They say nothing about any real dataset or metric
  scale.

## Stage 2: agent-state construction and validation

Modules: `pedsafety/validation.py` (issue codes, structured report, production
bounding-box validator, centre/footpoint) and `pedsafety/agent_state.py`
(`AgentStateConfig`, `build_agent_states`, `RecordProvenance`). The approved
schema is used unchanged; every record is validated against it before return.

### Identity policy (follows the Task 3 examples; new convention is only the prefix rule)
- `track_id = "<dataset>:<sequence>:<source_track>"`, `sequence_id = "<dataset>:<sequence>"`,
  `scene_id = "<dataset>:<scene>"`, where `<dataset>` is `source_dataset` stripped and
  lower-cased. The same source track id in two sequences (or datasets) therefore gives
  different canonical ids. `source_dataset` itself is kept as given.
- `agent_id` is null by default (no cross-scene identity is invented). The opt-in policy
  `source_track_id` sets it to the raw source track id; it is still only meaningful within
  its sequence.
- Two distinct raw (sequence, track) identities that yield the same canonical `track_id`
  (possible when ids contain `:`) are quarantined as `track_id_collision`.

### Classes
Schema `agent_class` enum: pedestrian, vehicle, bicyclist, motorcyclist, other, unknown.
The default map (configurable, case/space-insensitive) sends car/truck/bus/vehicle to
`vehicle`, bike/bicycle to `bicyclist` and motorcycle to `motorcyclist`. **`bike` -> `bicyclist`
is an assumption** (bike may mean motorbike in some sources). The schema cannot hold the
car/truck/bus distinction; `source_label` preserves it. Unmapped or missing labels are
issues; there is no default class. Map targets are checked against the schema enum.

### Box features
`bbox_xyxy` must be four real finite numbers with x_min < x_max and y_min < y_max
(zero-size boxes are rejected). Centre is the midpoint; footpoint is the bottom-centre
`((x_min+x_max)/2, y_max)` (image y grows downward). Boxes are never reordered or clipped,
and a missing box stays null.

### Coordinates and validity
- `world_position_valid` is **always false** in Stage 2. The schema cannot distinguish a
  declared world coordinate from a calibration-verified one, so declared world values and
  unverified planar values are **not written to the canonical record**; they are kept in
  the provenance sidecar (`withheld_position_status`, `withheld_position_xy`) and counted
  in `report.counters`. Stage 3 must consume the sidecar and make the validity decision.
- `image_x/y`, `image_coordinate_system` and `image_coordinate_unit` are populated only for
  `image_pixel` geometry. A named coordinate system is required (the schema needs one); none
  is invented. A box with non-image semantics is rejected (`bbox_semantics_mismatch`).

### Kinematics
Raw `vel_x/vel_y` are withheld (sidecar `raw_velocity`) unless the config declares a
`VelocityContract` (image space, pixel, seconds), in which case velocity and the three
`kinematic_*` fields are emitted. World-space contracts are refused. Speed, acceleration and
heading are never computed in Stage 2.

### Integrity checks and modes
Checked per input: identifiers, frame index (int >= 0), timestamp (finite, >= 0 seconds from
the sequence origin per Task 3), class, box, confidence, coordinate declaration, position and
velocity completeness. Checked across the batch, in input order and without sorting:
duplicates on (dataset, sequence, track, frame) - **all members quarantined**, never one
kept; frames that go backwards within a track (`out_of_order_frame`); timestamps that do not
increase with frame index; canonical-id collisions. A quarantined observation produces no
record but always produces a structured issue carrying its input index and source line.
Strict mode (`strict=True`) raises `ObservationValidationError` containing the full report if
any issue exists. Lenient mode returns records for clean observations plus the report. Output
order equals input order; identical input and config give identical output.

### Records are observed values
Every Stage 2 record is an observed value (`observation_valid=true`, `interpolation_flag=false`);
the sidecar marks `state_origin='observed'`. Missing-frame placeholder records are not created.
Optional fields the source cannot supply (visibility, occlusion, tracking confidence) stay null.

### Stage 2 limitations
No check that `pos_x/pos_y` agree with the box footpoint; no check that one sequence maps to a
single scene; velocity contract is a configurer's declaration, not verification; no image-size
clipping or bounds check (no image dimensions are available); observations are held in memory.

## Stage 3: coordinate transformation and calibration

Modules: `pedsafety/geometry.py` (homography maths, DLT fit, polygons, tolerances),
`pedsafety/calibration.py` (raster, manifest, eligibility, raster adaptation) and
`pedsafety/projection.py` (Stage 2 integration). The schema and Stage 1/2 behaviour are unchanged.

### Coordinate concepts kept apart
Image pixels (a defined raster) -> projective/road-plane coordinates (a homography output with
no inherent scale) -> metric coordinates (only when the manifest's evidence passes the software
checks). A matrix being finite, invertible or well conditioned says nothing about physical meaning.
Camera intrinsics, camera height or pitch (e.g. the PIE `calibration_data.json` fields), GPS and
agent trajectories are never inputs to a plane homography here; no extrinsics or ground plane are invented.

### Calibration manifest (`CalibrationManifest`, direction `image_to_plane` only)
Holds: provenance (id, version), the calibration `RasterSpec` (id, size, image coordinate-system
name, pixel-centre convention, pixel domain `as_captured`/`undistorted`/`unknown`, top-left origin),
the 3x3 homography, destination frame name/unit/origin/axes/handedness, scale basis, valid region
polygon, control points (role fit/holdout; source must be survey, metric_map, known_distance or
other_independent), a held-out error threshold, and a `VerificationClaim` (status, by, on, reference).
Construction rejects: bad shape, non-finite, singular/ill-conditioned matrices, other directions,
non-physical units, regions outside the raster / non-simple / zero-area / crossing the vanishing line,
control points outside the raster, duplicate ids, and any control-point source such as an agent trajectory.
A canonical-JSON SHA-256 identifies each manifest in all outputs. `from_dict` is strict.

### Eligibility (`evaluate_metric_eligibility`) is NOT physical verification
It reports `manifest_eligible` after ALL software checks (no short-circuit): named frame, physical unit,
documented origin/axes/handedness, scale basis, valid region, known pixel domain, >= 4 non-degenerate
fit correspondences (rank test on the DLT design), >= 1 held-out point distinct from fit points, all
control points inside the region, a declared threshold, fit and held-out residuals RECOMPUTED from the
matrix within that threshold, and a complete independent-verification claim. The code cannot confirm
that claim or that a survey was real, so `physical_verification_established_by_software` is always
False and every projection record carries `eligibility_basis = "software_manifest_checks_only"`.

### Raster / preprocessing consistency
Without a documented `RasterTransform` the observed raster must equal the calibration raster in id,
size, coordinate-system name, pixel convention and pixel domain. With one (crop -> scale -> pad,
including the integer-pixel-centre correction) the predicted size must match the observed size within
`size_rounding_px`, the pixel domain and convention must be unchanged, and the effective homography is
`H' = H A^-1` (Task 4). Resizing cannot change distortion state; no distortion model is implemented.

### Projection (`transform_agent_states`)
Uses the record's `image_x/image_y` (Task 3 footpoint convention; `footpoint` selectable; optional
cross-check tolerance). Only `image_pixel` records are transformed; unverified-planar and declared-world
records are skipped untouched and Stage 2's withheld declared values are never treated as calibration.
Per point it checks: finite input, inside the observed raster, inside the calibrated raster/region,
stable homogeneous denominator, same side of the vanishing line as the region, finite output. Failures
get a status and reason; nothing is clipped or repaired. Inputs are never mutated.
Canonical `world_*` fields are written ONLY if the manifest is eligible AND
`TransformConfig.promote_eligible_to_canonical=True` AND the point projected AND the result validates
against the schema. Otherwise they stay null with `world_position_valid=false`, and the planar
coordinates, frame, unit, `coordinate_kind` (`projective_unverified_scale` or
`metric_eligible_by_manifest_checks`), calibration id/version/hash and effective-homography hash live
in the `ProjectionRecord` sidecar.

### Tolerances (all in `NumericTolerances`, all configurable)
Condition number <= 1/sqrt(eps); relative denominator floor sqrt(eps); rank tolerance sqrt(eps)
(numerical degeneracy only; quality is judged by held-out residuals); region boundary tolerance 0.0;
resize rounding 1.0 px. Eligibility minima (4 fit, 1 held-out) are in `EligibilityPolicy`.

### Stage 3 limitations
Real calibrations (PIE, inD, TITAN, HBS) are NOT tested; PIE's published metadata does not contain a
road-plane homography. Declared-world datasets (e.g. source-provided metric maps) need their own
gate, not built here. Lens undistortion is not implemented. Residuals use Euclidean distance in the
destination unit; no uncertainty propagation or covariance is produced. A planar assumption is a
property of the scene that the software cannot check. The HBS smoke test reads a copy of 600 rows and
confirms only that declarations stay `unverified_planar` with no unit and no world values.

## Stage 4: trajectory preprocessing and kinematic features

Modules: `pedsafety/trajectory.py` (config, grouping, coordinate-mode selection, feature engine,
result types, window preparation), `pedsafety/causal_trajectory.py` (CAUSAL streaming + batch) and
`pedsafety/offline_trajectory.py` (OFFLINE / NON-CAUSAL interpolation and smoothing). Stages 1-3 are
unchanged. Discrepancies found between earlier reports and the actual code: none affecting Stage 4.

### Three layers kept apart
RAW = Stage 2 records, Stage 2 sidecar and Stage 3 projections (never modified; tests deep-compare).
PROCESSED = a frame grid with an `origin` mask (`observed` / `imputed` / `missing`), raw coordinates `x`,`y`
(never overwritten) and the positions actually differentiated `xs`,`ys` (equal to raw unless smoothing is on).
DERIVED = displacement, dt, velocity, speed, acceleration, heading, each with a validity STATE string.
Unavailable values are NaN plus a state (`insufficient_samples`, `gap_before`, `frame_identity_changed`,
`position_unavailable`, `missing_position`, `nonpositive_dt`, `non_finite_dt`, `dt_too_large`,
`imputed_support_disallowed`, `below_min_speed`, `non_finite_result`, `previous_velocity_unavailable`).
Zero is never substituted.

### Construction rules
Tracks are grouped by canonical `track_id` (dataset + sequence + source track), in input order; nothing is
sorted. Per track: duplicate frames quarantine every copy (as in Stage 2); a frame that goes backwards, a timestamp
that does not increase, an invalid time value or an off-grid frame (`frame_step`) is quarantined with a
`TrajectoryIssue`. Missing frames, per-step dt statistics and irregular intervals are reported in `SamplingReport`.
Short tracks (`min_valid_observations`) are flagged or excluded WITH an issue (`short_track_policy`).

### Coordinate modes (explicit, never mixed)
| Mode | Source | Units reported | Physical? |
|---|---|---|---|
| `image_pixel` | `image_x/y` | pixel, pixel/s, pixel/s^2 | no |
| `projective_unverified` | Stage 3 `plane_xy` (any manifest) | projective_plane_units (+/s, /s^2) | no |
| `metric_manifest_eligible` | canonical `world_*` with `world_position_valid` AND a promoted, eligible Stage 3 projection whose frame/unit/xy agree | the declared unit, e.g. m, m/s | yes, `physical_basis = software_manifest_checks_only` |
| `declared_world_unverified` | Stage 2 sidecar `withheld_position_xy` | `declared:<unit>` | no |
| `unverified_planar` | Stage 2 sidecar | unverified_planar_units | no |

A sample missing from the requested mode is marked unavailable; it is never taken from another mode. A change
of frame name, unit, calibration id/hash or agent class inside a track breaks differencing and interpolation.
Metric features without Stage 3 provenance, or with a record/projection disagreement, are refused. Declared-world
observations are not discarded: they are usable (labelled unverified, non-physical) and never promoted.
Heading is `atan2(vy, vx)` in the mode's own axes, wrapped to [-pi, pi); image headings are not world headings.
Stationary samples get `below_min_speed` and NaN heading (velocity is still the measured, valid zero).

### Kinematics
Backward differences with the ACTUAL dt: `v_i = (p_i - p_{i-1}) / (t_i - t_{i-1})`, `a_i = (v_i - v_{i-1}) / (t_i - t_{i-1})`.
Computed only between immediately consecutive grid frames of one coordinate identity; a gap, identity change, non-positive
or non-finite dt, or (by default) any imputed support makes the value unavailable.

### OFFLINE versus CAUSAL
CAUSAL (`causal_trajectory.py`): streaming `CausalFeatureStream.update` (one record at a time, state from the past only)
and `process_causal`. No interpolation (absent frames stay absent), only `causal_ema` smoothing, offline options rejected at
configuration. The module does not import `offline_trajectory`. Late/duplicate stream samples are rejected without touching state.
OFFLINE (`offline_trajectory.py`): bounded linear interpolation (`max_interpolation_gap_frames`, default 0 = off, optional seconds
limit) and `offline_centered_moving_average`. Output type `OfflineTrajectoryFeatures`, `view = OFFLINE_NON_CAUSAL`. Interpolated
rows are marked `imputed` with the method string; an anchor must be an observed row with a valid coordinate and the same identity,
so interpolation never crosses a long gap, an unavailable-coordinate row, a track or an identity change.
`require_causal(obj)` raises `NonCausalInputError` for any non-causal object; real-time consumers should call it.

### Leakage controls and Stage 5 integration requirement
Verified by tests: causal rows for times <= t are bit-identical after deleting or arbitrarily changing every later row (with
and without EMA), and a control test shows the OFFLINE pipeline does change under the same perturbation. `prepare_windows`
slices ONE trajectory into history (positions + causal features) and target (RAW positions); it refuses OFFLINE views unless
`allow_offline_view=True` (then `history_may_use_future=True`), requires consecutive frames, and attaches dataset / sequence /
track ids. Stage 4 computes NO normalisation statistics and knows nothing about splits. Stage 5 MUST assign whole recordings/scenes
to partitions first, then build trajectories and windows inside each partition, and fit any normalisation on the training
partition only. Interpolation and windows are per track, so they cannot cross a partition boundary once tracks are partitioned.

### Stage 4 limitations
Only backward differences (no Kalman, no Savitzky-Golay). Gap handling is by frame count on a regular grid; irregular frame
spacing beyond `frame_step` is quarantined, not resampled. No uncertainty propagation. Heading axes of non-image modes are
unverified. Metric features inherit every Stage 3 limitation (single static calibration, planarity, unverified claim). A moving
camera needs per-frame calibration, which changes the calibration hash and therefore breaks differencing, so such tracks yield
no metric velocity. Real PIE/inD/TITAN data are NOT tested; the HBS smoke test reads a copy of 1500 rows.

## Stage 5: leakage-aware splitting, window generation and train-only normalisation

Modules: `pedsafety/splitting.py` (grouping policy, split plan, validation, isolation proof),
`pedsafety/windows.py` (authoritative split-aware windows), `pedsafety/normalization.py` (train-only
statistics) and `pedsafety/dataset_builder.py` (orchestration + report). Stages 1-4 and the schema are
unchanged. Stage 4's `prepare_windows` is NOT used by Stage 5 (Stage 5 windows are frame-based, carry
masks/provenance and are generated per partition); it remains as a split-agnostic helper.

### What the records can and cannot say about grouping
Records carry `source_dataset`, `sequence_id` and `scene_id` (dataset-namespaced) only; there is no
recording or camera field. Therefore the grouping unit is an EXPLICIT `GroupingPolicy` with no default:
`scene` (the scene id), `sequence` (only if the user can state why sequences are independent) or
`recording` (a mandatory `{(source_dataset, raw_sequence_id): recording_id}` map; every sequence must be
mapped). `independence_basis` (why the groups are independent) is required free text stored in the manifest.
Sequence ids are never assumed unique or independent. Group keys include the dataset
(`scene:syn:sc0`), so equal ids in different datasets never collide.

### Failures instead of guesses (`SplitError.code`)
`MISSING_GROUPING_INFO` (unmapped sequences), `INSUFFICIENT_GROUPS` (< 3 groups, e.g. one scene; we never split inside a
group), `TRACK_SPANS_GROUPS`, `SEQUENCE_SPANS_GROUPS` (the same video under two scenes/groups), `CONFLICTING_RECORDING_MAP`,
`DUPLICATE_RECORDING_MAP`, `DUPLICATE_ASSIGNMENT`, `CONFLICTING_ASSIGNMENT`, `UNKNOWN_GROUP`, `INCOMPLETE_SPLIT`,
`INVALID_PARTITION`, `EMPTY_PARTITION`, `SCENE_CROSSES_PARTITIONS`, `PARTITION_LEAKAGE`.

### Assignment
Whole groups only. Default `hash_greedy_v1`: groups are ordered by `sha256("<seed>|<group>")` (no RNG library, independent of input order)
and placed one at a time into the partition with the largest relative deficit versus the target ratios, weighted by observation count
(an empty partition always wins first, so all three are non-empty). Explicit `(group, partition)` assignments must cover every group.
For units finer than a scene, `isolate_scenes=True` merges groups that share a scene id; otherwise scenes may span partitions and
the report warns. The plan is a hashable manifest (`SplitPlan.to_dict()/sha256()`: seed, ratios, unit, basis, assignments, counts).
Chronological (within-group) splitting is NOT implemented; it needs a derived purge and a protocol decision.

### Leakage controls, in order
1. group each record; 2. assign groups; 3. run Stage 4 on each partition's records ONLY (so interpolation, smoothing and trajectory
construction cannot read another partition); 4. build windows inside each partition; 5. re-verify, on the generated windows, that no
group, track, sequence or source frame appears in two partitions (`verify_window_isolation`); 6. fit normalisation from the training
windows only. The builder raises if a trajectory ends up in a foreign partition.

### Windows (`WindowConfig`, `SplitWindow`)
Frame-based: step `i` is at `start + i*frame_stride*frame_step`; starts advance by `window_stride*frame_step` from the first frame.
Features stay native-rate. Targets are RAW future positions; history holds `xs, ys, vx, vy, speed, ax, ay, heading` with per-channel
availability masks (NaN when unavailable), a per-step validity mask, origin / imputation masks, frame indices, timestamps and source
record indices (original list). Provenance: dataset, raw sequence / scene / track, group id and unit, partition, view, coordinate mode,
unit status, physical flag and basis, frame, calibration id/hash, units. Rejections are counted by reason: `unusable_target`,
`last_history_step_unusable`, `insufficient_history`, `gap_in_native_frames`, `mixed_coordinate_identity`,
`required_feature_unavailable`. By default every history step must be usable; `min_valid_history_steps` allows only LEFT padding, and
native frames from the first usable step to the last target step must all exist (`require_contiguous_native_frames`). Windows never mix
coordinate identities (frame name, unit, calibration, agent class). OFFLINE views need `allow_offline_view=True` in both the window
config and the builder, are flagged `history_may_use_future`, and imputed rows are unusable unless `allow_imputed`.

### Normalisation
`NormalizationState.fit` accepts training windows only (anything else raises `LEAKAGE_GUARD`); there is no refit or partial-fit.
Statistics are per `norm_key` (dataset, coordinate mode, unit status, unit, frame, calibration, manifest hash) from the HISTORY values of
UNIQUE (track, frame) observations that are available and finite. Default channels are the linear ones; heading (circular), ids,
timestamps, frame indices and labels are never normalised. Constant channels get scale 1.0 and a flag (or an error by policy);
channels with too few values are flagged and refuse data; an unseen key raises (no fallback); unavailable values stay NaN. The state
(with the split-plan hash, seed, grouping unit, train groups, training window-id digest, window/trajectory settings) serialises to JSON
and reloads exactly. Positions in targets use the same position statistics; `denormalize_xy` inverts them.

### Stage 5 limitations
Grouping reliability is a human claim (`independence_basis`); the software cannot verify that scenes are independent, only that its
own bookkeeping is consistent. Group-count weighting uses observation counts, so the achieved ratios can be far from the targets with few
groups (reported, not enforced). Windows from a track shorter than the span produce none. No chronological/purge protocol, no
class-balanced splitting, no stratification. Normalisation statistics are per key, so a rare coordinate space in validation/test fails
loudly rather than being approximated. Real PIE/inD/TITAN grouping is NOT tested (their scene/recording semantics are unverified).

## Stage 6: integration pipeline, CLI, visualisation and HBS smoke test

Stage 6 adds NO new science and changes NO earlier module or the schema. It wires Stages 1-5 in their documented
order, writes deterministic outputs, records provenance and maps failures to stable exit codes.

Modules: `pedsafety/pipeline_config.py` (strict JSON config), `pedsafety/pipeline.py` (ingest, features, build, HBS smoke, plot,
run manifest, error mapping), `pedsafety/viz.py`, `pedsafety/cli.py` + `pedsafety/__main__.py`,
`pedsafety/io/synthetic_corpus.py` (multi-scene synthetic CSV + config). Example config: `configs/phase2/pedsafety_hbs_smoke.json`.
Tests: `tests/task7/test_stage6_config.py`, `test_stage6_pipeline.py`, `test_stage6_hbs_viz.py`, fixtures `stage6_fixtures.py`.

### Running it (pcg-baseline interpreter, repo root)
```
python -m pedsafety synthetic --out <dir>                       # invented multi-scene CSV + config
python -m pedsafety validate-config <config.json>
python -m pedsafety ingest   <config.json> --out <dir>          # parse + agent states (+ projection) reports only
python -m pedsafety features <config.json> --out <dir>          # descriptive causal features, NO split, not a training input
python -m pedsafety build    <config.json> --out <dir>          # split, windows, train-only normalisation, run manifest
python -m pedsafety plot     <config.json> --out <dir> [--which trajectories,kinematics,projection,split,windows]
python -m pedsafety hbs-smoke configs/phase2/pedsafety_hbs_smoke.json --out <dir> [--max-rows N]
```
Common flags: `--input` (override the CSV, read-only), `--strict` (fail on the first parse/validation issue; it only tightens the config),
`--fail-on-issues` (exit 3 on any rejected row or warning), `--debug` (traceback).
Exit codes: 0 ok | 1 smoke check failed | 2 config error | 3 input/data error (also an empty training set) | 4 split refused |
5 calibration error | 6 other pedsafety error | 7 output directory refused (Stage 6.1) | 70 unexpected internal error. On failure `<out>/error.json` holds stage, code, message, details.

### Config (JSON, `config_version` 1, unknown keys are errors, relative paths resolve against the config file)
Sections: `input` (adapter), `agent_state`, `calibration` (optional: manifest file, observed raster, optional raster transform,
`promote_eligible_to_canonical`), `trajectory`, `grouping`, `split`, `windows`, `view`, `normalization`. `grouping`, `split` and `windows`
have NO defaults and are required only by `build`/split plots. Recording-level grouping is accepted only through an explicit
`recording_map` [[dataset, raw_sequence_id, recording_id], ...]; the tool never derives it.

### Outputs of `build` (all JSON is deterministic: sorted keys, LF, no timestamps)
`run_manifest.json` (tool versions, config hash, input SHA-256, schema SHA-256, counts, warnings, split plan hash, normalisation hash,
claims block, SHA-256 of every other output), `config_resolved.json`, `parse_report.json`, `validation_report.json`,
`projection_report.json` (only with calibration), `split_plan.json`, `split_report.json`, `normalization.json`,
`windows_<partition>.npz` + `windows_<partition>_index.json`. `.npz` files embed timestamps, so reproducibility is asserted on the
array CONTENT digest (`output_array_digests`) and array equality, not on npz bytes. Figures are not claimed byte-reproducible.
No wall-clock time is written anywhere.

### Safeguards verified by tests
Input CSV is hashed before and after and an in-run change fails with `INPUT_CHANGED_DURING_RUN`; two runs give byte-identical JSON and equal
arrays; a seed change changes the plan hash; windows never cross partitions and targets follow history; single-scene input exits 4
(`INSUFFICIENT_GROUPS`) and writes no windows; missing grouping exits 2; the offline view needs explicit consent; `claims` never report
independent verification, and a manifest-eligible calibration is labelled `software_manifest_checks_only`; metric mode without
promotion yields no metric windows (exit 3, `NO_TRAINING_WINDOWS`).

### HBS smoke test (`hbs-smoke`, read-only)
Hashes the source, optionally works on a COPY of the first N rows, forces `unverified_planar` mode, and CHECKS (does not assume) that:
declarations stay `unverified_planar` with no unit; no `world_*` or `image_*` field and no `world_position_valid=true` exists; no feature is
physical and no unit is metric; there is exactly one scene; a scene-level split attempt is refused with `INSUFFICIENT_GROUPS`; the source
hash is unchanged. It reports class counts and observed dt but claims no metric coordinates, units, calibration, recording independence,
partition, provenance or licence (`not_claimed` list). A negative-control test shows the checks can fail.
HBS has one scene (`scene_id` 0), so NO HBS windows, partitions or normalisation are produced.

### Visualisation
`trajectories`, `kinematics`, `projection` (needs calibration), `split`, `windows`. One panel per coordinate identity (never mixed);
axis labels come from the data's own unit status (`pixel`, `unverified_planar_units`, `m` only for a manifest-eligible identity, banner
"software manifest checks only"); image y is inverted. A figure that cannot be drawn is reported under `skipped` with a reason.

### Stage 6 limitations
No real PIE/inD/TITAN data and no real calibration were run. The pipeline holds all records in memory and the full HBS smoke takes about a
minute. Only the scene/sequence/recording grouping of Stage 5 exists (no chronological split). `features` is descriptive and must not feed
training. Plots are not golden-image tested (labelling rules are tested through pure functions plus a PNG-validity check). No model is trained
and no metric-world safety interpretation is made. Phase 2 exit is a separate review.

## Stage 6.1: output-directory policy, dependencies and version

Changes are limited to Stage 6 modules plus two new modules/files; Stage 1-5 processing code and the schema are untouched.
(`pedsafety/__init__.py` changed only its version string and docstring, as requested.)

### Output-directory policy (fixes the stale-output defect found by the Phase 2 exit audit)
Audit reproduction: a successful `build`, then a failing `build` into the SAME directory left the old `run_manifest.json` and windows
beside the new `error.json`. Now (`pedsafety/output_dir.py`, applied to `ingest`, `features`, `build`, `plot`, `hbs-smoke` and `synthetic`):
- A command writes only into a directory that does **not exist** or is **empty**. Any existing entry (hidden files included) or a path that is
  a file is refused with exit code **7** (`OUTPUT_DIR_NOT_EMPTY` / `OUTPUT_PATH_NOT_DIRECTORY`) BEFORE anything is read or written.
- Nothing is deleted or overwritten. A refused directory is not touched at all, not even by `error.json`; the structured error goes to stderr.
- An existing EMPTY directory is accepted (explicitly tested).
- Use a fresh directory per run, e.g. `--out runs/2026-11-12_a`. A directory left by a failed attempt (it contains `error.json`) cannot be reused;
  pick another path. Cleaning old results is the user's decision and is never done by the tool.
- Complete-run contract: a directory holds the valid result of a finished run only if its completion file exists (`run_manifest.json` for
  ingest/features/build, `plot_manifest.json`, `hbs_smoke_report.json`) AND `error.json` is absent (`output_dir.is_complete_run`). Completion files are
  written last. All JSON is written atomically (temp file + replace), so a reader never sees a half-written file.
- `error.json` is written only into a directory the failing run owns (new/empty at start). `--fail-on-issues` is now decided before any result file is
  written (before, a full result set and manifest were written and then the run was reported as failed).
- Library callers: `write_synthetic_corpus` applies the same policy; `write_error(..., owned=False)` refuses non-empty directories.
Remaining edge cases: the check is not atomic against two processes starting at the same moment in the same new directory; a crash mid-write can
leave partial files, but never a completion file; a file-system error while writing maps to exit 70 (internal), not a dedicated code.

### Dependencies and version
`requirements-pedsafety.txt` (runtime: `-r requirements.txt` plus `jsonschema>=4.0,<5`, tested with 4.25.1 only) and `requirements-dev.txt`
(`pytest>=8.0,<9`, tested with 8.4.2). numpy and matplotlib are already pinned in `requirements.txt`, which was left unchanged (it is the original
PCG file). `pedsafety.__version__` is `0.6.1.dev0` (PEP 440 development version). Python 3.9 compatible. Nothing was installed.

## Recorded design safeguards (apply to later stages)

1. **Purge is a conservative estimate.** The additive formula
   `(H+F-1)*dt + lookback + forward support + tolerance` will be treated as an
   upper-bound estimate. Stage 5 will additionally verify the *actual*
   dependency-interval overlap of every window pair across a partition boundary
   and fail on any overlap, rather than trusting the formula.
2. **Causal estimates are not representable in the approved schema.** The schema
   has only the boolean `interpolation_flag` (offline interpolation) and
   `additionalProperties: false`. This is checked by a test
   (`schema_represents_causal_estimate()` is `False`). The
   observed / causal-estimate / offline-interpolated distinction will therefore
   be carried in a **separate provenance sidecar** keyed to each record, and the
   schema will not be modified. A causal estimate must never be written as an
   observation.
3. **Software eligibility is not physical verification.** A calibration
   manifest passing the software completeness checks (Stage 3) will be reported
   as `manifest_eligible`, a statement only that the manifest's own claims are
   complete and internally consistent. It will never be reported or documented
   as physically verified calibration; that needs independent evidence outside
   this code.

## Not verified in Stage 1

Anything involving real PIE/inD/TITAN files, real calibration, metric units,
or performance. Stage 1 exercises only synthetic and hand-written CSV files.
