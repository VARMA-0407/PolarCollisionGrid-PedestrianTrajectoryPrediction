# Phase 2 Task 5 - Trajectory and Kinematic Preprocessing Design

**Status:** COMPLETED_WITH_LIMITATIONS  
**Date:** 11 November 2026  
**Scope:** Research and design only. No preprocessing pipeline was
implemented, no dataset was downloaded, and no model was trained or
evaluated.

## 1. Executive summary

This document defines a preprocessing design for converting tracked
observations into temporally consistent agent histories and kinematic features
for GATv2, TCN, multimodal trajectory prediction, and conflict forecasting.
It supports two deliberately separate modes:

1. **Offline dataset preparation**, where bounded interpolation and
   non-causal smoothing may use future observations if explicitly labelled.
2. **Causal real-time inference**, where only the current observation and
   past state may be used.

The required invariants are timestamp-aware differences, explicit coordinate
space and units, preserved observation/interpolation/validity masks, explicit
uncertainty, and strict prevention of future-observation leakage. Metric
world-space features are emitted only when Task 4 calibration and scale
evidence are validated. Otherwise the system remains in image or explicitly
relative/projective coordinates.

No parameter in this design is claimed to be optimal. Gap limits, filter
parameters, resampling interval, history length, and prediction horizon
require dataset-specific experiments after provenance and calibration are
approved.

## 2. Scope and dependencies

### In scope

- Track continuity and observation validity.
- Missing observations and interpolation.
- Smoothing and timestamp-aware kinematics.
- Heading, resampling, sequence construction, quality, and uncertainty.
- Offline and causal online preprocessing architectures.
- Downstream representation requirements.

### Out of scope

- Detector, tracker, coordinate-transform, homography, GATv2, TCN, decoder,
  TTC, PET, risk model, or deployment implementation.
- Dataset acquisition, access applications, train/validation/test split
  generation, training, and evaluation.

### Required dependencies

- Task 3 common schema:
  [`agent_state.schema.json`](schemas/agent_state.schema.json)
- Task 3 semantics:
  [`task3_agent_state_schema.md`](task3_agent_state_schema.md)
- Task 4 geometry and calibration limitations:
  [`task4_coordinate_systems_camera_geometry.md`](task4_coordinate_systems_camera_geometry.md)

Task 3 is treated as the current schema contract. Task 4 is accepted with
limitations; it does not establish metric world coordinates for the project.

## 3. Input data contract

Each input row represents one agent at one source timestamp or frame. The
following Task 3 fields are used without changing their schema semantics:

| Group | Fields | Processing meaning |
|---|---|---|
| Identity | `track_id`, `agent_id`, `sequence_id`, `scene_id`, `agent_class` | Track scope and grouping keys; never join tracks across scenes without evidence. |
| Time | `timestamp`, `frame_index` | Primary ordering and finite-difference basis. Timestamps are seconds from the sequence time origin. |
| Image geometry | `bbox_xyxy`, `bbox_center`, `footpoint`, `image_x`, `image_y` | Pixel-space observations. Boxes use `[x_min, y_min, x_max, y_max]`. |
| World geometry | `world_x`, `world_y`, `world_coordinate_system`, `world_coordinate_unit`, `world_position_valid` | Usable only when the Task 4 calibration/scale contract is verified. |
| Kinematics | `velocity_x`, `velocity_y`, `speed`, `acceleration_x`, `acceleration_y`, `kinematic_coordinate_space`, `kinematic_position_unit`, `kinematic_time_unit`, `heading` | Derived or source kinematics with explicit space and units. |
| Quality | `detection_confidence`, `tracking_confidence`, `visibility`, `occlusion_status`, `observation_valid` | Quality and validity evidence; not one interchangeable score. |
| Provenance | `interpolation_flag`, `source_dataset`, `source_label` | Distinguishes observed from derived/interpolated data and retains source labels. |

The preprocessing layer may maintain internal sidecar fields such as
`position_observed`, `position_interpolated`, `position_filtered`,
`timestamp_valid`, `kinematic_valid`, and `uncertainty`. These are not silent
changes to the Task 3 schema. If persisted, they require a reviewed extension
or a separate preprocessing provenance record.

## 4. Track continuity and observation-validity policy

### Track lifecycle

- **Initiation:** accept a new track only after the tracker/dataset identity
  and first valid observation are recorded.
- **Continuation:** retain a track while identity continuity and timestamps
  remain credible.
- **Short occlusion:** keep identity state but mark the current observation
  missing; do not pretend a predicted position was observed.
- **Long occlusion:** terminate or quarantine the track according to a
  configurable gap policy. Re-identification must not silently merge IDs.
- **Exit:** record the final valid timestamp and distinguish field-of-view
  exit from tracker failure where evidence permits.
- **Fragmentation/ID switch:** preserve source IDs, record the event, and
  require an explicit re-linking policy before combining fragments.

### Observation classes

| Class | Meaning | Can train as measured position? | Required flags |
|---|---|---:|---|
| Observed | Valid source annotation or current detector/tracker measurement. | Yes, subject to quality filters. | `observation_valid=true`, `interpolation_flag=false`. |
| Filtered observed | An observed value after a declared causal/offline filter. | Yes, but retain raw value and filter provenance. | Observation remains distinct from interpolation. |
| Interpolated | Value reconstructed between observations. | Only with explicit policy and mask. | `interpolation_flag=true`. |
| Missing/invalid | No defensible position or invalid geometry. | No. | `observation_valid=false`; preserve missingness. |
| Causally estimated | State predicted from current/past values only. | Not a measurement. | Separate internal estimate flag/provenance. |

Reject or mark invalid when boxes have fewer than four values, inverted
ordering, non-finite values, impossible dimensions, or coordinates outside
the declared image bounds without a documented clipping policy. Footpoints
must remain distinguishable from box centres and must not be invented for
fully occluded agents.

Duplicate timestamps, duplicate track rows, conflicting observations, and
non-monotonic frame indices require deterministic quarantine rules. They must
not be silently averaged.

Very short tracks can be retained for detection statistics but should be
excluded from velocity, acceleration, heading, or long-horizon forecasting
when the minimum evidence window is not met.

## 5. Missing observations and interpolation policy

### Method comparison

| Method | Suitable conditions | Advantages | Limitations and leakage |
|---|---|---|---|
| No interpolation | Sparse, uncertain, or safety-critical data. | Preserves truth and missingness. | Encoders need masks or missing-value handling. |
| Linear interpolation | Short gaps with valid endpoints and near-linear motion. | Simple and auditable. | Uses the future endpoint offline; poor for abrupt motion and invalid over long gaps. |
| Constant velocity | Short gaps with stable timestamp-aware velocity. | Physically interpretable and useful for bounded extrapolation. | Error grows with manoeuvres and noisy velocity. Future endpoint is unavailable online. |
| Kalman estimate | A declared motion/noise model and sequential filtering. | Provides causal state and uncertainty. | Model mismatch, tuning sensitivity, and smoothing must be distinguished from filtering. |

Offline interpolation may use both sides of a gap only when the record is
explicitly marked offline/interpolated. The real-time path must never use the
future endpoint to reconstruct the current state. Long gaps must remain
missing or terminate the usable history.

The maximum interpolation gap is a configurable duration or frame count,
selected from an experiment using held-out occlusion simulations and
downstream performance. It must not be chosen as a universal arbitrary
constant. A gap beyond the selected limit receives a missing mask rather than
silent filling.

Illustrative offline rule:

```text
if valid_left and valid_right and gap <= max_offline_gap:
    position = interpolate(left, right, timestamp)
    interpolation_flag = true
else:
    position = missing
    observation_valid = false
```

Illustrative causal rule:

```text
if current_measurement_is_valid:
    update_filter(current_measurement, timestamp)
elif elapsed_since_last_valid <= max_causal_gap:
    estimate_from_past_state_only()
else:
    emit_missing_state()
```

## 6. Smoothing-method comparison

| Method | Position noise | Velocity/acceleration | Lag | Online suitability |
|---|---|---|---|---|
| Moving average | Reduces high-frequency noise. | Differentiation can remain unstable; window causes delay. | Usually significant and symmetric offline. | Trailing window only, with lag. |
| Savitzky-Golay | Preserves local polynomial shape. | Often improves derivatives on regular samples. | Symmetric version uses future values. | Causal version needs separate design and has boundary effects. |
| Kalman filter | Model-based noise reduction. | Produces state/covariance when model is adequate. | Tunable responsiveness. | Strong candidate for causal operation. |
| Exponential smoothing | Simple adaptive reduction. | Derivatives still require care. | Controlled by smoothing factor. | Causal and inexpensive. |

The provisional default is a timestamp-aware Kalman filter or causal
exponential filter for real-time state, retaining raw observations and
uncertainty. Offline analysis may compare Savitzky-Golay and non-causal
smoothing as labelled alternatives. The smoothest trajectory is not
necessarily the most accurate trajectory, especially during sudden stops,
turns, or evasive motion.

Filter choice, process noise, measurement noise, window, polynomial order,
and causal delay remain experimentally unresolved.

## 7. Velocity-estimation methodology

For position `p_i` at time `t_i`, the timestamp-aware backward difference is:

```text
v_i = (p_i - p_(i-1)) / (t_i - t_(i-1))
```

Use the coordinate space of `p`. Image-space velocity is pixels/second.
World-space velocity is declared physical units/second only when world
calibration and scale are validated. It must never be labelled metres/second
by inference from pixel motion, camera height, GPS, or homography scale.

Rules:

- Require `dt = t_i - t_(i-1) > 0` and finite.
- Reject or quarantine duplicate/non-monotonic timestamps.
- Do not calculate velocity across an invalid or over-limit missing gap unless
  the interpolation/estimate policy explicitly permits it.
- Retain whether velocity is measured, filtered, interpolated, or causal.
- Use central differences only in offline mode, because they use a future
  sample relative to `t_i`.
- Propagate position uncertainty into velocity uncertainty where possible.

For irregular timestamps, use actual `dt`; do not substitute the nominal FPS.
Velocity should be null when supporting positions or timing are unreliable.

## 8. Acceleration-estimation methodology

With valid timestamp-aware velocities:

```text
a_i = (v_i - v_(i-1)) / (t_i - t_(i-1))
```

Image-space acceleration is pixels/second². World-space acceleration uses the
declared physical unit/second² only after metric geometry is verified.

Differentiation amplifies position and velocity noise. The recommended order
is to compare:

1. smoothing/filtering followed by differentiation;
2. a state-space model estimating position and velocity jointly; and
3. offline polynomial derivatives as an explicitly non-causal baseline.

Acceleration is invalid when velocity support, timestamps, or gap quality is
invalid. It must not be fabricated to fill a feature vector. The method,
window, causal status, and units belong in preprocessing provenance.

## 9. Heading-estimation methodology

For a non-zero displacement or velocity vector:

```text
heading = atan2(v_y, v_x)
```

The sign and interpretation of `v_y` depend on the declared coordinate
system. Every dataset adapter must specify:

- reference axis;
- positive rotation direction;
- valid angle range, such as `[-pi, pi)`;
- whether heading is image-space, relative/projective, or world-space.

Wrap angle differences to the selected range before comparing headings.
Require a minimum speed or displacement over a configured evidence window.
For a stationary or near-zero-speed pedestrian, heading should be null or
invalid rather than assigned an arbitrary direction. Sudden direction
changes should retain quality indicators rather than being erased by
unvalidated smoothing.

Image-space and world-space heading are not interchangeable. The selected
world-frame axes and handedness must match the Task 4 calibration manifest.

## 10. Temporal resampling and sequence construction

Source datasets may use different frame rates, timestamp precision, and
missing-frame policies. The pipeline should use a parameterized target
interval `Delta_t`, not assume all data is 30 FPS or 10 Hz.

Recommended sequence construction:

1. Sort each track by validated timestamp.
2. Align multiple agents to a common scene time grid.
3. Resample only within valid track spans and permitted gaps.
4. Preserve observed, interpolated, filtered, missing, and causal masks.
5. Recompute or transform velocity, acceleration, and heading consistently
   with the target time basis.
6. Attach uncertainty and source/calibration provenance.
7. Construct observed-history and future-target windows without overlap that
   leaks future labels into input features.

Resampling changes derivative estimates and uncertainty. It does not create
new information. The target interval, history length, forecast horizon, and
minimum valid-observation ratio must be selected against the intended safety
warning time and each dataset's temporal resolution. Final values are not
selected by this design.

## 11. Coordinate-system and calibration compatibility

| Available evidence | Permitted output | Restrictions |
|---|---|---|
| Image coordinates only | Pixel trajectories and pixel/second kinematics. | No metre labels; image heading uses image axes. |
| Relative/projective transform | Explicit relative/projective trajectories. | Preserve transform identity and do not call it metric. |
| Validated metric world coordinates | World trajectories and declared physical kinematics. | Require calibration ID/version, frame, units, valid region, residuals, and uncertainty. |
| Calibration unavailable or insufficient | Image-space or missing output. | Set world validity false; do not fabricate projection. |

Do not mix trajectories from incompatible cameras, scenes, map frames, or
units. A calibration sidecar must identify the transform direction, source
image dimensions, resize/crop/padding transform, projection region,
reprojection residuals, and uncertainty. The Task 3 schema is not silently
extended by this design.

## 12. Confidence, validity, and uncertainty handling

Quality dimensions must remain separate:

- detection confidence;
- tracking confidence;
- observation validity;
- visibility and occlusion status;
- interpolation/causal-estimate status;
- projection/calibration validity;
- position uncertainty;
- velocity and acceleration reliability;
- heading evidence quality.

Downstream models should receive masks and quality features, not only a
single unexplained confidence score. Low-quality states may be excluded from
edge construction or loss terms, while their missingness remains available to
the model. Safety metrics should report coverage and invalid-mask rates
alongside values.

Uncertainty should be propagated approximately where possible. For a finite
difference, position uncertainty and timestamp uncertainty affect velocity
uncertainty; repeated differentiation generally increases uncertainty.
Calibration uncertainty should remain distinct from detector/tracker
uncertainty.

## 13. Offline versus real-time processing architecture

### Offline dataset preparation

```text
raw annotations/tracks
  -> schema and geometry validation
  -> continuity and ID-switch analysis
  -> optional bounded offline interpolation [may use future observations]
  -> labelled smoothing [may use future observations]
  -> timestamp-aware kinematics
  -> quality, validity, and uncertainty annotations
  -> parameterized resampling
  -> observed-history/future-target sequence construction
```

Every future-dependent operation is marked as offline. Interpolated and
smoothed values must remain traceable to their source observations.

### Causal real-time inference

```text
current detections/tracks + past state
  -> causal validation
  -> causal filter
  -> timestamp-aware causal kinematics
  -> validity, quality, and uncertainty checks
  -> current agent state
  -> GATv2 graph and TCN history
```

No future frame, future annotation, bidirectional smoother, centred
difference, or future endpoint may enter this path. If current evidence is
insufficient, emit a missing/invalid state or a clearly marked causal
estimate.

## 14. GATv2 and TCN compatibility

### GATv2 node features

Possible node features include normalized image/world position in one declared
space, velocity, acceleration, heading encoding, class, visibility,
observation/interpolation masks, age, and quality features. Normalization
statistics must be fitted on training data only and remain space-specific.

### GATv2 edge features

Edges may use relative position, relative velocity, distance, bearing, time
since observation, and uncertainty. Edge features must be computed only
between agents sharing a compatible coordinate frame and time index. If only
image coordinates are available, the edge is an image-space relation, not a
physical separation.

### TCN histories and multimodality

TCN inputs are fixed or parameterized temporal windows with masks and
time-delta features. Variable numbers of agents require padding plus an agent
mask, set/graph batching, or another explicit variable-cardinality strategy.
Multimodal decoders should receive observed histories only; future labels are
targets, not input features.

Conflict forecasting can use quality-aware relative motion, but metric TTC,
PET, and minimum separation require validated metric geometry or must be
labelled as image/relative surrogates.

## 15. Recommended preprocessing pipeline

| Order | Operation | Classification |
|---:|---|---|
| 1 | Preserve source IDs, timestamps, dimensions, and provenance. | Required invariant |
| 2 | Validate schema, boxes, footpoints, timestamps, and coordinate metadata. | Required invariant |
| 3 | Analyze continuity, gaps, duplicates, fragmentation, and occlusion. | Required invariant |
| 4 | Select image/relative/world coordinate mode from evidence. | Required invariant |
| 5 | Apply bounded offline interpolation only when explicitly enabled. | Provisional design recommendation |
| 6 | Apply a declared causal or offline filter. | Provisional design recommendation |
| 7 | Estimate timestamp-aware velocity, acceleration, and heading. | Required invariant |
| 8 | Attach masks, confidence dimensions, uncertainty, and provenance. | Required invariant |
| 9 | Resample to a parameterized time grid. | Dataset-specific parameter |
| 10 | Construct leakage-free observed-history and future-target windows. | Required invariant |
| 11 | Run downstream coverage and quality audits before modelling. | Provisional design recommendation |

No step is experimentally validated by this document.

## 16. Adjustable configuration parameters

The following must remain configuration rather than hard-coded claims:

- target time interval and timestamp tolerance;
- maximum offline interpolation gap;
- maximum causal prediction gap;
- minimum track length;
- minimum speed/displacement for heading;
- filter type and noise/window parameters;
- derivative method;
- valid-observation ratio;
- projection residual and valid-region thresholds;
- uncertainty and quality masks;
- history length, forecast horizon, and stride;
- agent padding/max-cardinality policy;
- class and occlusion handling;
- normalization statistics and clipping policy.

Values must be recorded with each preprocessing run and selected through
held-out experiments without using future test information.

## 17. Failure cases and mitigation strategies

| Failure | Mitigation |
|---|---|
| Invalid/inverted box or footpoint | Reject, quarantine, and report; do not silently repair. |
| Duplicate or non-monotonic timestamp | Preserve raw row, quarantine conflict, and require deterministic resolution. |
| Short occlusion | Use bounded, labelled interpolation offline or causal estimate online. |
| Long occlusion | Retain missing mask or terminate track; do not silently fill. |
| ID switch | Report fragment and require explicit re-linking evidence. |
| No calibration | Use image/relative output and set world validity false. |
| Mixed coordinate frames | Partition by frame and use adapters; never concatenate directly. |
| Derivative noise | Compare filters/state models and retain uncertainty. |
| Near-zero heading | Emit null/invalid heading. |
| Future leakage | Enforce causal data-access boundaries and audit feature timestamps. |
| Resampling artifacts | Retain source timestamps, masks, and target-grid metadata. |
| Projection outside valid region | Emit invalid projection mask and preserve image state. |

## 18. Unresolved research questions

1. Which dataset and release will be approved for acquisition?
2. What are the exact source timestamp, frame-rate, and identity semantics for
   the selected release?
3. What interpolation gap preserves useful coverage without hiding
   uncertainty?
4. Which causal filter and noise model best balance responsiveness and
   derivative stability?
5. What minimum evidence window is appropriate for heading and acceleration?
6. Which target interval, observation history, and forecast horizon match the
   warning-time objective?
7. What calibration sidecar and residual thresholds will authorize world
   kinematics?
8. How should uncertainty enter graph edges, losses, TTC/PET estimates, and
   alert thresholds?
9. How will ID switches and cross-camera identity boundaries be evaluated?
10. Which quality masks should exclude samples from training versus only
    weighting their losses?

## 19. Verified findings, assumptions, and recommendations

### Verified findings

- The Task 3 schema separates image/world metadata and requires explicit
  kinematic coordinate space and units when numeric kinematics exist.
- The Task 3 schema requires `world_position_valid` and does not permit
  numeric world coordinates when it is false.
- Task 4 documents image/camera/world distinctions, calibration limitations,
  homography direction, and the need for a versioned calibration sidecar.
- The repository's existing Phase 2 evidence does not establish HBS metric
  units or a validated PCG world-coordinate transform.

### Assumptions

- A bottom-centre footpoint is a useful candidate ground-contact proxy.
- A state-space filter may be a useful causal baseline.
- A parameterized resampling grid will simplify multi-agent batching.

These are design hypotheses, not validated results.

### Recommendations

- Preserve raw observations and all quality masks.
- Use image-space-only operation until calibration and metric scale are
  independently verified.
- Compare no-interpolation, bounded interpolation, and causal filtering in
  controlled experiments.
- Keep offline and online preprocessing implementations and evidence separate.
- Record every configuration and calibration version with generated features.

## 20. References and repository files inspected

Repository files:

- `Pedestrian_Safety_Prediction_Master_Project_File_UPDATED_11_Nov_2026_Phase2_Handoff(2).md`
- `docs/phase2/task1_dataset_research.md`
- `docs/phase2/task2_dataset_feasibility.md`
- `docs/phase2/task3_agent_state_schema.md`
- `docs/phase2/schemas/agent_state.schema.json`
- `docs/phase2/task4_coordinate_systems_camera_geometry.md`
- `experiments/PCG_T4_5_Epoch_Run/reports/TASK7_HBS_COORDINATE_PROVENANCE_FINAL_AUDIT.md`
- `experiments/PCG_T4_5_Epoch_Run/reports/TASK6_VALIDATION_PROTOCOL_REVIEW.md`

The official dataset references remain those recorded in Tasks 1, 2, and 4.
No release was downloaded or locally inspected for this design.

## 21. Acceptance checklist

- [x] Executive summary and scope are defined.
- [x] Actual Task 3 fields and semantics are used.
- [x] Track continuity and observation validity are specified.
- [x] Missing observations, interpolation, and leakage are distinguished.
- [x] Smoothing methods are compared.
- [x] Timestamp-aware velocity and acceleration equations and units are given.
- [x] Heading reference axis, rotation direction, range, and coordinate space
  are required.
- [x] Temporal resampling and sequence construction are parameterized.
- [x] Image, relative/projective, and metric-world modes are separated.
- [x] Confidence, validity, masks, and uncertainty remain distinct.
- [x] Offline and causal online workflows are separate.
- [x] GATv2, TCN, variable-agent, multimodal, and conflict uses are covered.
- [x] Required invariants, recommendations, dataset parameters, and unresolved
  choices are classified.
- [x] No future observation enters the real-time design.
- [x] No metric coordinates are fabricated.
- [x] No source, schema, Task 1-4 report, master, dataset, checkpoint,
  prediction artifact, or experiment artifact was modified.
- [x] No dataset was downloaded and no access application was submitted.
- [x] No code tests were created for this documentation-only task.

**Task 5 recommendation:** Ready for review with limitations. Approval should
wait for dataset approval, timestamp/identity evidence, calibration evidence,
and experiments selecting gap, filter, resampling, and horizon parameters.
