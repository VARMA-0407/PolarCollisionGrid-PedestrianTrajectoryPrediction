# Phase 3 Stage 1: Leakage-Safe Evaluation Foundation

## Status

**COMPLETE_WITH_LIMITATIONS.** The new chronological assignment and non-oracle
evaluation foundation is implemented and its focused tests pass. This does not
produce experimental metrics, does not alter the historical PCG evaluator, and
does not establish scene-independent HBS generalization.

## Existing implementation audit

The audit covered `pedsafety/splitting.py`, `dataset_builder.py`,
`windows.py`, `normalization.py`, `trajectory.py`, `causal_trajectory.py`,
`offline_trajectory.py`, `schema.py`, `agent_state.py`, and the Stage 5 tests.

* The existing splitter assigns complete groups (normally scenes) and
  intentionally does not perform chronological splitting.
* The existing window defaults are not inferred by this stage: `obs_len`,
  `pred_len`, `frame_stride`, and `window_stride` are explicit in
  `WindowConfig`. A window contains `obs_len` history steps followed by
  `pred_len` raw target steps. Steps are spaced by
  `frame_stride * trajectory.frame_step`; starts advance by
  `window_stride * trajectory.frame_step`.
* The repository's Stage 5 fixture uses `obs_len=4`, `pred_len=3`,
  `frame_stride=1`, and `window_stride=1`; these are fixture values, not a
  project-wide protocol.
* `FeatureEngine.update` is sequential. Velocity uses the immediately
  preceding valid feature state and acceleration may use the preceding
  velocity, so the maximum causal feature look-back is a protocol input
  (one frame for velocity-only, two frames when acceleration is required).
* `causal_trajectory.py` rejects interpolation and centred smoothing.
  `offline_trajectory.py` explicitly labels interpolation and centred moving
  average smoothing as non-causal; centred smoothing can read future samples.
* `NormalizationState.fit` accepts only windows whose partition is `train`,
  deduplicates repeated `(track, frame)` observations, and has no refit or
  partial-fit API. Validation/test windows cannot supply statistics.
* Canonical IDs are dataset-namespaced. Windows retain dataset, raw sequence,
  scene, track, frame, timestamp, source-index, coordinate-space, unit, and
  calibration provenance.
* The approved agent-state schema remains read-only. No schema change was
  needed.

Unresolved values such as a project-wide observation/prediction horizon,
physical sampling rate, and HBS metric calibration are not guessed. They must
be supplied by a future approved protocol.

## Chronological split methodology

`pedsafety.chronological.RawSequence` validates finite timestamps, non-negative
integer frames, strict frame order, and strict timestamp order before any
window is considered. `SplitBoundaries` requires explicit increasing
`train_end` and `validation_end`, with an optional explicit `test_end`.
Periods are half-open:

* train: `(-inf, train_end]` for complete window end checks;
* validation: `[train_end, validation_end]`;
* test: `[validation_end, test_end]` (or unbounded when `test_end` is omitted).

The implementation does not randomize or fall back to group/random splitting.
It enumerates raw sequence windows using the configured observation length,
prediction length, frame stride, and window stride. A candidate is retained
only when all requested raw frames are contiguous and its complete dependency
interval is inside one period. Boundary-crossing and dependency-overlap
candidates are returned in `ChronologicalSplitResult.excluded`.

For a single-scene input, the result carries the warning:
“temporal generalization within one scene only; scene-independent generalization
is not established.” This is the required HBS interpretation.

## Purge derivation

The purge is not a fixed duration. The caller supplies the actual causal
feature look-back in native frames through `feature_lookback_frames`.
`WindowConfig.frame_stride` and `frame_step` determine the raw window grid;
the dependency interval begins at the actual timestamp
`lookback` native frames before the observation start when that history exists.
The target end is the last prediction timestamp.

Thus a window is eligible for partition `P` only when:

```
dependency_start >= P.start
and target_end <= P.end
```

The returned per-sequence purge duration is the observed timestamp difference
between observation start and dependency start. This is derived from the raw
timestamps, not an arbitrary seconds constant. A candidate touching a
boundary through its dependency interval is recorded as
`purged_dependency_overlap`; another incomplete candidate is recorded as
`crosses_split_boundary`. Insufficient sequences, duplicate sequence IDs,
invalid timestamps, invalid frames, and non-monotonic rows fail explicitly.

For the current causal engine, use one look-back frame when only velocity is
consumed and two when acceleration is consumed. Offline centred smoothing is
not represented as a one-sided causal purge; it remains explicitly offline and
must not enter real-time evaluation.

## Window-generation rules

The new module assigns raw windows but does not create model tensors. Existing
`pedsafety.windows.generate_windows` remains responsible for feature masks,
coordinate identity, target validity, and provenance. The two layers are
intentionally separated so a future builder cannot silently convert a rejected
boundary candidate into a shortened window.

## Hidden-future-input threat model

The Stage 1 tests cover:

1. chronological ordering and boundary integrity;
2. crossing/dependency-overlap rejection and purge boundary behavior;
3. invalid timestamps and insufficient sequence lengths;
4. explicit HBS single-scene warning;
5. rejection of offline centred smoothing in the causal configuration;
6. mutation of hidden future labels without changing causal history tensors;
7. training-only normalization and rejection of mixed train/non-train fitting;
8. mutation of test labels without changing model-score selection;
9. invalid/missing prediction accounting;
10. refusal to report image-space errors as metres.

Existing Stage 5 tests additionally verify truncated-history equivalence,
causal processing, offline separation, and train-only statistics.

## Evaluation selection policy

`pedsafety.evaluation` separates candidate selection from metric calculation.
The only supported selection rules are deterministic `first` and
model-provided `model_score`. `best_of_k`, `oracle`, and ground-truth-dependent
rules raise an error. ADE/FDE are calculated only for finite, shape-compatible
`(steps, 2)` arrays and optional validity masks. Missing or invalid cases are
retained with a status and reason.

The interface records the declared coordinate space. `physical_units=True` is
allowed only for a validated metric-labelled space (`metric`, `metres`, or
`meters`). Image-space errors are therefore not labelled as metres, and no
physical conflict/TTC metric is implemented.

## Files created and modified

Created:

* `pedsafety/chronological.py`
* `pedsafety/evaluation.py`
* `tests/phase3/conftest.py`
* `tests/phase3/test_stage1_evaluation_foundation.py`
* `docs/phase3/PHASE3_STAGE1_EVALUATION_FOUNDATION.md`

No existing `pedsafety` or original PCG source/data/checkpoint/result file was
modified by this stage. The agent-state schema was not changed.

## Exact validation commands and results

* `pytest -q tests\phase3\test_stage1_evaluation_foundation.py` — **11 passed**.
* `pytest -q tests\phase3\test_stage1_evaluation_foundation.py tests\task7\test_stage5_split.py tests\task7\test_stage5_windows.py tests\task7\test_stage5_normalization.py` — **72 passed**.
* `pytest -q tests\phase2\test_ttc_masks.py` — **collection blocked**:
  `ModuleNotFoundError: No module named 'torch'`. No package was installed and
  the original `TTC_mask` implementation/test was not changed.
* Pylance/editor problem checks for the new Python modules and tests — no
  diagnostics.

## Known failures and unresolved issues

* The environment does not have PyTorch, so the pre-existing TTC mask test
  could not run. Its known defect remains untouched.
* HBS has one scene; chronological HBS evaluation is temporal generalization
  within that scene, not scene-independent generalization.
* HBS coordinate units, calibration, original FPS, and physical metric validity
  remain unresolved.
* The historical PCG best-of-five evaluator and historical preprocessing are
  unchanged. The new interface prevents oracle selection only when the new
  interface is used.
* The repository-wide observation/prediction horizon and the exact model
  feature look-back must be approved per experiment. The new API requires the
  caller to state them rather than inventing values.

## Stage 1 acceptance checklist

- [x] Chronological splitting occurs before window generation.
- [x] Explicit train/validation/test periods are required.
- [x] Boundary-crossing windows are rejected and recorded.
- [x] Purge is derived from configured windowing and explicit feature look-back.
- [x] Purge boundary behavior is tested.
- [x] Hidden-future-input invariance is tested.
- [x] Train-only normalization is verified.
- [x] Oracle-based test selection is rejected in the new evaluator.
- [x] Related existing `pedsafety` regression tests remain compatible.
- [x] Known TTC mask failure is documented and unmodified.
- [ ] Production-ready evaluator claim — intentionally not made until coordinate,
  horizon, and protocol assumptions are approved.

