# P5.7.2 Train-Only Scaler Integration

## Scope

P5.7.2 adds fitting, persistence, validation, and encoder application of
feature statistics. It does not add a training loop, optimizer, checkpoint
selection, validation-based model selection, neural evaluation, or risk
calibration.

## API and fitting semantics

`pedsafety.phase5.training_scaler.fit_feature_statistics` accepts explicit
collated history batches and requires `fit_partition="train"`. The
`TrainingWindowDataset.fit_scaler_statistics` helper accepts an explicit,
non-empty, unique sequence of dataset indices and batches only those windows.
It preserves the run-manifest source identity and requires a caller-supplied
split identity.

Statistics are computed in the exact P5.5 derived feature spaces:

- node order: `rel_u`, `rel_v`, `vel_u`, `vel_v`, `speed`, `heading_sin`,
  `heading_cos`, `box_h`, `frames_since_prev_obs`;
- edge order: the fourteen fields in P5.3 `EDGE_FEATURES`.

The node reference is the final observed target-history position, matching the
encoder. Values are included only where their corresponding P5.5 validity mask
is true. A non-finite value under a valid mask is rejected. Missing values
remain excluded; they are not interpolated, forward-filled, or converted to
valid observations. Population standard deviation is used. Constant features
use scale `1.0`, and every feature must have at least one valid observation.

Future targets, future masks, anchors, predictions, and metadata are not read
as fitting inputs. Anchors and future targets remain raw image-space pixels.

## Persistence and application

`TrainOnlyFeatureStatistics.save()` writes deterministic JSON containing:

- schema version;
- node/edge feature names and order;
- means, positive scales, and valid counts;
- `fit_partition` and split identity;
- exact source identity;
- fitting configuration;
- canonical SHA-256 over the payload excluding the hash field.

`load()` verifies the schema, feature order, dimensions, finite/positive
statistics, hash, and optional expected source/split identities. Tampering or
incompatible provenance is rejected. `feature_scaler()` returns the existing
P5.5 `FeatureScaler` with source
`TRAIN_PARTITION_STATISTICS`; the encoder applies the statistics without
changing P5.5 internals. Raw batches are not mutated.

## Limitations

The pilot contains one video/group and is development-only and
non-independent. This task does not define a grouped train/validation/test
split, fit decoder-target scaling, or provide model-quality evidence.

## Validation

Focused tests are in
`tests/phase5/test_training_scaler.py`. The pilot smoke uses the committed
P5.3 artifact and fits only two explicitly selected windows; it is not an
independent validation. Exact test counts and command results are recorded in
`phase5_task7_2_report.json`.


## Integration review (consolidated P5.7 pass, 16 November 2026)

The existing implementation was reviewed and integrated with the P5.7.3-P5.7.5 pipeline. Result: **no defect found in the scaler logic**; one
repository-integration issue was fixed.

| Review item | Evidence |
|---|---|
| Training-only fitting | `fit_partition` must be exactly `train`; `validation` / `test` are rejected; exactly the supplied batches determine the statistics (hash differs when a batch is added); the pilot harness fits on all 2,327 windows, the only training partition, and records split identity `PILOT_ALL_WINDOWS_AS_TRAIN_DEVELOPMENT_ONLY` |
| Derived feature ordering | `tests/phase5/test_training_scaler_review.py::test_derived_feature_space_and_order_equal_the_encoders`: the encoder's scaled node inputs equal `(derived value - mean) / scale` computed independently in numpy for the nine features, masked entries are exactly 0, and the fitted data is standardised per feature |
| Validity-mask-aware statistics | values under a False mask (including 9e9 garbage) do not change the statistics hash; NaN under a True mask is rejected |
| Constant features | scale 1.0, scaled constant = 0; mean preserved |
| Deterministic serialization and SHA-256 | repeated fits give identical bytes and hash; load -> save is byte-identical; key order does not change the canonical hash |
| Provenance verification | wrong source identity, wrong split identity, a tampered value (hash mismatch), a non-train partition and a re-ordered feature list are rejected; the neural evaluator additionally requires the statistics to equal the scaler buffers stored in the checkpoint |
| Raw anchors and future targets unchanged | anchors, future targets and masks are byte-identical before and after fitting and after a scaled encoder pass; mutating them does not change the statistics |
| Independent recomputation | `scripts/phase5_validate_training.py --recompute-scaler` recomputes every node and edge statistic and valid count in plain numpy on the pilot (PASS) |

**Fix:** `training_scaler.py` imported `torch` at the Phase 5 package root, which made the Phase 2 Task 7 dependency guard fail (torch is
permitted only under `pedsafety/phase5/models`, see `requirements-phase5.txt`). `feature_scaler()` now builds the scaler through the new
`FeatureScaler.from_arrays(...)` classmethod in `pedsafety/phase5/models/scaling.py`; the public API, statistics format, hash and the original
six tests are unchanged.

Pilot application: the statistics used for training are `ed5fd6824995c056b161564fb40ce93b0c96320de03a8672eba12af36e7fe169`
(file `cc270aa6562931f5c66b909dcc5941ba20869101862ec91c35e924c81b0ce723`), fitted on all 2,327 pilot windows (development only).
