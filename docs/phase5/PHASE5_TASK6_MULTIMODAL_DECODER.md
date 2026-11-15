# Phase 5 Task 6 — Multimodal Trajectory Decoder and Forecasting Head

**Status:** `IMPLEMENTED_WITH_LIMITATIONS`  
**Scope:** direct multimodal image-space trajectory decoding and masked training losses.  
**Training:** not performed.

## Approved contract

P5.6 uses `K = 5`, `H = 6`, `F = 15`, and 10 Hz sampling. The decoder consumes the P5.5 target summary `[B, 64]`, the final observed target anchor `[B, 2]`, and an inference-time target validity mask `[B]`.

The decoder is direct and non-autoregressive. It predicts raw pixel displacement offsets:

```text
trajectory_offsets_px [B, 5, 15, 2]
trajectories_uv_px    [B, 5, 15, 2]
mode_logits           [B, 5]
mode_probabilities    [B, 5]
prediction_valid      [B, 5, 15]
```

Absolute trajectories are reconstructed by adding the final observed anchor position. No standardisation, MinMax scaling, pilot-fitted statistics, or hidden rescaling is applied.

## Loss

For each usable window, masked ADE is computed for every mode:

```text
ADE[b,k] = sum_t(mask[b,t] * ||prediction[b,k,t] - target[b,t]||_2)
           / sum_t(mask[b,t])
winner[b] = argmin_k ADE[b,k]
```

The regression term is SmoothL1 on the winning mode, summed over valid coordinates and divided by `2 * valid_timestep_count`. The batch reduction averages only usable windows. The classification term is cross-entropy over mode logits with the ADE winner as the class target:

```text
L = L_regression + classification_weight * L_classification
```

The initial classification weight is `0.1`; it is an engineering default, not an experimentally tuned value. For `K = 1`, classification loss is exactly zero. A batch with no usable target raises an explicit error. Invalid target timesteps are excluded, and masked NaNs are not evaluated.

## Evaluation

`evaluator_arrays` converts the decoder output to the existing P5.4 evaluator representation. The existing evaluator remains unchanged. The most-probable mode is the primary trajectory. `minADE@5` and `minFDE@5` remain ground-truth-selected oracle diagnostics and must not be reported as model-selected performance.

All coordinates remain image-space pixels in the existing top-left-origin convention. No metric-world, TTC, PET, collision, conflict, risk, or alerting output is introduced.

## Validation and limitations

Focused decoder and loss tests pass: 12 passed in Python 3.9.25 with PyTorch 2.2.1+cpu. The full Phase 5 suite passes with 153 passed and 1 CUDA test skipped. Evaluator compatibility was exercised through `evaluator_arrays` and the existing Phase 5 evaluator tests.

The full repository regression has 1,161 passed, 7 skipped, and 2 unrelated failures:

- `tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension` raises `UnboundLocalError` for `TTC_mask` in the legacy occupancy path.
- `tests/phase3/test_measurement_helpers.py::test_monitored_runner_records_batches_epochs_resources_and_isolates` observes generated `__pycache__` files in its fake repository snapshot.

These failures are outside P5.6 and no unrelated files were changed. No dependency installation or training was performed.

The pilot remains one PIE video with perceived-track pseudo-targets and is labelled `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`. This implementation does not establish accuracy, generalization, or real-time performance. Training, checkpoints, train-only normalization statistics, and model-selection infrastructure remain P5.7 responsibilities.
