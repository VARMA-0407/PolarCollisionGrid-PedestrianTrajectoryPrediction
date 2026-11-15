# P5.7.3 Integrated Training Step

**Status:** implemented (`COMPLETED_WITH_LIMITATIONS`: pilot-only, CPU-verified). **Label:** `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`. Report: `phase5_task7_3_report.json`. Tests: `tests/phase5/test_training_step.py`.

## What exists
`pedsafety/phase5/models/forecaster.py` — `ForecastingModel`: the unchanged P5.5 encoder and P5.6 decoder behind one `forward(encoder_input, anchor_uv_px, target_valid)`. The three arguments are history-only: the history `EncoderInput`, the raw pixel anchor (the target's last observed footpoint) and the history-derived target-validity mask. There is no argument that could carry a future value. `check_compatibility` rejects encoder / decoder / window configurations that disagree (embedding size, history / future steps, sampling rate). The decoder has its own `decoder_seed`; initialisation never touches the caller's global RNG.

`pedsafety/phase5/models/training_step.py`:
```
collated batch ──TrainingWindowDataset.encoder_input (history keys only)──► EncoderInput ─┐
raw anchor, target_valid ─────────────────────────────────────────────────────────────────┼─► ForecastingModel ─► [B,5,15,2] raw-pixel trajectories + [B,5] logits
target_future_uv_px / valid  ──(supervision tensors, never passed to the model)────────────► masked_wta_loss(prediction, supervision)
```
`make_train_batch` builds a `TrainBatch` (device-aware); `forward_loss`, `train_step` (zero-grad → forward → finite-loss check → backward → finite-gradient check → optional clipping → AdamW step → scheduler step) and `eval_step` (eval mode, no gradient, no update, restores the training flag) form the reusable interface. A non-finite loss or gradient raises `TrainingNumericsError` (`NONFINITE_LOSS` / `NONFINITE_GRADIENT`) before any parameter update. `StepResult` reports loss components, usable / excluded windows, gradient norms of encoder and decoder (before clipping), clipping, parameters without gradient, learning rate and mode diagnostics (winner counts, selected-mode and oracle-mode ADE, pairwise mode distance, entropy, mean max probability) — all training-batch, in-sample values.

## Guarantees verified by tests (17 pass, 1 CUDA test skipped: no GPU)
* Shapes: `target_summary [B,64]`, trajectories `[B,5,15,2]`, probabilities `[B,5]` summing to 1; trajectories equal `anchor + raw offsets` exactly (raw pixel anchor preserved).
* Gradient propagation: after one step every parameter of the default configuration has a gradient (`parameters_without_grad == []`), encoder (spatial + temporal) and decoder gradient norms are positive and the parameters change; scaler buffers are not parameters and do not change.
* History-only: the forward signature is exactly `(self, encoder_input, anchor_uv_px, target_valid)`; `EncoderInput` has no future / target field; shifting the future targets by +123 px leaves predictions bit-identical while the loss changes; a spy confirms the model is called with three arguments and none equals a future tensor; a meta-test shows that writing future positions into the history input *would* change predictions (so the invariance test can fail).
* Masked supervision: NaN or 1e9 under a False future mask leave the loss bit-identical and gradients finite; a window with no valid future step, or with `target_valid = False`, is excluded and counted (never zero-filled); a batch with no usable window is an error; a non-finite value under a True mask is rejected.
* Numerics: an injected infinite gradient raises `NONFINITE_GRADIENT` and leaves every parameter unchanged; NaN parameters raise; clipping caps the applied gradient norm; the cosine scheduler advances per step.
* Determinism: three steps from the same seed on one CPU thread give identical losses and parameters; overfitting a six-window batch reduces the loss by > 50 % (measured 23.9 → 6.6), showing that gradients drive learning.
* Compatibility: configuration mismatches are rejected; CPU tensors stay on CPU.

## Device compatibility (honest status)
All code paths are device-agnostic (`EncoderInput.to`, `device=` arguments, buffers registered on the modules, no hard-coded CPU tensors in the forward / loss path), and a CUDA test exists. **No CUDA device is available in this environment, so CUDA execution was not exercised**; the CUDA test is skipped, not passed.

## Limitations
CPU-verified only; the decoder predicts raw-pixel offsets (P5.6 contract, unchanged) so convergence needs a moderately high learning rate; multi-thread CPU backward passes are not bitwise reproducible (see P5.7.4); everything is pilot, in-sample and image-space.
