# P5.7.4 Pilot Training Harness

**Status:** implemented and run (`COMPLETED_WITH_LIMITATIONS`). **Label:** `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT` — one PIE video, one scene group, every loss is **in-sample**. Report: `phase5_task7_4_report.json`. Code: `pedsafety/phase5/models/{training_harness,checkpoint}.py`, `scripts/phase5_train_pilot.py`, `scripts/phase5_validate_training.py`; config `configs/phase5/training_default.json` (`phase5-training-config-1.0.0`); tests `tests/phase5/test_training_harness.py`.

## Harness
AdamW (decoupled weight decay on weight matrices only; biases and normalisation parameters are not decayed), cosine learning-rate schedule stepped per optimizer step (`T_max` = total steps, floor 1 % of the peak), gradient-norm clipping, configurable epochs / batch size / learning rate / weight decay / betas / epsilon, strict configuration validation (unknown or missing keys, invalid values and policies are rejected). Seeds: the order of epoch *e* is `default_rng([seed, e])` (so a resumed run continues exactly), torch global seed, separate fixed encoder / decoder initialisation seeds. Checkpoints (`torch.save`, loaded with `weights_only=True`) hold model, optimizer and scheduler state, global step, CPU RNG state and a provenance mapping (training / encoder / decoder / window configuration hashes, window-run manifest hash, scaler statistics hash and split identity); a checkpoint carries two hashes (file bytes, canonical model-state hash) and an existing checkpoint is never overwritten. Resumption verifies provenance (core configuration, scaler, dataset, seed, batch size) and refuses a different schedule horizon (`RESUME_SCHEDULE_MISMATCH`). Each run directory contains the resolved configuration, scaler statistics, a deterministic JSONL log (no wall-clock fields), checkpoints, `environment.json` and `run_manifest.json` (deterministic `data_artifacts` separated from `run_metadata`: git state, versions, timings, peak memory).

## Policies for the pilot (recorded in every manifest)
* `split_policy = PILOT_ALL_WINDOWS_AS_TRAIN_DEVELOPMENT_ONLY`: no independent split exists, so none is invented and no random window split is made (it would be a correlated, misleading "validation"). The train-only scaler is fitted on all 2,327 windows because they are the only training partition.
* `checkpoint_selection = FINAL_EPOCH_ONLY`: the final epoch is the only designated checkpoint; the epoch-10 checkpoint exists for resumption and inspection. **Limitation:** best-checkpoint selection would require held-out data; the only available "validation" windows are strongly overlapping neighbours of the training windows of the same video, so any selection on them would be indefensible. Nothing is selected.
* Hyperparameters are a-priori engineering defaults (20 epochs, batch 64, peak learning rate 0.003, weight decay 0.01, clip 10, cosine to 1 %). A single two-value, 2-epoch probe of the learning rate (0.001 vs 0.003, looking only at in-sample training loss and at gradient flow) preceded the run; no other search was done and nothing was tuned for accuracy.
* `torch_threads = 1`: with several CPU threads some backward reductions are not bitwise reproducible (measured: identical forward, 12 parameter gradients differing between identical runs). With one thread two runs — and a resumed run — are byte-identical, at the cost of wall-clock time.

## Pilot experiment (in-sample, development only)
`experiments/phase5_task7_training/20261115T190117Z_pie_video_0004/` and the identical-configuration rerun `…_determinism_rerun/`. 2,327 windows, 84 tracks, batch 64 → 37 steps/epoch, 20 epochs = **740 steps**, 172,699 parameters (encoder 144,128, decoder 28,571), CPU, 1 thread, seed 20261116, no step needed clipping.

| Item | Value |
|---|---|
| Window run (P5.3) | manifest SHA-256 `3369ea80…1e8`; window configuration `65eebeed…1034` |
| Training / encoder / decoder configuration (canonical SHA-256) | `6a98708b…bf40e` / `15dbb985…78c1` / `ff2a38d9…69ea` |
| Train-only scaler statistics | SHA-256 `ed5fd682…7fe169`, file `cc270aa6…0ce723`; `fit_partition = train`, split identity `PILOT_ALL_WINDOWS_AS_TRAIN_DEVELOPMENT_ONLY`, 2,327 windows, minimum valid count per feature 126,517 |
| Epoch-10 checkpoint | file SHA-256 `d0f0f417…37d2`, model state `78a8d920…da6b54` (2.17 MB) |
| **Final (epoch-20) checkpoint** | file SHA-256 `40aff17f1a40eb2da24d0e0b63245eecefa5bba4e3f6e44c272dc07f3ab71b44`, model state `9c953c1adfc81c53d686f42431d42c8e5197db18f7c5530342bc49b17d15d4c7` |
| Log | `train_log.jsonl` `882bf9e3…fe37c` (148 step records + 20 epoch records) |
| Runtime / memory (one machine, offline) | total 668 s (scaler fit 4.3 s, batch collation 150 s, forward + backward 511 s); peak working set 583 MB |
| Environment | Python 3.9.25, torch 2.2.1+cpu, numpy 1.26.4, Windows 10 build 26300, 1 torch thread, no CUDA |

In-sample training diagnostics (window-weighted, per epoch; these are training-batch values during training, not evaluation):

| epoch | loss | selected-mode ADE (px) | oracle-mode ADE (px) | pairwise mode distance (px) | winners per mode |
|---|---|---|---|---|---|
| 1 | 29.16 | 52.8 | 51.3 | 5.6 | 1003 / 286 / 189 / 241 / 608 |
| 5 | 19.56 | 40.7 | 34.0 | 30.3 | 744 / 541 / 180 / 349 / 513 |
| 10 | 14.91 | 36.9 | 25.8 | 39.6 | 597 / 549 / 191 / 476 / 514 |
| 20 | 11.63 | 31.4 | 20.3 | 42.3 | 643 / 458 / 191 / 503 / 532 |

The loss decreased monotonically; all five modes keep winning windows (no complete mode collapse in-sample), the modes spread (5.6 → 42 px), the mean maximum mode probability ranged 0.48–0.69 over the epochs and ended at 0.49 (uncalibrated, close to uniform at the end); no step needed gradient clipping. These are in-sample optimisation facts, not evidence of forecasting skill.

## Reproducibility and validation (all verified)
* Two runs with identical configuration produced **byte-identical** data artifacts: configuration, scaler statistics, log and both checkpoints (SHA-256 equal).
* `scripts/phase5_validate_training.py --compare-with … --recompute-scaler`: **PASS 17/17** (artifact and environment hashes, labels / claims, configuration hashes, window-run re-verification, scaler provenance, **independent numpy recomputation of all node and edge statistics and valid counts**, log parse / finiteness / epoch records, every checkpoint's hashes, metadata, scheduler horizon and scaler buffers, absence of future fields in the model input, bitwise equality with the rerun); the rerun validates 16/16.
* Unit-level: a resumed run reproduces the continuous run's final checkpoint byte-for-byte; provenance, seed, schedule and tamper mismatches are rejected (tests).

## Limitations
In-sample, one video, pseudo-targets; no held-out data, no checkpoint selection, no hyperparameter selection; WTA training can collapse modes on one scene (diagnostics are logged, quality is unvalidated); CPU single-thread only (CUDA code path exists but was not run: no GPU here); timings are offline measurements, not real-time claims.
