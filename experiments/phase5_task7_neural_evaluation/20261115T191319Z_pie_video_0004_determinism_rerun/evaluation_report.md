# Phase 5 P5.7.5 neural evaluation — PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT, IN-SAMPLE

**The neural model was trained on every window evaluated here** (one PIE video, one scene group, overlapping windows, perceived-track pseudo-targets). These numbers are in-sample development diagnostics. They do not measure generalisation and do not show that the model is better than any baseline.

Model `NEURAL_GATV2_TCN_K5` (checkpoint `checkpoints/checkpoint_epoch_0020.pt`, epoch 20), K = 5 modes. Windows evaluated: 2327 of 2327; tracks: 84. Miss rate: NOT_CONFIGURED.

## Primary (most-probable mode) — pixels

| aggregation | ADE mean | ADE median | ADE std | FDE mean | FDE median | FDE std |
|---|---|---|---|---|---|---|
| window-weighted | 31.46 | 15.31 | 41.22 | 68.61 | 27.26 | 102.74 |
| track-weighted (84 tracks) | 45.36 | 35.18 | — | 113.92 | 67.76 | — |

Error by horizon (window-weighted mean, px): 0.5s: 17.93, 1.0s: 36.19, 1.5s: 68.61

## Best-of-K ORACLE (ground truth selects the mode) — reported separately, not a single-trajectory result

| metric | mean | median |
|---|---|---|
| minADE@5 | 20.28 | 8.82 |
| minFDE@5 | 46.21 | 11.80 |

Mode diagnostics (in-sample): selected mode equals the oracle-best mode in 52.1% of windows; selected-mode histogram [712, 619, 0, 347, 649]; mean pairwise mode distance 42.3 px; mean max-probability 0.491 (uncalibrated).

## Comparison on common windows (2327 windows, 84 tracks) — IN_SAMPLE_MODEL_TRAINED_ON_THESE_WINDOWS; the baselines have no trainable parameters

| model | ADE mean | FDE mean | ADE median | FDE median | err @0.5 s | @1.0 s | @1.5 s |
|---|---|---|---|---|---|---|---|
| NEURAL_GATV2_TCN_K5 (most-probable mode) | 31.46 | 68.61 | 15.31 | 27.26 | 17.93 | 36.19 | 68.61 |
| B0_zero_velocity (single trajectory) | 54.26 | 109.93 | 31.16 | 54.74 | 31.79 | 65.75 | 109.93 |
| B1_last_valid_causal_velocity (single trajectory) | 92.95 | 181.59 | 61.71 | 119.89 | 56.53 | 114.14 | 181.59 |
| B2_history_linear_regression (single trajectory) | 35.66 | 79.30 | 23.31 | 46.29 | 18.64 | 41.99 | 79.30 |

This table is not evidence of superiority: the neural row is in-sample on one video.

Units: pixels of `captured_1920x1080`. No metric distance, TTC, PET or collision probability is computed.
