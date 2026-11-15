# Phase 5 P5.4 baseline evaluation — PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT

**All numbers below are PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT diagnostics**: one PIE video (set01/video_0004), perceived-track pseudo-targets, heavily overlapping windows, image-space pixels (`captured_1920x1080`). They do not establish forecasting accuracy, generalization, or that any baseline is better than another.

Windows in dataset: 2327. Policies: STRICT_FULL_HORIZON targets, EXCLUDE_AND_COUNT predictions. Miss rate: NOT_CONFIGURED.

## Window-weighted (each evaluated window weight 1; windows are not independent)

| Baseline | evaluated | not evaluated | ADE mean | ADE median | ADE std | FDE mean | FDE median | FDE std | tracks |
|---|---|---|---|---|---|---|---|---|---|
| B0_zero_velocity | 2327 | 0 | 54.26 | 31.16 | 66.64 | 109.93 | 54.74 | 147.28 | 84 |
| B1_last_valid_causal_velocity | 2327 | 0 | 92.95 | 61.71 | 103.67 | 181.59 | 119.89 | 197.70 | 84 |
| B2_history_linear_regression | 2327 | 0 | 35.66 | 23.31 | 37.66 | 79.30 | 46.29 | 94.89 | 84 |

## Track-weighted (mean of per-track means; each track weight 1)

| Baseline | tracks | ADE mean | ADE median | FDE mean | FDE median |
|---|---|---|---|---|---|
| B0_zero_velocity | 84 | 84.70 | 63.87 | 188.76 | 122.11 |
| B1_last_valid_causal_velocity | 84 | 99.63 | 85.92 | 209.50 | 187.74 |
| B2_history_linear_regression | 84 | 53.78 | 40.90 | 136.00 | 84.81 |

## Common evaluated windows (2327 windows, 84 tracks; same set for every baseline)

| Baseline | ADE mean | FDE mean | error @0.5 s | @1.0 s | @1.5 s |
|---|---|---|---|---|---|
| B0_zero_velocity | 54.26 | 109.93 | 31.79 | 65.75 | 109.93 |
| B1_last_valid_causal_velocity | 92.95 | 181.59 | 56.53 | 114.14 | 181.59 |
| B2_history_linear_regression | 35.66 | 79.30 | 18.64 | 41.99 | 79.30 |

## Invalid predictions

- B0_zero_velocity: {'anchor_position_unavailable': 0}; evaluator exclusions {'target_incomplete': 0, 'prediction_invalid': 0}
- B1_last_valid_causal_velocity: {'anchor_position_unavailable': 0, 'no_valid_causal_velocity_in_history': 0}; evaluator exclusions {'target_incomplete': 0, 'prediction_invalid': 0}
- B2_history_linear_regression: {'anchor_position_unavailable': 0, 'too_few_valid_history_samples': 0, 'history_time_span_too_short': 0, 'ill_conditioned_fit': 0}; evaluator exclusions {'target_incomplete': 0, 'prediction_invalid': 0}

minADE@1 / minFDE@1 equal ADE / FDE (K = 1 for every baseline; best-of-K oracle metrics are not reported as primary).
Units: pixels. No metric distance, TTC, PET or collision risk is computed.
