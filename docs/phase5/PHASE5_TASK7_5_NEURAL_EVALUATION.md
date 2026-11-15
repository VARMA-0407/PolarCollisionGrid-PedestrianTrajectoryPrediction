# P5.7.5 Neural Evaluator

**Status:** implemented and run (`COMPLETED_WITH_LIMITATIONS`). **Label:** `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`, sample status **`IN_SAMPLE_MODEL_TRAINED_ON_THESE_WINDOWS`**. Report: `phase5_task7_5_report.json`. Code: `pedsafety/phase5/models/neural_evaluation.py`, `scripts/phase5_evaluate_neural.py`, `scripts/phase5_validate_neural_evaluation.py`; config `configs/phase5/neural_evaluation_default.json`; tests `tests/phase5/test_neural_evaluation.py`.

## Design
1. **Verification before inference** (any mismatch is an error): training manifest and every data-artifact hash; checkpoint file hash and canonical model-state hash; train-only scaler statistics (canonical hash, source identity = the verified P5.3 window run, split identity, `fit_partition = train`); equality of the scaler buffers stored *inside* the checkpoint with the verified statistics; training / encoder / decoder / window configuration hashes; checkpoint provenance metadata; the P5.3 window run itself.
2. **History-only inference.** The model is called with the history `EncoderInput`, the raw pixel anchor and the history-derived target mask; ground truth is read afterwards, inside the evaluator. Overwriting the dataset's future arrays in memory leaves predictions bit-identical (tested).
3. **Tensor contract to the unchanged P5.4 evaluator** (`evaluate_predictions`): predictions `[N,K,F,2]` (K = 5, F = 15, raw pixels), prediction masks `[N,K,F]`, probabilities `[N,K]`, ground truth `[N,F,2]`, ground-truth masks `[N,F]`, window ids, track ids, `captured_1920x1080` / `pixel`. Policies from the P5.4 baseline configuration (strict full-horizon targets, exclude-and-count invalid predictions, miss rate `NOT_CONFIGURED`). The P5.4 evaluator and baselines were not modified.
4. **Primary vs oracle.** Primary metrics use the **most-probable mode** (argmax of the model's own probabilities); `minADE@5` / `minFDE@5` select the closest mode with ground truth, are labelled *best-of-K oracle*, and are reported separately (hand-computed test: a window whose most-probable mode is 10 px off and whose other mode is exact has primary ADE 10 and oracle ADE 0).
5. **Reporting:** window-weighted and track-weighted aggregation, per-horizon errors (0.5 / 1.0 / 1.5 s), per-track summaries, mode diagnostics (how often the most-probable mode is the oracle-best mode, histograms, spread, entropy), and a comparison with B0 / B1 / B2 recomputed on the same windows with the same evaluator and cross-checked against the committed P5.4 results (identical ADE means).

## Pilot evaluation — IN-SAMPLE, development only
Run `experiments/phase5_task7_neural_evaluation/20261115T191319Z_pie_video_0004/` (+ `…_determinism_rerun/`), final epoch-20 checkpoint `40aff17f…b44` (model state `9c953c1a…15d4c7`), scaler statistics `ed5fd682…7fe169`, 2,327 windows, 84 tracks, 0 excluded windows, K = 5. **The model was trained on every one of these windows.**

| Pixels (`captured_1920x1080`) | ADE mean | ADE median | ADE std | FDE mean | FDE median | FDE std |
|---|---|---|---|---|---|---|
| Neural, most-probable mode, window-weighted | 31.46 | 15.31 | 41.22 | 68.61 | 27.26 | 102.74 |
| Neural, most-probable mode, track-weighted (84 tracks) | 45.36 (median 35.18) | | | 113.92 (median 67.76) | | |
| Neural, **oracle** minADE@5 / minFDE@5 (window-weighted) | 20.28 | 8.82 | | 46.21 | 11.80 | |

Mean error by horizon, neural most-probable mode: 17.9 / 36.2 / 68.6 px at 0.5 / 1.0 / 1.5 s.

Comparison on the common 2,327 windows (window-weighted mean; baselines have no trainable parameters; **in-sample for the neural model only**):

| Model | ADE mean | FDE mean | ADE median | FDE median | track-weighted ADE / FDE | @0.5 / 1.0 / 1.5 s |
|---|---|---|---|---|---|---|
| Neural (most-probable mode) | 31.46 | 68.61 | 15.31 | 27.26 | 45.36 / 113.92 | 17.9 / 36.2 / 68.6 |
| B0 zero velocity | 54.26 | 109.93 | 31.16 | 54.74 | 84.70 / 188.76 | 31.8 / 65.8 / 109.9 |
| B1 last valid causal velocity | 92.95 | 181.59 | 61.71 | 119.89 | 99.63 / 209.50 | 56.5 / 114.1 / 181.6 |
| B2 historical linear regression | 35.66 | 79.30 | 23.31 | 46.29 | 53.78 / 136.00 | 18.6 / 42.0 / 79.3 |

Mode diagnostics (in-sample): the most-probable mode is the oracle-best mode in 52.1 % of windows; most-probable-mode histogram 712 / 619 / **0** / 347 / 649 (mode 3 of five — index 2 — is never selected although it is the oracle-best mode of 191 windows); mean pairwise mode distance 42.3 px; mean maximum probability 0.49 (near uniform, uncalibrated).

**Interpretation boundary.** The neural row is a training-set number on one video with perceived-track pseudo-targets; it only shows that the pipeline fits its own training windows moderately better than a simple trend baseline. It does not show generalisation, that the architecture is better than B0–B2, or that the model's mode probabilities mean anything. No independent evaluation exists (P5.7.6 is blocked). The oracle numbers are best-of-5 selections by ground truth and are not single-trajectory performance.

## Reproducibility and validation
Two evaluations produced byte-identical data artifacts (`predictions_neural.npz` `c070134f…`, `evaluation_neural.npz` `80988e8e…`, `evaluation_results.json` `e7b8c8ff…`, `evaluation_report.md` `524a8df3…`). `scripts/phase5_validate_neural_evaluation.py --compare-with …`: **PASS 17/17** (artifact hashes, labels / in-sample status / claims, checkpoint and scaler re-verification, **bitwise recomputation of the predictions from the checkpoint**, shapes / units / probabilities, raw anchors, **independent plain-Python recomputation of the primary and oracle ADE / FDE and of the summary means**, oracle ≤ primary, oracle labelling, baseline comparison, equality of the recomputed baselines with the committed P5.4 results, bitwise equality of the two runs). Runtime 15.9 s (inference + evaluation + baselines, 1 thread), peak memory 378 MB.

*Validation history (kept for transparency):* the first validation attempt used the default 10 CPU threads and failed one check (`predictions_recomputed_from_checkpoint_bitwise`; max difference 1.8e-4 px); the cause was that the validator did not apply the evaluation's recorded `torch_threads = 1` (multi-thread matmuls differ in the last bits). That report is preserved as `validation_report_attempt1_validator_used_default_threads.json`; the validator now applies the recorded thread count and records that it did.

## Limitations
In-sample, one scene; pseudo-targets from perceived tracks; no confidence intervals (correlated windows, one video); miss rate not configured; mode probabilities uncalibrated; pixels are depth-dependent and include camera ego-motion; bitwise reproducibility is guaranteed for the recorded thread count (1) on this platform; CUDA not exercised; timings are not real-time claims.
