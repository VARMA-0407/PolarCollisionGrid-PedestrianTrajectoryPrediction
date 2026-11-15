# Phase 5 Evaluation Protocol (leakage-safe, non-oracle by default)

**Version:** 0.1.1 (H = 6 reconciliation for P5.3, sections 1 and 3). **Date:** 15 November 2026. **Status:** `PROPOSED — NOT IMPLEMENTED` (no evaluator, split or metric is implemented by P5.3).
Built on the Phase 2 Task 6 protocol (`docs/phase2/task6_data_splitting_evaluation_protocol.md`, `configs/phase2/split_protocol.template.json`) and the existing Stage 5 split code (`pedsafety/splitting.py`: grouping policy, explicit group assignment, `verify_window_isolation`) and the non-oracle evaluator (`pedsafety/evaluation.py`). Nothing here overrides them; Phase 5 specialisations are listed explicitly. All errors are in **pixels** of `captured_1920x1080` (no validated metric unit exists).

## 1. Split protocol
1. Split unit = an **independent group**, chosen before any window is generated. Proposal for PIE: the recording set (`set01`, …), after verifying that a set is a recording session; at minimum whole videos. Never individual windows, frames or tracks.
2. Explicit group → partition assignment recorded in a split manifest (Phase 2 template) with input hashes and a seed, frozen before training.
3. Windows generated **inside** each partition only (Phase 2 §5 step 7).
4. Isolation assertions: no video, track, or source frame of any window's dependency interval appears in two partitions (`verify_window_isolation` extended to dependency intervals).
5. Train-only fitting of normalization statistics, any learned preprocessing and any filter thresholds.
6. Validation-only model, checkpoint, hyperparameter and K selection; test evaluated once with the frozen selection.
7. **Pilot exception:** with one video, only a labelled `smoke_split` (chronological blocks of `video_0004` separated by a purge gap) is allowed, to test code. Its results are development diagnostics, not evidence of generalization, and must never be reported as test results.

**Dependency interval and purge.** For anchor `t0`, history `H`, horizon `F`, step `s`: interval `[t0 − (H−1)s − 1, t0 + F·s]` (input contract §10). For chronological blocks the purge gap between partitions must be at least the interval length `(H−1+F)s + 2` frames, plus any longer derived-feature lookback; it is computed from the configuration, never chosen ad hoc. For the approved H = 6, F = 15, s = 3: interval `[t0 − 16, t0 + 45]` (sampled frames `t0−15 … t0+45` plus frame `t0−16` behind the one-frame backward-difference velocity at `h_0`) and minimum purge gap **62 frames**, defined as the minimum difference between the last anchor of one chronological partition and the first anchor of the next (63 on the `t0 % 3 == 0` anchor grid); derivation and the past-only counter caveat (`track_age_frames`, `frames_since_prev_obs`) in input contract §10. The kickoff's H = 5 value, 59 frames, is superseded. P5.3 creates **no** smoke split: the pilot window dataset carries a single `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT` listing for loading/batching tests only.

## 2. Sample construction
One sample = (target pedestrian track, anchor frame `t0`) satisfying input-contract §7: target present at all `H` history frames and all `F` target frames; inputs = per-history-frame heterogeneous graphs; target = `F` future positions relative to the target's `t0` position. Anchors on the stride grid; overlapping windows of the same track are allowed **within** a partition only, and their correlation is handled in uncertainty estimates (section 5).

## 3. Horizons (provisional; finalised by the P5.3 window study on training data only)
Default `s = 3` (10 Hz), `H = 6` (t−0.5 … t, 0.5 s first-to-last; approved for P5.3, superseding the kickoff's provisional `H = 5`), `F = 15` (1.5 s). Also report per-horizon errors at 0.5 s, 1.0 s, 1.5 s. Longer horizons (2.0 s) as a sensitivity study. The PCG HBS setting (2 Hz, 6 + 6) is not adopted.

## 4. Metrics (pixels; per window i, valid target steps t = 1…F, prediction p̂, ground truth p)
**Single-trajectory (deterministic or most-likely mode, chosen without ground truth):**
- `ADE_i = (1/F) Σ_t ‖p̂_t − p_t‖₂`, `FDE_i = ‖p̂_F − p_F‖₂`; aggregate = mean (and median) over windows.
- Most-likely mode = `argmax_k π_k` (model probability), never the closest mode.
**Best-of-K (oracle upper bound, always labelled):**
- `minADE_K = min_k ADE_i^k`, `minFDE_K = min_k FDE_i^k`. Uses ground truth to select; reported only beside single-trajectory metrics, never alone.
**Probability-aware:**
- `brier-minFDE_K = FDE_i^{k*} + (1 − π_{k*})²` with `k*` the minFDE mode (rewards putting probability on the good mode).
- Negative log-likelihood of the ground truth under the predicted mixture (if the decoder outputs a parametric mixture, e.g. Laplace/Gaussian per step).
- Mode-probability reliability (calibration of `π_{k*}`) on validation only; no calibration claim without this check.
**Miss rate:**
- `MR_K(τ) = fraction of windows with minFDE_K > τ` (and the single-mode `MR_1`). τ must be **predeclared before any test evaluation**. Proposed: τ in pixels chosen on training data (e.g. a fixed value) **and** a scale-aware variant `FDE / box_h(t0) > τ_h` (heuristic, labelled; box height is not calibrated depth). Both reported; neither is a physical miss threshold.
**Breakdowns:** per agent category (pedestrian primary); per video/group; per interaction density at `t0` (number of neighbours by relation); per presence of P-V vs only P-P neighbours; per truncation flag at `t0`; per horizon step.
**Coverage:** number of eligible windows, excluded windows by reason, valid-prediction rate.
**Runtime and memory:** model forward time per window and per batch (CPU and T4 where measured), peak memory; offline measurements only, never real-time claims.

## 5. Statistical reporting
Windows are correlated (same track, same scene). Confidence intervals by **cluster bootstrap over tracks** (and over groups when several exist); paired comparisons between models on the same windows. With one video, intervals describe within-video variability only.

## 6. Baselines
| # | Baseline | Definition | Comparability |
|---|---|---|---|
| B0 | Zero-velocity | `p̂_t = p_{t0}` | always comparable (sanity floor) |
| B1 | Constant velocity (CV) | velocity estimated from history positions over the last `m` samples (`m` fixed on validation), extrapolated linearly; uses positions only (robust to Task 2 one-frame jitter) | always comparable |
| B2 | CV with Task 2 velocity | uses the `t0` node velocity if valid | comparable; shows effect of one-frame jitter |
| B3 | Temporal-only neural (target track only) | same TCN encoder and decoder as the full model, no graph | comparable; ablation of interactions |
| B4 | PCG (historical) | original HBS checkpoints and best-of-5 results | **not comparable**: different dataset (HBS), coordinates (unresolved units, not PIE pixels), horizon (2 Hz, 3 s / 3 s), oracle best-of-5 selection, future information in rollout. Kept as a historical reference only; never placed in the same results table as PIE metrics without these caveats |
| B5 | PCG-style re-implementation on PIE | only if its TTC grid can be defined in image space without metric thresholds; otherwise not attempted | decision deferred |

## 7. Proposed model and ablation matrix (not implemented)
Full model: per-frame heterogeneous GATv2 (relation-specific attention, edge features with masks) → per-agent embedding sequence → causal dilated TCN over the `H` history samples → multimodal decoder (K trajectories + mode probabilities).

| ID | Variant | Question |
|---|---|---|
| A1 | TCN only (no graph) = B3 | do interactions help at all? |
| A2 | GATv2 at `t0` only, no temporal encoder (MLP over history of the target) | value of temporal modelling (if meaningful) |
| A3 | Full GATv2 + TCN | main model |
| A4 | Relation ablations: drop P→P / drop V→P,P→V / drop B relations | which relation types carry signal |
| A5 | Edge-feature ablations: positions only vs + relative velocity vs + scale proxies | sensitivity to noisy/heuristic features |
| A6 | Neighbour subsets: complete vs `knn_4` / `knn_8` vs `sep_over_h ≤ 5` | does pruning help (flags from Task 3) |
| A7 | GATv2 vs simpler aggregator (mean/attention-free) | is GATv2 justified |
| A8 | Unimodal (K=1, L2) vs multimodal | value of multimodality |
| A9 | K ∈ {1, 3, 6} (validation) | number-of-modes sensitivity |
| A10 | `s`/`H`/`F` sensitivity | horizon dependence |
All selections on validation; one final test run of the frozen choices.

## 8. Training objective proposal (to be validated, not assumed)
Winner-takes-all regression on the best mode (by ADE to ground truth during **training** only) with a smooth-L1/Laplace loss, plus cross-entropy on mode probabilities toward the best mode; alternative: full mixture NLL. Monitor mode collapse (pairwise mode distance, entropy of π) and mode diversity on validation. Seeds fixed; at least 3 seeds for headline comparisons.

## 9. What is excluded
Metric ADE/FDE, metric TTC/PET/minimum separation, conflict or collision probabilities, safety thresholds, deployment claims, comparisons that mix PIE pixels with HBS units, and any oracle metric presented as single-sample performance.
