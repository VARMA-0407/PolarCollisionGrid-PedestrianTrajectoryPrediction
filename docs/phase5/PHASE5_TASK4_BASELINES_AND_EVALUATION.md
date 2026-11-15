# Phase 5 Task P5.4 — Evaluation Framework and Classical Baselines

**Date:** 15 November 2026. **Status:** `COMPLETED_WITH_LIMITATIONS — PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT` (uncommitted; awaiting review).
**Scope:** a reusable, model-agnostic trajectory evaluator and three classical single-trajectory baselines (B0–B2), run on the P5.3 pilot windows. No neural model, GATv2, TCN, decoder, training, split or metric-world quantity exists. Machine-readable companion: `phase5_task4_evaluation_report.json`.

> **Every measured number in this document is `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`.** One PIE video (set01/video_0004), one scene group, 2,327 heavily overlapping windows from 84 pedestrian tracks, perceived-track pseudo-targets (not ground truth), image-space pixels. The numbers validate the pipeline; they do not measure forecasting accuracy, generalization or cross-scene performance, and they do not show that any baseline is scientifically better than another.

## 1. Starting state and Git
| | Value |
|---|---|
| Branch | `feature/phase4-task3-interaction-representation` (unchanged) |
| Initial HEAD | `c3137b448686a9211153c1582cf7177a861071a0` "feat(phase5): implement pilot trajectory window generation" — the P5.3 commit (41 files: 40 added + `.gitignore`), parent `5d58ab7146ea4611733393769071b02a6bf9d187` |
| Initial working tree | clean (no staged, unstaged or untracked changes; no stash) |
| Final HEAD | `c3137b4` (no commit, amend, stage or push) |
| Final working tree | untracked P5.4 files only; one tracked file modified: `docs/phase5/PHASE5_IMPLEMENTATION_ROADMAP.md` (status update, as instructed) |

(A first attempt at this task stopped because P5.3 was staged but not committed; after the user committed it as `c3137b4`, the checks above passed.)

## 2. Files
**Created:** `pedsafety/phase5/baselines/` (`__init__`, `base`, `history_input`, `predictors`); `pedsafety/phase5/evaluation/` (`__init__`, `metrics`, `experiment`); `configs/phase5/baseline_evaluation_default.json`; `scripts/phase5_run_baselines.py`, `scripts/phase5_evaluate_predictions.py`, `scripts/phase5_validate_evaluation.py`; `tests/phase5/` (`p54_helpers`, `test_baselines`, `test_evaluation_metrics`, `test_baseline_pipeline`); this report; `phase5_task4_evaluation_report.json`; experiment directory `experiments/phase5_task4_baselines_pie_video_0004/` (section 9).
**Modified:** `docs/phase5/PHASE5_IMPLEMENTATION_ROADMAP.md` (P5.4 status); in the hygiene pass also `docs/phase5/PHASE5_EVALUATION_PROTOCOL.md` (§6 baseline table: B1/B2 renamed to the approved definitions; text only) and `.gitignore` (one narrow P5.4 rule).
**Not modified:** P5.3 code (`pedsafety/phase5/*.py` from P5.3), configuration, reports and experiment outputs; Phase 2/3/4 code and outputs; PCG baseline; master files; existing tests.

## 3. Verified P5.3 input (no schema change needed)
From `experiments/phase5_task3_windows_pie_video_0004/20261115T035825Z_pie_video_0004/primary/` (manifest and all four output hashes re-verified on every run; window config hash `65eebeed…1034` equals the baseline config reference):

| Needed | Stored field (verified) |
|---|---|
| history positions | `windows.npz` `history_target_uv_px` float32 [2327, 6, 2] (equal to graph-bank `u_px, v_px`) |
| history timestamps | `windows.npz` `history_timestamps` float64 [2327, 6] |
| position / velocity validity | graph-bank `masks[:, 0]` / `masks[:, 1]` at `history_target_bank_node` [2327, 6] |
| causal velocity | graph-bank `features[:, 2:4]` (`vel_u_px_s`, `vel_v_px_s`): Task 2 causal one-frame backward difference (P5.3 verified all 65,388 rows `view = CAUSAL`) |
| targets (evaluator only) | `target_future_uv_px` float32 [2327, 15, 2], `target_future_valid` bool [2327, 15] |
| identity | `window_id`, `target_track_index`, `target_track_id`, `anchor_frame` |

The offline (non-causal) sidecar is not reachable from any of these fields.

## 4. Predictor interface and isolation
* `HistoryInput` (history only; `__slots__`, immutable, arrays copied and read-only): window/track ids, anchor frame and timestamp, `history_timestamps [N,6]`, `history_uv_px [N,6,2]`, position and velocity masks, `history_velocity_px_s [N,6,2]`, `horizon_offsets_s [15]` (= (m+1)·3/30 s from the window configuration), coordinate system and unit (validated: only `captured_1920x1080` / pixel). It has no future, target or target-node field.
* `Predictor.predict(HistoryInput) -> Prediction`: `positions` float64 **[N, 15, 2]** (u, v px at t+0.1 … t+1.5 s), `valid [N,15]`, `window_valid [N]`, `invalid_reason [N]`, `baseline_id`, `config`, per-window `aux` diagnostics. Invalid entries are NaN, never zero. Any other input type is refused (`PREDICTOR_INPUT_TYPE`); output shape and finiteness are enforced.
* Future targets are read only by the evaluator, after all predictions exist. The validator re-runs every predictor on a copy whose future arrays were overwritten (1e9 / −1e6) and confirms bit-identical predictions.

## 5. Baseline definitions (all K = 1, deterministic, not learned)
Let t be the anchor (last history) time, p(t) the last observed footpoint, τ ∈ {0.1, …, 1.5} s.
* **B0 zero velocity:** p̂(t+τ) = p(t). Invalid only if p(t) is unavailable.
* **B1 last valid causal velocity:** p̂(t+τ) = p(t) + v(t_k*)·τ, where k* is the latest of the 6 history samples (searching backward from the anchor; `max_lookback_samples = 5`) whose Task 2 causal velocity is valid. **Policy: no fallback** — if no valid velocity exists the window is invalid (`no_valid_causal_velocity_in_history`); zero velocity is never substituted. Extrapolation always starts from p(t), not from the sample whose velocity was used.
* **B2 history linear regression:** least squares x(t) = a_x + b_x t, y(t) = a_y + b_y t over the *valid* history samples, using actual timestamps **centred at their mean** (t − t̄) for numerical conditioning; masked samples are excluded, never interpolated. p̂(t+τ) = p(t) + (b_x, b_y)·τ — anchored at the last observation, not the intercept. Reliability rules, fixed before any pilot result: ≥ 3 valid samples; time span ≥ 0.2 s (with a 1e−9 s tolerance for floating-point differences of absolute timestamps); centred time variance > 1e−6 s²; last sample observed. Failures are invalid with a reason code.

## 6. Evaluator (`pedsafety/phase5/evaluation/metrics.py`, model-agnostic)
Inputs: ground truth [N,F,2] + mask [N,F]; predictions [N,F,2] or [N,K,F,2] + masks; window ids (unique), track ids, coordinate system and unit (validated), model id, horizon offsets, optional mode probabilities [N,K]. Values flagged valid must be finite (NaN/inf → error); masked values are ignored.
* e_t = ‖p̂_t − p_t‖₂ (pixels). **ADE** = mean of e_t over the 15 steps; **FDE** = e_15.
* **Partial-target policy `STRICT_FULL_HORIZON`:** a window is evaluated only if all 15 target steps are valid; otherwise excluded and counted (`target_incomplete`). **Invalid-prediction policy `EXCLUDE_AND_COUNT`:** a hypothesis is usable only if all 15 steps are valid; otherwise the window is excluded and counted (`prediction_invalid`). Nothing is zero-filled.
* **Primary hypothesis:** K = 1 → the only one; K > 1 → argmax mode probability (never the closest); K > 1 without probabilities → no single-trajectory metric.
* **minADE@K / minFDE@K:** min over usable hypotheses, labelled **BEST-OF-K ORACLE**; for K = 1 identical to ADE/FDE (verified on every window).
* **Miss rate:** only with a configured pixel threshold; the default configuration has `miss_rate_threshold_px = null`, so it is reported as `NOT_CONFIGURED` (no defensible depth-independent pixel threshold has been approved).
* **Aggregation, separately labelled:** window-weighted (each evaluated window weight 1: mean, median, population std (ddof 0), min, max, p90) and track-weighted (mean/median of per-track means); per-track summaries; per-horizon mean error at every step and at 0.5/1.0/1.5 s; and a **common-window comparison** on the windows evaluated by every baseline. No confidence intervals or significance tests are reported (correlated windows, one video).

## 7. Results — `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`
Run `experiments/phase5_task4_baselines_pie_video_0004/20261115T042053Z_pie_video_0004/primary`. Windows evaluated: **2,327** of 2,327 for every baseline; 84 tracks; common window set = all 2,327.

| Baseline | valid / invalid predictions | invalid reasons | diagnostics |
|---|---|---|---|
| B0 | 2,327 / 0 | — | — |
| B1 | 2,327 / 0 | none | velocity taken at the anchor sample in 2,322 windows, one sample earlier (backward search) in 5 |
| B2 | 2,327 / 0 | none | all windows have 6 valid history samples |

Window-weighted (pixels; each window weight 1; windows are **not** independent):

| Baseline | ADE mean | ADE median | ADE std | ADE p90 | FDE mean | FDE median | FDE std | FDE p90 |
|---|---|---|---|---|---|---|---|---|
| B0 zero velocity | 54.26 | 31.16 | 66.64 | 145.31 | 109.93 | 54.74 | 147.28 | 316.02 |
| B1 last causal velocity | 92.95 | 61.71 | 103.67 | 205.28 | 181.59 | 119.89 | 197.70 | 401.72 |
| B2 linear regression | 35.66 | 23.31 | 37.66 | 84.41 | 79.30 | 46.29 | 94.89 | 202.41 |

Track-weighted (mean / median of the 84 per-track means):

| Baseline | ADE mean | ADE median | FDE mean | FDE median |
|---|---|---|---|---|
| B0 | 84.70 | 63.87 | 188.76 | 122.11 |
| B1 | 99.63 | 85.92 | 209.50 | 187.74 |
| B2 | 53.78 | 40.90 | 136.00 | 84.81 |

Mean displacement error by horizon (window-weighted, common windows): B0 31.79 / 65.75 / 109.93 px at 0.5 / 1.0 / 1.5 s; B1 56.53 / 114.14 / 181.59; B2 18.64 / 41.99 / 79.30. minADE@1 / minFDE@1 equal ADE / FDE. Miss rate: `NOT_CONFIGURED`.

**Track-level structure.** Windows per track range from 1 to 248 (median 13); the largest track (`pie:video_0004:7514`) contributes 248 windows (10.7 %), so window-weighted numbers are dominated by a few long tracks — which is why the track-weighted figures are higher and are reported separately. Per-track ADE ranges: B0 2.6–229.2 px, B1 12.5–412.6 px, B2 3.2–199.9 px. Descriptively, B2's ADE is lower than B0's in 56 % of windows and in 66 of 84 tracks.

**Interpretation (engineering only).** The ordering B2 < B0 < B1 on this pilot is consistent with expectations: B1 extrapolates a one-frame (1/30 s) Task 2 velocity with ~48 px/s per-axis jitter over 1.5 s (≈ 70 px of noise at the horizon before any motion error), whereas B2 averages six samples over 0.5 s. Errors grow roughly linearly with horizon for all three. Because the windows are correlated, come from one video and use perceived-track targets, this ordering is **not** evidence that B2 is superior in general; it is a reference for later models on the same pipeline.

## 8. Correctness tests (hand-calculated, synthetic)
`tests/phase5`: **82 passed** (39 P5.3 + 43 new). New tests cover all 20 required behaviours: perfect prediction (0/0); constant horizontal (3 px) and vertical (4 px) error; diagonal (3, 4) → 5 px and growing error (ADE 8, FDE 15); B0 last position; B1 exact extrapolation, backward search to the latest valid velocity, lookback limit, garbage under invalid masks ignored, no zero substitution; B2 exact recovery of a linear trajectory, irregular timestamps (an index-based fit would differ by > 1 px/s), anchoring at the last observation, masked samples ignored without interpolation, each reliability rule; invalid predictions and invalid targets excluded and counted (not zero); strict partial-mask policy; shape validation for inputs, predictions and evaluator; coordinate/unit validation; NaN/inf rejected where valid and tolerated where masked; determinism; window-id traceability and uniqueness; window- vs track-weighted aggregation on a hand example (2.0 vs 3.0); K > 1 oracle vs most-probable primary; miss rate configured vs not configured; per-horizon values; recomputation from saved predictions (bit-identical evaluation files); predictor refusal of non-`HistoryInput` objects, immutability and absence of future fields; invariance of predictions to overwritten future arrays; B1 velocity equal to the graph-bank causal velocity; input hash/config mismatch rejection; output-directory protection; CLI exit codes. The end-to-end tests run on windows generated by the real P5.3 → Task 3 → Task 2 pipeline from the synthetic scene.

Four defects in the new code were found by the tests, the validator and the first pilot run, and were fixed before the reported pilot runs (the numbered items below are the first two; the rest follow): (1) `HistoryInput` made the caller's arrays read-only because `np.asarray` did not copy — it now copies (regression test added); (2) the B2 span rule rejected an exact 0.2 s span computed as 0.19999999999999 s from absolute timestamps — a 1e−9 s tolerance was added (no pilot window was affected: all have six samples and a 0.5 s span). A third issue, found on the pilot: strings saved by `write_npz` become 1-element arrays; a `scalar_str` reader was added (the validator caught it).

Regression: see section 11.

## 9. Reproducibility and provenance
Experiment root `experiments/phase5_task4_baselines_pie_video_0004/`:
* `20261115T042053Z_pie_video_0004/primary` and `/determinism_rerun` — **reported run** (identical configuration). All 8 data artifacts have identical SHA-256 in both: `predictions_B0…` `a62d7f08…`, `predictions_B1…` `0c34a884…`, `predictions_B2…` `95dba916…`, `evaluation_B0…` `34bf0b14…`, `evaluation_B1…` `eaa0c1d5…`, `evaluation_B2…` `b8fef815…`, `evaluation_results.json` `fa089f5e…`, `evaluation_report.md` `aa593c9b…`.
* `…/recomputed_from_predictions` — evaluation recomputed by `phase5_evaluate_predictions.py` from the saved prediction files alone: the three evaluation `.npz` files are bit-identical to the primary run's.
* `…/validation_report.json` — `phase5_validate_evaluation.py`: **PASS 26/26** (artifact hashes; P5.3 input re-verification; predictions regenerated from history are bit-identical; predictions invariant to overwritten future targets; shapes, units, coordinate system, ids; invalid entries NaN; per-window ADE/FDE recomputed by an independent pure-Python loop within 1e−9 px; summary statistics; min = ADE/FDE for K = 1). Same result on `determinism_rerun`.
* `20261115T042023Z_pie_video_0004/` — an earlier run of identical code except that the manifest's torch-version lookup imported torch, inflating the recorded peak memory (188.6 MB). Its 8 data artifacts are byte-identical to the reported run (a third determinism witness); only its run metadata is superseded. Kept, not overwritten.

**Code-baseline provenance.** Each P5.4 manifest records `git.commit = c3137b4` (the P5.3 commit, branch `feature/phase4-task3-interaction-representation`) with `working_tree_dirty = true`: the dirty files are exactly the untracked P5.4 files, and `tracked_changes` is empty. Its input record `windows_run_git = 5d58ab7` is the commit the P5.3 window run itself recorded (that run was made before P5.3 was committed); the P5.3 window-run files used here were re-verified by hash (`run_manifest.json` and four outputs).

Each `run_manifest.json` separates **`data_artifacts`** (deterministic hashes) from **`run_metadata`** (git commit `c3137b4`, branch, working-tree state `dirty` with the untracked P5.4 file list, Python 3.9.25, numpy 1.26.4, torch 2.2.1+cpu installed but not imported, timestamps, command, runtime, peak memory) and records input hashes (P5.3 manifest `run_manifest.json` and four outputs), baseline-config hashes (canonical and file), window-config hash, counts per baseline, coordinate system, units, target source and the pilot label. `evaluation_results.json` and `evaluation_report.md` contain no run-specific metadata.

## 10. Runtime and memory (offline CPU, i5-1235U; not a real-time claim)
Input verification and loading 0.15 s; prediction 7.2 ms total for 2,327 windows (B0 0.9 ms, B1 1.6 ms, B2 3.1 ms); evaluation 35 ms; total before writing 0.19 s. Peak process working set 66.9 MB (67.4 MB on the rerun). Output 4.7 MB per run directory set; the largest files are the evaluation `.npz` files (≈ 0.34 MB each) and predictions (0.03–0.41 MB).

**`.gitignore` (hygiene pass, applied):** one narrow rule plus a comment line, `/experiments/phase5_task4_baselines_pie_video_0004/**/*.npz`. It ignores exactly the 30 generated `.npz` arrays under the P5.4 experiment directory (≈ 9.1 MB; ≈ 1.8 MB per run), matches no tracked file, and leaves the P5.3 rule (`/experiments/phase5_task3_windows_pie_video_0004/**/*.npz`) and every JSON manifest, report, validation report and Markdown report trackable.

## 11. Tests
| Command | Result |
|---|---|
| `python -m pytest tests/phase5 -q -p no:cacheprovider` | **82 passed** |
| `python -m pytest tests/phase5 tests/phase4 tests/phase3 tests/task7 tests/phase2 -q -rfs -p no:cacheprovider` | **1086 passed, 6 skipped, 1 failed** (90 s); previous suite 1043 + 43 new = 1086 |
| the only failure | `tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension` — the known pre-existing PCG `TTC_mask` defect (unchanged) |
| skips | cv2/ultralytics not installed in `pcg-baseline` (5); original notebook absent (1) — pre-existing |

The full regression started before the final, manifest-only edit (torch version read from package metadata instead of importing torch); `tests/phase5` was re-run on the final code: 82 passed.

## 12. Limitations
1. Single video, single scene group, correlated overlapping windows; no split; no confidence intervals; no generalization, cross-scene or research-grade claim; no statistically supported ranking.
2. Targets are perceived-track pseudo-targets; Phase 3 detection/tracking quality on PIE is unmeasured.
3. Pixel errors are depth-dependent and include camera ego-motion; not metric; no TTC/PET/collision quantities.
4. B1 inherits Task 2 one-frame velocity jitter; strict windows bias towards long, continuous tracks; window-weighted statistics are dominated by a few long tracks.
5. Miss rate is not configured (no approved pixel threshold); a scale-aware variant (FDE / box height) is not implemented in P5.4.
6. Evaluator supports K > 1 and mode probabilities, but only K = 1 is exercised on real data.

## 13. Deviations
* Prediction arrays are [N, F, 2] for the baselines; the evaluator additionally accepts [N, K, F, 2] for future multimodal models (instructed preference kept for baselines).
* Added a common-window comparison and per-horizon errors (from the evaluation protocol) beyond the minimum metric list.
* The cluster bootstrap described in the evaluation protocol §5 is not implemented: with one video it would only describe within-video variability; deferred until independent groups exist.
* The roadmap planned `pedsafety/forecasting/{metrics,baselines}.py`; P5.4 uses `pedsafety/phase5/{evaluation,baselines}/` as instructed. Baseline naming is standardised everywhere (code, configuration, manifests, results, evaluation protocol §6): B0 zero velocity, B1 last valid causal velocity, B2 historical linear regression. The kickoff-era protocol table had listed B1/B2 in the opposite order; it was updated to the approved definitions in the hygiene pass (naming only; no predictor was changed).

## 14. Acceptance
| Criterion | Result |
|---|---|
| B0, B1, B2 have independent unit tests | PASS |
| Evaluator passes hand-calculated metric tests | PASS |
| No baseline accesses future targets | PASS (type-restricted input without future fields; invariance to overwritten futures on synthetic and pilot data) |
| Causal velocity use verified | PASS (B1 velocity = graph-bank Task 2 causal velocity; offline sidecar unreachable) |
| Invalid masks handled correctly | PASS (NaN, excluded, counted; never zero) |
| Correct dimensions and units | PASS ([2327, 15, 2] px, `captured_1920x1080`) |
| Prediction and metric artifacts reproducible | PASS (identical hashes; recomputation from saved predictions bit-identical) |
| Traceable to source windows | PASS (window_id, track, anchor in every artifact; P5.3 hashes recorded) |
| Two identical runs → identical data hashes | PASS (8/8; also equal to an earlier third run) |
| Pilot-only limitation explicit | PASS (every artifact labelled) |
| No protected source files modified | PASS |
| Focused and relevant regression tests | PASS (82 Phase 5; 1086 full, only the known pre-existing failure) |

**P5.4 status: `COMPLETED_WITH_LIMITATIONS`** — the evaluation framework and reference baselines are validated on the pilot; any performance interpretation waits for independent scene groups.

## 15. Recommended next task
**P5.5 — heterogeneous GATv2 + causal dilated TCN encoder** (plain PyTorch, masked dense relation-specific attention; synthetic and pilot shape/causality tests; no training claims), evaluated later through this same evaluator against B0–B2 on the same windows. In parallel and outside the code: P5.2 data acquisition so that independent groups, a real split and confidence intervals become possible.
