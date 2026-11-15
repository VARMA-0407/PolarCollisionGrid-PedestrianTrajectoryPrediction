# Phase 5 Implementation Roadmap

**Version:** 0.2.0 (consolidated P5.6-P5.7 status). **Date:** 16 November 2026. **Status:** `P5.3-P5.5 COMMITTED; P5.6 COMMITTED (afdf822); P5.7.1-P5.7.5 IMPLEMENTED AND RUN ON THE PILOT (in-sample, development only); P5.7.6 BLOCKED_PENDING_SCENE_DIVERSE_DATA; P5.2, P5.8, P5.9 BLOCKED`. See `PHASE5_CONSOLIDATED_ACCEPTANCE.md` and `PHASE5_ARCHITECTURE.md`.
**P5.7 revision (evidence: `PHASE5_TASK7_2_TRAIN_ONLY_SCALER.md`, `PHASE5_TASK7_3_INTEGRATED_TRAINING_STEP.md`, `PHASE5_TASK7_4_PILOT_TRAINING_HARNESS.md`, `PHASE5_TASK7_5_NEURAL_EVALUATION.md`, `PHASE5_TASK7_6_INDEPENDENT_EVALUATION_STATUS.md`, JSON reports `phase5_task7_*_report.json`, `phase5_task7_6_status.json`; runs `experiments/phase5_task7_training/20261115T190117Z_pie_video_0004/` and `experiments/phase5_task7_neural_evaluation/20261115T191319Z_pie_video_0004/`):** history-only encoder -> decoder -> masked multimodal loss training step; AdamW + cosine + clipping harness (single CPU thread for bitwise reproducibility; resume equals the continuous run); neural evaluator on the unchanged P5.4 evaluator with most-probable-mode primary metrics, separate best-of-5 oracle metrics, window- and track-weighted aggregation and a comparison with B0-B2. All neural numbers are in-sample (one video; the model is evaluated on its own training windows). No independent split exists, so no checkpoint or hyperparameter selection was made (final epoch only).
**P5.5 revision (evidence: `PHASE5_TASK5_SPATIAL_TEMPORAL_ENCODER.md`, `phase5_task5_encoder_report.json`, run `experiments/phase5_task5_encoder_smoke_pie_video_0004/20261115T044455Z_pie_video_0004/`):** plain-PyTorch relation-aware heterogeneous GATv2 (per history frame, per-relation attention with edge attributes in logits and messages, root term) + causal dilated TCN (kernel 3, dilations 1,2, receptive field 7 ≥ H = 6) in `pedsafety/phase5/models/` (not `pedsafety/forecasting/`), config `configs/phase5/encoder_default.json` (144,128 parameters). History-only input; no decoder, head, loss or training. 59 new passing tests (141 in `tests/phase5`); forward-pass smoke on the 2,327 pilot windows of an UNTRAINED, randomly initialised encoder: finite, masked, deterministic, independently validated (25/25), `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`, no accuracy or baseline comparison. Dependency policy (Option A): `requirements-phase5.txt` declares `torch>=2.2`; only `pedsafety/phase5/models` may use it (Phase 2 Task 7 guard extended path-scoped; `requirements-pedsafety.txt` unchanged). Nominal development feature scaling only; train-only statistics belong to P5.7.
**P5.4 revision (evidence: `PHASE5_TASK4_BASELINES_AND_EVALUATION.md`, `phase5_task4_evaluation_report.json`, run `experiments/phase5_task4_baselines_pie_video_0004/20261115T042053Z_pie_video_0004/`):** model-agnostic evaluator (`pedsafety/phase5/evaluation/`: single-trajectory ADE/FDE as the primary metrics, per-horizon error, best-of-K minADE/minFDE as labelled oracle metrics (identical to ADE/FDE for K = 1), miss rate only with a configured pixel threshold — none is approved, so it is reported `NOT_CONFIGURED` — window-weighted and, separately, track-weighted aggregation, common-window comparison) and baselines (`pedsafety/phase5/baselines/`): B0 zero velocity, B1 last valid Task 2 causal velocity (no fallback), B2 history least-squares velocity anchored at the last observation. Pilot: 2,327 overlapping, non-independent windows (84 tracks, one video) evaluated against perceived-track pseudo-targets for every baseline (`PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`; no accuracy or ranking claim); deterministic (identical data hashes across runs; recomputation from saved predictions bit-identical); independent validation 26/26; 82 Phase 5 tests. The planned B3 (temporal-only neural) belongs to P5.5+. Baseline naming (standardised, identical in code, configuration, manifests, results and the evaluation protocol §6): B0 zero velocity, B1 last valid causal velocity, B2 historical linear regression. Cluster bootstrap deferred until independent groups exist.
**P5.3 revision:** implemented under `pedsafety/phase5/` (as instructed for P5.3) rather than the planned `pedsafety/forecasting/`; configuration `configs/phase5/window_generation_default.json` (H = 6, F = 15, s = 3); no split manifest is produced for the single pilot video (only a `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT` listing); the window study on training data is deferred until independent groups exist. Report: `PHASE5_TASK3_WINDOW_GENERATION.md`.
Order revised from the kickoff proposal using repository evidence: window generation and baselines can be built and tested on the pilot now, but final training depends on scene-diverse data (dataset readiness report). Code lives in a new package area (`pedsafety/forecasting/`), never in the PCG baseline files.

**Compute plan.** Preprocessing, window generation, baselines and all unit tests run on the local CPU (`pcg-baseline`: Python 3.9.25, torch 2.2.1+cpu, numpy 1.26.4). Training targets a Colab T4; batch size, `H`, `F`, `s`, hidden size, heads and K are configuration. No GPU is assumed for tests. `torch_geometric` is **not installed**; recommendation: implement relation-specific GATv2 as a masked dense layer in plain PyTorch (≤ 23 agents per frame on the pilot), avoiding a new dependency; if PyG is preferred, the version/compatibility request is raised first (no installation without approval).

| Task | Title | Blocked? |
|---|---|---|
| P5.1 | Readiness and input contract | done as planning (this kickoff); acceptance = your review |
| P5.2 | Dataset acquisition and scene-diverse preparation | **blocked**: licence archive, subset choice, Colab perception runs |
| P5.3 | Window generation, split manifests, window study | **pilot window generation implemented** (PILOT_SMOKE, H = 6, F = 15; no split); final splits and window study blocked by P5.2 |
| P5.4 | Corrected evaluator and baselines (B0–B3) | **evaluator + B0–B2 implemented and validated on the pilot** (development metrics only); B3 with P5.5; final numbers blocked by P5.2 |
| P5.5 | Heterogeneous GATv2 + dilated TCN encoder | **implemented and smoke-tested on the pilot (untrained)**; torch declared via `requirements-phase5.txt` |
| P5.6 | Multimodal decoder and loss | **implemented** (`pedsafety/phase5/models/{decoder,losses}.py`, K = 5, raw-pixel offsets, masked WTA SmoothL1 + mode cross-entropy; committed `afdf822`) |
| P5.7 | Training/validation infrastructure | **P5.7.1 data adapter, P5.7.2 train-only scaler, P5.7.3 training step, P5.7.4 pilot harness and P5.7.5 neural evaluator implemented and run on the pilot** (20-epoch in-sample pilot training, deterministic, checkpoints and scaler hash-verified; no best-checkpoint selection; `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`); **P5.7.6 independent evaluation `BLOCKED_PENDING_SCENE_DIVERSE_DATA`** |
| P5.8 | Ablations and baseline comparison | blocked by P5.2 (needs independent groups, a validation partition and a frozen split); the in-sample pilot comparison in `PHASE5_TASK7_5_NEURAL_EVALUATION.md` is not a substitute |
| P5.9 | Locked final evaluation and report | blocked by P5.8 |

## P5.1 — Readiness and input contract (this kickoff)
- **Objective:** verify Phase 4 outputs; define input, dataset, evaluation and roadmap documents.
- **Inputs:** Task 3 committed code/schemas, pilot run manifests. **Outputs:** the five `docs/phase5/` documents.
- **Tests:** document review; commands in the kickoff report. **Acceptance:** your approval of the contract and open decisions Q1–Q6.
- **Risks:** stale master files (noted). **Blocked:** no.

## P5.2 — Dataset acquisition and scene-diverse preparation
- **Objective:** obtain enough independent PIE groups and process them through Phase 3 → Task 2 → Task 3.
- **Inputs:** PIE release (Drive), licence terms, Phase 3 Colab notebook, Task 2/Task 3 CLIs. **Outputs:** dataset manifest (videos, sets, archive hashes, licence reference), one Task 3 run directory per video, Phase 3 Task 4 evaluation of at least the pilot.
- **Files:** `configs/phase5/dataset_manifest.json`, `docs/phase5/PHASE5_TASK2_DATASET_PREPARATION.md`; no changes to Phase 3/4 code.
- **Dependencies:** your licence/subset decision; Colab GPU sessions. **Tests:** each Task 3 run passes the input-contract checks; per-video manifests complete.
- **Acceptance:** ≥ 3 independent groups (proposal: sets) with documented independence; all runs validated; storage plan respected (no data in Git).
- **Risks:** Colab time and storage; per-video calibration/identity differences; perception quality unknown until Task 4 evaluation. **Blocked:** yes (external data and approvals).

## P5.3 — Window generation, split manifests and window study
- **Objective:** build samples per input contract §6–§10 and split manifests per evaluation protocol §1.
- **Inputs:** Task 3 run directories; split configuration. **Outputs:** window index files (target track, anchor, frames, provenance), split manifest with hashes, exclusion counts, window-study report (counts vs `s/H/F` on training data only).
- **Files:** `pedsafety/forecasting/{inputs.py, windows.py, splits.py}`, `scripts/build_forecast_windows.py`, `configs/phase5/windows_default.json`, `tests/phase5/test_windows*.py`. Reuses `pedsafety/splitting.py` concepts and `verify_window_isolation`.
- **Tests:** hand-built node tables: window eligibility, no interpolation, gap exclusion, dependency-interval computation, no future edges in inputs, partition isolation (including purge), determinism, contract rejections (hash/NaN/schema).
- **Acceptance:** zero dependency-interval overlap across partitions; exclusion reasons complete; pilot smoke split labelled non-independent.
- **Risks:** too few distinct tracks at long horizons; correlated windows. **Blocked:** final splits by P5.2.

## P5.4 — Corrected evaluator and baselines
- **Objective:** metrics of evaluation protocol §4–§5 and baselines B0–B2 (B3 after P5.5).
- **Files:** `pedsafety/forecasting/{metrics.py, baselines.py}`, extending (not modifying) `pedsafety/evaluation.py` semantics; `tests/phase5/test_metrics.py`, `test_baselines.py`.
- **Tests:** hand-computed ADE/FDE/minADE/MR/brier-minFDE; oracle metrics labelled and never used for selection; CV exact on synthetic constant-velocity tracks; cluster bootstrap reproducibility with a seed.
- **Acceptance:** all metrics documented with units (px); pilot development numbers produced and labelled `smoke_split`.
- **Risks:** pixel thresholds are depth-dependent (scale-aware variant reported). **Blocked:** no (pilot), final numbers by P5.2.

## P5.5 — Heterogeneous GATv2 + dilated TCN encoder
- **Objective:** per-frame relation-specific GATv2 with edge features and masks; causal dilated TCN over history samples.
- **Files:** `pedsafety/forecasting/{graph_encoder.py, temporal_encoder.py}`, `tests/phase5/test_graph_encoder.py`, `test_temporal_encoder.py`.
- **Tests:** permutation equivariance over agents; masked/padded agents have no influence; masked edge features have no influence; relation-specific parameters; causal TCN (output at step k unaffected by later steps); receptive field covers `H`; CPU forward on a 23-agent frame; gradient flow.
- **Acceptance:** tests pass on CPU; parameter count and memory reported for default config.
- **Risks:** overfitting on small data; noisy relative-velocity features. **Blocked:** no.

## P5.6 — Multimodal decoder and loss
- **Objective:** K trajectories + mode probabilities; winner-takes-all + classification loss (and mixture-NLL alternative).
- **Files:** `pedsafety/forecasting/{decoder.py, losses.py}`, `tests/phase5/test_decoder_losses.py`.
- **Tests:** probabilities sum to 1; loss decreases on a synthetic two-mode toy problem; both modes recovered (no collapse) on that toy; K=1 reduces to unimodal regression; NaN-safe with masks.
- **Acceptance:** toy multimodality recovered; diversity/entropy diagnostics implemented. **Risks:** mode collapse. **Blocked:** no.

## P5.7 — Training and validation infrastructure
- **Objective:** reproducible training with validation-only selection.
- **Files:** `pedsafety/forecasting/{dataset.py, train.py, checkpoint.py, logging.py}`, `scripts/train_forecaster.py`, `configs/phase5/model_default.json`, Colab notebook `notebooks/phase5_training_colab.ipynb`; tests `tests/phase5/test_training_smoke.py`.
- **Tests:** fixed-seed determinism on CPU for a few steps; overfit a tiny batch; checkpoint save/load round trip with config and data hashes; normalization fitted on train only (asserted).
- **Acceptance:** pilot smoke training runs on CPU and on a T4; manifest records code commit + dirty flag, config, data and split hashes.
- **Risks:** Colab session limits; CPU-only local runs are slow. **Blocked:** real training by P5.2.

## P5.8 — Baseline comparison and ablations
- **Objective:** run B0–B3 and A1–A10 on validation; choose frozen configuration.
- **Outputs:** validation report with cluster-bootstrap intervals; selection record. **Tests:** experiment manifests complete; no test-partition access (asserted by the data loader).
- **Acceptance:** every claim supported by validation evidence; PCG reported only as a non-comparable historical reference. **Blocked:** yes (P5.2, P5.7).

## P5.9 — Locked final evaluation and report
- **Objective:** one evaluation of the frozen model and baselines on the test partition; final Phase 5 report with limitations.
- **Acceptance:** single test run recorded with hashes; oracle metrics labelled; per-category, per-group and per-interaction breakdowns; image-space limitations stated. **Blocked:** yes (P5.8).

## Cross-cutting rules
No Phase 2/3/4 or PCG files modified; no data, checkpoints or large arrays in Git (a `.gitignore` proposal will be raised before the first large artifact); no package installation without approval; every run has a manifest; no planned experiment reported as executed.
