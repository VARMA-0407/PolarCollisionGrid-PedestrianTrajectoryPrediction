# Phase 5 Architecture: from PIE agent histories to reproducible multimodal pedestrian trajectory predictions

**Status:** implemented and smoke-validated on one pilot video (`PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`). **Coordinate system:** `captured_1920x1080`, image-space pixels only (origin top-left, +u right, +v down). No metric, physical, TTC/PET, calibrated-probability, deployment or real-time claim is made anywhere in this pipeline.

## 1. Data flow

```
PIE video ──Phase 3──► tracked agents ──Phase 4 Task 2──► causal kinematics ──Phase 4 Task 3──► per-frame heterogeneous graphs (nodes, 7 relations)
                                                                                                        │
P5.3  window generation (30 fps -> 10 Hz, H = 6 history, F = 15 future, strict target, masked context)   ▼
        windows.npz + graph_bank.npz  (history graphs shared across windows; pseudo-targets from perceived tracks)
                                                                                                        │
P5.7.1  TrainingWindowDataset  ── collate (history graphs only) ──► history batch ┐                      │
                                                                  future targets ─┼─ supervision only ──┐│
P5.7.2  train-only feature statistics  (fit on training windows) ──► FeatureScaler (encoder INPUT features only)
                                                                                                        │
P5.5  encoder: per-frame relation-aware GATv2 ─► per-agent sequences ─► causal dilated TCN ─► target_summary [B,64]
P5.6  decoder: target_summary + raw pixel anchor ─► K = 5 trajectories [B,5,15,2] (raw pixels) + mode probabilities [B,5]
P5.6  masked winner-takes-all SmoothL1 + mode cross-entropy  ◄── future targets (loss only, after the prediction exists)
P5.7.3  train step: backward, finite checks, clipping, AdamW + cosine schedule
P5.7.4  pilot harness: seeds, checkpoints (model + optimizer + scheduler + RNG), logs, manifests, resume
P5.7.5  neural evaluator: verified checkpoint + scaler ─► history-only inference ─► P5.4 evaluator ([N,K,F,2]) ─► most-probable-mode metrics,
        oracle minADE@5 / minFDE@5 (separate), per-horizon, window- and track-weighted, comparison with B0/B1/B2
P5.7.6  independent evaluation: BLOCKED_PENDING_SCENE_DIVERSE_DATA (no scene-diverse, annotation-verified data exist)
```

## 2. Modules

| Stage | Module | Role |
|---|---|---|
| P5.3 | `pedsafety/phase5/{window_config,window_builder,graph_window_adapter,runner,...}.py` | windows, shared history graph bank, validation (unchanged) |
| P5.4 | `pedsafety/phase5/baselines/`, `pedsafety/phase5/evaluation/` | B0/B1/B2 and the model-agnostic evaluator (unchanged; accepts `[N,K,F,2]`) |
| P5.5 | `pedsafety/phase5/models/{config,scaling,spatial,temporal,encoder}.py` | GATv2 + causal dilated TCN encoder (P5.7.2 adds `FeatureScaler.from_arrays` only) |
| P5.6 | `pedsafety/phase5/models/{decoder,losses}.py` | direct K-mode decoder (raw pixel offsets) and masked WTA loss (unchanged) |
| P5.7.1 | `pedsafety/phase5/training_data.py` | `TrainingWindowDataset`: verified, deterministic window / batch adapter |
| P5.7.2 | `pedsafety/phase5/training_scaler.py` | train-only statistics, deterministic JSON + SHA-256, provenance verification |
| P5.7.3 | `pedsafety/phase5/models/{forecaster,training_step}.py` | `ForecastingModel` (encoder + decoder), `train_step` / `eval_step` |
| P5.7.4 | `pedsafety/phase5/models/{checkpoint,training_harness}.py`, `scripts/phase5_train_pilot.py` | pilot training harness, checkpoints, manifests, resume |
| P5.7.5 | `pedsafety/phase5/models/neural_evaluation.py`, `scripts/phase5_evaluate_neural.py` | neural evaluator on the unchanged P5.4 evaluator |
| P5.7.6 | `pedsafety/phase5/independent_data_status.py`, `scripts/phase5_check_independent_data.py` | availability check; blocked status |
| validation | `scripts/phase5_validate_{training,neural_evaluation}.py` | independent re-verification and recomputation |

**Dependency policy.** PyTorch is declared in `requirements-phase5.txt` and may be imported only under `pedsafety/phase5/models/` (enforced by the extended Phase 2 dependency guard). Everything else, including `training_data.py`, `training_scaler.py`, the baselines and the evaluator, is torch-free. (During P5.7 the untracked `training_scaler.py` imported torch directly and made the guard fail; it now builds its scaler through `FeatureScaler.from_arrays`, with its API and tests unchanged.)

## 3. Tensor contracts

| Tensor | Shape / dtype | Space |
|---|---|---|
| history node features | `[B,6,A,10]` float32 + masks `[B,6,A,4]`, presence `[B,6,A]` | raw Task 3 columns, NaN where masked |
| history edges (per relation) | `bhsd [E,4]`, `edge_attr [E,14]`, `edge_mask [E,14]` | raw Task 3 features |
| encoder derived node input | `[B,6,A,13]` = 9 scaled features + 3 validity bits + is-target | target-centred, train-only standardised, masked → 0 |
| `target_summary` | `[B,64]` | embedding |
| `anchor_uv_px` | `[B,2]` float32 | **raw pixels**, last observed target footpoint |
| decoder offsets / trajectories | `[B,5,15,2]` | **raw pixels** (trajectory = anchor + offset) |
| mode logits / probabilities | `[B,5]` | uncalibrated softmax |
| supervision | `target_future_uv_px [B,15,2]`, `target_future_valid [B,15]`, `target_valid [B]` | raw pixels, loss / evaluator only |
| evaluator input | predictions `[N,5,15,2]`, masks `[N,5,15]`, probabilities `[N,5]`, ground truth `[N,15,2]`, masks `[N,15]` | raw pixels |

## 4. Invariants (each enforced by tests)

1. **History only.** `ForecastingModel.forward(encoder_input, anchor_uv_px, target_valid)` has no future argument; `EncoderInput` has no future field; mutating future arrays leaves predictions bit-identical; a meta-test shows that injecting future information into the history input would be detected.
2. **Raw pixels preserved.** Scaling touches encoder input features only; anchors, offsets, trajectories and targets are never scaled or modified.
3. **Train-only scaling.** Statistics require `fit_partition == "train"`, carry a split identity, source identity and canonical SHA-256, are validity-mask aware, treat constant features with scale 1, and are verified (hash, provenance, equality with the checkpoint's scaler buffers) before any evaluation.
4. **Masks, not fill.** Missing agents and invalid values are masked; nothing is interpolated, forward-filled or merged; invalid targets / windows are excluded and counted, never zero-filled; non-finite values under a valid mask are rejected.
5. **Primary vs oracle.** Primary metrics use the most-probable mode (argmax of the model's own probabilities); minADE@K / minFDE@K select with ground truth, are labelled oracle and reported separately.
6. **Reproducibility.** Seeds drive window order and initialisation; with one CPU thread two runs, and a resumed run, produce byte-identical checkpoints; manifests separate deterministic data-artifact hashes from run-specific metadata.
7. **No overwrite.** Every run writes to a new timestamped directory; existing checkpoints cannot be overwritten.

## 5. Experiment directories

`experiments/phase5_task7_training/<UTC>_pie_video_0004[_determinism_rerun]/` (config, scaler statistics, log, checkpoints, environment and run manifest) and `experiments/phase5_task7_neural_evaluation/<UTC>_pie_video_0004[...]/` (predictions, per-window metrics, results, report, manifest). Checkpoints (`*.pt`) and evaluation arrays (`*.npz`) are git-ignored by narrow rules; their hashes are recorded in the manifests.

## 6. Limitations

One PIE video and one scene group; windows overlap strongly; targets are perceived-track pseudo-targets; all neural results are in-sample. No independent train / validation / test split exists, so no best-checkpoint selection, generalisation estimate or ranking is possible. Image-space pixels are depth-dependent and include camera ego-motion. Mode probabilities are not calibrated. See `PHASE5_TASK7_6_INDEPENDENT_EVALUATION_STATUS.md` for what is needed to lift the restriction.
