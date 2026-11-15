# Phase 5 Task P5.5 — Heterogeneous GATv2 + Causal Dilated TCN Encoder

**Date:** 15 November 2026. **Status:** `COMPLETED_WITH_LIMITATIONS — PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT` (uncommitted; awaiting review).
**Scope:** the spatial–temporal *encoder* only: per-frame heterogeneous interaction graphs → relation-aware GATv2 → per-agent temporal sequences → causal dilated TCN → context-aware agent embeddings. No decoder, trajectory/conflict head, loss, training, calibration, alerting, accuracy figure or baseline comparison exists. Machine-readable companion: `phase5_task5_encoder_report.json`.

> The only measurements in this task are an **engineering forward-pass smoke test of an untrained, randomly initialised encoder** on the P5.3 pilot windows (one PIE video, perceived-track pseudo-targets, image-space pixels). They show shapes, masking, finiteness, determinism, size and runtime. They say nothing about forecasting quality, generalization, or whether the architecture is useful. Label: `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`.

## 1. Starting state and Git
Branch `feature/phase4-task3-interaction-representation`; HEAD `95aa920769bcd0eaa66a90887ff0fe2255bb8c12` ("feat(phase5): add classical trajectory baselines and evaluator", parent `c3137b4`); working tree clean at the start (no staged/unstaged/untracked changes, no stash). No commit, stage or push was made. P5.4 commit verified present.

## 2. Verified interfaces the encoder is built on
* P5.3 `collate_windows` (history only): `node_features [B,6,A,10]` = `u_px, v_px, vel_u_px_s, vel_v_px_s, speed_px_s, heading_sin, heading_cos, box_h_px, frames_since_prev_obs, track_age_frames` (NaN where masked); `node_masks [B,6,A,4]` = `position_valid, velocity_valid, heading_valid, frames_since_prev_obs_valid`; `present [B,6,A]`; `node_type [B,A]` codes 0 pedestrian / 1 vehicle / 2 bicyclist (−1 padding); `track_index [B,A]` (sequence-scoped Task 3 index; slot 0 = target); per-relation COO `bhsd [E,4]` = (window, step, source slot, destination slot), `edge_attr [E,14]`, `edge_mask [E,14]`, `flags [E,6]`. Edge direction: source = neighbour j, destination = target i (Task 3 `edge_index[0]`, `[1]`).
* Edge attributes (Task 3 dictionary): `du_px, dv_px, sep_px, bearing_sin/cos, rel_bearing_sin/cos, sep_over_h, log_h_ratio, dv_row_px, dvel_u_px_s, dvel_v_px_s, closing_rate_px_s, overlap_ios`; relations present by default: P→P, V→P, P→V (+ B→P, P→B, B→V, V→B only with the optional bicyclist context). No V→V and no B→B in the pilot.
* P5.3 pilot: 2,327 windows, 84 tracks; agent slots per window (default context) mean 12, max 22; with bicyclist context mean 15, max 27; at most 18 (23 with bicyclists) nodes in one frame.
* P5.4 evaluator accepts `[N, K, F, 2]` predictions and is unchanged; the encoder produces no trajectories, so it is not yet connected to it.
* torch 2.2.1+cpu (10 threads, no CUDA), numpy 1.26.4, no PyTorch Geometric (none added).

## 3. Package and tensor interface
`pedsafety/phase5/models/`: `config.py` (strict `EncoderConfig`), `scaling.py` (`FeatureScaler`), `spatial.py` (heterogeneous GATv2), `temporal.py` (causal dilated TCN), `encoder.py` (`EncoderInput`, `SpatioTemporalEncoder`, `EncoderOutput`), `smoke.py`. Config: `configs/phase5/encoder_default.json` (`phase5-encoder-config-1.0.0`, canonical SHA-256 `15dbb985…78c1`).

**Input (`EncoderInput`, history only).** `window_id [B]`, `anchor_frame [B]`, `history_frames [B,H]`, `track_index [B,A]`, `node_type [B,A]`, `slot_valid [B,A]`, `present [B,H,A]`, `node_features [B,H,A,10]`, `node_masks [B,H,A,4]`, `edges {relation: EdgeBatch(bhsd, attr, mask[, flags])}`. The dataclass has **no field** for future positions, future frames, future graphs or targets; `from_collated_batch` copies only `HISTORY_KEYS`, so `target_future_*` and `future_frames` of a collated batch can never reach the model; the forward pass rejects anything that is not an `EncoderInput`. Construction validates: shapes/dtypes, slot 0 is a valid pedestrian present at the last history step, present ⇒ slot valid, distinct non-negative tracks per window, history frames strictly increasing and ending at the anchor, finite values wherever a mask is true, edge endpoints in range.

**Derived, causal model inputs.** Node (13 = 9 + 4): target-centred position (u, v minus the target's position at the *last* history sample), velocity (2), speed, heading sin/cos, box height, `frames_since_prev_obs`, scaled by explicit constants with masked entries set to exactly 0, plus validity bits for velocity, heading and gap, plus an is-target bit. `track_age_frames` is not used (unbounded past-only counter, input contract §10); absolute pixel position is not an input. Edge (28): the 14 scaled attributes (masked → 0) and their 14 validity bits (+6 flags if `use_edge_flags`). `frames_since_prev_obs` keeps its Task 3 meaning: the number of **missing native frames** since the same track's previous accepted observation (0 = consecutive; masked for a track's first observation) — it is relative to the previous 30 fps frame, not the previous 10 Hz sample.

**Output (`EncoderOutput`).** `sequence_embeddings [B,A,H,D]` + `presence [B,A,H]` (exactly 0 / False where an agent is absent or the slot is padding), `spatial_embeddings [B,H,A,D]` (per-frame GATv2 output), `target_summary [B,D]` (target, last step), `last_step_embeddings [B,A,D]` + `last_step_mask [B,A]`, and the identifiers a decoder needs (`track_index`, `node_type`, `slot_valid`, `is_target`, `window_id`, `anchor_frame`, `history_frames`, `receptive_field`). D = 64.

## 4. Spatial encoder (relation-aware GATv2, plain PyTorch)
Flattened over batch and history step, so every frame is an independent graph (edges exist only inside a frame). For destination node i of type t(i), relation r = (s → d) with d = t(i), heads k = 1..K (K = 4, per-head width c = 16); `W` matrices are **relation-specific**, `self` matrices **type-specific**:

```
ê_ij        = [ scaled edge attributes (masked -> 0) , validity bits (, flags) ]                     (28-dim)
z_ij^{r,k}  = LeakyReLU( Ws^{r,k} h_j + Wd^{r,k} h_i + We^{r,k} ê_ij )                               (GATv2: nonlinearity before the attention vector)
logit       = a^{r,k} · z_ij^{r,k}
alpha_ij    = softmax_j over the VALID incoming edges (j -> i) of the SAME relation r
m_ij        = alpha_ij ( Ws^{r,k} h_j + Wm^{r,k} ê_ij )
agg_i^r     = Wo^r concat_k Σ_j m_ij                                  (0 if i has no valid incoming edge of r)
h_i'        = LayerNorm( h_i + GELU( Wself^{t(i)} h_i + Σ_r agg_i^r ) )
```
Edge attributes enter **both** the attention logit (through `We`) and the message (through `Wm`); masked attributes are exactly 0 and their validity bits tell the model they are missing, so NaN can never reach the logits. The self/root term `Wself h_i` keeps nodes with no valid incoming edge fully representable. Attention is normalised per destination **and per relation** (relations do not compete in one softmax; their messages are summed), so relation identity is preserved and the neighbour set is exactly the valid incoming edges of that relation. Edges with an absent/padded endpoint are dropped; invalid nodes output exactly 0 and send nothing. Input projection is type-specific (preserves node-type distinctions). Two layers, residual + per-node LayerNorm (no statistic is shared across nodes, agents or time). Permutation-equivariant in node order, invariant to edge order (up to float summation). No pixel radius, kNN pruning or other pruning is applied (the complete Task 3 candidate set is used); nothing is interpreted as physical.

## 5. Temporal encoder (causal dilated TCN)
Per agent slot, input = the 64-dim spatial embedding sequence over the H history samples plus the presence bit; one **causal** convolution per dilation level (`kernel 3`, dilations `[1, 2]`), left-only zero padding `(k−1)·d`, residual + per-timestep LayerNorm over channels, absent samples masked (via `torch.where`) before every convolution and on every output. No normalisation across time or agents.
* **Receptive field** = 1 + (k−1)·Σd = 1 + 2·3 = **7** samples ≥ H = 6: the output at the last history sample can depend on all six samples (the seventh position is left padding). `EncoderConfig.check_history` (and the forward pass) reject a history longer than the receptive field unless `require_full_history_coverage` is set to false. H comes from the window configuration / input tensors, not from the model.
* Causality is verified by prefix-invariance, truncation equality and an empirical dependency test (the output at step T−1 changes iff the perturbed input is within the last RF samples).
* Absent observation: zero input plus presence bit, zero output, never a copied or fabricated embedding; a gap is visible downstream through the mask.

## 6. Normalization and numerical stability
`FeatureScaler` applies `(x − offset)/scale` to valid values and writes exactly 0 to masked ones. Default source `NOMINAL_DEVELOPMENT_DEFAULT_NOT_FITTED`: fixed constants from the 1920×1080 / 30 fps geometry (positions/offsets/separations 1000 px, velocities 500 px/s, box height 1080 px, gap 30 frames, `sep_over_h` 10); **nothing was computed from the pilot or any dataset**. The future workflow supplies `TRAIN_PARTITION_STATISTICS` (must declare `fit_partition == "train"` and `statistics_sha256`; anything else is refused with `SCALER_NOT_TRAIN_ONLY`). Non-finite values flagged valid are rejected; masked NaN/1e30 values are shown to have no influence; synthetic backward passes give finite gradients and every parameter of the default configuration receives a gradient. Model construction seeds weights with `init_seed` inside `fork_rng` (the caller's global RNG state is untouched).

## 7. Relations and node types
Default context = pedestrian + vehicle with P→P, V→P, P→V (as the P5.3 default). Bicyclist nodes (COCO `bicycle` detections, not evidence of a rider) are off by default and require `node_types` to include `bicyclist`, `bicyclist_context_enabled = true` and the four bicyclist relations; the default encoder rejects bicycle-derived input (`UNSUPPORTED_NODE_TYPE` / `UNSUPPORTED_RELATION`). V→V needs `allow_vehicle_vehicle = true` *and* a Task 3 run that produced it (not the pilot); B→B is never allowed. Unsupported relations in an input are rejected, not silently dropped.

## 8. Parameters and size (default configuration)
Total **144,128** parameters (576 KB float32): spatial 114,816 (input projections 1,792 + norm 128; per layer 56,448 = three relations × 16,000 + type-specific self projections 8,320 + norm 128), temporal 29,312 (blocks 16,832 + 12,480). Enabling bicyclist context (7 relations, 3 node types): 281,344. Per-batch activation memory was not isolated; process peak working set during the smoke run: 347 MB (includes Python, numpy, torch and the loaded P5.3 dataset).

## 9. Tests (`tests/phase5`)
| File | Tests | Covers |
|---|---|---|
| `test_encoder_spatial.py` | 16 | shapes for variable node/edge counts (incl. 0 edges); every approved relation changes exactly its destination nodes (others bit-identical); unsupported/disabled relations and node types rejected; invalid nodes zero and edges touching them dropped; nodes without incoming edges; node-order equivariance; edge-order invariance; segment-softmax normalisation over incoming neighbours only (stability, shift invariance); edge attributes affect attention (`We`) and messages (`Wm`); relation-specific parameters and gradients; type-specific projection; finite gradients; determinism; frame independence |
| `test_encoder_temporal.py` | 13 | shapes for H ∈ {1,2,4,6,10,16}; causal prefix invariance (even when later presence changes); truncation equality; left-only padding; receptive-field formula **and** empirical dependency for four (kernel, dilation) schedules; config RF vs H; absent steps exactly zero (NaN/garbage ignored); no filling of gaps; presence bit; agent independence; finite gradients, zero gradient at absent inputs; validation; determinism |
| `test_encoder_integrated.py` | 25 (+1 CUDA test skipped: no GPU) | shapes/roles/masks; variable agent counts; padding invariance; slot-permutation equivariance (track identity travels with embeddings); duplicate-track rejection; no future fields, collated future arrays ignored and mutating them changes nothing; **leakage-sensitivity meta-test** (injecting future information into history changes outputs); translation invariance (no absolute position); masked-value non-influence (edge NaN/1e30, velocity/heading/gap garbage, absent-node garbage) and a valid-gap-value sensitivity check; non-finite rejection; integrated causality; centring uses only the last sample; node-type distinctions; default bicycle exclusion and opt-in; flags; configurable H and RF guard; input validation; scaler train-only interface; forward/backward smoke; deterministic and seed-isolated; real P5.3 adapter path (synthetic scene through Task 2 → Task 3 → P5.3) incl. optional bicycle context and mutated future arrays |
| `test_encoder_smoke.py` | 5 | manifest/results, determinism, independent validation, tamper and forgery detection, P5.3 hash rejection, output protection, CLI exit codes |

In-memory mutation checks (not committed) confirmed the tests catch: symmetric (non-causal) padding (4 tests fail), masks ignored for edge attributes (1), absolute position as an input (1).

## 10. Pilot forward-pass smoke test — `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`
Experiment: `experiments/phase5_task5_encoder_smoke_pie_video_0004/20261115T044455Z_pie_video_0004/` (`primary`, `determinism_rerun`, `validation_report.json`). CPU, eval mode, `torch.no_grad`, batch 64, seed 0, **no training**.

| Item | Value |
|---|---|
| P5.3 input | `…/20261115T035825Z_pie_video_0004/primary`; manifest SHA-256 `3369ea80…1e8`; `windows.npz` `49b4ca39…`, `graph_bank.npz` `fc0821a3…`, `rejected_candidates.npz` `6efbe92a…`, `smoke_manifest.json` `02c21aab…`; window config `65eebeed…` |
| Encoder config | `configs/phase5/encoder_default.json`, canonical SHA-256 `15dbb985…78c1` (file `a2419c69…`) |
| Windows / batches | 2,327 / 37; max agent slots 22; 130,720 present node-steps |
| Output shapes | `sequence_embeddings [B,A,6,64]` (first batch `[64,16,6,64]`, last `[23,9,6,64]`), `target_summary [B,64]`, `spatial_embeddings [B,6,A,64]` |
| Numerical validity | 0 non-finite values; 0 non-zero values at absent steps; slot 0 equals the window's target track in all windows; target present at all steps |
| Output tensor hashes | sequence `a44c09a6…bc8b`, target summary `b7170d2a…2a15`, spatial `aece42ab…d7be` (float32 LE bytes, batch order; 14,603,904 / 148,928 / 14,603,904 values) |
| Optional bicycle context | first 128 windows, 281,344-parameter configuration, 0 non-finite, sequence hash `2dd957b0…da6f4` |
| Result artifact | `smoke_results.json` SHA-256 `fd485ece…d01b` (7,269 B); **identical in the determinism rerun** |
| Independent validation | `phase5_validate_encoder_smoke.py`: **PASS 25/25** on both runs (hashes, P5.3 re-verification, config hash, labels/claims, full recomputation of all output hashes and counts, manual parameter count, numpy re-check of presence/zero/finite/slot-0, no future/target fields) |
| Runtime (CPU, 10 threads) | forward incl. collation 4.3 s for 2,327 windows (0.110 s mean / 0.209 s max per batch of 64); optional pass 0.35 s; not a real-time claim |
| Peak memory | 347 MB process working set (350 MB rerun) |
| Device | CPU (CUDA not available here; the optional CUDA test was skipped) |

No target, future array or offline kinematics was read; no metric, B0–B2 comparison or accuracy is computed or implied.

## 11. Limitations and unresolved decisions
1. Untrained, randomly initialised encoder: output values are not features; nothing shows the architecture helps forecasting. The A-series ablations (GATv2 vs simpler aggregator, TCN-only, relation/edge-feature ablations) remain for P5.8.
2. One video / one scene group: nothing here supports generalization; nominal scaling constants are engineering defaults (not tuned, not fitted); train-only statistics await P5.2/P5.7.
3. Design choices made without data, open for review: per-relation (not joint) softmax; summation of relation messages; two spatial layers; 4 heads; hidden 64; TCN kernel 3 / dilations [1, 2] (RF 7 vs H = 6, only one sample of slack); LayerNorm placement; `track_age_frames` and the six edge flags excluded by default; target-centred position as the only position input (an absolute-position or scene-context ablation is possible later).
4. Image-space features: relative image velocity and closing rate mix agent motion with moving-camera ego-motion and perspective; one-frame Task 2 velocities carry ~48 px/s jitter; `sep_over_h`/`log_h_ratio` are heuristic scale proxies. Nothing is a physical quantity.
5. Absent context agents are zero-masked with a presence bit; a "steps since last seen" feature was not added. `frames_since_prev_obs` reflects native-frame gaps only.
6. Windows with the target absent at an intermediate history step are not produced by P5.3 (strict policy), so the encoder's tolerance of an absent target at earlier steps is only unit-tested.
7. CUDA code paths are written device-agnostically (`EncoderInput.to`) but were not exercised (no GPU). Batched sparse attention scales with edges (≈ 30–60 k per batch of 64 windows here); no performance optimisation was attempted.
8. The encoder is not connected to the P5.4 evaluator: that needs the P5.6 decoder.
9. **Dependency declaration (resolved, Option A).** The Phase 2 Task 7 dependency guard initially rejected `torch` (imported by `pedsafety/phase5/models`) as undeclared, and a first attempt to declare it in `requirements-pedsafety.txt` conflicted with the pinned-lines guard and was reverted. Resolution: new `requirements-phase5.txt` (`-r requirements-pedsafety.txt` + `torch>=2.2`, the bound already used in `requirements-perception.txt`; nothing installed or upgraded; tested only with torch 2.2.1+cpu) and a path-scoped extension of `test_every_third_party_import_of_the_package_is_declared`: **only** `pedsafety/phase5/models` may additionally use `requirements-phase5.txt`; the rest of `pedsafety/` (including the other `pedsafety/phase5` modules) and the perception frontend keep their existing checks; `requirements-pedsafety.txt` and the pinned-lines test are unchanged. Five new guard tests cover the narrow file, the exact exemption scope and the whitelist behaviour; end-to-end probes confirmed the real guard still fails on an undeclared import inside `models` and on `torch` outside `models`.

## 12. Deviations
* Package `pedsafety/phase5/models/` (the accepted Phase 5 layout) instead of the roadmap's `pedsafety/forecasting/`; config `encoder_default.json` (the roadmap's `model_default.json` stays reserved for P5.7).
* No `.gitignore` change: the smoke experiment stores only JSON (output tensors are represented by hashes, not saved).
* The roadmap tests "CPU forward on a 23-agent frame" is covered by the pilot run (frames up to 18 default / 23 optional nodes) and synthetic graphs of up to 50 nodes.

## 13. Acceptance
| Criterion | Result |
|---|---|
| Plain-PyTorch relation-aware GATv2 with edge attributes, masks, root term, per-relation normalisation, permutation equivariance | PASS |
| Causal dilated TCN with left-only padding, documented schedule and receptive field, presence masks | PASS |
| History-only interface; future target arrays cannot reach the model; leakage-sensitive tests | PASS |
| Masked/invalid values cannot contaminate outputs; finite outputs and gradients | PASS |
| Synthetic tests (spatial, temporal, integrated) and real P5.3 adapter path | PASS (59 new passing tests + 1 skipped CUDA test) |
| Pilot smoke test with hashes, shapes, runtime, memory, validity; repeatable; independently validated | PASS (25/25; identical reruns) |
| No training, decoder, head, accuracy, baseline comparison, generalization or physical claim | PASS |
| Protected files unchanged; no commit/push | PASS (section 14) |
| Relevant regression suite | PASS apart from the known pre-existing `TTC_mask` failure (dependency guard resolved, section 11, item 9) |

**P5.5 status: `COMPLETED_WITH_LIMITATIONS`** — implementation, tests and smoke evidence meet the criteria; the dependency-declaration guard failure found earlier is resolved (section 11, item 9); remaining limitations are those listed above.

## 14. Tests run and repository state
| Command | Result |
|---|---|
| `python -m pytest tests/phase5 -q -p no:cacheprovider` | **141 passed, 1 skipped** (82 before P5.5 + 59 new; skipped = CUDA test, no GPU) |
| `python -m pytest tests/phase5 tests/phase4 tests/phase3 tests/task7 tests/phase2 -q -rf -p no:cacheprovider` | **1150 passed, 7 skipped, 1 failed** (119 s) after the dependency resolution; before it: 1144 passed, 2 failed |
| failure 1 (known, pre-existing, unrelated) | `tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension` (PCG `TTC_mask` defect) |
| (resolved) former failure 2 | `tests/task7/test_stage61_dependencies_version.py::test_every_third_party_import_of_the_package_is_declared` passes after Option A (section 11, item 9); `tests/task7`: 586 passed |
| `python scripts/phase5_validate_encoder_smoke.py --run …/primary` | PASS 25/25 (also on `determinism_rerun`) |
| `git diff --check` | exit 0 (only Git's LF→CRLF notices) |

Skips: cv2 / ultralytics not installed in `pcg-baseline` (5), original notebook absent (1), no CUDA device (1).

**Git:** branch `feature/phase4-task3-interaction-representation`, HEAD `95aa920` (unchanged); nothing staged or committed. Modified tracked files after the dependency resolution: `docs/phase5/PHASE5_IMPLEMENTATION_ROADMAP.md` and `tests/task7/test_stage61_dependencies_version.py`; new `requirements-phase5.txt`. New untracked: `pedsafety/phase5/models/`, `configs/phase5/encoder_default.json`, the three `scripts/phase5_*encoder_smoke*.py`, `tests/phase5/{enc_helpers,test_encoder_*}.py`, this report and its JSON, `experiments/phase5_task5_encoder_smoke_pie_video_0004/` (JSON only). No `.gitignore` change. Protected files (Phase 2/3/4 source, tests and outputs; P5.3 and P5.4 code, configs, reports and experiment outputs; baselines and evaluator; master files; `requirements-pedsafety.txt`, `requirements.txt`, `requirements-perception.txt`, `requirements-dev.txt`) verified unchanged against HEAD.
