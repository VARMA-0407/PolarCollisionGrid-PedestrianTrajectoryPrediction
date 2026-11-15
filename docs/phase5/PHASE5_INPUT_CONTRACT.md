# Phase 5 Input Contract: Phase 4 Task 3 → Phase 5 Forecasting

**Version:** 0.2.0 (revised for P5.3). **Date:** 15 November 2026. **Status:** `SECTIONS 6–10 IMPLEMENTED FOR PILOT_SMOKE (P5.3, uncommitted, under review)`; the rest remains proposed.
**Revision 0.2.0 (P5.3):** the approved window configuration is **H = 6** history samples (t−0.5 … t, 0.5 s first-to-last), F = 15, stride 3 at 30 fps (10 Hz) — superseding the kickoff's provisional H = 5. Stale H = 5 counts were removed (section 6); the dependency interval/purge values are given for H = 6 (section 10); the bicycle semantic limitation is added (section 8); the Task 3 input is the regenerated run from commit `5d58ab7` (section 2, 13). Implementation report: `PHASE5_TASK3_WINDOW_GENERATION.md`.
**Branch/commit at writing:** `feature/phase4-task3-interaction-representation`, `5d58ab7146ea4611733393769071b02a6bf9d187`.
Source of truth for every field below: the committed Task 3 code (`pedsafety/interaction/`), its schemas (`docs/phase4/schemas/interaction_feature_dictionary.schema.json`, `interaction_run_manifest.schema.json`) and the pilot run's `feature_dictionary.json` (`interaction-feature-dictionary-1.0.0`). No field is invented; derived Phase 5 quantities are marked **[derived]**.

## 1. Scope
Phase 5 forecasts future **image-space** footpoint trajectories from past observations only. Model input = Task 3 per-frame nodes and relation-specific edges over an observation window; targets = future node positions of the same track. Image space only (`captured_1920x1080`, pixels, pixel/s, seconds, image-plane radians); `world_position_valid = false`; no world coordinates, metric distances, TTC/PET, collision probabilities or ego-motion compensation.

## 2. Required input files (one Task 3 run directory per video)
| File | Content | Required checks before use |
|---|---|---|
| `run_manifest.json` | provenance, counts, claims, output hashes | validates against `interaction_run_manifest.schema.json`; `status = completed`; `inputs.unchanged_after_run = true`; `validation.all_input_contracts_passed = true`; `validation.edges_equal_theoretical_bound = true`; all claims false; every output hash and size matches |
| `nodes.npz` | node table | hash = manifest; dtypes/shapes below; NaN exactly where masks are False |
| `edges_<src>__<dst>.npz` (7 by default) | relation-specific edges | hash = manifest; dtypes/shapes below; endpoints in the same frame as the edge |
| `feature_dictionary.json` | names, formulas, units, masks | `dictionary_version = interaction-feature-dictionary-1.0.0`; validates against its schema; feature/flag name lists equal the arrays' `feature_names`/`flag_names` |
| `tracks.json` | track index ↔ track id, node type, source labels, first/last frame | track count = `max(track_index)+1` |
| `config_resolved.json` | Task 3 configuration | hash = manifest |

Pilot run (PIE `set01/video_0004`): `experiments/phase4_task3_pie_video_0004/20261114T214634Z_pie_video_0004/primary/` (manifest SHA-256 `e320ea6b…f28a`, feature dictionary `d3cb194c…`, tracks `01cdb886…`). Its `.npz` arrays are gitignored and exist locally; the manifests, dictionary and diagnostics are committed.
**Phase 5 input (P5.3):** the regeneration of that run from committed code `5d58ab7`, `experiments/phase4_task3_pie_video_0004/20261115T034308Z_pie_video_0004_regen_5d58ab7/primary/`; the 13 data artifacts listed in the manifest's `outputs` (nodes, 7 edge files, tracks, feature dictionary, diagnostics, summary, resolved config) are byte-identical to the original run (same SHA-256); the run manifest itself is not byte-identical, differing only in run-specific metadata (`git.commit`, `started_utc`, `finished_utc`, `command` output path, `runtime_s`, `peak_memory_bytes`). The original run is preserved unchanged.

**Not inputs:** the Task 2 `offline_kinematics.jsonl` (never read; Task 3 refuses it); Phase 3 raw videos; PIE annotation XML (ground truth for Phase 3 evaluation, not a model input; see Q2 in the kickoff report for its possible role as a target source).

## 3. Arrays, columns and types (exact)
**`nodes.npz`** — N nodes, ordered by frame, then natural track order:

| Array | dtype, shape | Meaning |
|---|---|---|
| `frame` | int64 (N) | video frame index |
| `timestamp` | float64 (N) | seconds, `frame / fps` (Phase 3 `timestamp_source = frame_index_over_fps`) |
| `track_index` | int32 (N) | index into `tracks.json` (sequence-scoped identity; never joined across videos) |
| `node_type` | int8 (N) | 0 pedestrian, 1 vehicle, 2 bicyclist (`node_type_names`) |
| `source_line` | int64 (N) | provenance into Phase 3 `agent_states.jsonl` |
| `features` | float32 (N, 10) | `u_px`, `v_px`, `vel_u_px_s`, `vel_v_px_s`, `speed_px_s`, `heading_sin`, `heading_cos`, `box_h_px`, `frames_since_prev_obs`, `track_age_frames` |
| `boxes` | float32 (N, 4) | `box_x_min_px`, `box_y_min_px`, `box_x_max_px`, `box_y_max_px` |
| `border` | bool (N) | inherited Task 2 `bbox_touches_border` (possible truncation) |
| `masks` | bool (N, 4) | `position_valid`, `velocity_valid`, `heading_valid`, `frames_since_prev_obs_valid` |

**`edges_<src>__<dst>.npz`** — E edges, ordered by frame, destination node, source node:

| Array | dtype, shape | Meaning |
|---|---|---|
| `relation` | str (2) | (source type, destination type) |
| `frame` | int64 (E) | frame of both endpoints |
| `edge_index` | int64 (2, E) | row 0 source node j, row 1 destination node i (global node indices) |
| `edge_attr` | float32 (E, 14) | `du_px`, `dv_px`, `sep_px`, `bearing_sin`, `bearing_cos`, `rel_bearing_sin`, `rel_bearing_cos`, `sep_over_h`, `log_h_ratio`, `dv_row_px`, `dvel_u_px_s`, `dvel_v_px_s`, `closing_rate_px_s`, `overlap_ios` |
| `edge_mask` | bool (E, 14) | validity per feature |
| `flags` | bool (E, 6) | `scale_truncation_affected`, `sep_over_h_le_2`, `sep_over_h_le_5`, `sep_over_h_le_10`, `knn_4`, `knn_8` |

Relations present by default: P→P, V→P, P→V, B→P, P→B, B→V, V→B. V→V only if the Task 3 run enabled it; B→B is not represented (Task 3 open item). Missing values are NaN with a False mask; Phase 5 must never read a value whose mask is False.

## 4. Coordinate conventions and units
Origin top-left, +u right, +v down; positions px; velocities px/s (Task 2 causal backward difference over **one frame**, 1/30 s); headings image-plane radians from +u toward +v (clockwise on screen); time s. Edge displacement `p_j − p_i` for edge j → i; closing rate positive when **pixel** separation shrinks. These are image-plane quantities: on PIE's moving vehicle-mounted camera they mix agent motion with ego-motion and perspective (equal pixel distances are unequal road distances; image velocity is not physical velocity). `sep_over_h` and `log_h_ratio` are heuristic scale proxies, not calibrated depth.

## 5. Sampling frequency and timestamp handling
- Native rate: 30 fps (Phase 3 manifest `fps = 30.0`, container FPS; constant-frame-rate assumption). Timestamps must equal `frame / fps` (checked; one timestamp per frame is a Task 3 contract).
- Phase 5 samples windows on the frame grid with a configurable **step `s` frames** (sampling interval `s/30` s). Time deltas are computed from timestamps, never from nominal FPS alone.
- **[derived] history displacement features**: if `s > 1`, Phase 5 may compute per-sample displacement `(p(t) − p(t − s)) / (t − (t − s))` from positions inside the history (causal), because Task 2 velocities are one-frame derivatives with ~48 px/s per-axis jitter (Task 2 estimate). Such features are labelled derived, use only history samples, and are masked when either endpoint is absent.

## 6. Observation and prediction windows (definition; values provisional)
For target track `a`, anchor frame `t0` (last observed sample), history length `H`, horizon `F`, step `s`:
```
history frames  h_k = t0 − (H − 1 − k)·s,   k = 0 … H−1
target frames   f_m = t0 + (m + 1)·s,        m = 0 … F−1
```
**Approved configuration for P5.3** (`configs/phase5/window_generation_default.json`, `phase5-window-generation-config-1.0.0`):

| Parameter | Approved value | Rationale | Alternative to test later |
|---|---|---|---|
| `s` | 3 frames (30 fps → 10 Hz) | reduces 30 Hz jitter and correlation; common practice in trajectory prediction | 1 (30 Hz), 5 (6 Hz) |
| `H` | **6 samples**: t−0.5, t−0.4, t−0.3, t−0.2, t−0.1, t (0.5 s between first and last sample) | pilot tracks are short (median contiguous run 7 frames); a 0.5 s observation matches the PIE benchmark convention reported in the literature (to be verified against the PIE paper) | 11 (1.0 s span) |
| `F` | 15 samples: t+0.1 … t+1.5 (1.5 s after the last history sample) | same convention | 10 (1.0 s), 20 (2.0 s) |
| anchor grid | global: `frame % 3 == 0` (anchor stride 3, phase 0) | every window samples frames on one shared grid, so per-frame graphs are stored once and shared by all target pedestrians | per-track phase (see readiness §3) |

**Reconciliation (P5.3).** The kickoff used a provisional `H = 5` described as "first-to-anchor span 0.4 s, i.e. a 0.5 s observation". That description counted the observation as H × Δt; the approved definition is the **first-to-last span** (H − 1) × Δt = 0.5 s, hence H = 6. The kickoff's pilot estimate "2,482 pedestrian windows from 88 tracks at (s=3, H=5, F=15)" is **superseded** and must not be quoted. Recalculated feasibility estimates for H = 6 (exploratory counts from `nodes.npz`, not a dataset): 2,380 windows / 83 tracks with a per-track anchor phase (the kickoff's counting convention), 2,327 windows / 84 tracks with the global anchor grid adopted here. The actual P5.3 implementation count is reported separately in `PHASE5_TASK3_WINDOW_GENERATION.md` and `phase5_task3_window_report.json`.

The historical PCG configuration (6 + 6 samples at 2 Hz on HBS) is a reference point only; at that setting the pilot yields 211 pedestrian windows from 27 tracks (exploratory count, kickoff).

## 7. Missing-observation, track-gap and continuity policy
- **No interpolation, no forward-fill, no stationary assumption.** A track absent at a sampled frame is absent.
- **Target agent history:** the target track must have a node (valid position) at **all H** history frames (initial strict policy; a masked-history variant may be studied later with explicit masks).
- **Target future:** primary metrics require valid positions at **all F** target frames; windows failing this are excluded and counted (coverage reported). Partial-target windows may be reported separately with masks, never mixed silently.
- **Velocity validity:** node velocity is used only where `velocity_valid`; at the first history sample a valid velocity requires the frame `h_0 − 1` (dependency interval, section 10).
- **Track gaps:** `frames_since_prev_obs` and `track_age_frames` are available features; gaps between sampled frames (for `s > 1`) do not invalidate positions but do invalidate the one-frame Task 2 velocity at a sample whose previous frame is missing (its mask is already False).
- **Fragments / possible ID switches:** never merged; a fragmented identity simply ends one track and starts another (Task 2 indicators may be used to report affected windows, not to repair them).
- **Neighbours:** a neighbour is present at a history frame iff it has a node there; neighbours appearing only in the future are never visible.

## 8. Agent filtering policy
- Primary forecasting targets: **pedestrians** (project focus). Vehicles and bicyclists are context nodes; forecasting them is a secondary, separately reported experiment.
- Optional quality filters (configurable, default off, reported if used): exclude target windows with `border = true` at `t0` (possible truncation); minimum box height. Filters must be fixed on training/validation data, never tuned on test.
- No filtering by future behaviour.
- **Bicycle semantic limitation (P5.3).** Task 3 `bicyclist` nodes come from the COCO `bicycle` detection class: Phase 3 maps `bicycle` → `bike` (`pedsafety/perception/config.py` `DEFAULT_LABEL_MAP`), and the agent-state mapping turns `bike` into `bicyclist` by an explicitly recorded assumption (`pedsafety/agent_state.py`). On the pilot all 7,049 bicyclist nodes carry source label `bike`. A bicycle detection does **not** establish a person riding it (parked bicycles are possible). Phase 5 therefore keeps these nodes and their relations unchanged in storage, flags them `semantically_uncertain`, excludes them from the default model context (default context = pedestrian, vehicle; bicyclist = optional context) and never requires them for a window. The approved Task 3 relation schema is unchanged.

## 9. Graph construction and batching
- One heterogeneous graph per history frame `h_k`: node set = Task 3 nodes at `h_k` (all types); edges = Task 3 edges with `frame = h_k` for the enabled relations; per-feature masks carried as tensors.
- **Edges at target frames are never loaded into model inputs.** Future node positions are read only into the target tensor.
- Variable agent counts: batch as dense padded per-frame tensors with an agent mask (pilot maximum 23 agents per frame), or as concatenated COO graphs; the maximum agent count is configurable and an overflow is an error, not a silent truncation.
- Relation identity is preserved (relation-specific attention parameters in GATv2); V→V and B→B absent unless the Task 3 run provides them.
- Sensitivity flags (`sep_over_h_le_*`, `knn_*`) may define edge subsets for ablations; the complete candidate set is the default.

## 10. Dependency interval (for splits and purges)
A window for target `a` at anchor `t0` depends on source frames
```
[ h_0 − 1 , f_{F−1} ]  =  [ t0 − (H−1)·s − 1 ,  t0 + F·s ]
```
(the `−1` is the Task 2 one-frame backward difference behind velocities and Task 3 relative velocities at `h_0`). Any [derived] feature with a longer lookback extends the interval accordingly. Partitions must not share any frame of this interval (evaluation protocol).

**Worked values for the approved H = 6, F = 15, s = 3** (verified against `WindowConfig` in P5.3):
```
history frames  h_k = t0 − (5 − k)·3,  k = 0..5   ->  t0−15, t0−12, t0−9, t0−6, t0−3, t0
future frames   f_m = t0 + (m + 1)·3,  m = 0..14  ->  t0+3, t0+6, ..., t0+42, t0+45
sampled span    [t0−15, t0+45]  = 61 frames (60 frame intervals = 2.0 s)
dependency      [t0−16, t0+45]  = 62 frames
```
*Why `t0 − 16` and not `t0 − 15`.* Node velocity (`vel_u_px_s`, `vel_v_px_s`, `speed_px_s`), heading (`heading_sin/cos`) and the edge features derived from them (`dvel_*`, `closing_rate_px_s`, `rel_bearing_*`) at the oldest history sample `h_0 = t0 − 15` are Task 2 causal one-frame backward differences `(p(h_0) − p(h_0 − 1)) · 30`; they use the observation at frame `t0 − 16` (and are masked invalid when that frame is missing, since `max_derivative_gap_frames = 0`). Frame `t0 − 16` therefore influences the window's input values although it is not a sampled frame. Position, box and all other edge features use only the sampled frame itself.

*Purge check.* Two windows with anchors `a < b` have disjoint dependency intervals iff `b − 16 > a + 45`, i.e. `b − a ≥ 62 = (H − 1 + F)·s + 2 = (5 + 15)·3 + 2` = the interval length. "Minimum purge gap = 62 frames" means: the last anchor of an earlier chronological partition and the first anchor of the next must differ by at least 62 frames (equivalently, no frame of `[a − 16, a + 45]` of any earlier-partition window may appear in a later-partition window's interval). On the P5.3 anchor grid (`t0 % 3 == 0`) the smallest admissible anchor difference is therefore **63** frames. Without the velocity lookback it would be 61. (Kickoff H = 5 values, `[t0 − 13, t0 + 45]` / 59 frames, are superseded.)

*Scope caveat (unbounded past-only counters).* Two node features summarise a track's earlier existence and are not bounded by this interval: `track_age_frames` (= frame − first frame of the track) and `frames_since_prev_obs` (frames since the previous observation, any distance back). Both are causal (past-only) and never carry future information, so they do not create forward leakage. They matter only for **chronological blocks within one video**: a track that crosses a block boundary carries counter values computed from frames in the earlier block. Splits by independent group (whole videos/sets, the planned protocol) are unaffected because tracks never cross videos. If chronological blocks are ever used, either exclude windows whose target or context tracks started in another block, or drop/recompute these two counters per block. P5.3 creates no split, so this is a documented constraint for P5.2+/P5.7, not a P5.3 defect.

These values must be recomputed whenever H, F, s or a derived-feature lookback changes, and applied when independent scene groups become available.

## 11. Feature normalization
- Target-centred translation: subtract the target's position at `t0` from all history positions, neighbour positions and target future positions **[derived]** (image axes kept; no rotation by heading initially because headings of slow agents are noise-dominated).
- Scale statistics (per feature, per node/edge type) **fitted on the training partition only**, frozen and stored as a hashed artifact; applied unchanged to validation and test.
- Masked values excluded from statistics; NaN replaced by 0 **only after** normalization and only together with the mask tensor passed to the model.

## 12. Invalid-input rejection (fail, never repair)
Reject a Task 3 run if: manifest schema invalid or claims not all false; any output hash/size mismatch; dictionary version or feature/flag names differ; coordinate system ≠ `captured_1920x1080`; NaN where a mask is True or a finite value where it is False; edge endpoints in a different frame or of the wrong type; `edges_equal_theoretical_bound` false; any world-coordinate value present. Reject a window if it violates section 7. Every rejection is counted with a reason code.

## 13. Provenance and manifest requirements
Every Phase 5 dataset/window artifact records: Task 3 run manifest SHA-256 and every consumed `.npz` hash; feature dictionary version and hash; Phase 5 code commit **and a working-tree-dirty flag**; configuration and its hash; split manifest hash; window parameters (`H`, `F`, `s`, stride); counts of eligible/excluded windows by reason; coordinate system and units; claims (all physical/metric claims false).

**Known provenance limitation:** the pilot Task 3 manifests record `git.commit = 2306309` because the Task 3 code was not yet committed when the runs were made (it was committed afterwards as `5d58ab7`); the runner does not record a dirty flag. Recommended before Phase 5 data generation: regenerate the pilot Task 3 output from the committed `5d58ab7` code into a new run directory and confirm byte-identical outputs by hash (about 45 s), so that consumed inputs trace to a commit. This is a recommendation, not done in this kickoff. **Resolved in P5.3:** regenerated from `5d58ab7` into a new directory (section 2); data artifacts byte-identical, run manifest differs only in run-specific metadata. Residual: the Task 3 runner records no working-tree flag (the Phase 5 regeneration record does), and the Phase 3 and Task 2 manifests record no git commit.

## 14. Leakage rules (summary; full protocol in `PHASE5_EVALUATION_PROTOCOL.md`)
Only history frames enter inputs; future positions are targets only; no future edges; no offline kinematics; no test-partition information in training or normalization; windows generated after partition assignment; dependency intervals never cross partitions.
