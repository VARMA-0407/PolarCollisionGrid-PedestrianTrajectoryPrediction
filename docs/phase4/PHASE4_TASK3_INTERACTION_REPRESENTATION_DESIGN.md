# Phase 4 Task 3 (PROPOSED): Image-Space Interaction Representation — Design

**Status:** `PROPOSED — NOT STARTED`. Design only; no implementation code exists. Requires your approval before any implementation.
**Date:** 15 November 2026. **Branch where this was written:** `feature/phase4-task2-image-kinematics` (Task 3 should get its own branch).
**Depends on:** Phase 4 Task 1 (coordinate contract: `captured_1920x1080`, image space only) and Phase 4 Task 2 (causal image-space kinematic sidecar, `COMPLETED_WITH_LIMITATIONS`).

## 1. Objective and scope

### 1.1 Task boundary (evidence)
- Project architecture (`PROJECT_MASTER.md`): video → YOLO → ByteTrack → geometric and kinematic preprocessing → **interaction graph construction** → heterogeneous GATv2 → dilated TCN → multimodal forecasting → conflict prediction → calibration → warning.
- The handoff roadmap puts "construct heterogeneous agent graphs with P-P, P-V and, if useful, V-V edges" and "define edge features" at the start of **Phase 5**, together with the GATv2/TCN models. Phase 4 items 2 (image → world transform) and 5 (validation against known geometry) are blocked by Task 1 (no metric calibration), and Phase 4's exit criterion ("consistent world-coordinate agent states with verified units") cannot be met as written.
- Phase 2 Task 5 §14 already specifies what edges may carry (relative position, relative velocity, distance, bearing, time since observation, uncertainty) and requires that "if only image coordinates are available, the edge is an image-space relation, not a physical separation".
- Nothing in `pedsafety/` builds pairwise or graph structures today (searched: `neighbo`, `interaction`, `pairwise`, `adjacen`, `edge`). Stage 5 (`windows.py`, `dataset_builder.py`, `normalization.py`) builds single-track windows only. The PCG baseline (`utils/Interaction.py`) builds a polar TTC grid with distance thresholds in HBS units; its TTC and `d_min` require metric geometry and cannot be transferred to PIE pixels.

### 1.2 Recommended boundary
Task 3 = **a deterministic, model-agnostic, image-space interaction representation**: per-frame node sets, typed candidate pairs, pairwise relational features with units and validity masks, stored in a graph-ready format, plus diagnostics on the real PIE pilot. It contains **no learning**, no model code, no training windows and no conflict metric. This keeps preprocessing in Phase 4 and leaves GATv2/TCN, training-window assembly and learning to Phase 5. *This moves "graph construction" from the start of Phase 5 into Phase 4 Task 3 as a data-representation step; it needs your explicit agreement (open question Q1).*

### 1.3 Objectives
1. Assemble frame-synchronous scenes from the Task 2 causal sidecar.
2. Represent agents as typed nodes with image-space features and masks.
3. Build typed candidate pairs (P-P, P-V, P-C; V-V optional) without physical radii.
4. Compute mathematically valid image-space relational features with explicit units, validity masks and provenance.
5. Store per-frame heterogeneous graphs compactly, in a layout a future heterogeneous GATv2 can consume.
6. Report what the representation looks like on the real PIE pilot (degrees, edge counts, missingness, feature distributions).

## 2. Architecture and data flow
```
Phase 3 agent_states.jsonl ──► Task 2 runner ──► causal_kinematics.jsonl (+ run_manifest.json)
                                                         │  (CAUSAL rows only; offline file never read)
                                                         ▼
                         T3.1 input adapter + contract check (require_causal_rows, units, world null)
                                                         ▼
                         T3.2 frame index: frame -> list of node rows (stable order by track_id)
                                                         ▼
                         T3.3 node table (type, position, velocity, masks, scale proxies, age)
                                                         ▼
                         T3.4 candidate pairs per frame, typed by (src_type, dst_type)
                                                         ▼
                         T3.5 pairwise features (vectorised numpy) + per-feature masks
                                                         ▼
                         T3.6 storage: per-relation COO edge_index + edge_attr + masks (.npz),
                              feature dictionary (JSON), manifest with hashes, diagnostics
                                                         ▼
                 (Phase 5) temporal window assembly ─► heterogeneous GATv2 per frame ─► dilated TCN over time
```
Reused, not duplicated: Task 2 sidecar schema and `require_causal_rows`; `output_dir.check_output_dir`; the atomic-write and manifest conventions of `image_kinematics_run.py`; `trajectory.wrap_angle`. No Stage 4/5 or Phase 3 file is modified.

## 3. Input/output contracts

### 3.1 Input
- `causal_kinematics.jsonl` from a completed Task 2 run (sidecar schema `image-kinematics-sidecar-1.0.0`) and that run's `run_manifest.json` (hashes, configuration, claims).
- Every row must be `view = CAUSAL`, `velocity_method = backward_difference_actual_dt`, `coordinate_space = image`, units `pixel`, `pixel/s`, `pixel/s^2`, `rad`, `s`, `world_x = world_y = null`, `world_position_valid = false`, and a single `coordinate_system` across the file (`captured_1920x1080` for PIE). Violations stop the run (they indicate a broken upstream contract, not data noise).
- Box height is needed as a scale proxy. The Task 2 sidecar does not carry the box; it is read from the Phase 3 `agent_states.jsonl` through `source_line`, with both files' hashes checked against the Task 2 manifest (open question Q4: add box height to a future sidecar version instead).
- The offline file is never an input to this task.

### 3.2 Output (one run directory, absent/empty-directory policy)
| File | Content |
|---|---|
| `nodes.npz` | per node-row: frame, track index, node type, position (u, v), velocity (u, v), speed, heading, box height, masks (position, velocity, heading), border flag, frames since previous observation |
| `edges_<src>__<dst>.npz` | per relation: `frame`, `edge_index` (2 × E, node-row indices), `edge_attr` (E × F float32), `edge_mask` (E × F bool), candidate-selection flags |
| `feature_dictionary.json` | every node and edge feature: name, formula, coordinate space, unit, valid range, mask rule, causal status, "physical: false" |
| `tracks.json` | track index ↔ `track_id`, node type, source labels |
| `interaction_diagnostics.json` | counts per relation, degree distributions, missingness per feature, feature distributions, selection sensitivity |
| `run_manifest.json` (last) | input hashes (sidecar, its manifest, Phase 3 file), configuration, versions, output hashes, claims, limitations |

A versioned JSON schema for the manifest and feature dictionary would be added under `docs/phase4/schemas/`. Canonical agent-state schema: unchanged.

## 4. Coordinate and unit restrictions
- Space: `captured_1920x1080`; origin top-left; +u right; +v down; pixels, pixel/s, s, image-plane radians (from +u toward +v, clockwise on screen, `[−π, π)`).
- No metric distance, physical velocity, TTC, PET, minimum separation, homography, `world_*` value or `world_position_valid = true`.
- Every feature name carries its space and unit (e.g. `du_px`, `sep_px`, `dvel_u_px_s`), and the feature dictionary marks it `physical: false`.
- **Image-space validity vs physical meaning:**

| Quantity | Mathematically valid in image space? | Physically meaningful? |
|---|---|---|
| Δu, Δv, pixel separation, image bearing | Yes (exact geometry of the image) | No: pixel distance depends on depth; equal pixel gaps are unequal road gaps |
| Separation / box height | Yes | Only a heuristic depth-normalised proximity; assumes comparable object heights within a class |
| log(box-height ratio) | Yes | Heuristic relative-depth proxy; class-dependent; breaks under truncation |
| Footpoint row ordering (v_i − v_j) | Yes | Depth order only under a flat road below the horizon (assumption, Task 1) |
| Relative image velocity, closing rate of pixel separation | Yes (px/s) | No: contains depth-dependent ego-motion parallax and perspective; not relative road velocity |
| Box overlap (IoU, intersection over smaller box) | Yes, scale-free | Occlusion indicator only |
| Any "time to contact" from pixels | Computable | **Excluded** (would be mistaken for TTC; deferred) |

## 5. Node representation
One node per (frame, track) causal row with a valid position (`position_state = ok`). Node types follow the canonical `agent_class`: `pedestrian`, `vehicle` (car/bus/truck, source label kept as an attribute), `bicyclist`. Bicyclists are a separate type because they are vulnerable road users whose motion differs from both pedestrians and vehicles (open question Q3 if they should be merged).

Node features (all image space, all masked): `u_px`, `v_px` (footpoint); `vel_u_px_s`, `vel_v_px_s`, `speed_px_s` (Task 2 raw causal, masked by `velocity_state`); `heading_sin`, `heading_cos` (masked by `heading_state`; sin/cos to avoid the ±π seam); `box_h_px` (scale proxy); `border` (truncation flag); `frames_since_prev_obs` (from `frame_gap_before`, 0 = continuous); `track_age_frames` (frames since the track's first row, causal); one-hot type. **Acceleration is excluded from default node features**: Task 2 showed frame-to-frame raw acceleration is dominated by non-smooth components (available behind a flag for ablation). Positions are not normalised here; normalisation statistics belong to Phase 5 training (fit on training data only, Task 5 §14).

## 6. Edge representation
Directed edges `j → i` (message from neighbour j to target i) within one frame. Features for each ordered pair:

| Feature | Formula | Unit | Mask |
|---|---|---|---|
| `du_px`, `dv_px` | p_j − p_i | px | both positions valid (always, by node definition) |
| `sep_px` | ‖p_j − p_i‖ | px | same |
| `bearing_sin`, `bearing_cos` | of atan2(dv, du) | – | sep > 0 |
| `rel_bearing_sin/cos` | bearing − heading_i (wrapped) | – | heading_i valid and sep > 0 |
| `sep_over_h` | sep_px / max(h_i, h_j) | – | always (heuristic) |
| `log_h_ratio` | ln(h_j / h_i) | – | always (heuristic; flag if either box touches the border) |
| `dv_row_px` | v_j − v_i | px | always (depth-order heuristic) |
| `dvel_u_px_s`, `dvel_v_px_s` | vel_j − vel_i | px/s | both velocities valid |
| `closing_rate_px_s` | −d(sep)/dt = −(Δp · Δvel) / sep | px/s | both velocities valid and sep > 0 |
| `overlap_ios` | intersection area / smaller box area | – | always |
| `same_track_family` | reserved (fragments are not linked) | – | always false in Task 3 |

Antisymmetric features (`du`, `dv`, `dvel`, `dv_row`, `log_h_ratio`) satisfy f(j→i) = −f(i→j); symmetric ones (`sep`, `sep_over_h`, `closing_rate`, `overlap_ios`) are equal both ways. These identities are test oracles.

## 7. P-P and P-V interaction design
Relations are typed `(src_type, relation, dst_type)` for a heterogeneous GATv2:
- `pedestrian → pedestrian` (P-P), both directions.
- `vehicle → pedestrian` and `pedestrian → vehicle` (P-V): stored as two relation types because influence is asymmetric (a pedestrian reacts to a vehicle differently from the reverse); the same pair features are computed once and mirrored with the antisymmetric sign rule.
- `bicyclist ↔ pedestrian` and `bicyclist ↔ vehicle`: included (bicyclists share space with both).
Relation-specific selection rules (section 9) are configuration, not hard-coded constants.

## 8. Optional V-V justification
V-V edges can carry traffic-flow context (queues, a vehicle yielding because the vehicle ahead stopped) that affects when vehicles meet pedestrians, but the project's targets are pedestrian trajectories and P-V conflicts, PIE has no vehicle ground truth beyond `car`, and V-V edges are the most numerous (PIE frames are vehicle-heavy: 34,678 of 65,388 rows). **Recommendation: compute V-V candidates behind a flag, default off; report their count; decide in Phase 5 by ablation.** No V-V-based claim is made in Task 3.

## 9. Neighbour-selection strategy (no physical units)
- **Evidence:** at most 23 agents in one PIE frame (Phase 3 manifest), about 7 on average (65,388 rows / 9,326 frames with a track); a complete directed graph per frame has at most 23 × 22 = 506 edges, so pruning is not needed for tractability.
- **Recommendation:** the stored candidate set is **complete within the enabled relation types** (no physical radius, no arbitrary pixel radius). Attention in GATv2 can learn relevance.
- **Sensitivity flags, not filters:** each edge stores boolean flags for (a) depth-normalised proximity `sep_over_h ≤ R` for R in a configured list (for example 2, 5, 10), and (b) k-nearest by `sep_over_h` for k in a list (for example 4, 8). Diagnostics report how many edges each rule would keep. Phase 5 can then choose a rule by validation rather than by assumption.
- Not used: pixel radii (perspective makes them depth-inconsistent), metric radii (unauthorised), PCG's `neighborhood_size` (HBS units).

## 10. Missing-data handling
- A track with no valid position at frame t has **no node** at t. No interpolation, no carried-forward position, no causal prediction (deferred).
- Edge features that need velocity or heading are masked when either endpoint lacks a valid value; the edge itself still exists (position features are valid).
- `frames_since_prev_obs` and `track_age_frames` expose gaps and young tracks to the model instead of hiding them.
- Rejected/invalid Task 2 rows (`observation_type ≠ observed`) never become nodes; counted in diagnostics.
- Border-touching boxes keep their node but their scale-dependent features carry a truncation flag.

## 11. Temporal consistency
- Graphs are per frame and causal: features at frame t use only node values at t, which Task 2 computed from t and earlier.
- Track identity is the sequence-scoped `track_id`; pairs are identified by `(frame, src_track, dst_track)`, so a pair's feature series over time is recoverable for a TCN.
- Fragments are not merged: if a pedestrian's track breaks into two ids, the pair series breaks too (the Task 2 fragment indicators are reported alongside, never applied).
- Frame indexing follows Phase 3 (`timestamp = frame / fps`); Task 3 checks one timestamp per frame across rows (a mismatch stops the run).
- Training-window assembly over time (history length, stride, masks, split isolation) is **Phase 5**, extending Stage 5 windows to scenes.

## 12. Complexity and performance
- Pairs per frame: n(n−1) for n agents, ≤ 506 on PIE; total directed candidate pairs on the pilot are bounded by Σ n_t(n_t−1). The exact number will be measured (rough order of magnitude: a few hundred thousand), not assumed.
- Vectorised per-frame numpy (broadcast n × n), O(Σ n_t²) time; memory dominated by `edge_attr` (E × ~15 float32), i.e. tens of MB for the pilot, far below the 106 MB Task 2 JSONL.
- Storage: `.npz` (numpy, already a dependency); no new dependency. A PyTorch Geometric `HeteroData` adapter is a Phase 5 concern (the layout maps one-to-one: `edge_index` per relation, `edge_attr`).
- Runtime and size are measured on the real pilot and reported; no real-time claim.

## 13. Test plan (synthetic first, then real pilot)
Hand-computed oracles, no model, no network:
1. Two agents, known positions → `du`, `dv`, `sep`, bearing exact; antisymmetry and symmetry identities for every pair.
2. Relation typing: P-P, V→P, P→V, C↔P, C↔V present; V-V absent by default and present with the flag; no self-edges.
3. Velocity masks: one endpoint with `velocity_state ≠ valid` → velocity features masked, position features kept.
4. Heading seam: target heading near ±π → `rel_bearing` correct across the seam.
5. Closing rate: agents approaching / receding / moving in parallel → sign and value exact.
6. Overlap: disjoint, nested, partially overlapping boxes → `overlap_ios` exact.
7. Missing frame: a track absent at t → no node, no edge at t; present at t+1 with `frames_since_prev_obs` correct.
8. Causal guard: an offline row or a row with populated world fields in the input → run refused.
9. Input-order invariance: shuffled rows within a frame → identical outputs (after canonical ordering by `track_id`).
10. Determinism: two runs → identical output hashes.
11. Units and claims: every feature in the dictionary has a unit and `physical: false`; no `world_*` produced.
12. Selection flags: `sep_over_h ≤ R` and kNN flags match brute force.
13. Scale: a synthetic 23-agent frame and a 65k-row synthetic file → bounded runtime (measured).
14. Mutation checks on the safeguards (sign rule, mask rule, V-V default, causal guard).
15. Real PIE pilot run on the Task 2 output: diagnostics only (edge counts per relation, degree distribution, missingness, feature distributions); no accuracy claim.

## 14. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Pixel features read as physical gaps or speeds | Units in every name; `physical: false` in the dictionary; manifest claims; no TTC-like ratios |
| Perspective: equal pixel distance ≠ equal road distance | Depth-normalised features (`sep_over_h`, `log_h_ratio`, row ordering) labelled heuristic; selection flags instead of radii |
| Ego-motion contaminates relative velocity (depth-dependent parallax) | Documented; relative velocity masked where invalid; ego-motion compensation listed as a separate future task (Q6) |
| Velocity jitter (~48 px/s per axis per agent, Task 2) dominates relative velocity of slow agents | Raw values plus masks; smoothing decision deferred to Task 2 follow-up; diagnostics report relative-speed distributions |
| Box truncation corrupts scale proxies | Border flags on nodes and edges |
| Fragments / possible ID switches break pair continuity | Reported, never merged; Task 2 indicators carried as diagnostics |
| Class flips (4,994 dropped Phase 3 rows) leave holes | Nodes absent; counted; Phase 3 logging is a separate decision |
| Over-engineering before the model exists | Model-agnostic arrays; selection and V-V left to Phase 5 ablation |

## 15. Acceptance criteria
1. Implementation reuses Task 2 contracts and modifies no Phase 2/3/Stage 4/5 or Task 1/2 file.
2. Every node/edge feature is documented with formula, coordinate space, unit, mask rule and `physical: false`.
3. Inputs are verified causal; offline rows and populated world fields are refused.
4. Synthetic tests 1–14 pass; mutation checks caught; full regression unchanged apart from the known `TTC_mask` defect.
5. Output deterministic (hash-identical re-run) and input-order invariant.
6. Real PIE pilot run completed with diagnostics, measured runtime and size, input hashes unchanged.
7. Report states image-space limitations and makes no metric, TTC/PET, risk or deployment claim.

## 16. Explicit exclusions
GATv2, TCN or any learned component; training-window assembly and dataset splits; normalisation statistics; trajectory forecasting; conflict prediction, TTC, PET, minimum separation or any time-to-contact surrogate; metric distances, homography, `world_*`; ego-motion compensation; interpolation or causal prediction of missing nodes; merging of track fragments; changes to Phase 3 (including class-flip handling); smoothing or gap-bridging changes to Task 2; updating the master project file.

## 17. Deferred until metric geometry is authorised
Physical distances and radii; relative road velocity; TTC, PET, minimum separation and conflict zones; road-plane relative heading; physically motivated thresholds (social distances, PCG `d_min`, speed limits); crosswalk/road-geometry features; transfer of PCG's polar TTC grid.

## 18. Integration with the future GATv2 + TCN (Phase 5)
- Spatial: per-frame heterogeneous graphs → relation-specific GATv2 layers (`edge_attr` used as edge features in attention), producing per-node embeddings at each frame.
- Temporal: per-track embedding sequences over an observation window → dilated TCN, with node presence masks and `frames_since_prev_obs` for gaps.
- Pair series (stable `(src_track, dst_track)` keys) allow an edge-level temporal encoder if needed.
- Causality is preserved end-to-end because every Task 3 input is causal and every feature is computed at its own frame.

## 19. Proposed implementation breakdown (not started)
| Step | Content | Done when |
|---|---|---|
| T3.0 | Approval of boundary (Q1), node types (Q3), V-V default, box-height source (Q4); new branch | Decisions recorded |
| T3.1 | Input adapter: read Task 2 causal sidecar + manifest, verify hashes, contract and causality; join box heights via `source_line` | Contract tests pass |
| T3.2 | Frame index and node table with masks, types, scale proxies, age | Node tests pass |
| T3.3 | Pairwise features (vectorised) with masks and sign rules | Oracle, symmetry and mask tests pass |
| T3.4 | Relation typing, V-V flag, selection-sensitivity flags | Typing and flag tests pass |
| T3.5 | Storage (`.npz`, feature dictionary, tracks, schema), runner and CLI with output policy and manifest | Determinism, order-invariance and output-policy tests pass |
| T3.6 | Diagnostics report | Report fields complete on synthetic data |
| T3.7 | Mutation checks, full regression | All caught; regression unchanged |
| T3.8 | Real PIE pilot run and report (design-vs-observed edge counts, degrees, missingness, runtime, size) | Report written; status decided honestly |

## 20. Open scientific questions
- **Q1** Task boundary: is deterministic graph construction Phase 4 Task 3 (recommended) or the first Phase 5 task?
- **Q2** Phase 4 exit: Phase 4's world-coordinate exit criterion cannot be met; should Phase 4 close with an explicit image-space limitation record?
- **Q3** Should bicyclists be their own node type (recommended) or merged with pedestrians/vehicles?
- **Q4** Box height: join from Phase 3 via `source_line` (recommended now) or add to a new Task 2 sidecar version?
- **Q5** Which neighbour-selection rule (complete vs depth-normalised radius vs kNN) helps the model? To be answered by Phase 5 validation, not assumed.
- **Q6** Is image-based ego-motion compensation (frame-to-frame background registration, which needs no metric calibration) worth a dedicated task before relative-velocity features are trusted?
- **Q7** Should Task 2 smoothing (time-aware EMA) be decided before Task 3, given relative velocity inherits node-velocity jitter?
- **Q8** How should uncertainty (Task 2 jitter estimate) propagate into edge features (e.g. a per-edge noise scale) for the model?
