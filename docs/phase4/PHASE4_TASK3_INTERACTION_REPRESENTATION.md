# Phase 4 Task 3: Image-Space Interaction Representation

**Date:** 15 November 2026. **Branch:** `feature/phase4-task3-interaction-representation`, HEAD `2306309` (Task 2 commit); nothing committed.
**Status:** `COMPLETED_WITH_LIMITATIONS` (criteria in section 14).
**Specification:** `PHASE4_TASK3_INTERACTION_REPRESENTATION_DESIGN.md` (unchanged, SHA-256 `1433b227…74ad0`) and `PHASE4_TASK3_DECISION_RECORD.md` v1.1.0 (SHA-256 `5bb7fd28…3a11`). **Machine-readable report:** `phase4_task3_interaction_report.json`.

| Stage | Status | Evidence |
|---|---|---|
| T3.0 Git, branch, decisions | done | decision record v1.1.0 |
| T3.1 Input adapter and contracts | done | `pedsafety/interaction/input_adapter.py`; 40 contract codes, all exercised by tests; real-data verification of P7/P13 |
| T3.2 Frame index and node table | done | `nodes.py`; node tests |
| T3.3 Pairwise features and masks | done | `pairwise.py`; hand-computed oracle tests; reversal identities exact on real data |
| T3.4 Relation typing and sensitivity flags | done | `pairwise.py`, `config.py`; typing, V-V, kNN tie and radius tests |
| T3.5 Storage, runner, CLI | done | `storage.py`, `runner.py`, `feature_dictionary.py`, `scripts/run_interaction_representation.py`, two JSON schemas |
| T3.6 Diagnostics | done | `diagnostics.py`; `interaction_diagnostics.json` |
| T3.7 Tests, mutation checks, regression | done | 75 Task 3 tests; 28/28 mutations caught; full regression 1004 passed, 6 skipped, 1 known failure |
| T3.8 Real PIE pilot | done | `experiments/phase4_task3_pie_video_0004/20261114T214634Z_pie_video_0004/` |

## 1. Objective and scope
A deterministic, model-agnostic, image-space interaction representation: per-frame typed nodes, complete typed candidate edges, pairwise image-space features with validity masks and sensitivity flags, stored graph-ready, with a feature dictionary, diagnostics and a manifest. Excluded: learning, GATv2, TCN, training windows, forecasting, conflict prediction, TTC/PET/minimum separation, metric geometry, homography, ego-motion compensation, interpolation, fragment merging, Phase 2/3 and Task 1/2 changes.

## 2. Architecture
```
Task 2 causal_kinematics.jsonl + run_manifest.json (+ config_resolved.json)   Phase 3 agent_states.jsonl
              └──────────────► input_adapter.load_inputs ◄────────────────────────────┘
                 hashes, full schema, explicit contract, strict source_line join, P7/P12 checks
                                   ▼
                     nodes.build_node_table (P6-P9, P11)
                                   ▼
          pairwise.build_edges: typed complete pairs, features, masks, flags (P1-P5)
                                   ▼
     diagnostics.build_diagnostics · feature_dictionary · storage.write_npz/json · runner (manifest last)
```
Reused: Task 2 `load_sidecar_validator`, `sha256_file`, `_atomic_write_text`, sidecar constants; `output_dir.check_output_dir`. No existing module was modified.

## 3. Input/output contracts
**Inputs (only these; the Task 2 offline file is refused by name and by content):**

| Input | Path (pilot) | SHA-256 |
|---|---|---|
| Task 2 causal sidecar | `experiments/phase4_task2_pie_video_0004/20261114T202653Z_pie_video_0004/primary_causal/causal_kinematics.jsonl` | `9e5c5f3d6185028194b25f520a74aa0e9d751e66a16754881f366c87c7ac82e3` |
| Task 2 run manifest | `…/primary_causal/run_manifest.json` | `ad2089591d704ca403d530b7237bac3ebdee524852f09310f21087c5beac144b` |
| Task 2 resolved config | `…/primary_causal/config_resolved.json` | `804f699369c84eeec75cfc06bf3ffd08191b2f3cc508ac42a74647ca4429c515` |
| Phase 3 agent states | `PIE_pilot_inputs/agent_states.jsonl` | `ff8b9b217144a9a6792069aa2db11cd9e9ca43fbf00077ad4da6e2b3fe485dc7` |
| Task 2 sidecar schema | `docs/phase4/schemas/image_kinematics_sidecar.schema.json` | `1e64be9ac541f068a7a95e4743ba94dfb7e15c924e238df25a2fe3558f51ed57` |

**Contract checks** (each failure raises `InteractionContractError` with a stable code; 40 distinct codes, every one triggered by at least one test): Task 2 manifest identity and status; causal hash and row count = manifest; Phase 3 hash = manifest input before and after; schema version and hash; `config_resolved.json` hash, `max_derivative_gap_frames = 0`, smoothing `none`, raster; every causal row: explicit view/method/space/system/unit/world checks, then full sidecar-schema validation; `source_line` one-based physical, in range, unique, non-blank, JSON object, identical `track_id`/`frame_index`/`timestamp`; bbox four finite numbers with positive height and width; one timestamp per frame; boolean border flag and known type on observed rows; `frame_gap_before` recomputed and equal on every row; any malformed non-blank Phase 3 line.

**Outputs** (run directory, absent/empty-directory policy, temp-file + rename, manifest last and not self-hashed): `nodes.npz`, `edges_<src>__<dst>.npz` (one per enabled relation), `tracks.json`, `feature_dictionary.json`, `interaction_diagnostics.json`, `config_resolved.json`, `interaction_summary.md`, `run_manifest.json`. Schemas: `docs/phase4/schemas/interaction_feature_dictionary.schema.json`, `interaction_run_manifest.schema.json` (both validated before writing).

## 4. Node and edge schemas
**`nodes.npz`** (N nodes, canonical order: frame, then natural track order): `frame` int64 (N), `timestamp` float64, `track_index` int32, `node_type` int8 (0 pedestrian, 1 vehicle, 2 bicyclist), `source_line` int64, `features` float32 (N, 10), `feature_names`, `boxes` float32 (N, 4), `box_names`, `border` bool, `masks` bool (N, 4), `mask_names`, `node_type_names`.

| Node feature | Formula / source | Unit | Mask |
|---|---|---|---|
| `u_px`, `v_px` | footpoint (Task 2 position) | px | always valid |
| `vel_u_px_s`, `vel_v_px_s`, `speed_px_s` | Task 2 causal backward difference | px/s | `velocity_valid` |
| `heading_sin`, `heading_cos` | of Task 2 image heading (from +u toward +v) | 1 | `heading_valid` |
| `box_h_px` | Phase 3 `y_max − y_min` via `source_line` | px | always valid |
| `frames_since_prev_obs` | Task 2 `frame_gap_before` (missing frames; 0 = consecutive) | frames | `frames_since_prev_obs_valid` (false at track start) |
| `track_age_frames` | frame − first observed frame of the track | frames | always valid |

Acceleration is not a node feature (A1). Source labels (car/bus/truck/bike/pedestrian) are in `tracks.json`.

**`edges_<src>__<dst>.npz`** (E edges, order: frame, destination node, source node): `relation` (2 strings), `frame` int64 (E), `edge_index` int64 (2, E) with row 0 = source node j, row 1 = destination node i, `edge_attr` float32 (E, 14), `edge_mask` bool (E, 14), `flags` bool (E, 6), `feature_names`, `flag_names`.

## 5. Feature formulas, units and masks (edge j → i, P1–P3)
| Feature | Formula | Unit | Reversal | Mask (invalid → NaN) |
|---|---|---|---|---|
| `du_px`, `dv_px` | p_j − p_i | px | anti | always valid |
| `sep_px` | ‖p_j − p_i‖ (pixel, not physical) | px | sym | always valid |
| `bearing_sin/cos` | dv/sep, du/sep | 1 | anti | sep > 0 |
| `rel_bearing_sin/cos` | sin/cos(bearing − heading_i), destination heading | 1 | – | sep > 0 and heading_i valid |
| `sep_over_h` | sep / max(h_j, h_i) (heuristic) | 1 | sym | always valid; truncation flag |
| `log_h_ratio` | ln(h_j / h_i) (heuristic) | 1 | anti | always valid; truncation flag |
| `dv_row_px` | v_j − v_i (identical to `dv_px` by definition) | px | anti | always valid |
| `dvel_u/v_px_s` | vel_j − vel_i (relative IMAGE velocity) | px/s | anti | both velocities valid |
| `closing_rate_px_s` | −(Δp·Δvel)/sep; positive = pixel separation shrinking | px/s | sym | both velocities valid and sep > 0 |
| `overlap_ios` | intersection / smaller box area | 1 | sym | always valid |

Every feature: image space, `physical: false`, causal inputs only (feature dictionary). Features are computed in float64 from the stored float32 node values, so nodes and edges are mutually consistent. Closing rate is the rate of change of **pixel** separation and is affected by perspective and ego-motion; it is not a road-relative closing speed.

## 6. Relation typing
Enabled by default: `pedestrian→pedestrian`, `vehicle→pedestrian`, `pedestrian→vehicle`, `bicyclist→pedestrian`, `pedestrian→bicyclist`, `bicyclist→vehicle`, `vehicle→bicyclist`. `vehicle→vehicle` only with `enable_vehicle_vehicle: true` (default false, also in code). No self-edges. Bicyclists are a separate node type. **Bicyclist→bicyclist** is neither in the approved list nor in the design; it is conservatively not represented and its pair count is reported (open item, section 13).

## 7. Candidate selection
The complete candidate set for each enabled relation is stored (pilot: equal to Σ n_src·n_dst per frame for every relation). Flags never remove an edge: `scale_truncation_affected` (either endpoint's inherited `bbox_touches_border` true; possible, not confirmed, truncation), `sep_over_h_le_{2,5,10}` (inclusive), `knn_{4,8}` (per destination, all enabled incoming relations ranked by `sep_over_h`, then `sep_px`, then source canonical track order). The radius and k values are exploratory (P5).

## 8. Missing-data handling
Nodes only for `observation_type = observed` and `position_state = ok` (P9); invalid, rejected and Task 2 quarantined rows are counted separately. No interpolation, no carry-forward, no fragment merging. Unavailable values are NaN with a False mask (never zero); verified on real data: NaN occurs exactly where masks are False, in every node and edge array.

## 9. Causality
Only the Task 2 causal file is read (refused by name if it is the offline file; every row must be `CAUSAL` with the backward-difference method). Each edge uses only node values at its own frame, which Task 2 computed from that frame and earlier frames.

## 10. Synthetic test evidence
Commands (environment `pcg-baseline`, Python 3.9.25):
```
python -m pytest tests/phase4/task3 -q -p no:cacheprovider                                   -> 75 passed
python -m pytest tests/phase4 -q -p no:cacheprovider                                         -> 152 passed (77 Task 2 + 75 Task 3)
python -m pytest tests/phase4 tests/phase3 tests/task7 tests/phase2 -q -rs -p no:cacheprovider -> 1004 passed, 6 skipped, 1 failed
D:\envs\pedsafety-perception\python.exe -m pytest tests/phase4 -q -p no:cacheprovider        -> 152 passed (Python 3.11.18)
```
The single failure is the known baseline defect `tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension` (unchanged, not modified). 1004 = 929 (before Task 3) + 75. The 6 skips: 5 Task 1 files need OpenCV/Ultralytics; 1 needs `PCG_ORIGINAL_NOTEBOOK`. The Stage 6.1 dependency-declaration test passes (Task 3 imports only numpy and jsonschema).
Coverage: strict physical-line joins (including a blank line shifting numbering), blank/malformed/out-of-range/duplicate lines, track/frame/timestamp mismatch, no row-order fallback (swapped lines), invalid/rejected/quarantined rows, null border flag, non-finite and non-positive boxes, multiple timestamps per frame, NaN/mask handling, P1 direction and antisymmetry, P2 closing-rate signs (approaching/receding/parallel), P3 relative bearing and the ±π seam, zero separation, kNN tie-breaking by canonical track order (natural, not lexicographic) and across relations, inclusive radius flags, complete graph despite flags, separate bicyclist type, V-V default off (code and shipped config) and on by configuration, truncation propagation, causal-only guard, hash/manifest integrity, schema validation of dictionary and manifest, deterministic output hashes, input-order invariance (rows shuffled within frames), source immutability, output-directory policy, CLI exit codes, 23-agent scalability (9,200 nodes, 202,400 pairs).
**Mutation checks:** 28 safeguards broken one at a time in a scratch copy; **28 caught** (one, the V-V default in code, was initially missed; a test was added). Result file: `experiments/phase4_task3_synthetic_validation/20261114T214617Z_synthetic_scene/mutation_check_results.txt`.
**Synthetic end-to-end run** (`scripts/phase4_task3_synthetic_validation.py`, synthetic Phase 3 records → real Task 2 runner → Task 3): 177 nodes, 750 edges; counts match hand arithmetic (e.g. P-P 54 = 27 frames with both pedestrians × 2; V→P 114 = 2 vehicles × 57 pedestrian nodes).

## 11. Real PIE pilot diagnostics (PIE set01/video_0004; image space only)
Run: `experiments/phase4_task3_pie_video_0004/20261114T214634Z_pie_video_0004/` with `primary/`, `determinism_rerun/`, `memory_measured_rerun/`, `analysis/output_check.json`.

**Provenance:** 65,388 causal rows, all `observed`; 0 invalid, 0 rejected, 0 Task 2 quarantined; 65,388 Phase 3 physical lines, 0 blank, all referenced exactly once; one-based physical `source_line` confirmed by identity match on every row; `frame_gap_before` semantics verified on all 65,388 rows; one timestamp per frame in all 9,326 frames.

**Nodes:** 65,388 in 9,326 frames (1,500 tracks): pedestrian 23,661, vehicle 34,678, bicyclist 7,049. Nodes per frame: median 6, p95 16, max 23 (= Phase 3 maximum). Velocity/heading valid 95.1 %; `frames_since_prev_obs` valid 97.7 % (1,500 track starts). Border-touching nodes 3,336 (5.1 %). Track age median 54 frames, max 1,191. Box height median 104.9 px (17.4–1,040.9).

**Edges (each equals the complete-graph bound):**

| Relation | Edges | dvel masked | rel. bearing masked | truncation-affected | sep_over_h median |
|---|---|---|---|---|---|
| pedestrian→pedestrian | 105,634 | 8.6 % | 4.4 % | 3.4 % | 2.34 |
| vehicle→pedestrian | 81,618 | 7.9 % | 4.8 % | 10.4 % | 3.26 |
| pedestrian→vehicle | 81,618 | 7.9 % | 3.3 % | 10.4 % | 3.26 |
| bicyclist→pedestrian | 42,205 | 8.3 % | 3.9 % | 4.2 % | 1.98 |
| pedestrian→bicyclist | 42,205 | 8.3 % | 4.6 % | 4.2 % | 1.98 |
| bicyclist→vehicle | 25,715 | 7.7 % | 2.2 % | 13.5 % | 4.19 |
| vehicle→bicyclist | 25,715 | 7.7 % | 5.7 % | 13.5 % | 4.19 |
| **total** | **404,710** | | | | |

Not represented: vehicle→vehicle 141,544 pairs (disabled by default), bicyclist→bicyclist 18,878 pairs (no approved relation). Edges per frame: median 14, p95 208, max 452 (theoretical max for 23 agents incl. all types: 506). In/out degree: median 5, max 22; 11,319 nodes have no edge (alone in their frame, or only with partners of an unrepresented type pair). No zero-separation edge occurred (bearing valid 100 %).

**Distributions (image space, not physical):** relative image speed median 112–159 px/s, p95 580–746 px/s by relation; closing rate median −16 to −41 px/s (p05 −389 to −589, p95 191–259); box overlap > 0 in 1.1 % (B-V) to 16.6 % (B-P) of edges.

**Selection sensitivity (fraction of edges flagged; nothing removed):** `sep_over_h ≤ 2`: 16–50 %; `≤ 5`: 64–77 %; `≤ 10`: 94–96 %; `knn_4`: 25–70 %; `knn_8`: 66–90 % (asymmetric by direction because ranking is per destination).

**Uncertainty diagnostics (P17, exploratory; no feature; not a threshold):** component-wise noise reference √2 × 48 ≈ 67.9 px/s under an independence assumption. Robust SD of relative image velocity components per relation: 58–132 px/s. Share of edges with both components within one reference: 24–37 %; within two: 48–60 %. These describe the spread of image-space relative velocity, which mixes real relative motion, ego-motion parallax and correlated jitter; they are not calibrated uncertainty.

**Independent output check** (`scripts/phase4_task3_output_check.py`, `analysis/output_check.json`): manifest hashes/sizes, dtypes, canonical node order, endpoints in bounds, same frame, correct types, no self-edges, NaN exactly where masked, truncation and radius flags recomputed and equal, complete-graph bound per relation, all features `physical: false`, no world values — all pass. Reversal identities on all 255,172 mirrored edge pairs: maximum absolute deviation **0.0** for every antisymmetric and symmetric feature.

## 12. Determinism, runtime and storage
- **Determinism:** `determinism_rerun` and `memory_measured_rerun` outputs are byte-identical to `primary` (all 13 files). Manifests differ only in `command`, `started_utc`, `finished_utc`, `runtime_s` (and `peak_memory_bytes`, see below). `.npz` files are written with fixed zip member timestamps (NumPy's `savez` would embed the current time).
- **Source integrity:** all four input files hashed before and after all three runs: unchanged.
- **Runtime** (offline preprocessing on an Intel i5-1235U, Python 3.9.25; not a real-time claim): primary 44.0 s total (input validation 28.9 s, of which full schema validation of 65,388 rows 27.0 s; join 1.4 s; nodes 0.24 s; edges 12.9 s; writing 1.5 s); re-runs 42.1 s and 46.8 s.
- **Memory:** peak working set 913 MB (memory-measured re-run). The first two runs recorded `peak_memory_bytes: null` because a Windows API call lacked explicit ctypes types (64-bit handle truncation); the fix was applied before the third run, and a test now checks that memory is measured.
- **Storage:** 22.5 MB per run directory (edges 18.98 MB, nodes 3.15 MB, JSON 0.42 MB); 65 MB for the three run directories.

## 13. Limitations, deviations and deferred work
**Limitations:** image space only, no metric geometry; relative image velocity and closing rate include ego-motion parallax and perspective; box-height normalisation is heuristic and affected by truncation, pose and class size; node velocities carry Task 2 jitter (no smoothing, Q7); missing agents not interpolated, fragments not merged, Phase 3 class-flip holes remain; radius/kNN values exploratory; runtime is not real-time evidence; reproducing T3.8 needs the gitignored local inputs.

**Deviations and notes (recorded in every manifest):** (1) `dv_row_px` equals `dv_px` by definition, kept as specified; (2) `same_track_family` (design table, always false) not stored; (3) bicyclist→bicyclist not represented; (4) P10 applied to box width as well as height; (5) any malformed Phase 3 line fails the run, even if unreferenced; (6) peak memory null in the first two pilot runs (fixed; third run measured); (7) runner code changed after the primary run only in the memory helper; outputs are byte-identical across all three runs.

**Open items for your decision:** whether to add a bicyclist→bicyclist relation; whether to enable V-V for Phase 5 ablation; `.gitignore` for Task 3 outputs (below).

**`.gitignore` recommendation (not applied):** measured 22.5 MB per run, 65 MB for the pilot directory (three runs). Recommend ignoring `experiments/phase4_task3_pie_video_0004/**/*.npz` (≈ 19 MB of edges and 3 MB of nodes per run, regenerable from committed code plus the hashed local inputs) while keeping the JSON/Markdown reports, manifests and `analysis/` visible. The synthetic run (0.9 MB) can be committed.

## 14. Acceptance criteria and status
| Criterion | Result |
|---|---|
| Input contracts enforced | PASS (40 codes, all tested; mutations caught) |
| Node and edge features fully documented | PASS (feature dictionary, schema-validated; this report) |
| Feature masks correct | PASS (tests; NaN exactly where masked on real data) |
| Synthetic tests pass | PASS (75 Task 3; 152 Phase 4) |
| Mutation checks catch deliberate faults | PASS (28/28) |
| Full regression documented | PASS (1004 passed, 6 skipped, only the known `TTC_mask` failure) |
| Outputs deterministic | PASS (three byte-identical runs) |
| Input-order invariance | PASS (test) |
| Real PIE pilot completes | PASS |
| Input hashes unchanged | PASS |
| Coordinate and physical-claim restrictions | PASS (`physical: false` everywhere; no world values; manifest claims schema-enforced) |
| Reports and manifests agree | PASS (numbers in this report are taken from the manifests/diagnostics) |

**Status: `COMPLETED_WITH_LIMITATIONS`.** No commit, push, `PROJECT_MASTER.md` change or `.gitignore` change was made.
