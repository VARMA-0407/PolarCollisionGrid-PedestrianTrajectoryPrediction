# Phase 5 Task P5.3 — Pilot Window Generation and Dataset Pipeline

**Date:** 15 November 2026. **Status:** `COMPLETED_WITH_LIMITATIONS — PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT` (uncommitted; awaiting review).
**Scope:** causality-safe, reproducible trajectory-window generation from existing Phase 3/4 outputs for the single PIE pilot video. No model, training, GATv2, TCN, decoder, metric, split or forecasting result exists. Machine-readable companion: `phase5_task3_window_report.json`.

**Phase 2 status.** Phase 2 has **not** been formally closed (`docs/phase2/task7_implementation.md`: Stage 6/6.1 awaiting review). P5.3 is pilot-only engineering on top of it; nothing in Phase 2 was changed and no Phase 2 closure is claimed.

## 1. Git state
| | Initial (verified) | Final |
|---|---|---|
| Branch | `feature/phase4-task3-interaction-representation` | same |
| HEAD | `5d58ab7146ea4611733393769071b02a6bf9d187` (parent `2306309`) | same (no commit, amend, reset, stage or push) |
| Tracked changes | none | none (`git diff --quiet HEAD` → 0) |
| Untracked | 5 `docs/phase5/` planning documents | those 5 (edited) + the new P5.3 files listed in section 2 |

## 2. Files
**Created:** `pedsafety/phase5/` (`__init__`, `errors`, `window_config`, `task3_source`, `upstream`, `trajectory_targets`, `window_builder`, `graph_window_adapter`, `window_storage`, `window_manifest`, `window_validation`, `runner`); `configs/phase5/window_generation_default.json`; `scripts/phase5_generate_windows.py`, `scripts/phase5_validate_windows.py`, `scripts/phase5_task3_regeneration_record.py`; `tests/phase5/` (`conftest`, `phase5_helpers`, `test_window_config`, `test_window_generation`, `test_window_pipeline`); this report and `phase5_task3_window_report.json`.
**Generated (new directories, nothing overwritten):** `experiments/phase4_task3_pie_video_0004/20261115T034308Z_pie_video_0004_regen_5d58ab7/` (Task 3 regeneration + `regeneration_record.json`); `experiments/phase5_task3_windows_pie_video_0004/20261115T035825Z_pie_video_0004/` (`primary/`, `determinism_rerun/`, `validation_report.json`).
**Modified (untracked planning docs only):** `PHASE5_INPUT_CONTRACT.md` (v0.2.0), `PHASE5_DATASET_READINESS.md`, `PHASE5_EVALUATION_PROTOCOL.md` (v0.1.1), `PHASE5_IMPLEMENTATION_ROADMAP.md` (v0.1.1), `PHASE5_KICKOFF_REPORT.md` (marked revisions).
**Not modified:** Phase 2/3/4 code, schemas, documents and outputs; PCG baseline; master files; `.gitignore`; existing tests.

## 3. Window configuration reconciliation (H = 6)
Approved: 30 fps source, stride 3 → 10 Hz; history H = 6 at t−0.5 … t (0.5 s first-to-last); future F = 15 at t+0.1 … t+1.5.
The kickoff's `H = 5` was described as "0.4 s first-to-anchor, i.e. a 0.5 s observation" (H × Δt counting). The approved definition uses the first-to-last span, so H = 6.

| Affected item (kickoff, H = 5) | Location | Revised (H = 6) |
|---|---|---|
| Pedestrian windows 2,482 / 88 tracks (also vehicle 5,034/132, bicyclist 684/28) | readiness §3, input contract §6, kickoff §4 | **superseded**; recalculated **estimates**: 2,380 / 83 (per-track anchor phase, kickoff convention); 2,327 / 84 (global grid adopted) — vehicle 4,893/124 vs 4,807/121, bicyclist 653/28 vs 646/25 |
| Dependency interval `[t0 − 13, t0 + 45]`, 59 frames | implied by contract §10, protocol §1 | `[t0 − 16, t0 + 45]`, **62 frames** |
| Purge gap `(H−1+F)s + 2` = 59 frames | protocol §1 | **62 frames** |
| "H = 5 (0.5 s)" | protocol §3, kickoff §5.2, contract §6 | H = 6 (t−0.5 … t) |

**Dependency interval, exact (verified with `WindowConfig`, H = 6, F = 15, s = 3).**
History frames `h_k = t0 − (5 − k)·3` = t0−15, t0−12, t0−9, t0−6, t0−3, t0; future frames `f_m = t0 + (m + 1)·3` = t0+3, t0+6, …, t0+45. The sampled span `[t0−15, t0+45]` is 61 frames (2.0 s). The dependency interval starts one frame earlier, at **t0−16**, because velocity, speed, heading and the velocity-based edge features at `h_0 = t0−15` are Task 2 one-frame causal backward differences that read the observation at frame `t0−16` (masked when that frame is absent). Interval `[t0−16, t0+45]` = **62 frames** = `(H−1+F)·s + 2`. Purge check: anchors `a < b` have disjoint intervals iff `b − 16 > a + 45` ⇔ `b − a ≥ 62` (brute-force minimum 62; 63 on the `t0 % 3 == 0` grid; 61 without the velocity lookback). The reported 62 is correct; the wording is now precise: it is the minimum *anchor-to-anchor* separation across a chronological partition boundary. Caveat: `track_age_frames` and `frames_since_prev_obs` are past-only counters with unbounded lookback, relevant only to chronological blocks inside one video (see input contract §10); no split exists in P5.3.

The estimates were computed from `nodes.npz` with the counting rule only. The **implementation count** (section 7) is separate and equals the global-grid estimate (2,327 / 84), which is an independent consistency check of the builder. Other exploratory rows in readiness §3 (different s/H/F) are untouched and labelled as using the kickoff's H × Δt convention.

## 4. Task 3 regeneration and provenance
Command: `python scripts/run_interaction_representation.py --causal <task2>/causal_kinematics.jsonl --task2-manifest <task2>/run_manifest.json --phase3 PIE_pilot_inputs/agent_states.jsonl --config configs/phase4/interaction_representation_default.json --out experiments/phase4_task3_pie_video_0004/20261115T034308Z_pie_video_0004_regen_5d58ab7/primary` → exit 0, 43.4 s, "65388 nodes in 9326 frames, 404710 edges". No YOLO/ByteTrack inference; Phase 3 outputs reused.

| Item | Value |
|---|---|
| Git (manifest) | branch `feature/phase4-task3-interaction-representation`, commit `5d58ab7` |
| Working tree before run | tracked clean; untracked only the 5 `docs/phase5/` planning docs (no code) |
| Python / numpy / jsonschema | 3.9.25 / 1.26.4 / 4.25.1 (`pcg-baseline`) |
| Config hash (resolved) | `2770586a…90e0` (unchanged) |
| Regenerated manifest SHA-256 | `366b8d0cf33000965dc410268db8db0045ce4e318937071c322ff03a8bfdbe6e` |
| Schema validation | PASS (manifest and feature-dictionary schemas; output hashes; NaN/mask contract) for both runs |
| Data artifacts (the 13 files hashed in the manifest's `outputs`) | **byte-identical** to the original `20261114T214634Z…/primary`: `nodes.npz`, the 7 `edges_*.npz`, `tracks.json`, `feature_dictionary.json`, `interaction_diagnostics.json`, `interaction_summary.md`, `config_resolved.json` — same SHA-256 and size. Consequently: 65,388 nodes, 404,710 edges, equal relation counts, node features 65,388 × 10 and edge features × 14, equal node/edge masks, equal source-line references, equal coordinate metadata |
| Run manifest (`run_manifest.json`, the 14th file; not in its own `outputs` list) | **not byte-identical, by design.** Original SHA-256 `e320ea6b…f28a`, regenerated `366b8d0c…be6e`. Differing keys (exactly six, all run-specific metadata): `git.commit` (2306309 → 5d58ab7), `started_utc`, `finished_utc`, `command` (only the `--out` path), `runtime_s` (measured timings), `peak_memory_bytes` (null → 956,784,640 B; the original's null came from the since-fixed Windows API typing issue). Equal keys include `inputs` (all input hashes), `config`, `counts`, `validation`, `conventions`, `outputs` (all 13 hashes), `claims`, `limitations`, `versions` |

So the regeneration reproduces every **data artifact** exactly; only the **run manifest** differs, and only in metadata that necessarily changes between runs. The record `…_regen_5d58ab7/regeneration_record.json` (field `all_output_files_byte_identical: true`) refers to the 13 manifest-listed outputs and lists the six differing manifest keys in `manifest_keys_that_differ`. Both runs and the record are unchanged by this documentation pass; the original run is preserved untouched (manifest SHA-256 still `e320ea6b…f28a`).

## 5. Inputs and verification
| Input | Path | SHA-256 | Use |
|---|---|---|---|
| Task 3 nodes | `…regen_5d58ab7/primary/nodes.npz` | `f904c4dc…c13c` | history graphs, pseudo-targets |
| Task 3 edges (7) | `…/edges_<rel>.npz` | per manifest (e.g. P→P `2f2a804a…`) | history graphs |
| Task 3 tracks / dictionary / config / manifest | `tracks.json`, `feature_dictionary.json`, `config_resolved.json`, `run_manifest.json` | `01cdb886…`, `d3cb194c…`, `4f2dd0b3…`, `366b8d0c…` | identity, names, contract |
| Task 2 causal sidecar | `experiments/phase4_task2_pie_video_0004/20261114T202653Z_pie_video_0004/primary_causal/causal_kinematics.jsonl` | `9e5c5f3d…82e3` | verification only (via Task 3) |
| Task 2 manifest / config | same dir `run_manifest.json`, `config_resolved.json` | `ad208959…144b`, `804f6993…c515` | verification |
| Sidecar schema | `docs/phase4/schemas/image_kinematics_sidecar.schema.json` | `1e64be9a…ed57` | verification |
| Phase 3 agent states | `PIE_pilot_inputs/agent_states.jsonl` | `ff8b9b21…5dc7` | source-line join |
| Phase 3 manifest | `PIE_pilot_inputs/run_manifest.json` | `e8c2a895…c7c2` | fps 30 (container), 1920×1080, frames 0–9839, label map |
| Phase 3 tracked agents | `PIE_pilot_inputs/tracked_agents.jsonl` | `182babb3…3ff5` | hashed only, not consumed |
| Task 2 offline sidecar | (listed in Task 2 manifest) | — | **never opened** |

Checks performed on every run (fail, never repair): file existence; size and SHA-256 equal to the manifests at every link (Phase 3 → Task 2 → Task 3); Task 3 manifest/dictionary schemas, status, `unchanged_after_run`, validation flags, all physical claims false; dictionary version `interaction-feature-dictionary-1.0.0`; coordinate system `captured_1920x1080` and unit pixel at Task 3, dictionary and every Phase 3 line; timestamps = frame/30 (±1e−6 s), Phase 3 fps = configured 30; frames inside the Phase 3 frame range; one node per (track, frame); one node type per track; unique one-based source lines; **source-line join of all 65,388 nodes** to Phase 3 lines (track id, frame, footpoint, world fields null, `interpolation_flag` false); all 65,388 matching Task 2 rows are `view = CAUSAL`, not interpolated, unfiltered, with position, velocity and velocity validity equal to the node; NaN exactly where masks are False; edge endpoints in the edge's frame and of the relation's types. Track-gap semantics: `frames_since_prev_obs` / Task 2 `velocity_valid` (velocity invalid after any gap, `max_derivative_gap_frames = 0`).

## 6. Implementation
* **Config** (`phase5-window-generation-config-1.0.0`, SHA-256 `65eebeed…1034`): every required field explicit; strict loader rejects unknown/missing keys and any inconsistency (fps/stride ≠ Hz, span ≠ (H−1)s/fps, horizon ≠ F·s/fps, non-pixel unit, non-pedestrian targets, uncertain types in default context).
* **Candidates:** every (pedestrian track, anchor t0) with t0 on the global grid `t0 % 3 == 0` inside the track's lifespan. **Policy `TARGET_STRICT_CONTEXT_MASKED`:** the target must have a Task 3 node at all 6 history and all 15 future sampled frames; otherwise exactly one reason in the fixed precedence `history_before_video_start, future_after_video_end, target_absent_at_anchor, history_before_track_start, target_history_gap, future_after_track_end, target_future_gap`. Gaps at *unsampled* frames do not reject (contract §7) and are counted.
* **Targets (`PILOT_SMOKE`):** future footpoints (u, v) of the same track from Task 3 nodes, plus box height and border flag for later scale-aware diagnostics. `ANNOTATED_TARGET` is a documented mode that raises `TARGET_SOURCE_NOT_IMPLEMENTED` and never substitutes perceived positions.
* **Context:** a context agent is present at a history frame iff Task 3 has its node there; absence is `present = False` with NaN features and False masks; never a reason to reject; agents seen only in the future are invisible.
* **Graph storage:** a shared **graph bank** holds the unchanged Task 3 nodes (all types and columns) and edges (all 7 relations) for exactly the 1,295 frames that are a history frame of some window; windows reference frames, so no scene graph is duplicated per target. Frames used only as future targets are not stored; future edges are never loaded.
* **Adapter:** `window_graphs` / `collate_windows` read only `history_frames[w]` (asserted ≤ anchor), assign per-window agent slots (slot 0 = target, consistent across the 6 steps), emit relation-specific COO edges with masks and flags; `max_agents` overflow is an error. Default context = pedestrian + vehicle (relations P→P, V→P, P→V); `include_optional=True` adds bicyclist (all 7 relations) with a `semantically_uncertain` flag.
* **Validation:** 38 in-memory checks at generation and 45 independent checks on the files re-read from disk (`phase5_validate_windows.py`).

## 7. Results on the pilot (PILOT_SMOKE, development only)
| Quantity | Value |
|---|---|
| Candidate windows | **8,698** (551 of 586 pedestrian tracks have ≥ 1 grid anchor) |
| Accepted windows | **2,327** |
| Rejected windows | **6,371**: history_before_video_start 17; future_after_video_end 3; target_absent_at_anchor 829; history_before_track_start 1,936; target_history_gap 822; future_after_track_end 2,158; target_future_gap 606 |
| Unique pedestrian target tracks | **84** |
| Videos / scene groups | 1 / 1 |
| Accepted windows whose target misses ≥ 1 *unsampled* frame | 166 (allowed; velocities there are masked by Task 2) |
| Target box touches border: at anchor / anywhere in future | 0 / 41 windows |
| Graph bank | 1,295 frames; 12,999 nodes (pedestrian 6,181, vehicle 4,765, bicyclist 2,053); edges P→P 31,756, V→P 21,532, P→V 21,532, B→P 13,540, P→B 13,540, B→V 7,712, V→B 7,712 |
| Distinct context tracks per window (mean / median / max; windows with ≥ 1) | pedestrian 6.68 / 7 / 15 (2,232); vehicle 4.31 / 4 / 19 (2,321); bicyclist 3.16 / 3 / 13 (1,656) |
| Max nodes in one history frame | 18 (default types), 23 (all types) |

These are data-preparation counts, not forecasting results.

## 8. Semantic limitation: bicycles
Phase 3 maps COCO `bicycle` → `bike` (`DEFAULT_LABEL_MAP`); `pedsafety/agent_state.py` maps `bike` → `bicyclist` by a recorded assumption; Task 3 uses node type `bicyclist`. On the pilot all 7,049 bicyclist nodes have source label `bike` (vehicles: car 32,981, bus 1,034, truck 663; pedestrians: pedestrian 23,661; no mixed-label tracks). A bicycle detection does not establish a rider. P5.3 keeps the nodes and the approved relations unchanged, flags them `semantically_uncertain`, excludes them from the default context, and never requires them. Original source labels remain in Phase 3 lines (joined by `source_line`) and `tracks.json`.

## 9. Leakage and temporal validation (all PASS on the pilot)
Anchor = last history frame; anchors on grid; history offsets −15…0 and future offsets +3…+45 frames for every window; frame stride 3 throughout (= 30 fps / 10 Hz); history and future timestamps strictly increasing; first future timestamp > last history timestamp; history/future frame sets disjoint; every interval 0.1 s within 1e−6 s (max deviation recorded); timestamps = frame/30 and equal the Task 3 timestamps; history span 0.5 s, horizon 1.5 s; bank frames = union of history frames; inputs never after the anchor; targets are pedestrians of the same track at every step; every stored history/future target value, source line, box height and border flag equals the Task 3 node (no interpolation, no filling, no merging); bank nodes/edges equal and complete relative to Task 3; masks consistent; approved relations only (no B→B, no V→V). Each window records video id, group id, target track index and id, anchor, history/future frames and timestamps; target source, input source and configuration version are recorded in the manifest.

**No split.** One video, one group: no train/validation/test partition is created. `smoke_manifest.json` lists all windows in one `pilot_development` partition labelled `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT` for loading/batching tests only. **Overlap risk:** windows of one track start every 0.1 s and span 2.0 s, so consecutive windows share most frames; a formal split must assign whole independent groups; if chronological blocks are ever used, the last anchor of one block and the first anchor of the next must differ by ≥ 62 frames (63 on the anchor grid), and the past-only counters `track_age_frames` / `frames_since_prev_obs` must not carry over a block boundary (recompute if H/F/s or feature lookback change).

## 10. Storage
Per run: `windows.npz` 147 KB, `graph_bank.npz` 6.38 MB, `rejected_candidates.npz` 8 KB, `smoke_manifest.json`, `run_manifest.json` (≈ 6.5 MB per run; 13 MB for the run directory with the determinism rerun). Deterministic `.npz` (Task 3 `write_npz`: fixed member timestamps, no pickle); the determinism rerun produced identical SHA-256 for all four outputs (`windows.npz` `49b4ca39…`, `graph_bank.npz` `fc0821a3…`, `rejected_candidates.npz` `6efbe92a…`, `smoke_manifest.json` `02c21aab…`). Generation time ≈ 2 s (1.5 s of which is upstream verification).

**`.gitignore` (approved and applied in the hygiene pass):** exactly one rule plus a comment line appended, mirroring the Task 3 rule; it matched no previously tracked file (`git ls-files -i -c --exclude=<rule>` → 0):
```
# Phase 5 P5.3 pilot window arrays (regenerable from committed code + hashed local Task 3 run; ~6.5 MB per run); manifests and reports stay visible
/experiments/phase5_task3_windows_pie_video_0004/**/*.npz
```
It ignores the 6 window `.npz` files (primary + determinism rerun); `run_manifest.json`, `smoke_manifest.json` and `validation_report.json` stay trackable. The regenerated Task 3 `.npz` files are covered by the existing rule `/experiments/phase4_task3_pie_video_0004/**/*.npz`; its JSON/Markdown outputs and `regeneration_record.json` stay trackable.

*Note on manifest wording:* the generated run manifests (unchanged) describe the purge as "the dependency length between chronological blocks"; the precise definition (minimum anchor-to-anchor separation 62 frames, 63 on the grid, plus the counter caveat) is in section 3 and input contract §10. The `WindowConfig` docstrings say the same in short form; their behaviour is correct and they were not edited.

## 11. Tests
| Command | Result |
|---|---|
| `python -m pytest tests/phase5 -q -p no:cacheprovider` | **39 passed** |
| `python -m pytest tests/phase5 tests/phase4 tests/phase3 tests/task7 tests/phase2 -q -rs -p no:cacheprovider` | **1043 passed, 6 skipped, 1 failed** (88 s) |
| `python -m pytest tests/phase2 -q -rf -p no:cacheprovider` | the only failure: `tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension` — the known pre-existing PCG `TTC_mask` defect (unchanged; previous suite 1004 passed + 39 new = 1043) |
| `python scripts/phase5_validate_windows.py --run …/primary --report …/validation_report.json` | **PASS 45/45** |

Skips: cv2/ultralytics not installed in `pcg-baseline` (5) and the original notebook absent (1) — pre-existing. Phase 5 tests use a synthetic scene run through the **real** Task 2 and Task 3 runners (9 tracks: continuous, gap at a sampled frame, gap at an unsampled frame, vehicle, `bike`-labelled bicyclist, fragmented identity, short track, future-only vehicle) and cover all 22 required behaviours, plus tampering (timestamp order, edited target, output hash/size, true claim, upstream hash), refusal of `ANNOTATED_TARGET`, output-directory protection, agent overflow and CLI exit codes.

## 12. Limitations
1. One video, one scene group, correlated windows: development diagnostics only; no generalization or accuracy claim is possible.
2. Pseudo-targets are perceived tracks (detector/tracker errors included); Phase 3 perception quality on PIE is still unmeasured (Phase 3 Task 4 not run on the real XML, which is not on this machine).
3. Image space only; velocities mix ego-motion and perspective; one-frame Task 2 velocities carry ~48 px/s jitter.
4. Strict target policy favours long, continuously tracked pedestrians (84 of 586 pedestrian tracks contribute).
5. Fragments/ID switches are not repaired; Phase 3 class-flip holes remain (4,994 class-change issues in the Phase 3 manifest).
6. Bicyclist semantics uncertain (section 8).
7. Provenance: Phase 3 and Task 2 manifests record no git commit; Phase 3 records no per-file hashes; P5.3 code is uncommitted, which the Phase 5 manifest records (`working_tree_dirty: true` with the list of untracked files).

## 13. Deviations from the plan
* Package `pedsafety/phase5/` (as instructed) instead of the roadmap's `pedsafety/forecasting/`; module set adds `task3_source`, `upstream`, `runner`, `errors`.
* Global anchor grid (`frame % 3 == 0`) instead of the kickoff's per-track phase: enables shared per-frame graphs; costs 53 windows vs the per-track estimate (2,327 vs 2,380). Configurable (`anchor_phase_frames`).
* No smoke **split** (the evaluation protocol allowed a purged chronological smoke split); only a single non-independent development listing, per the P5.3 instruction.
* Extra script `phase5_task3_regeneration_record.py` for the required regeneration provenance.
* Bicyclist = optional, non-default context (instructed); Task 3 relation schema unchanged.

## 14. Acceptance
| Criterion | Result |
|---|---|
| H = 6 reconciled; stale H = 5 counts removed/marked; estimates distinguished from implementation counts | PASS |
| Task 3 regenerated from `5d58ab7` into a new directory; provenance and comparison recorded; originals preserved | PASS (13 data artifacts byte-identical; run manifest differs only in 6 run-specific metadata keys) |
| Inputs verified (existence, hashes, git, coordinates, units, frames, timestamps, identity, source-line join, validity, gaps, causal status) | PASS (git recorded where available; Phase 3/Task 2 record none) |
| Exactly 6 history + 15 future samples at 10 Hz; strict target policy; reasons counted | PASS |
| No interpolation, forward-fill, ID merging, stationary assumption; no future inputs | PASS (validated against Task 3 on every window) |
| Context with masks; vehicles kept; bicycles flagged optional; relations filtered | PASS |
| Manifest with every required field; deterministic outputs; independent validation | PASS |
| Tests: focused and regression | PASS (only the known pre-existing failure) |
| No split, no training, no accuracy/generalization/metric claims | PASS |

**P5.3 status: `COMPLETED_WITH_LIMITATIONS`** for the pilot. Final independent splits, the training-data window study and annotated targets remain blocked by P5.2 (scene-diverse data) and by the missing PIE annotations.

## 15. Recommended next task
**P5.4 — corrected evaluator and baselines B0–B2 on the pilot windows** (zero-velocity, constant velocity from history positions, CV from Task 2 velocity), with metrics in pixels labelled `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`. In parallel, outside the code: P5.2 data acquisition (PIE licence archive, subset choice, Colab perception runs) and running the Phase 3 Task 4 evaluator once `video_0004_annt.xml` is available locally.
