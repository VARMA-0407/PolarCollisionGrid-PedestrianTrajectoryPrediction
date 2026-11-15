# Phase 4 Task 3 Decision Record: Image-Space Interaction Representation

**Record version:** 1.1.0. **Date:** 15 November 2026.
**Task status:** T3.0 complete. All design decisions (Q1–Q8, V-V, A1–A4) and all implementation details P1–P17 approved by the user on 15 November 2026 (P17 amended, P13 extended). T3.1 not started.

## 1. Repository identity at T3.0
| Item | Value |
|---|---|
| Branch | `feature/phase4-task3-interaction-representation` (already existed when T3.0 ran; it was not re-created) |
| HEAD | `2306309c786fa5608da96062123633cbb18acf7e`, "feat(phase4): implement image-space kinematic preprocessing" |
| Parent | `b66cbc5f45a4f68c04ca1051eee1d4a2e4870da7` (Phase 2–4 baseline) |
| Task 2 branch | `feature/phase4-task2-image-kinematics` → same commit `2306309` |
| Working tree | clean except the untracked design document and this record; nothing staged; no stash |

## 2. Design document version
`docs/phase4/PHASE4_TASK3_INTERACTION_REPRESENTATION_DESIGN.md`, SHA-256 `1433b22776a9d4b11f7acb24b63bb77544d3fa127798086cfe396a2735e74ad0`. It is **untracked**: it was not part of the Task 2 commit. It still carries the header "PROPOSED — NOT STARTED" and is deliberately left unchanged; this record supersedes that status for the approved items. Recommendation: include it in the first Task 3 commit unchanged, so the approved version is preserved.

## 3. Approved design decisions (approved by the user on 15 November 2026)
| ID | Decision | Rationale (evidence) |
|---|---|---|
| Q1 | Deterministic image-space interaction representation is Phase 4 Task 3 | Preprocessing, not learning; GATv2/TCN and training windows stay in Phase 5 (design §1.2) |
| Q2 | Phase 4 may close with explicit image-space limitations; metric geometry is not claimed complete | Task 1: no metric calibration evidence for the moving PIE camera |
| Q3 | Pedestrian, vehicle and bicyclist are separate node types | Bicyclists are vulnerable road users with distinct motion (design §5) |
| Q4 | Box height joined from Phase 3 `agent_states.jsonl` via `source_line`, with integrity checks; Task 2 sidecar not modified | Task 2 sidecar carries no box; avoids changing a committed deliverable |
| V-V | V-V edges optional, disabled by default | Pedestrian-centred targets; no evidence yet; Phase 5 ablation (design §8) |
| Q5 | Complete directed candidate graph within enabled relations; depth-normalised radius and kNN stored as sensitivity flags, never as filters; no pixel or metric radius | ≤ 23 agents per PIE frame (≤ 506 directed pairs); perspective makes pixel radii depth-inconsistent (design §9) |
| Q6 | Ego-motion compensation deferred to a separate task | Out of Task 3 scope; documented limitation of relative image velocity |
| Q7 | Task 2 not modified; its existing causal output (smoothing disabled) is consumed | Task 2 committed as `COMPLETED_WITH_LIMITATIONS` |
| Q8 | Uncertainty reported as diagnostics only; no unvalidated uncertainty features in the default representation | Task 2 jitter estimate is evidence, not a validated uncertainty model |
| A1 | Acceleration excluded from default node features | Task 2 review: frame-to-frame raw acceleration dominated by non-smooth components |
| A2 | Only Task 2 `causal_kinematics.jsonl` is a kinematic input; the offline file is never read | Causal integrity |
| A3 | Canonical agent-state schema unchanged; no world values; every feature `physical: false` | Coordinate contract (`captured_1920x1080`, image space only) |
| A4 | No interpolation, forward-fill or causal prediction of missing agents; no fragment merging; no Phase 3 or Task 2 changes; no commits or pushes by the agent | Task 2 policies; user instructions |

## 4. Approved implementation details P1–P17 (approved by the user on 15 November 2026)
These make details the design leaves unspecified explicit; they do not change any approved design decision.

| ID | Detail | Approved decision | Reason |
|---|---|---|---|
| P1 | Edge direction and sign | Edge `j → i` = source j (neighbour) to destination i (target). `du_px = u_j − u_i`, `dv_px = v_j − v_i`, `dvel_* = vel_j − vel_i`, `dv_row_px = v_j − v_i`, `log_h_ratio = ln(h_j / h_i)`. Antisymmetric features flip sign under reversal | Design §6 |
| P2 | Closing rate | `closing_rate_px_s = −(Δp · Δvel) / sep_px` with Δp = p_j − p_i, Δvel = vel_j − vel_i; positive = image-plane separation shrinking. Masked when `sep_px = 0` or either velocity invalid. It is the rate of change of **pixel** separation, not a road-relative closing speed | Design §6 |
| P3 | Relative bearing | `rel_bearing = wrap(atan2(dv, du) − heading_i)` stored as sin/cos; uses the **destination's** heading; masked if heading_i invalid or `sep_px = 0` | Design §6 |
| P4 | kNN scope and ties | For each destination node, rank **all** candidate sources across all enabled incoming relation types by (`sep_over_h`, `sep_px`, source canonical track order) and flag the first k. Used only for sensitivity flags; the complete candidate edge set is always stored | Relation-agnostic neighbourhood; total, deterministic order |
| P5 | Sensitivity values | Configurable exploratory values: `sep_over_h ≤ R` for R in [2, 5, 10]; kNN k in [4, 8]. **Not scientifically validated thresholds**; diagnostics only | Design §9 examples |
| P6 | Missing-value representation | float32 arrays with **NaN** for unavailable values plus explicit boolean validity masks; no zero substitution | Unambiguous; filling for model input belongs to Phase 5 |
| P7 | `frames_since_prev_obs` | Task 2 `frame_gap_before`, used only after its exact semantics are verified in T3.1. From the committed Task 2 code (`pedsafety/image_kinematics.py`, `CausalImageKinematics.update`): `frame_gap_before = (f − f_prev) // frame_step − 1`, where `f_prev` is the frame of the same track's previous **accepted** row; it is null for the track's first accepted row and for rejected rows. So it counts **missing** frames (0 = consecutive). Masked where null | Exact Task 2 semantics; name kept from the design, meaning documented |
| P8 | `track_age_frames` | `current frame index − first observed frame index of that track` (0 at first observation; counts calendar frames including gaps) | Causal (uses only the past) |
| P9 | Node eligibility | A node exists iff the causal row has `observation_type = observed` and `position_state = ok`. Rows that are `invalid` (position unavailable) or `rejected` (Task 2 grouping/stream rejection) are **never** nodes; they are counted **separately** in the diagnostics (invalid rows by `position_state` reason, rejected rows by state), together with Task 2 quarantined rows, which never appear in the causal file | Design §10 |
| P10 | Box-height safety | Non-finite or non-positive bounding-box height (and any non-finite box coordinate) is an **upstream contract failure** that stops the run; it is not masked | Contradicts the Phase 3 contract (`y_max > y_min`, finite) |
| P11 | Truncation | The node border flag is **inherited** from the Task 2 row's `bbox_touches_border` (Task 2 sets it when any box edge is within `border_tolerance_px = 1.0` of the image frame; 3,336 such rows on the PIE pilot); it is not recomputed. Edge flag `scale_truncation_affected` = either endpoint's border flag is true; `sep_over_h` and `log_h_ratio` keep their values but carry this flag. A null Task 2 flag is a contract failure for an observed row | Design §5/§10 |
| P12 | Frame-time consistency | All causal rows of one frame must share one timestamp (exact equality; Phase 3 uses `frame_index / fps`); otherwise the contract fails | Synchronous scenes |
| P13 | Provenance join (`source_line`) | Strict join from each causal row to the Phase 3 `agent_states.jsonl` record at `source_line`. Requirements: **(a) indexing base** — established from the committed Task 2 code (`pedsafety/image_kinematics_run.iter_jsonl`, `enumerate(fh, start=1)`): `source_line` is **one-based**; T3.1 must verify this on the actual files; **(b) physical vs logical** — it is the **physical** line number in the file, not a logical record index; blank lines are skipped by Task 2 but still counted, so physical and logical numbering diverge if blank lines exist (the PIE file has 65,388 non-blank lines and 65,388 records); **(c) blank and malformed lines** — the join resolves physical line numbers explicitly; a `source_line` that points to a blank or malformed line, a malformed line anywhere in the Phase 3 file that a causal row depends on, or a malformed Phase 3 file overall, fails the run (Task 2 quarantined malformed lines; none exist on PIE); **(d) no row-order fallback** — matching is by `source_line` only, never by position in the file or row order; **(e) identity** — the Phase 3 record at that line must have identical `track_id`, `frame_index` and `timestamp`; **(f) uniqueness and bounds** — each `source_line` is used at most once and lies within `[1, number of physical lines]`; **(g) box validity** — the joined `bbox_xyxy` has four finite coordinates and positive height (`y_max − y_min > 0`), else P10 applies. Any violation stops the run with a specific error; nothing is repaired | Do not rely on row order; provenance must be exact |
| P14 | Sidecar validation | **Every** causal row is validated against the Task 2 sidecar JSON schema, plus explicit checks (view `CAUSAL`, velocity method `backward_difference_actual_dt`, coordinate space/system, units, world fields null, `world_position_valid = false`). Schema-validation runtime is **measured** and reported (earlier "≈ 1 min" was an estimate extrapolated from the Task 2 profile) | Full contract enforcement |
| P15 | Locations | Code `pedsafety/interaction/`; CLI `scripts/run_interaction_representation.py`; config `configs/phase4/interaction_representation_default.json`; tests `tests/phase4/task3/`; real run `experiments/phase4_task3_pie_video_0004/<UTC>_…/`; synthetic runs `experiments/phase4_task3_synthetic_validation/` | Mirrors Task 2 conventions; no existing `pedsafety/interaction` |
| P16 | Large outputs | After the real pilot run, **measure** the output sizes and only then decide whether `*.npz` (or other files) under the real-run directory need a `.gitignore` rule (pre-run estimate "tens of MB" is unmeasured). Any `.gitignore` change is reported and requires approval | Measured size first, then decide |
| P17 | Uncertainty diagnostics (Q8), **amended** | Report relative image-velocity component distributions (`dvel_u_px_s`, `dvel_v_px_s`) per relation, the estimated component-wise noise reference, and clearly labelled **exploratory** low-relative-speed diagnostics. The reference ≈ 68 px/s is the approximate **component-wise standard deviation of relative velocity under an independence assumption** when each agent has ≈ 48 px/s per-axis velocity noise (√2 × 48 ≈ 68), from the Task 2 single-stride jitter estimate. It is **not** a relative-speed threshold, not a validated speed threshold, not calibrated uncertainty and not a physical quantity; the independence assumption is known to be imperfect (Task 2 stride test). No uncertainty feature enters the default representation | User amendment; Task 2 review R2 |

## 5. Risks identified at T3.0
1. **Design-document status:** untracked and still marked "PROPOSED"; risk of a later edit changing the approved basis. Mitigation: hash recorded here; commit with Task 3.
2. **Task 2 manifest paths** use Windows backslashes and repository-relative form (`PIE_pilot_inputs\agent_states.jsonl`); the adapter must resolve paths explicitly and treat hashes, not paths, as identity.
3. **Large local inputs are gitignored** (`PIE_pilot_inputs/`, Task 2 JSONL). They are present locally and verified by hash; a fresh clone cannot reproduce T3.8 without them.
4. **Relative image velocity** inherits ego-motion parallax and ~48 px/s per-axis node jitter; features remain raw with masks, and P17 reports the noise reference as a diagnostic only.
5. **kNN scope (P4)** is an approved interpretation of the design; it affects only diagnostic flags, never stored edges.
6. **Edge volume:** the exact count is unknown until measured; the theoretical bound Σ n_t(n_t − 1) over enabled relation types will be computed and compared.
7. **Provenance assumptions (P7, P13):** the indexing base and `frame_gap_before` semantics are established from committed code; T3.1 must confirm them on the actual files before relying on them.

## 6. Change log
| Date | Version | Change |
|---|---|---|
| 2026-11-15 | 1.0.0 | T3.0: approved decisions recorded; open implementation details P1–P17 proposed for confirmation |
| 2026-11-15 | 1.1.0 | User approval of P1–P17 recorded; P17 amended (≈ 68 px/s is a component-wise relative-velocity noise reference under independence, not a threshold); P13 extended (indexing base, physical vs logical lines, blank/malformed lines, no row-order fallback, identity, box validity); P7, P9, P11, P14, P16 detailed; risk 7 added; design document unchanged |
