# Phase 4 Task 2: Image-Space Kinematic Preprocessing and Track Continuity

**Date:** 15 November 2026. **Branch:** `feature/phase4-task2-image-kinematics` (base `b66cbc5`, nothing committed).
**Status:** `COMPLETED_WITH_LIMITATIONS`. Implementation, synthetic validation and the **real PIE `set01/video_0004` pilot validation** are done. Gap bridging and smoothing remain deliberately disabled; their decisions are deferred (section 15).
**Machine-readable record:** `docs/phase4/phase4_task2_kinematic_report.json`. **Final review:** 15 November 2026 (section 0).

## 0. Final review and corrections (15 November 2026)
Evidence for every item: `experiments/phase4_task2_pie_video_0004/20261114T202653Z_pie_video_0004/analysis/review_addendum_20261115.json`, produced by the read-only `scripts/phase4_task2_pie_review_addendum.py` (no kinematics re-processed, nothing overwritten).

| # | Finding | Correction |
|---|---|---|
| R1 | The report gave the longest gap as "101 frames, track 8431, frames 3994–4159". **101 is correct; 3994–4159 is the track's first/last frame, not the gap.** Records: last observation before the gap at frame **4009**, first after at **4111**; frame difference **102**; missing frames **101** (implemented convention: difference − 1); elapsed **3.40 s**. Gap lists recomputed from the original input agree with the continuity report for all 1,500 tracks. | Section 11.3 corrected; JSON updated; regression test `test_gap_convention_separates_gap_boundaries_from_track_span` added (mutation-checked: an off-by-one and a span-for-boundaries substitution are both caught). |
| R2 | The earlier "predicted 2,955 vs measured 3,372 px/s²" comparison was presented as evidence that raw acceleration is noise-dominated. **It is not independent:** on uniformly spaced frames the causal acceleration *is* the second difference divided by dt², and σ was estimated from the same second differences, so the comparison only describes the shape of one distribution. | Section 11.4–11.6 rewritten; a stride test (σ_k for k = 1…8 frames) added as better, still assumption-dependent, evidence. |
| R3 | The ByteTrack-buffer argument for gaps > 30 frames was stated without code evidence. | Verified in Ultralytics 8.4.212 `byte_tracker.py`: `max_frames_lost = track_buffer` (30); stale lost tracks are removed after matching when `frame_id − end_frame > 30`, so coasting cannot produce more than 30 missing frames. The class-flip attribution remains an inference. |
| R4 | Consistency re-verified: primary and re-run outputs byte-identical except `processing_log.txt` (wall-clock times); manifests differ only in `command`, `started_utc`, `finished_utc` and the log's hash entry; input hashes unchanged; 0 schema violations in 130,776 rows; 0 world values; all causal rows backward-difference, all offline rows central-difference; `max_derivative_gap_frames = 0`; smoothing `none`. | None needed. |

| Item | Status |
|---|---|
| Causal velocity / acceleration / image heading, validity states, sidecar output | Implemented; tested synthetically; **validated on real PIE** |
| Offline central differences (separate module and file) | Implemented; tested synthetically; run on real PIE (offline file only) |
| Track continuity diagnostics and anomaly indicators | Implemented; tested synthetically; **run on real PIE** |
| Time-aware causal EMA; evaluation of the existing fixed-alpha EMA | Implemented; evaluated on synthetic data; **disabled** for the real run (no τ selected) |
| Runner/CLI, manifest, hashes, output policy, determinism | Implemented; tested; **verified on real PIE** |
| Gap bridging (`max_derivative_gap_frames > 0`) | Not implemented by design; refused by configuration; real gap distribution now measured |
| Offline interpolation, Kalman filter | Not implemented (not justified by evidence) |

## 1. Scope and objectives
Derive image-space motion from the existing Phase 3 tracked observations: velocity (pixel/s), acceleration (pixel/s²), image-plane heading (rad), track-continuity diagnostics, explicit validity and provenance, and separate causal and offline modes. Image space only (Phase 4 Task 1 contract); no inference rerun; Phase 3 outputs, canonical schema, Stage 4 code and Task 1 deliverables untouched.

## 2. Design decisions
- **Reuse, not duplication.** Derivative arithmetic and validity states are the Phase 2 Task 7 Stage 4 `FeatureEngine` (`pedsafety/trajectory.py`); duplicate/out-of-order/non-monotonic quarantine is Stage 4 `group_tracks`; the fixed-alpha EMA is Stage 4 `SmoothingPolicy('causal_ema')`. No Stage 4 file was modified.
- **No provenance reconstruction.** The Phase 3 `agent_states.jsonl` has no Stage 2 provenance sidecar. The new driver reads `image_x`/`image_y` directly after a contract check and feeds `FeatureEngine` with a Stage 4 `PositionInfo`; fields missing from a record stay null.
- **Canonical schema preserved.** Derived values go to a separately versioned sidecar, `docs/phase4/schemas/image_kinematics_sidecar.schema.json` (`image-kinematics-sidecar-1.0.0`, `additionalProperties: false`). Canonical records are neither rewritten nor extended (schema SHA-256 `784c8ca9…3c6c` unchanged, checked by a test).
- **Never repair.** A record that breaks the contract keeps its row with `observation_type = invalid`, a reason code, null kinematics and an engine restart. Unplaceable records and malformed JSON lines go to `quarantine.jsonl`.
- **Conservative defaults.** No interpolation, no smoothing, no gap bridging, `min_heading_speed = 0` (only an exactly zero vector has no heading); thresholds are reported as sensitivity tables rather than chosen.

New modules: `pedsafety/image_kinematics.py`, `pedsafety/image_kinematics_offline.py`, `pedsafety/track_continuity.py`, `pedsafety/image_kinematics_run.py`, `scripts/run_image_kinematics.py`, `scripts/phase4_task2_synthetic_validation.py`, and for the real run `scripts/phase4_task2_pie_input_integrity.py` and `scripts/phase4_task2_pie_analysis.py` (both read-only).

## 3. Algorithms and equations
Positions `p_i = (u_i, v_i)` are the footpoint (bbox bottom-centre) in pixels; `t_i` the record timestamp in seconds.

**Causal (only current and past samples):** velocity `v_i = (p_i − p_{i−1}) / (t_i − t_{i−1})`; acceleration `a_i = (v_i − v_{i−1}) / (t_i − t_{i−1})` (needs a valid `v_{i−1}` on the immediately preceding accepted frame); speed `|v_i|`; heading `atan2(v_y, v_x)` wrapped to `[−π, π)`, only if `speed > 0` and `speed ≥ min_heading_speed`.

**Offline (`OFFLINE_NON_CAUSAL`; uses the next sample):** velocity `v_i = (p_{i+1} − p_{i−1}) / (t_{i+1} − t_{i−1})`; acceleration `a_i = 2[(p_{i+1} − p_i)/(t_{i+1} − t_i) − (p_i − p_{i−1})/(t_i − t_{i−1})] / (t_{i+1} − t_{i−1})`.

**Time-aware causal EMA (available, disabled):** `f_i = f_{i−1} + α_i (p_i − f_{i−1})`, `α_i = 1 − exp(−(t_i − t_{i−1}) / τ)`, restarted wherever the engine breaks differencing; filtered derivatives come from a second, independent engine.

## 4. Coordinate and unit conventions
Raster `captured_1920x1080`; origin top-left; +u right; +v down; pixel, pixel/s, pixel/s², s, rad; heading `atan2(v_y, v_x)` from +u toward +v (**clockwise on screen**), `[−π, π)`, not a world heading. Every row carries these as schema constants plus `world_x = world_y = null`, `world_position_valid = false`. No metric, physical-speed, physical-acceleration, world-heading, TTC, distance or calibrated-probability claim is made; image motion from the moving PIE camera contains ego-motion.

## 5. Timestamp handling
Actual record timestamps; nominal FPS never substituted. `dt` must be finite and > 0. Non-finite timestamps → quarantined; repeated/decreasing timestamps → `rejected_timestamp_not_monotonic`; frames going backwards → `rejected_out_of_order_frame`; duplicate frames quarantine all copies. Input order per track (no sorting).

## 6. Gap handling
`max_derivative_gap_frames = 0`, enforced by the configuration loader. Any missing frame, rejected row, unavailable position or identity change restarts differencing (`gap_before` etc., null values). No interpolation in either mode.

## 7. Filtering and smoothing methodology (SYNTHETIC evidence)
Default `none`. Synthetic evaluation (`experiments/phase4_task2_synthetic_validation/20261114T201025Z_smoothing/`, fixed seeds, 12 s at 30 fps, 2 s warm-up excluded):

| Synthetic footpoint noise σ = 1 px, constant velocity | velocity RMSE uniform / irregular (px/s) | heading RMSE uniform (rad) |
|---|---|---|
| raw backward difference | 64.3 / 69.3 (theory 2σ/dt = 60) | 0.543 |
| causal_ema α = 0.3 | 14.7 / **29.8** | 0.111 |
| causal_ema α = 0.5 | 26.2 / 34.5 | 0.202 |
| time_aware_ema τ = 0.1 s | 13.8 / 13.5 | 0.104 |
| time_aware_ema τ = 0.2 s | 7.2 / 7.1 | 0.054 |
| time_aware_ema τ = 0.4 s | 3.7 / 3.6 | 0.027 |

Noise-free 90° manoeuvre, heading settle time: raw 0.033 s; time-aware τ = 0.1 / 0.2 / 0.4 s → 0.20 / 0.37 / 0.73 s (theory 0.18 / 0.36 / 0.71 s). Conclusions (synthetic model only): with σ = 1 px independent noise, raw acceleration is noise-dominated at 30 fps; the fixed-α EMA is sampling-dependent (degrades under irregular timestamps), the time-aware EMA is not; every τ trades noise for lag. No Kalman filter was added.

## 8. Causal versus offline processing
`image_kinematics.py` never imports the offline module (AST test); `CausalImageKinematics.update` uses only the current record and earlier state; offline rows go only to `offline_kinematics.jsonl`; `require_causal_rows` and the sidecar schema both refuse an offline row labelled causal. Tests: changing all observations from index 20 on leaves rows 0–19 byte-identical (none / causal_ema / time_aware_ema); streaming outputs equal batch outputs. **Real data:** all 62,212 valid causal velocities recomputed from the causal file's own positions as backward differences match the stored values exactly (max difference 0.0 px/s); all 65,388 causal rows are `CAUSAL|backward_difference_actual_dt` and all 65,388 offline rows `OFFLINE_NON_CAUSAL|central_difference_actual_dt`.

## 9. Track continuity diagnostics
Per track: counts, first/last frame and time, duration, every gap (frames, elapsed s), duplicate/out-of-order/non-monotonic/invalid-time/off-grid/sequence-conflict counts, border-touching rows. **Anomaly indicators** (never confirmed): suspicious jumps (displacement / mean box height, box-height ratio, sensitivity tables, combined "possible ID switch indicator" rule) and possible fragment pairs (same sequence and class, 1–30 frames apart, within 1 box height). `confirmed_id_switches` is always null; no track id is changed, merged or split.

## 10. Synthetic validation results (SYNTHETIC DATA ONLY)
- `tests/phase4/`: 76 tests at the time of the real run, **77 after the final review** (gap-convention regression test added), all passing in `pcg-baseline` (Python 3.9.25); the 76 were also run in `pedsafety-perception` (Python 3.11.18). Coverage: all 16 required cases plus boundaries, contract violations, output policy, CLI exit codes, determinism, source immutability, schema validity, manifest completeness, gap convention.
- Mutation checks: 13 safeguards broken one at a time in a scratch copy; 13 caught; plus 2 for the gap convention, both caught.
- Phase 3 *format* check on the local mock replay run (608 rows, 7 tracks, all schema-valid, input unchanged; recovered the scripted 6-frame dropout and the one-frame edge id).
- Synthetic scale run: 60,429 rows in 117 s.

## 11. Real PIE pilot validation (REAL DATA: PIE set01/video_0004, Phase 3 run 20261114T134036Z)
Run directory: `experiments/phase4_task2_pie_video_0004/20261114T202653Z_pie_video_0004/` containing `input_integrity.json`, `primary_causal/` (primary run), `determinism_rerun/` (identical second run) and `analysis/real_data_summary.json`.

### 11.1 Input integrity
Input folder: `D:\BTech-Research\PCG-Baseline\PIE_pilot_inputs\` (**inside the repository**, not `D:\BTech-Research\PIE_pilot_inputs\` as stated in the request; untracked; must not be committed).

| File | SHA-256 | Bytes | Rows |
|---|---|---|---|
| `agent_states.jsonl` | `ff8b9b217144a9a6792069aa2db11cd9e9ca43fbf00077ad4da6e2b3fe485dc7` | 67,565,978 | 65,388 |
| `tracked_agents.jsonl` | `182babb386a821da720b9520afa024cf4e9195baea3031c4970b64000c593ff5` | 38,774,621 | 65,388 |
| `run_manifest.json` | `e8c2a895543c6e222b58d9974a409faddab3b98e742fa048a4286bcd9b1dc7c2` | 5,456 | – |

The Phase 3 manifest records **no file hashes**, so byte-level verification against it is impossible; the hashes above are now the reference. Instead 24 structural checks against the manifest and the contract all passed: video `video_0004.mp4`, not mock, YOLO backend, 1920×1080 at 30 fps, 9,840 frames processed; 65,388 rows in both files (= manifest `track_rows` and `agent_states.records`); 1,500 unique track ids (= manifest); per-class counts identical (bike 7,049, bus 1,034, car 32,981, pedestrian 23,661, truck 663); row-by-row correspondence of frame, track number, bbox and timestamp between the two files (0 mismatches); all rows `captured_1920x1080`, unit pixel, world fields null, `world_position_valid = false`, kinematic fields null, footpoint = bbox bottom-centre = `image_x/y`, `observation_valid = true`, `interpolation_flag = false`, finite timestamps. Track rows span frames 0–9818 inside the processed range 0–9839 (an initial check wrongly compared the two ranges for equality; it was corrected to containment and the one-file report it produced was replaced). Hashes were unchanged after all runs.

Pre-run configuration: `configs/phase4/image_kinematics_default.json` (SHA-256 `757f5588…f601`, resolved `1f5e931c…5c28`), raster `captured_1920x1080`, `max_derivative_gap_frames = 0`, smoothing `none`, `min_heading_speed = 0`.

### 11.2 Rows, tracks, lengths
65,388 input rows; 0 quarantined; 65,388 valid positions; 0 invalid; 0 rejected (no duplicate, out-of-order or non-monotonic rows; no sequence conflicts). 1,500 tracks in one sequence; schema classes pedestrian 586 tracks / 23,661 rows, vehicle 729 / 34,678 (car, bus, truck), bicyclist 185 / 7,049. Track length (valid positions): median 13, p25 4, p75 44, p95 173, max 1,187; duration median 0.53 s, p95 6.2 s, max 39.7 s. Short tracks: 118 with 1 position, 225 with < 3, 385 with < 5, 615 with < 10, 982 with < 30 (65% of tracks).

### 11.3 Missing-frame gaps
1,676 gaps in 665 of 1,500 tracks; 7,706 missing frames in total.

| Missing frames | Gaps | % of gaps | Cumulative % |
|---|---|---|---|
| 1 | 688 | 41.1 | 41.1 |
| 2 | 266 | 15.9 | 56.9 |
| 3–5 | 337 | 20.1 | 77.0 |
| 6–10 | 188 | 11.2 | 88.2 |
| 11–30 | 183 | 10.9 | 99.2 |
| > 30 | 14 | 0.8 | 100 |

Gap convention (implemented in `pedsafety/track_continuity.py`): for consecutive accepted observations of one track at frames `f_a < f_b`, **missing frames = f_b − f_a − 1** (frame step 1) and **elapsed = t(f_b) − t(f_a)**; a track's first/last frame are not gap boundaries.

Gap duration (elapsed): median 0.10 s, p95 0.60 s, max 3.40 s. **14 gaps have more than 30 missing frames**. Longest five (last observed frame before → first after, missing frames, elapsed):

| Track | Class | Gap boundaries | Missing | Elapsed | Track span |
|---|---|---|---|---|---|
| `pie:video_0004:8431` | bicyclist | 4009 → 4111 | 101 | 3.40 s | 3994–4159 |
| `pie:video_0004:13632` | vehicle | 7405 → 7464 | 58 | 1.97 s | 7405–7469 |
| `pie:video_0004:3280` | vehicle | 1944 → 2003 | 58 | 1.97 s | 1937–2030 |
| `pie:video_0004:15242` | vehicle | 8789 → 8847 | 57 | 1.93 s | 8727–8908 |
| `pie:video_0004:1114` | vehicle | 938 → 982 | 43 | 1.47 s | 500–1015 |

In Ultralytics 8.4.212 a lost track is removed after matching once `frame_id − end_frame > track_buffer` (= 30), so tracker coasting cannot produce more than 30 missing frames between two rows of one id. The Phase 3 manifest reports 4,994 rows dropped for `track_class_changed` (the tracker keeps the id; the pipeline drops the row), which would leave longer holes. That attribution is an **inference**: the dropped rows are not in these files.

### 11.4–11.6 Velocity, acceleration, heading (causal, raw)
| | valid | invalid: track start | invalid: gap | invalid: other |
|---|---|---|---|---|
| velocity | 62,212 (95.1%) | 1,500 | 1,676 (2.6% of rows) | 0 |
| acceleration | 59,642 (91.2%) | 2,769 | 2,977 (4.6% of rows) | 0 |
| heading | 62,212 (95.1%) | 1,500 | 1,676 | 0 |

`dt` of every valid velocity is 1/30 s (no irregular sampling). Speed (pixel/s): median 105.5, p25 45.5, p75 230.9, p95 563.9, max 3,292. Acceleration magnitude (pixel/s²): median 3,372, p95 17,293, max 162,612. Heading availability vs a minimum-speed threshold (sensitivity only; default 0): 62,212 at 0, 59,776 at 10, 52,079 at 30, 42,205 at 60, 28,407 at 120 px/s. No row has exactly zero speed. Offline central differences: 59,642 valid, 2,882 track ends, 2,864 gap-adjacent.

**Noise evidence (for a later smoothing decision; no τ selected; corrected in the final review, R2).**

*Single-stride estimate (empirical statistic under a model).* For rows of one track on three consecutive frames, `d2 = p(f+1) − 2p(f) + p(f−1)` per axis; `σ = 1.4826 · MAD(d2) / √6`. The model assumes independent, identically distributed per-frame footpoint noise of standard deviation σ and negligible true change of velocity over two frames (then `Var(d2) = 6σ²`). Observations: 59,642 triples (pedestrian 21,579, vehicle 31,756, bicyclist 6,307). Result: σ ≈ 1.13 / 1.15 px (x / y) overall; pedestrian 1.01 / 1.70, vehicle 1.22 / 0.95, bicyclist 1.22 / 0.82 px; about 1.0–1.3 % of box height. Under the same model the implied raw velocity noise is `√2 σ / dt` ≈ 48 px/s per axis, and the implied median acceleration magnitude `1.1774 · √6 σ / dt²` ≈ 2,955 px/s² against the measured median 3,372 px/s². **This comparison is not an independent test**: on uniform 1/30 s spacing the causal acceleration equals `d2 / dt²`, so both numbers come from the same second differences; it shows only that the distribution is close to, but heavier-tailed than, an isotropic Gaussian.

*Stride test (better separation, still assumption-dependent).* `d2_k = p(f+k) − 2p(f) + p(f−k)`; white position noise gives a constant `σ_k`, a locally constant acceleration adds `a(k·dt)²`:

| k (frames) | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| σ_k x / y (px), all classes | 1.13 / 1.15 | 1.47 / 1.44 | 1.74 / 1.65 | 2.00 / 1.86 | 2.56 / 2.17 | 3.19 / 2.46 |
| triples | 59,642 | 56,459 | 53,976 | 51,922 | 48,524 | 45,725 |

σ_k grows with k, so the second differences are **not** pure white noise. If the whole k = 8 spread were locally constant acceleration over 16 frames, its contribution at k = 1 would be at most σ_8/64 ≈ 0.05 / 0.04 px, far below the observed 1.13 / 1.15 px. Under that assumption most of the frame-to-frame second difference is **not smooth motion**: it is detector box jitter and/or high-frequency image motion such as camera vibration (real image motion, not detector error). The assumption fails for abrupt motion; noise has **not** been isolated from real movement.

*Synthetic model vs real detections.* The synthetic evaluation (section 7) used independent Gaussian footpoint noise; real jitter is correlated across frames (the stride growth shows this), class- and size-dependent, affected by occlusion, truncation and box looseness, and mixed with ego-motion. Synthetic RMSE values therefore indicate filter behaviour, not expected PIE error.

Consequence: frame-to-frame raw acceleration on PIE is very likely dominated by components that are not smooth agent motion, and headings of slow agents (below roughly 50–100 px/s) are unreliable; both statements are evidence-based judgements, not proofs.

### 11.7 Suspicious jumps and scale changes (indicators only)
62,212 consecutive-frame pairs evaluated. Displacement > 0.25 / 0.5 / 1.0 / 2.0 box heights: 866 / 40 / 1 / 0 pairs (318 / 22 / 1 / 0 tracks). Box-height ratio > 1.5 / 2.0 / 3.0: 49 / 0 / 0 pairs (41 tracks at 1.5). Combined possible-ID-switch rule (> 2.0 box heights and ratio > 1.5): **0 pairs**. Largest jump: `pie:video_0004:11604` frames 5670→5671, 109.7 px = 1.03 box heights (ratio 1.12). Tracks with most > 0.25 box-height steps: `pie:video_0004:12382` (vehicle, frames 6435–6476, 39 of 41 steps), `pie:video_0004:13191` (vehicle, 7061–7242, 36), `pie:video_0004:9798` (vehicle, 4587–4904, 18; 52 border rows). Vehicles moving fast relative to a small box height produce this; none is a confirmed ID switch.

### 11.8 Possible fragment pairs (indicators only; nothing merged)
Pairs (same class, later track starts ≤ N frames after an earlier one ends, ≤ 1 box height apart): 53 at N = 1, 274 at 5, 642 at 15, 1,116 at 30. Closest examples: `5717 → 5726` (pedestrian, frames 3048 → 3049, 0.03 box heights), `6414 → 7225` (vehicle, 3536 → 3537), `13106 → 13108` (vehicle, 6940 → 6941), `7395 → 7980` (pedestrian, 3843 → 3844). A pair can mean a fragmented identity, or two different nearby agents (common in crowds); counts are pairs, not unique tracks. No independent identity evidence exists, so none is confirmed.

### 11.9 Other problematic-track examples
Most gaps: `pie:video_0004:3380` (vehicle, frames 1971–2325, 19 gaps, 136 missing frames), `13210` (vehicle, 7109–7273, 18 gaps), `13396` (vehicle, 7226–7400, 17), `15242` (vehicle, 8727–8908, 16 gaps incl. a 57-frame gap), `1114` (vehicle, 500–1015, 15 gaps incl. 43). Longest gaps: see the table in 11.3 (gap boundaries are listed there; the frame ranges in this paragraph are track spans). Most border-touching rows (possible truncation, Task 1): `8390` (vehicle, frames 3976–4533, 363 of 558 rows), `2648` (vehicle, 1457–1643, 161 of 165), `13666` (vehicle, 7428–8301, 98). 3,336 border-touching rows in total.

### 11.10 Runtime and output size (measured, local CPU i5-1235U, Python 3.9.25)
Primary run 167.4 s wall (manifest 20:27:44 → 20:30:30 UTC); determinism re-run 224.0 s (the machine was also running other work). Outputs per run: `causal_kinematics.jsonl` 105.9 MB, `offline_kinematics.jsonl` 106.3 MB, `track_continuity_report.json` 1.5 MB; whole run directory 408 MB.

### 11.11 Schema and contract violations
None. 0 input contract violations (11.1); 0 quarantined rows; 0 sidecar-schema violations in 130,776 rows re-validated from disk; 0 rows with populated world fields.

### 11.12 Determinism
Every output of the re-run is byte-identical to the primary run except `processing_log.txt` and `run_manifest.json`, which differ only in `command`, `started_utc`, `finished_utc` and the log hash (wall-clock and directory name). Primary hashes: causal `9e5c5f3d…82e3`, offline `9e19ff3f…09e8`, continuity `64f8b734…144b`, quality `a7b71acc…8167313`, config `804f6993…c515`, manifest `ad208959…29c515` (full values in the JSON report).

## 12. Known limitations
1. Image space only; ego-motion not removed; nothing here is a physical speed, acceleration, heading, distance, TTC or calibrated probability.
2. Phase 3 recorded no file hashes; input integrity rests on structural cross-checks and the reference hashes recorded here.
3. `max_derivative_gap_frames = 0` costs 1,676 velocities (2.6%) and 2,977 accelerations (4.6%) on PIE; 41% of gaps are one frame. Bridging remains disabled pending review.
4. Frame-to-frame raw causal acceleration on PIE is very likely dominated by non-smooth components (jitter and/or vibration), and headings of slow agents are unreliable (11.4–11.6, stride test). Noise is not isolated from real motion. Smoothing remains disabled; the evidence does not select a τ.
5. Gaps > 30 frames are attributed to Phase 3 class-flip row drops by inference only.
6. Causal acceleration divides by the current-step `dt` (equal spacing on PIE, so no effect here).
7. Anomaly indicators are image-geometry heuristics, not ID-switch detections.
8. Outputs are large (408 MB per run directory, almost all in four JSONL sidecars); `PIE_pilot_inputs/` (102 MB) sits inside the repository. Both are now protected by `.gitignore` (inputs folder; `*.jsonl` under the real-run directory), leaving the run's reports, manifests and analysis JSON visible.
9. Two superseded synthetic run directories are retained as historical evidence and ignored by `.gitignore` until a removal or inclusion decision: `experiments/phase4_task2_synthetic_validation/20261114T195051Z_smoothing` and `…/20261114T195107Z_phase3_synthetic_replay_format_check`.
10. The original analysis summary (`analysis/real_data_summary.json`, script v1.0.0, SHA-256 `ea989bba…55dd`) lists track spans, not gap boundaries, under `examples.longest_single_gap`; the addendum gives the boundaries. The original file is kept unchanged.

## 13. Reproducibility instructions
Environment: `C:\Users\revan\anaconda3\envs\pcg-baseline\python.exe` (Python 3.9.25, numpy 1.26.4, jsonschema 4.25.1). From the repository root:

```
python -m pytest tests/phase4 -q -p no:cacheprovider
python scripts/phase4_task2_pie_input_integrity.py --input-dir PIE_pilot_inputs --out experiments/phase4_task2_pie_video_0004/<UTC>_pie_video_0004/input_integrity.json
python scripts/run_image_kinematics.py --input PIE_pilot_inputs/agent_states.jsonl --config configs/phase4/image_kinematics_default.json --out experiments/phase4_task2_pie_video_0004/<UTC>_pie_video_0004/primary_causal --run-manifest PIE_pilot_inputs/run_manifest.json --tracked-agents PIE_pilot_inputs/tracked_agents.jsonl
python scripts/run_image_kinematics.py ... --out experiments/phase4_task2_pie_video_0004/<UTC>_pie_video_0004/determinism_rerun   (same arguments)
python scripts/phase4_task2_pie_analysis.py --run <run>/primary_causal --rerun <run>/determinism_rerun --input PIE_pilot_inputs/agent_states.jsonl --out <run>/analysis/real_data_summary.json
python scripts/phase4_task2_pie_review_addendum.py --run-dir <run> --input-dir PIE_pilot_inputs --out <run>/analysis/review_addendum_<date>.json
python scripts/phase4_task2_synthetic_validation.py --out experiments/phase4_task2_synthetic_validation/<UTC>_smoothing
```

## 14. Task completion status
**`COMPLETED_WITH_LIMITATIONS`.** Evidence by category:

1. **Implementation evidence:** modules in section 2 reuse the unmodified Stage 4 engine; sidecar schema; runner with manifest, hashes and output policy; 77 passing Phase 4 tests; 15 of 15 mutation checks caught. Full regression after the final review (`pytest tests/phase4 tests/phase3 tests/task7 tests/phase2`, `pcg-baseline`): 929 passed, 6 skipped, 1 failed: the known, unmodified baseline defect `tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension`. (At the time of the real run it was 928 passed; the difference is the new regression test.)
2. **Synthetic validation:** sections 7 and 10 (synthetic data only).
3. **Real PIE validation:** section 11 (65,388 rows, 1,500 tracks, 1,676 gaps in 665 tracks, 7,706 missing frames under the stated convention; no contract, schema, world-field or causal/offline violation; deterministic; inputs unchanged).
4. **Known limitations:** section 12.
5. **Deliberately deferred decisions:** section 15.

Nothing here claims metric-world kinematics, physical speed or acceleration, physical TTC/PET, calibrated risk probabilities or deployment-ready safety performance. Task 3 has not been started.

## 15. Deliberately deferred decisions
- **Gap bridging** (labelled cross-gap derivatives with actual elapsed time): 41.1% of gaps miss one frame, 77.0% at most five. Not enabled.
- **Causal smoothing** (time-aware EMA, τ): evidence in 7 and 11.4–11.6; τ not selected; Kalman filter not justified yet.
- **Minimum heading speed**: sensitivity table in 11.4–11.6; default stays 0.
- **Handling of class-flip holes**: requires Phase 3 to emit (or log) the dropped rows; a Phase 3 change, not part of Task 2.
- **Deletion of the two superseded synthetic runs**: awaiting your approval.
