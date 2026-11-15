# Phase 8 — Early warning and ITS alert integration (RESEARCH PROTOTYPE)

Label `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`. Branch `feature/phase8-early-warning`, cut from Phase 7 commit `209b738`. Alerts produced here are **heuristic, in-sample,
image-space pilot events**. They are not validated, not calibrated, not collision predictions and not for operational or safety use.

## 1. Handoff verification (Phase 7)
Phase 7 is committed (`209b738`, parent `d6ac175`), working tree was clean. Actual contract consumed: `RiskOutput.to_records()` with schema `phase7-risk-output-1.0.0`
(fields: `pedestrian_id`, `vehicle_id`, `anchor_frame`, `window_id`, `conflict_score(+kind)`, `calibrated_probability`, `calibration_status`, `temporal_urgency`, `severity`(null),
`risk_score`, `risk_basis`, `risk_category`, `category_status`, `reason`, `masks{conflict_valid, category_assigned, out_of_distribution}`).
**Handoff gap (documented, not bypassed):** Phase 7 persisted `risk_arrays.npz` (no pedestrian/vehicle ids) and only the first 25 records. Phase 8 therefore regenerates the full record
list through the *public* Phase 7 API (`build_conflict_pairs` → `assess_risk` → `to_records`) from the same committed artifacts and **aborts unless the result equals `risk_arrays.npz`
field by field** (window id, anchor frame, risk score, conflict score, category, status, probabilities all NaN). No Phase 4–7 file was modified. A `from-records` mode consumes any
JSONL(.gz) of Phase 7 records, so a future Phase 7 that persists its records needs no Phase 8 change.
Facts about Phase 7 values that matter downstream: categories are `PROVISIONAL_ENGINEERING` (thresholds 0.15/0.4/0.7, no empirical evidence), probabilities are null, `vehicle_id` is the
perceived-track index inside the video (a string like `"3"`), pedestrian ids are `pie:video_0004:<track>`; there is no `set01` prefix and no persisted video fps (taken from the
Phase 5 window configuration, 30 fps).

## 2. Pipeline
Video → Phase 3 perception (YOLO + ByteTrack) → Phase 4 image-space geometry and interaction representation → Phase 5 multimodal trajectory forecasting → Phase 6 pairwise
image-space conflict representation → Phase 7 risk scoring / provisional categories (calibration blocked) → **Phase 8 early-warning event generation**.

## 3. Modules (`pedsafety/phase8/`)
| Module | Role |
|---|---|
| `config.py` | strict versioned `AlertConfig`; `config_status` must be `ENGINEERING_CONFIGURATION_NOT_VALIDATED`; hysteresis ordering enforced |
| `phase7_input.py` | `parse_record` → `Observation`; malformed ⇒ `InputContractError`, no usable evidence ⇒ `valid=False` with reason |
| `manager.py` | `AlertManager`: per-pair state machine, persistence, hysteresis, dedup, cooldown, clearance, event log, alert table |
| `schema.py` | JSON Schema (draft 2020-12) `phase8-alert-event-1.0.0`, `validate_event` |
| `lead_time.py` | lead-time interface, blocked without verified events |
| `runner.py`, `__main__.py` | Phase 7 regeneration + cross-check, run, JSONL/JSON/manifest writers, CLI |
No HTTP server, dashboard or external connection was built: the repository has no web framework, alert service or ITS dependency (`requirements*.txt`), so the boundary is JSON files,
the Python API and a CLI. No connection to any traffic-management system exists or is implied.

## 4. Information discipline (no future leakage)
`step(frame, observations)` accepts one anchor frame; an observation carrying another frame raises `FRAME_MISMATCH`; earlier frames raise `OUT_OF_ORDER`; a repeated frame is ignored
only when its content is identical. A decision at frame *f* depends only on frame *f* and the machine's past (tested: output for a stream prefix equals the prefix of the full output).
`process_stream` is a replay scheduler for recorded Phase 7 output: it groups records by frame and hands them over in ascending order.

## 5. Timing vocabulary
*Observation time* = anchor frame / source fps (stream time). *Warning generation time* = observation time of the anchor at which the state was decided (identical in stream time; no wall
clock is recorded, to keep replay deterministic). *Predicted conflict time*: not exposed by Phase 7 (urgency is unitless, not a time) ⇒ `UNAVAILABLE`. *Verified event time*: none ⇒
`BLOCKED_INSUFFICIENT_GROUND_TRUTH`. No lead time, TTC or PET is reported. The alert manager was timed alone (≈2×10⁴ observations/s, 0.38 s for 8233 records, in `run_manifest.json`); this
excludes video decoding, perception, forecasting and risk computation and is **not** an end-to-end latency or real-time claim.

## 6. Limitations
Pilot single video, in-sample Phase 5 model, perceived tracks as the only reference; heuristic uncalibrated scores; provisional categories; engineering thresholds; vehicle ids not stable
across sessions; replay-only (no streaming latency evidence); no verified events, hence no precision/recall/false-alarm/lead-time evaluation of the alerts.
