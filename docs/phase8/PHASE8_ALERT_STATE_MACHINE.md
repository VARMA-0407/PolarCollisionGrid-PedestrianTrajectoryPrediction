# Phase 8 — Alert state machine

One machine per pedestrian–vehicle pair, key `video_id|pedestrian_id|vehicle_id`. Different ids are different pairs: identities are never merged or inherited.
Category rank: LOW 0, MODERATE 1, HIGH 2, CRITICAL 3 (Phase 7 `risk_category`; PROVISIONAL_ENGINEERING). A HIGH or CRITICAL category means "elevated heuristic image-space proximity
score", **not** a verified or imminent collision. All parameters below are **engineering choices** (`configs/phase8/alert_default.json`, `config_status =
ENGINEERING_CONFIGURATION_NOT_VALIDATED`); none is derived from verified events. Counts are consecutive anchors (3 frames = 0.1 s at 30 fps).

| Parameter | Default | Meaning |
|---|---|---|
| `enter_min_category` / `enter_persistence` | HIGH / 3 | consecutive observations ≥ HIGH needed (≥ 2 enforced) |
| `hold_min_category` | MODERATE | category ≥ this keeps an alert open (hysteresis below enter) |
| `escalate_min_category` / `escalate_persistence` | CRITICAL / 2 | consecutive observations ≥ CRITICAL |
| `escalated_hold_min_category` / `deescalate_persistence` | HIGH / 2 | consecutive observations < HIGH step ESCALATED back to ACTIVE |
| `clear_persistence` | 3 | consecutive valid observations < hold clear an alert |
| `max_insufficient_observations` | 5 | invalid / absent steps tolerated before an alert is closed as EVIDENCE_LOST |
| `max_gap_frames` | 6 | a larger jump between processed frames counts as missing frames |
| `cooldown_frames` | 30 | after clearance the same pair cannot enter WARNING_PENDING again |
| `conflicting_indicators_policy` | flag_only | or `treat_as_insufficient` |

## States and transitions (anything else raises `InvalidTransition`)
| From | To | Condition (trigger_reason) |
|---|---|---|
| NO_ALERT | WARNING_PENDING | valid obs ≥ enter, not in cooldown (`ENTER_THRESHOLD_MET`); suppressed obs are counted, not logged |
| NO_ALERT | INSUFFICIENT_EVIDENCE | invalid / missing obs |
| WARNING_PENDING | ACTIVE_WARNING | `enter_persistence` consecutive obs ≥ enter (`PERSISTENCE_SATISFIED`); alert id created |
| WARNING_PENDING | NO_ALERT | category < enter before persistence (`RISK_BELOW_ENTER_BEFORE_PERSISTENCE`) |
| WARNING_PENDING | INSUFFICIENT_EVIDENCE | invalid / missing obs, or a frame gap (`MISSING_FRAMES`) |
| ACTIVE_WARNING | ESCALATED_WARNING | `escalate_persistence` consecutive obs ≥ escalate |
| ACTIVE_WARNING | CLEARED | `clear_persistence` consecutive valid obs < hold (`RISK_SUBSIDED`) |
| ACTIVE_WARNING | INSUFFICIENT_EVIDENCE | invalid / absent obs or gap |
| ESCALATED_WARNING | ACTIVE_WARNING | `deescalate_persistence` consecutive obs < escalated_hold (clear streak is kept) |
| ESCALATED_WARNING | CLEARED / INSUFFICIENT_EVIDENCE | as for ACTIVE |
| INSUFFICIENT_EVIDENCE | ACTIVE / ESCALATED | (with alert) valid obs; ESCALATED only if the pre-loss state was ESCALATED and the obs ≥ escalated_hold (`EVIDENCE_RESTORED`) |
| INSUFFICIENT_EVIDENCE | CLEARED | (with alert) more than `max_insufficient_observations` invalid steps, `clearance_kind = EVIDENCE_LOST`, or end of stream |
| INSUFFICIENT_EVIDENCE | WARNING_PENDING / NO_ALERT | (no alert) valid obs ≥ enter and not in cooldown / otherwise, or evidence not recovered |
| CLEARED | NO_ALERT | the next step for that pair (consumes it; cooldown keeps running) |

Properties (each covered by a test): hysteresis (MODERATE neither opens nor closes an alert; a single dip never clears), persistence (a one-observation spike never alerts or escalates),
deduplication (one open alert per pair; repeated ACTIVE observations create no events; one alert id per episode, deterministic hash of pair, activation frame and config),
cooldown, invalid observations are never treated as LOW and never counted toward clearing or carried forward (counters restart), absent pairs and frame gaps become
INSUFFICIENT_EVIDENCE (`PAIR_NOT_OBSERVED`, `MISSING_FRAMES`), `EVIDENCE_LOST` and `END_OF_STREAM` closures are recorded as such and are **not** "risk subsided". At most one transition per
pair and step, except that a frame gap may move an alert to INSUFFICIENT_EVIDENCE and the same frame's valid observation then restores it (both events are logged). Pairs are processed in
sorted key order, so output is independent of input order within a frame. Probabilities are never used in a decision.
`ALERT_UPDATED` events (no state change) are logged when the Phase 7 category or the conflicting-indicator flag of an open alert changes. `warning_level` (ADVISORY for ACTIVE, ELEVATED for
ESCALATED, UNDETERMINED_EVIDENCE_LOST while evidence is missing) is a provisional state label, not physical severity.
