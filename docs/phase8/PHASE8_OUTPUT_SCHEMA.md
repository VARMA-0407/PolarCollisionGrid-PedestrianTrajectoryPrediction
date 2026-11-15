# Phase 8 — Output schema

Machine-readable definition: `pedsafety/phase8/schema.py` (`EVENT_SCHEMA`, JSON Schema draft 2020-12, additionalProperties false, hashed into every event as `event_schema_sha256`).

## Alert event `phase8-alert-event-1.0.0` (one JSON object per line in `alert_events.jsonl.gz`)
`schema_version`, `event_id` (deterministic hash), `event_seq` (0-based, replay order), `event_type` (`STATE_TRANSITION` | `ALERT_UPDATED`), `alert_id` (null until a pair becomes
ACTIVE_WARNING and again after it returns to NO_ALERT), `dedup_key` (`video|pedestrian|vehicle`), `video_id`, `pedestrian_id`, `vehicle_id`, `pair_type` (`pedestrian_vehicle`), `frame`,
`observation_time_s`, `warning_generation_time_s`, `from_state`, `to_state`, `warning_level` (+`warning_level_status`), `trigger_reason`,
`phase7` {`risk_score`, `risk_category`, `category_status`, `conflict_score`, `conflict_score_kind`, `temporal_urgency`, `calibrated_probability` (null unless Phase 7 supplies a CALIBRATED one),
`calibration_status`, `masks`, `reason`, `conflicting_indicators`} (all null for events caused by an absent pair), `evidence_valid`, `evidence_limitations[]`, `first_detection_frame/_time_s`,
`last_update_frame/_time_s`, `clearance_frame/_time_s`, `clearance_kind` (`RISK_SUBSIDED` | `EVIDENCE_LOST` | `END_OF_STREAM`, only on a CLEARED event),
`predicted_conflict_time`, `verified_event_time`, `severity` (each `{status, value:null}`; statuses `UNAVAILABLE_NOT_EXPOSED_BY_PHASE7_AND_NOT_PHYSICAL`, `BLOCKED_INSUFFICIENT_GROUND_TRUTH`,
`NOT_SUPPORTED_NO_METRIC_GEOMETRY`), `source` {`phase7_record_index`, `phase7_records_sha256`}, `hashes` {`alert_config_sha256`, `event_schema_sha256`, `phase7_risk_config_sha256`,
`source_checkpoint_file_sha256`, `calibration_artifact_sha256`}, `provenance` (dataset, label, sample status, Phase 5/6/7 hashes, coordinate system), `prototype_notice`.
Unavailable values are explicit `null` / status objects; nothing is invented. Times are stream time (`frame / source_fps`), never wall clock.

## Other files of a run
`alerts.json` (`phase8-alert-table-1.0.0`: one row per alert with first detection, activation, last update, clearance frame/time/kind, peak state/category/score, event count, evidence-lost steps,
final state); `summary_report.json` (counts, transitions, clearance kinds, blocked lead time, claims all false); `run_manifest.json` (config + hash, schema hash, input record hashes, Phase 7
cross-check result, Phase 7 provenance, component throughput with caveat); `phase7_records_input.jsonl.gz` (the exact Phase 7 records consumed).

## Interfaces
Python: `AlertConfig.load`, `parse_record`, `AlertManager(cfg, provenance, source_records_sha256=, hashes=).step(frame, observations)` / `.finalize()` / `.process_stream(obs)` / `.alert_table()`,
`validate_event`. CLI: `python -m pedsafety.phase8 from-phase7 ...` (regenerate + cross-check, then run) and `python -m pedsafety.phase8 from-records --records R --alert-config C --out DIR`.
