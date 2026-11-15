"""Versioned alert-event schema (JSON Schema draft 2020-12) and validation."""

import hashlib
import json
from typing import Any, Dict, Mapping

from jsonschema import Draft202012Validator

from .errors import AlertContractError

EVENT_SCHEMA_VERSION = "phase8-alert-event-1.0.0"
STATES = ("NO_ALERT", "WARNING_PENDING", "ACTIVE_WARNING", "ESCALATED_WARNING", "CLEARED", "INSUFFICIENT_EVIDENCE")
EVENT_TYPES = ("STATE_TRANSITION", "ALERT_UPDATED")
WARNING_LEVELS = ("NONE", "ADVISORY", "ELEVATED", "UNDETERMINED_EVIDENCE_LOST")
CLEARANCE_KINDS = ("RISK_SUBSIDED", "EVIDENCE_LOST", "END_OF_STREAM")
PROTOTYPE_NOTICE = "RESEARCH PROTOTYPE: pilot, in-sample, image-space heuristic alerts; not validated, not for operational or safety use."

_nullable = lambda t: {"type": [t, "null"]}                                                         # noqa: E731
_status_value = {"type": "object", "required": ["status", "value"], "additionalProperties": False, "properties": {"status": {"type": "string"}, "value": {"type": "null"}}}

EVENT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": EVENT_SCHEMA_VERSION,
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "event_id", "event_seq", "event_type", "alert_id", "dedup_key", "video_id", "pedestrian_id", "vehicle_id", "pair_type", "frame",
                 "observation_time_s", "warning_generation_time_s", "from_state", "to_state", "warning_level", "warning_level_status", "trigger_reason", "phase7",
                 "evidence_valid", "evidence_limitations", "first_detection_frame", "first_detection_time_s", "last_update_frame", "last_update_time_s", "clearance_frame",
                 "clearance_time_s", "clearance_kind", "predicted_conflict_time", "verified_event_time", "severity", "source", "hashes", "provenance", "prototype_notice"],
    "properties": {
        "schema_version": {"const": EVENT_SCHEMA_VERSION},
        "event_id": {"type": "string", "minLength": 8},
        "event_seq": {"type": "integer", "minimum": 0},
        "event_type": {"enum": list(EVENT_TYPES)},
        "alert_id": _nullable("string"),
        "dedup_key": {"type": "string", "minLength": 1},
        "video_id": {"type": "string", "minLength": 1},
        "pedestrian_id": {"type": "string", "minLength": 1},
        "vehicle_id": {"type": "string", "minLength": 1},
        "pair_type": {"const": "pedestrian_vehicle"},
        "frame": {"type": "integer", "minimum": 0},
        "observation_time_s": {"type": "number", "minimum": 0},
        "warning_generation_time_s": {"type": "number", "minimum": 0},
        "from_state": {"enum": list(STATES)},
        "to_state": {"enum": list(STATES)},
        "warning_level": {"enum": list(WARNING_LEVELS)},
        "warning_level_status": {"const": "PROVISIONAL_ENGINEERING_NOT_PHYSICAL_SEVERITY"},
        "trigger_reason": {"type": "string", "minLength": 1},
        "phase7": {"type": "object", "additionalProperties": False,
                   "required": ["risk_score", "risk_category", "category_status", "conflict_score", "conflict_score_kind", "temporal_urgency", "calibrated_probability",
                                "calibration_status", "masks", "reason", "conflicting_indicators"],
                   "properties": {"risk_score": _nullable("number"), "risk_category": {"type": ["string", "null"]}, "category_status": {"type": ["string", "null"]},
                                  "conflict_score": _nullable("number"), "conflict_score_kind": {"type": ["string", "null"]}, "temporal_urgency": _nullable("number"),
                                  "calibrated_probability": _nullable("number"), "calibration_status": {"type": ["string", "null"]},
                                  "masks": {"type": ["object", "null"]}, "reason": _nullable("string"), "conflicting_indicators": {"type": ["boolean", "null"]}}},
        "evidence_valid": {"type": "boolean"},
        "evidence_limitations": {"type": "array", "items": {"type": "string"}},
        "first_detection_frame": _nullable("integer"), "first_detection_time_s": _nullable("number"),
        "last_update_frame": _nullable("integer"), "last_update_time_s": _nullable("number"),
        "clearance_frame": _nullable("integer"), "clearance_time_s": _nullable("number"),
        "clearance_kind": {"enum": list(CLEARANCE_KINDS) + [None]},
        "predicted_conflict_time": _status_value, "verified_event_time": _status_value, "severity": _status_value,
        "source": {"type": "object", "additionalProperties": False, "required": ["phase7_record_index", "phase7_records_sha256"],
                   "properties": {"phase7_record_index": _nullable("integer"), "phase7_records_sha256": _nullable("string")}},
        "hashes": {"type": "object", "additionalProperties": False, "required": ["alert_config_sha256", "event_schema_sha256", "phase7_risk_config_sha256",
                                                                                "source_checkpoint_file_sha256", "calibration_artifact_sha256"],
                   "properties": {k: _nullable("string") for k in ("alert_config_sha256", "event_schema_sha256", "phase7_risk_config_sha256", "source_checkpoint_file_sha256",
                                                                   "calibration_artifact_sha256")}},
        "provenance": {"type": "object"},
        "prototype_notice": {"const": PROTOTYPE_NOTICE},
    },
}
_VALIDATOR = Draft202012Validator(EVENT_SCHEMA)


def schema_sha256() -> str:
    return hashlib.sha256(json.dumps(EVENT_SCHEMA, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_event(event: Mapping[str, Any]) -> None:
    errs = sorted(_VALIDATOR.iter_errors(event), key=lambda e: list(e.path))
    if errs:
        raise AlertContractError("EVENT_SCHEMA", "; ".join("%s: %s" % ("/".join(map(str, e.path)) or "<root>", e.message) for e in errs[:5]))
    if event["event_type"] == "STATE_TRANSITION" and event["from_state"] == event["to_state"]:
        raise AlertContractError("EVENT_SCHEMA", "a STATE_TRANSITION must change the state")
    if event["clearance_kind"] is not None and event["to_state"] != "CLEARED":
        raise AlertContractError("EVENT_SCHEMA", "clearance_kind is only valid on a CLEARED event")


__all__ = ["EVENT_SCHEMA", "EVENT_SCHEMA_VERSION", "STATES", "validate_event", "schema_sha256", "PROTOTYPE_NOTICE", "WARNING_LEVELS", "CLEARANCE_KINDS"]
