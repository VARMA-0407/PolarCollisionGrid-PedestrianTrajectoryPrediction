"""Warning lead time against VERIFIED event timestamps. Blocked unless independent, verified, documented event labels are supplied.

Definition when supported: lead_time_s = verified_event_time_s - warning_generation_time_s for the first ACTIVE_WARNING of an alert whose pair matches a verified
event and whose generation time precedes the event. Warning generation time, observation time, predicted conflict time and verified event time are four different
quantities; only the first two exist in this repository."""

from typing import Any, Dict, Mapping, Sequence

from .errors import LeadTimeBlocked
from .schema import validate_event  # noqa: F401  (schema import keeps the dependency explicit)

VERIFIED_SOURCE_TYPES = ("official_dataset_annotation", "independent_human_annotation", "independent_sensor_event_log")


def blocked_status() -> Dict[str, Any]:
    return {"status": "BLOCKED_INSUFFICIENT_GROUND_TRUTH",
            "reason": "no independently established event timestamps or verified conflict labels exist (one PIE video, no annotation files, perceived tracks only)",
            "required": ["verified collision / near-conflict events with frame timestamps from a documented independent source (name, type, version, SHA-256)",
                         "pair identities that match the alert dedup keys", "multiple independent videos for any aggregate statement"]}


def compute_lead_times(alerts: Sequence[Mapping[str, Any]], verified_events: Sequence[Mapping[str, Any]], source: Mapping[str, Any], fps: float) -> Dict[str, Any]:
    if not verified_events:
        raise LeadTimeBlocked("BLOCKED_INSUFFICIENT_GROUND_TRUTH", "no verified events supplied")
    if any(not source.get(k) for k in ("name", "type", "version", "sha256")) or source.get("type") not in VERIFIED_SOURCE_TYPES or len(str(source["sha256"])) != 64:
        raise LeadTimeBlocked("BLOCKED_INSUFFICIENT_GROUND_TRUTH", "verified-event source lacks name/type/version/sha256 or has an unaccepted type")
    for e in verified_events:
        if not (e.get("verified") is True and isinstance(e.get("event_frame"), int) and e.get("dedup_key")):
            raise LeadTimeBlocked("BLOCKED_INSUFFICIENT_GROUND_TRUTH", "each verified event needs verified=true, an integer event_frame and a dedup_key")
    by_key = {}
    for a in alerts:
        by_key.setdefault(a["dedup_key"], []).append(a)
    rows = []
    for e in verified_events:
        cands = [a for a in by_key.get(e["dedup_key"], []) if a["activated_frame"] <= e["event_frame"]]
        if cands:
            first = min(cands, key=lambda a: a["activated_frame"])
            rows.append({"dedup_key": e["dedup_key"], "alert_id": first["alert_id"], "lead_time_s": (e["event_frame"] - first["activated_frame"]) / fps, "detected": True})
        else:
            rows.append({"dedup_key": e["dedup_key"], "alert_id": None, "lead_time_s": None, "detected": False})
    return {"status": "COMPUTED", "source": dict(source), "n_events": len(rows), "n_detected": sum(r["detected"] for r in rows), "rows": rows,
            "definition": "verified event frame minus first ACTIVE_WARNING generation frame, in seconds of video time (frame / source_fps)"}


__all__ = ["compute_lead_times", "blocked_status", "VERIFIED_SOURCE_TYPES"]
