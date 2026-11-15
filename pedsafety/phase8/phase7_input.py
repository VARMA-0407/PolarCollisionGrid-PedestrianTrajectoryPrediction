"""Phase 7 record contract on the Phase 8 side.

A Phase 7 record is one element of ``RiskOutput.to_records()`` (schema ``phase7-risk-output-1.0.0``). Malformed records (missing fields, wrong types, an
unknown category, a probability emitted without calibration) raise ``InputContractError``: the Phase 7 contract is broken and nothing is guessed.
Records that are well-formed but carry no usable evidence (``INSUFFICIENT_EVIDENCE``, invalid or inconsistent risk score) become ``Observation(valid=False)``
with an explicit reason; they are never converted to LOW and never carried forward.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .errors import InputContractError

PHASE7_SCHEMA = "phase7-risk-output-1.0.0"
CATEGORIES = ("LOW", "MODERATE", "HIGH", "CRITICAL")
REQUIRED = ("pedestrian_id", "vehicle_id", "anchor_frame", "window_id", "conflict_score", "conflict_score_kind", "calibrated_probability", "calibration_status",
            "temporal_urgency", "severity", "severity_status", "risk_score", "risk_basis", "risk_category", "category_status", "reason", "masks")
REQUIRED_MASKS = ("conflict_valid", "category_assigned", "out_of_distribution")


@dataclass(frozen=True)
class Observation:
    video_id: str
    pedestrian_id: str
    vehicle_id: str
    frame: int
    window_id: Optional[int]
    risk_category: str
    category_status: str
    risk_score: Optional[float]
    conflict_score: Optional[float]
    conflict_score_kind: str
    calibrated_probability: Optional[float]
    calibration_status: str
    temporal_urgency: Optional[float]
    masks: Dict[str, bool]
    reason: Optional[str]
    valid: bool
    invalid_reason: Optional[str]
    conflicting_indicators: bool
    source_index: Optional[int]

    @property
    def pair_key(self):
        return (self.video_id, self.pedestrian_id, self.vehicle_id)

    @property
    def dedup_key(self) -> str:
        return "%s|%s|%s" % self.pair_key


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def video_of(pedestrian_id: str) -> str:
    if ":" not in pedestrian_id:
        raise InputContractError("VIDEO_ID", "cannot derive the video id from pedestrian_id %r (expected '<video>:<track>')" % pedestrian_id)
    return pedestrian_id.rsplit(":", 1)[0]


def parse_record(rec: Mapping[str, Any], index: Optional[int] = None, *, policy: str = "flag_only") -> Observation:
    if not isinstance(rec, Mapping):
        raise InputContractError("RECORD", "record %s is not an object" % index)
    miss = [k for k in REQUIRED if k not in rec]
    if miss:
        raise InputContractError("MISSING_FIELDS", "record %s lacks %s" % (index, miss))
    if not (isinstance(rec["pedestrian_id"], str) and rec["pedestrian_id"] and isinstance(rec["vehicle_id"], str) and rec["vehicle_id"]):
        raise InputContractError("IDENTITY", "record %s: pedestrian_id and vehicle_id must be non-empty strings" % index)
    if not (isinstance(rec["anchor_frame"], int) and not isinstance(rec["anchor_frame"], bool) and rec["anchor_frame"] >= 0):
        raise InputContractError("FRAME", "record %s: anchor_frame must be a non-negative integer" % index)
    cat = rec["risk_category"]
    if cat not in CATEGORIES and cat != "INSUFFICIENT_EVIDENCE":
        raise InputContractError("CATEGORY", "record %s: unknown risk_category %r" % (index, cat))
    masks = rec["masks"]
    if not isinstance(masks, Mapping) or any(k not in masks or not isinstance(masks[k], bool) for k in REQUIRED_MASKS):
        raise InputContractError("MASKS", "record %s: masks must hold boolean %s" % (index, REQUIRED_MASKS))
    p, cs = rec["calibrated_probability"], rec["calibration_status"]
    if p is not None and (cs != "CALIBRATED" or not _finite(p) or not 0 < p < 1):
        raise InputContractError("PROBABILITY", "record %s: a probability must be null unless calibration_status == CALIBRATED and lie in (0,1)" % index)
    if cs == "CALIBRATED" and p is None and masks["conflict_valid"]:
        raise InputContractError("PROBABILITY", "record %s: CALIBRATED record without a probability" % index)
    for k in ("risk_score", "conflict_score", "temporal_urgency"):
        if rec[k] is not None and not _finite(rec[k]):
            raise InputContractError("NONFINITE", "record %s: %s must be null or finite (JSON null encodes invalid)" % (index, k))
    valid, why = True, None
    rs = rec["risk_score"]
    if cat == "INSUFFICIENT_EVIDENCE":
        valid, why = False, rec["reason"] or "INSUFFICIENT_EVIDENCE"
    elif rs is None or not 0 <= rs <= 1 or not masks["category_assigned"] or not masks["conflict_valid"]:
        valid, why = False, "INCONSISTENT_RECORD_CATEGORY_WITHOUT_VALID_RISK_SCORE"
    conflicting = isinstance(rec["category_status"], str) and rec["category_status"].endswith("_CONFLICTING_INDICATORS")
    if valid and conflicting and policy == "treat_as_insufficient":
        valid, why = False, "CONFLICTING_INDICATORS_POLICY"
    return Observation(video_of(rec["pedestrian_id"]), rec["pedestrian_id"], rec["vehicle_id"], int(rec["anchor_frame"]),
                       rec["window_id"] if isinstance(rec["window_id"], int) else None, cat, str(rec["category_status"]), rs, rec["conflict_score"],
                       str(rec["conflict_score_kind"]), p, str(cs), rec["temporal_urgency"], {k: masks[k] for k in REQUIRED_MASKS}, rec["reason"], valid, why, conflicting, index)


__all__ = ["Observation", "parse_record", "video_of", "CATEGORIES", "PHASE7_SCHEMA"]
