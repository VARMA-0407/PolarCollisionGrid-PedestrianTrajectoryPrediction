"""Observation-level validation primitives: issue codes, structured issues,
the validation report, and the production bounding-box validator.

Nothing here repairs, reorders, clips or deduplicates data. A problem is
always reported as a :class:`ValidationIssue` that references the offending
input observation, so no evidence is silently discarded.
"""

import math
import numbers
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import PedsafetyError


class ObservationIssueCode(str, Enum):
    INVALID_IDENTIFIER = "invalid_identifier"
    INVALID_FRAME_INDEX = "invalid_frame_index"
    INVALID_TIMESTAMP = "invalid_timestamp"
    MISSING_CLASS_LABEL = "missing_class_label"
    UNMAPPED_CLASS = "unmapped_class"
    BBOX_MALFORMED = "bbox_malformed"
    BBOX_NON_FINITE = "bbox_non_finite"
    BBOX_INVERTED = "bbox_inverted"
    BBOX_DEGENERATE = "bbox_degenerate"
    BBOX_SEMANTICS_MISMATCH = "bbox_semantics_mismatch"
    INVALID_CONFIDENCE = "invalid_confidence"
    COORDINATE_DECLARATION_INVALID = "coordinate_declaration_invalid"
    MISSING_COORDINATE_SYSTEM = "missing_coordinate_system"
    INCOMPLETE_POSITION = "incomplete_position"
    INCOMPLETE_VELOCITY = "incomplete_velocity"
    NON_FINITE_VALUE = "non_finite_value"
    VELOCITY_CONTRACT_MISMATCH = "velocity_contract_mismatch"
    DUPLICATE_OBSERVATION = "duplicate_observation"
    OUT_OF_ORDER_FRAME = "out_of_order_frame"
    TIMESTAMP_NOT_MONOTONIC = "timestamp_not_monotonic"
    TRACK_ID_COLLISION = "track_id_collision"
    SCHEMA_VIOLATION = "schema_violation"


@dataclass(frozen=True)
class ValidationIssue:
    """One problem tied to one input observation (``input_index`` is its
    0-based position in the sequence given to the builder)."""

    code: ObservationIssueCode
    message: str
    input_index: int
    source_line: int = 0
    source_dataset: Optional[str] = None
    sequence_id: Optional[str] = None
    track_id: Optional[str] = None
    frame_index: Optional[int] = None
    field: Optional[str] = None
    details: Tuple[Tuple[str, str], ...] = ()

    def __str__(self) -> str:
        who = "input #%d" % self.input_index
        if self.source_line:
            who += " (source line %d)" % self.source_line
        ident = "/".join(str(x) for x in (self.source_dataset, self.sequence_id, self.track_id) if x is not None)
        fr = "" if self.frame_index is None else " frame %s" % (self.frame_index,)
        return "%s [%s%s] %s: %s" % (who, ident, fr, self.code.value, self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value, "message": self.message, "input_index": self.input_index,
            "source_line": self.source_line, "source_dataset": self.source_dataset,
            "sequence_id": self.sequence_id, "track_id": self.track_id,
            "frame_index": self.frame_index, "field": self.field, "details": dict(self.details),
        }


@dataclass
class ValidationReport:
    n_input: int = 0
    n_records: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    # Evidence that was deliberately NOT promoted into canonical records.
    counters: Dict[str, int] = field(default_factory=dict)

    @property
    def quarantined_indices(self) -> List[int]:
        return sorted({i.input_index for i in self.issues})

    @property
    def n_quarantined(self) -> int:
        return len(self.quarantined_indices)

    @property
    def ok(self) -> bool:
        return not self.issues

    def counts_by_code(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i in self.issues:
            out[i.code.value] = out.get(i.code.value, 0) + 1
        return dict(sorted(out.items()))

    def summary(self) -> str:
        return "%d input, %d records, %d quarantined, %d issue(s) %s" % (
            self.n_input, self.n_records, self.n_quarantined, len(self.issues), self.counts_by_code())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_input": self.n_input, "n_records": self.n_records, "n_quarantined": self.n_quarantined,
            "counts_by_code": self.counts_by_code(), "counters": dict(sorted(self.counters.items())),
            "issues": [i.to_dict() for i in self.issues],
        }


class ObservationValidationError(PedsafetyError):
    """Raised in strict mode. ``report`` holds every issue found in the batch."""

    def __init__(self, report: ValidationReport):
        self.report = report
        shown = "; ".join(str(i) for i in report.issues[:5])
        more = "" if len(report.issues) <= 5 else " (+%d more; see .report)" % (len(report.issues) - 5)
        super().__init__("observation validation failed: %s. First issues: %s%s" % (report.summary(), shown, more))


class BoundingBoxError(ValueError):
    """Raised by :func:`validate_bbox_xyxy`; ``code`` is an ObservationIssueCode."""

    def __init__(self, code: ObservationIssueCode, message: str):
        self.code = code
        super().__init__(message)


def is_real_number(v: Any) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def bbox_problem(bbox: Any) -> Optional[Tuple[ObservationIssueCode, str]]:
    """Return ``(code, message)`` for the first problem with ``bbox`` or None.

    Convention (approved schema / Task 3): ``[x_min, y_min, x_max, y_max]`` in
    image pixels with a top-left origin. Requirements: exactly four real,
    finite numbers, x_min < x_max and y_min < y_max (strict, so width and
    height are positive). The box is never reordered or clipped.
    """
    if isinstance(bbox, (str, bytes)) or not hasattr(bbox, "__len__") or not hasattr(bbox, "__iter__"):
        return ObservationIssueCode.BBOX_MALFORMED, "bbox must be a sequence of four numbers, got %r" % (bbox,)
    vals = list(bbox)
    if len(vals) != 4:
        return ObservationIssueCode.BBOX_MALFORMED, "bbox must contain exactly four coordinates, got %d" % len(vals)
    for name, v in zip(("x_min", "y_min", "x_max", "y_max"), vals):
        if not is_real_number(v):
            return ObservationIssueCode.BBOX_MALFORMED, "bbox %s is not a number: %r" % (name, v)
    for name, v in zip(("x_min", "y_min", "x_max", "y_max"), vals):
        if not math.isfinite(v):
            return ObservationIssueCode.BBOX_NON_FINITE, "bbox %s is not finite: %r" % (name, v)
    x0, y0, x1, y1 = (float(v) for v in vals)
    if x0 > x1 or y0 > y1:
        return ObservationIssueCode.BBOX_INVERTED, (
            "bbox is reversed (x_min=%r, x_max=%r, y_min=%r, y_max=%r); boxes are never reordered" % (x0, x1, y0, y1))
    if x0 == x1 or y0 == y1:
        return ObservationIssueCode.BBOX_DEGENERATE, (
            "bbox has zero width or height (width=%r, height=%r)" % (x1 - x0, y1 - y0))
    return None


def validate_bbox_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return the box as a float 4-tuple or raise :class:`BoundingBoxError`."""
    problem = bbox_problem(bbox)
    if problem is not None:
        raise BoundingBoxError(*problem)
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return (x0, y0, x1, y1)


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Midpoint of the horizontal and vertical bounds (validated box)."""
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def bbox_footpoint(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Bottom-centre of the box. Image y grows downward, so 'bottom' is y_max.
    This is a ground-contact proxy, not a measured contact point (Task 3/4)."""
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, y1)
