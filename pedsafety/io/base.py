"""Adapter interface and the raw (pre-schema) observation record.

A RawObservation is what an adapter could read from a source file, with
units/semantics declared by the adapter configuration. It is NOT a schema
record: Stage 2 converts it, validates geometry (box ordering, duplicate
timestamps, ...) and maps classes. Adapters must not repair, reorder,
interpolate or drop data silently; unparseable rows become ParseIssues.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from ..errors import ParseError


class IssueCode(str, Enum):
    MISSING_VALUE = "missing_value"
    INVALID_NUMBER = "invalid_number"
    NON_FINITE = "non_finite"
    INVALID_FRAME_INDEX = "invalid_frame_index"
    OUT_OF_RANGE = "out_of_range"
    ROW_TOO_SHORT = "row_too_short"
    ROW_TOO_LONG = "row_too_long"


@dataclass(frozen=True)
class ParseIssue:
    """A problem in one source row. ``source_line`` is the 1-based file line
    on which the record ended (the header is line 1)."""

    source_line: int
    column: Optional[str]
    code: IssueCode
    message: str
    raw_value: Optional[str] = None

    def __str__(self) -> str:
        col = "" if self.column is None else " column '%s'" % self.column
        val = "" if self.raw_value is None else " (value %r)" % (self.raw_value,)
        return "line %d%s [%s]: %s%s" % (self.source_line, col, self.code.value, self.message, val)


@dataclass(frozen=True)
class RawObservation:
    """One agent at one frame, as supplied by the source.

    timestamp is seconds from the sequence origin. ``pos_x``/``pos_y`` are in
    the space named by ``position_semantics`` (e.g. ``image_pixel``) and unit
    ``position_unit`` - both declared by configuration, never inferred, and
    checked for mutual compatibility by the adapter config (``unverified_planar``
    carries no unit). ``position_coordinate_system`` names the frame when the
    config declared one. None of this asserts calibration or world validity.
    ``bbox_xyxy`` is passed through exactly as supplied (ordering unchecked).
    """

    source_dataset: str
    sequence_id: str
    scene_id: str
    track_id: str
    frame_index: int
    timestamp: float
    source_label: Optional[str] = None
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None
    position_semantics: Optional[str] = None
    position_unit: Optional[str] = None
    position_coordinate_system: Optional[str] = None
    vel_x: Optional[float] = None
    vel_y: Optional[float] = None
    detection_confidence: Optional[float] = None
    source_line: int = 0
    extras: Tuple[Tuple[str, str], ...] = ()


@dataclass
class ParseResult:
    observations: List[RawObservation] = field(default_factory=list)
    issues: List[ParseIssue] = field(default_factory=list)
    rows_read: int = 0

    @property
    def rows_rejected(self) -> int:
        return len({i.source_line for i in self.issues})

    def raise_if_issues(self) -> None:
        if self.issues:
            raise ParseError(self.issues)


class AnnotationAdapter(ABC):
    """Dataset adapter interface. Observations are returned in source order."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short adapter identifier used in provenance."""

    @abstractmethod
    def parse(self) -> ParseResult:
        """Read the source and return observations plus row-level issues.
        Configuration/file-level problems raise ConfigError."""
