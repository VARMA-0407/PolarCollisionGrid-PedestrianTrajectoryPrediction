"""Data contracts of the perception frontend (Phase 3, Stage 2A).

Runtime independent: this module imports only the standard library and ``pedsafety.validation``
(for the bounding-box rule shared with the agent-state schema). It does NOT import OpenCV,
Ultralytics, a tracker library, numpy or torch, so the forecasting side never depends on them.

Conventions (binding for every detector, tracker and frame source):

* **Bounding box**: ``(x_min, y_min, x_max, y_max)``, floats, IMAGE PIXELS of the frame named by
  ``FrameMeta.coordinate_system``; origin at the top-left, x to the right, y DOWNWARD. This is the
  schema's ``bbox_xyxy``. Boxes are never reordered; zero width/height is invalid. Pixel coordinates are
  NOT metres and nothing here converts them.
* **Footpoint**: bottom-centre ``((x_min + x_max) / 2, y_max)`` - a ground-contact PROXY for upright agents,
  not a measured contact point (unreliable for cropped, occluded or non-upright boxes).
* **Timestamp**: ``FrameMeta.timestamp_s`` is SECONDS from the sequence origin (finite, >= 0), strictly
  increasing within a sequence. ``timestamp_source`` says where it came from; a frame-rate is never guessed.
* **Frame index**: non-negative integer, strictly increasing within a sequence (gaps allowed and kept).
* **Confidence**: detector score in [0, 1]. It is a model score, not a calibrated probability.
* **Track id**: positive integer unique within one sequence and never reused for another object there.
  Source datasets' ids and these perception ids are different namespaces.
* **Classes**: pedestrian, car, truck, bus, bike. Anything else is rejected, never defaulted.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..errors import PedsafetyError
from ..validation import bbox_problem, is_real_number

BBOX_CONVENTION = "xyxy_image_pixels_top_left_origin_y_down"
TIMESTAMP_SOURCES = ("pts", "frame_index_over_fps", "dataset_annotation", "mock_clock")


class PerceptionClass(str, Enum):
    PEDESTRIAN = "pedestrian"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    BIKE = "bike"


class IssueCode(str, Enum):
    INVALID_FRAME_INDEX = "invalid_frame_index"
    INVALID_TIMESTAMP = "invalid_timestamp"
    INVALID_IMAGE_SIZE = "invalid_image_size"
    INVALID_FRAME_META = "invalid_frame_meta"
    FRAME_ORDER = "frame_order"
    SEQUENCE_CHANGED = "sequence_changed"
    UNSUPPORTED_CLASS = "unsupported_class"
    INVALID_BBOX = "invalid_bbox"
    BBOX_OUT_OF_FRAME = "bbox_out_of_frame"
    INVALID_CONFIDENCE = "invalid_confidence"
    DUPLICATE_DETECTION_ID = "duplicate_detection_id"
    INVALID_DETECTION = "invalid_detection"
    DETECTOR_ERROR = "detector_error"
    TRACKER_ERROR = "tracker_error"
    INVALID_TRACK_ID = "invalid_track_id"
    DUPLICATE_TRACK_ID = "duplicate_track_id"
    TRACK_CLASS_CHANGED = "track_class_changed"
    UNKNOWN_MATCHED_DETECTION = "unknown_matched_detection"
    INVALID_TRACK = "invalid_track"


class PerceptionError(PedsafetyError):
    """Raised in strict mode (or for unrecoverable misuse). ``issue`` carries the structured problem."""

    def __init__(self, issue: "PerceptionIssue"):
        self.issue = issue
        super().__init__("[%s/%s] %s" % (issue.stage, issue.code.value, issue.message))


@dataclass(frozen=True)
class PerceptionIssue:
    stage: str                       # 'source' | 'detector' | 'tracker' | 'adapter'
    code: IssueCode
    message: str
    frame_index: Optional[int] = None
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "code": self.code.value, "message": self.message, "frame_index": self.frame_index, "detail": self.detail}


@dataclass(frozen=True)
class ComponentInfo:
    """Identity of one pipeline component, stored in every record's provenance.

    ``is_mock=True`` marks a test double: its outputs are never detection/tracking results and must
    never be reported as performance. ``parameters`` are free-form (name, value) strings, e.g. weights
    hash and thresholds for a real detector."""

    kind: str                        # 'frame_source' | 'detector' | 'tracker'
    name: str
    version: str
    is_mock: bool
    parameters: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.kind not in ("frame_source", "detector", "tracker"):
            raise ValueError("kind must be frame_source/detector/tracker, got %r" % (self.kind,))
        if not (isinstance(self.name, str) and self.name.strip() and isinstance(self.version, str) and self.version.strip()):
            raise ValueError("component name and version must be non-empty strings")
        if not isinstance(self.is_mock, bool):
            raise ValueError("is_mock must be a bool (it must be stated explicitly)")


@dataclass(frozen=True)
class FrameMeta:
    """Metadata of one frame, as produced by a ``FrameSource``. Pixels are not carried here."""

    source_id: str                   # video file / stream / sequence identifier
    sequence_id: str
    frame_index: int
    timestamp_s: float
    width: int
    height: int
    coordinate_system: str           # names the pixel raster, e.g. 'captured_1920x1080'; never invented by the pipeline
    timestamp_source: str
    nominal_fps: Optional[float] = None


def frame_problem(f: Any) -> Optional[Tuple[IssueCode, str]]:
    """First problem with a ``FrameMeta`` or None (type, index, time, size, naming)."""
    if not isinstance(f, FrameMeta):
        return IssueCode.INVALID_FRAME_META, "frame must be a FrameMeta, got %r" % (type(f).__name__,)
    for name in ("source_id", "sequence_id", "coordinate_system"):
        v = getattr(f, name)
        if not (isinstance(v, str) and v.strip()):
            return IssueCode.INVALID_FRAME_META, "%s must be a non-empty string, got %r" % (name, v)
    if isinstance(f.frame_index, bool) or not isinstance(f.frame_index, int) or f.frame_index < 0:
        return IssueCode.INVALID_FRAME_INDEX, "frame_index must be a non-negative integer, got %r" % (f.frame_index,)
    t = f.timestamp_s
    if not is_real_number(t) or not math.isfinite(t) or t < 0:
        return IssueCode.INVALID_TIMESTAMP, "timestamp_s must be a finite number of seconds >= 0, got %r" % (t,)
    for name in ("width", "height"):
        v = getattr(f, name)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            return IssueCode.INVALID_IMAGE_SIZE, "%s must be an integer >= 1, got %r" % (name, v)
    if f.timestamp_source not in TIMESTAMP_SOURCES:
        return IssueCode.INVALID_FRAME_META, "timestamp_source must be one of %s, got %r" % (TIMESTAMP_SOURCES, f.timestamp_source)
    if f.nominal_fps is not None and not (is_real_number(f.nominal_fps) and math.isfinite(f.nominal_fps) and f.nominal_fps > 0):
        return IssueCode.INVALID_FRAME_META, "nominal_fps must be a positive finite number or None, got %r" % (f.nominal_fps,)
    return None


@dataclass(frozen=True)
class RawDetection:
    """What a detector returns for one frame (frame identity is bound later by the pipeline).

    ``detection_id`` must be unique within the frame. ``label`` is the detector's own label string. ``class_id`` is the
    detector's own integer class index (e.g. the COCO id) or None when the detector has none; it is provenance only and is
    never used for semantic mapping (that is the label map's job)."""

    detection_id: int
    label: str
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: Optional[int] = None


@dataclass(frozen=True)
class DetectionRecord:
    """A validated detection with full provenance."""

    sequence_id: str
    frame_index: int
    timestamp_s: float
    image_width: int
    image_height: int
    coordinate_system: str
    detection_id: int
    label_raw: str
    category: PerceptionClass
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    clipped: bool                    # True only if the out-of-frame policy clipped the box (original in bbox_raw)
    bbox_raw: Tuple[float, float, float, float]
    detector: ComponentInfo
    class_id: Optional[int] = None   # detector's own class index (provenance only); None when the detector has none

    @property
    def footpoint(self) -> Tuple[float, float]:
        x0, _, x1, y1 = self.bbox_xyxy
        return ((x0 + x1) / 2.0, y1)


@dataclass(frozen=True)
class TrackObservation:
    """What a tracker returns for one frame.

    ``is_predicted=True`` marks a box that is a motion-model estimate with NO matched detection in this frame. It
    is not an observation; the agent-state adapter drops such rows (the schema cannot represent an estimate)."""

    track_id: int
    category: PerceptionClass
    bbox_xyxy: Tuple[float, float, float, float]
    matched_detection_id: Optional[int]
    is_predicted: bool = False
    tracking_confidence: Optional[float] = None


@dataclass(frozen=True)
class TrackRecord:
    """A validated track state with full provenance."""

    sequence_id: str
    frame_index: int
    timestamp_s: float
    image_width: int
    image_height: int
    coordinate_system: str
    track_id: int
    category: PerceptionClass
    bbox_xyxy: Tuple[float, float, float, float]
    detection_confidence: Optional[float]      # confidence of the matched detection (None if none)
    matched_detection_id: Optional[int]
    is_predicted: bool
    tracking_confidence: Optional[float]
    detector: ComponentInfo
    tracker: ComponentInfo

    @property
    def footpoint(self) -> Tuple[float, float]:
        x0, _, x1, y1 = self.bbox_xyxy
        return ((x0 + x1) / 2.0, y1)


@dataclass(frozen=True)
class LabelMap:
    """Explicit detector-label -> supported-category map (case/space-insensitive). Unlisted labels are
    UNSUPPORTED. The default aliases the usual COCO names; ``motorcycle`` is deliberately absent."""

    mapping: Mapping[str, str] = field(default_factory=lambda: {
        "pedestrian": "pedestrian", "person": "pedestrian", "ped": "pedestrian",
        "car": "car", "truck": "truck", "bus": "bus", "bike": "bike", "bicycle": "bike"})

    def __post_init__(self):
        norm: Dict[str, PerceptionClass] = {}
        for k, v in self.mapping.items():
            if not (isinstance(k, str) and k.strip()):
                raise ValueError("label map keys must be non-empty strings, got %r" % (k,))
            norm[k.strip().lower()] = PerceptionClass(v)       # ValueError for an unsupported target
        object.__setattr__(self, "_norm", norm)

    def resolve(self, label: Any) -> Optional[PerceptionClass]:
        if not isinstance(label, str):
            return None
        return self._norm.get(label.strip().lower())   # type: ignore[attr-defined]


def detection_problem(d: Any) -> Optional[Tuple[IssueCode, str]]:
    """First structural problem with a ``RawDetection`` (class support is checked by the label map)."""
    if not isinstance(d, RawDetection):
        return IssueCode.INVALID_DETECTION, "detector must return RawDetection objects, got %r" % (type(d).__name__,)
    if isinstance(d.detection_id, bool) or not isinstance(d.detection_id, int):
        return IssueCode.INVALID_DETECTION, "detection_id must be an integer, got %r" % (d.detection_id,)
    c = d.confidence
    if not is_real_number(c) or not math.isfinite(c) or not 0.0 <= c <= 1.0:
        return IssueCode.INVALID_CONFIDENCE, "confidence must be a finite number in [0, 1], got %r" % (c,)
    p = bbox_problem(d.bbox_xyxy)
    if p is not None:
        return IssueCode.INVALID_BBOX, p[1]
    if d.class_id is not None and (isinstance(d.class_id, bool) or not isinstance(d.class_id, int)):
        return IssueCode.INVALID_DETECTION, "class_id must be an integer or None, got %r" % (d.class_id,)
    return None


def track_problem(t: Any) -> Optional[Tuple[IssueCode, str]]:
    if not isinstance(t, TrackObservation):
        return IssueCode.INVALID_TRACK, "tracker must return TrackObservation objects, got %r" % (type(t).__name__,)
    if isinstance(t.track_id, bool) or not isinstance(t.track_id, int) or t.track_id < 1:
        return IssueCode.INVALID_TRACK_ID, "track_id must be a positive integer, got %r" % (t.track_id,)
    if not isinstance(t.category, PerceptionClass):
        return IssueCode.INVALID_TRACK, "category must be a PerceptionClass, got %r" % (t.category,)
    p = bbox_problem(t.bbox_xyxy)
    if p is not None:
        return IssueCode.INVALID_BBOX, p[1]
    if t.tracking_confidence is not None and not (is_real_number(t.tracking_confidence) and math.isfinite(t.tracking_confidence)
                                                  and 0.0 <= t.tracking_confidence <= 1.0):
        return IssueCode.INVALID_CONFIDENCE, "tracking_confidence must be in [0, 1] or None, got %r" % (t.tracking_confidence,)
    if not isinstance(t.is_predicted, bool):
        return IssueCode.INVALID_TRACK, "is_predicted must be a bool"
    if t.is_predicted and t.matched_detection_id is not None:
        return IssueCode.INVALID_TRACK, "a predicted-only track cannot name a matched detection"
    if not t.is_predicted and t.matched_detection_id is None:
        return IssueCode.INVALID_TRACK, "an observed track must name its matched detection"
    return None
