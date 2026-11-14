"""Perception pipeline orchestration (Stage 2A): frames -> detector -> validated detections -> tracker -> validated tracks.

The pipeline owns validation; detectors and trackers are untrusted. It imports only interfaces, so any real YOLO or
ByteTrack wrapper (Stage 2B) plugs in without the forecasting side importing either library.

Error policy (``strict``):
* ``strict=True``  - the first issue raises :class:`PerceptionError` (carrying the structured issue).
* ``strict=False`` - issues are recorded and processing continues: an invalid FRAME (bad index/time/size, out of order,
  sequence change) is skipped entirely (detector and tracker are not called for it); an invalid DETECTION or TRACK is
  dropped; a detector/tracker exception skips the frame's remaining stages. Nothing is silently repaired.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from .detector import Detector
from .performance import PerformanceRecorder
from .sources import FrameSource
from .tracker import Tracker
from .types import (
    ComponentInfo, DetectionRecord, FrameMeta, IssueCode, LabelMap, PerceptionClass, PerceptionError, PerceptionIssue, TrackRecord,
    detection_problem, frame_problem, track_problem,
)

PIPELINE_VERSION = "perception-pipeline-0.1.0-stage2a"


@dataclass(frozen=True)
class PipelineConfig:
    """``out_of_frame``: 'reject' (default) drops a box that leaves the frame by more than ``bbox_tolerance_px``;
    'clip' clips it to the frame and records ``clipped=True`` with the original box. ``bbox_tolerance_px`` is the
    allowed overshoot in pixels for either policy before action is taken (0 = exact)."""

    strict: bool = False
    label_map: LabelMap = field(default_factory=LabelMap)
    out_of_frame: str = "reject"
    bbox_tolerance_px: float = 0.0

    def __post_init__(self):
        if self.out_of_frame not in ("reject", "clip"):
            raise ValueError("out_of_frame must be 'reject' or 'clip', got %r" % (self.out_of_frame,))
        if not (isinstance(self.bbox_tolerance_px, (int, float)) and not isinstance(self.bbox_tolerance_px, bool)
                and self.bbox_tolerance_px >= 0 and self.bbox_tolerance_px == self.bbox_tolerance_px):
            raise ValueError("bbox_tolerance_px must be a number >= 0")


@dataclass(frozen=True)
class FrameResult:
    frame: FrameMeta
    detections: Tuple[DetectionRecord, ...]
    tracks: Tuple[TrackRecord, ...]
    issues: Tuple[PerceptionIssue, ...]
    processed: bool                  # False if the frame was skipped


@dataclass
class PerceptionResult:
    frames: List[FrameResult]
    issues: List[PerceptionIssue]
    source: ComponentInfo
    detector: ComponentInfo
    tracker: ComponentInfo
    pipeline_version: str = PIPELINE_VERSION

    @property
    def is_mock(self) -> bool:
        return self.source.is_mock or self.detector.is_mock or self.tracker.is_mock

    @property
    def counters(self) -> Dict[str, int]:
        c = {"frames_seen": len(self.frames), "frames_processed": sum(1 for f in self.frames if f.processed),
             "frames_skipped": sum(1 for f in self.frames if not f.processed), "detections": sum(len(f.detections) for f in self.frames),
             "tracks": sum(len(f.tracks) for f in self.frames), "issues": len(self.issues)}
        for i in self.issues:
            c["issue:" + i.code.value] = c.get("issue:" + i.code.value, 0) + 1
        return dict(sorted(c.items()))

    @property
    def components(self) -> Tuple[ComponentInfo, ...]:
        return (self.source, self.detector, self.tracker)


class PerceptionPipeline:
    def __init__(self, source: FrameSource, detector: Detector, tracker: Tracker, config: Optional[PipelineConfig] = None,
                 recorder: Optional[PerformanceRecorder] = None):
        for obj, kind in ((source, "frame_source"), (detector, "detector"), (tracker, "tracker")):
            if obj.info.kind != kind:
                raise ValueError("component %r has kind %r, expected %r" % (obj.info.name, obj.info.kind, kind))
        self.source, self.detector, self.tracker = source, detector, tracker
        self.config = config or PipelineConfig()
        self.recorder = recorder
        self._orphan_issues: List[PerceptionIssue] = []     # issues of items that were not even FrameMeta objects

    # ------------------------------------------------------------- helpers
    def _issue(self, issues: List[PerceptionIssue], stage: str, code: IssueCode, msg: str, frame_index: Optional[int], detail: Optional[str] = None):
        iss = PerceptionIssue(stage, code, msg, frame_index, detail)
        if self.config.strict:
            raise PerceptionError(iss)
        issues.append(iss)

    def _check_box_in_frame(self, box, frame: FrameMeta) -> Tuple[Optional[Tuple[float, float, float, float]], bool]:
        """Return (box_to_use, clipped); box_to_use is None when the box is rejected."""
        tol = float(self.config.bbox_tolerance_px)
        x0, y0, x1, y1 = box
        inside = x0 >= -tol and y0 >= -tol and x1 <= frame.width + tol and y1 <= frame.height + tol
        if inside:
            return box, False
        if self.config.out_of_frame == "reject":
            return None, False
        cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(float(frame.width), x1), min(float(frame.height), y1)
        if cx0 >= cx1 or cy0 >= cy1:
            return None, False                       # nothing of the box remains in the frame
        return (cx0, cy0, cx1, cy1), True

    # ---------------------------------------------------------------- run
    def iter_results(self) -> Iterator[FrameResult]:
        det_info, trk_info = self.detector.info, self.tracker.info
        last_index: Optional[int] = None
        last_time: Optional[float] = None
        sequence: Optional[str] = None
        track_class: Dict[int, PerceptionClass] = {}
        self.tracker.reset()
        self._orphan_issues = []
        if self.recorder is not None:
            self.recorder.start_run()
        for frame, payload in self.source.frames():
            issues: List[PerceptionIssue] = []
            fi = getattr(frame, "frame_index", None)
            fp = frame_problem(frame)
            if fp is None and sequence is not None and frame.sequence_id != sequence:
                fp = (IssueCode.SEQUENCE_CHANGED, "sequence_id changed from %r to %r inside one run" % (sequence, frame.sequence_id))
            if fp is None and last_index is not None and not frame.frame_index > last_index:
                fp = (IssueCode.FRAME_ORDER, "frame_index %d does not increase (previous %d)" % (frame.frame_index, last_index))
            if fp is None and last_time is not None and not frame.timestamp_s > last_time:
                fp = (IssueCode.FRAME_ORDER, "timestamp %r does not increase (previous %r) at frame %d" % (frame.timestamp_s, last_time, frame.frame_index))
            if fp is not None:
                self._issue(issues, "source", fp[0], fp[1], fi if isinstance(fi, int) else None)
                meta = frame if isinstance(frame, FrameMeta) else None
                if meta is not None:
                    yield FrameResult(meta, (), (), tuple(issues), False)
                else:
                    self._orphan_issues.extend(issues)
                continue
            sequence, last_index, last_time = frame.sequence_id, frame.frame_index, frame.timestamp_s
            yield self._process(frame, payload, det_info, trk_info, track_class, issues)

    def _process(self, frame: FrameMeta, payload: Any, det_info: ComponentInfo, trk_info: ComponentInfo,
                 track_class: Dict[int, PerceptionClass], issues: List[PerceptionIssue]) -> FrameResult:
        rec = self.recorder
        fctx = rec.frame() if rec is not None else _null()
        with fctx:
            with (rec.stage("detect") if rec else _null()):
                try:
                    raw = list(self.detector.detect(frame, payload))
                except Exception as exc:                                  # noqa: BLE001 - boundary to untrusted code
                    self._issue(issues, "detector", IssueCode.DETECTOR_ERROR, "detector raised %s: %s" % (type(exc).__name__, exc), frame.frame_index)
                    return FrameResult(frame, (), (), tuple(issues), False)
            dets: List[DetectionRecord] = []
            seen_ids: Set[int] = set()
            dup_ids: Set[int] = set()
            for d in raw:
                p = detection_problem(d)
                if p is not None:
                    self._issue(issues, "detector", p[0], p[1], frame.frame_index)
                    continue
                if d.detection_id in seen_ids:
                    dup_ids.add(d.detection_id)
                seen_ids.add(d.detection_id)
            for d in raw:
                if detection_problem(d) is not None:
                    continue
                if d.detection_id in dup_ids:
                    self._issue(issues, "detector", IssueCode.DUPLICATE_DETECTION_ID, "detection_id %d is used more than once in the frame" % d.detection_id,
                                frame.frame_index)
                    continue
                cat = self.config.label_map.resolve(d.label)
                if cat is None:
                    self._issue(issues, "detector", IssueCode.UNSUPPORTED_CLASS, "label %r is not a supported class" % (d.label,), frame.frame_index,
                                detail=str(d.label))
                    continue
                box = tuple(float(v) for v in d.bbox_xyxy)
                use, clipped = self._check_box_in_frame(box, frame)
                if use is None:
                    self._issue(issues, "detector", IssueCode.BBOX_OUT_OF_FRAME, "box %r leaves the %dx%d frame" % (box, frame.width, frame.height),
                                frame.frame_index)
                    continue
                dets.append(DetectionRecord(frame.sequence_id, frame.frame_index, frame.timestamp_s, frame.width, frame.height, frame.coordinate_system,
                                            d.detection_id, d.label, cat, use, float(d.confidence), clipped, box, det_info, d.class_id))
            by_id = {d.detection_id: d for d in dets}
            with (rec.stage("track") if rec else _null()):
                try:
                    obs = list(self.tracker.update(frame, tuple(dets)))
                except Exception as exc:                                  # noqa: BLE001
                    self._issue(issues, "tracker", IssueCode.TRACKER_ERROR, "tracker raised %s: %s" % (type(exc).__name__, exc), frame.frame_index)
                    return FrameResult(frame, tuple(dets), (), tuple(issues), False)
            counts: Dict[int, int] = {}
            for o in obs:
                if track_problem(o) is None:
                    counts[o.track_id] = counts.get(o.track_id, 0) + 1
            tracks: List[TrackRecord] = []
            for o in obs:
                p = track_problem(o)
                if p is not None:
                    self._issue(issues, "tracker", p[0], p[1], frame.frame_index)
                    continue
                if counts[o.track_id] > 1:
                    self._issue(issues, "tracker", IssueCode.DUPLICATE_TRACK_ID, "track_id %d appears more than once in the frame" % o.track_id, frame.frame_index)
                    continue
                prev = track_class.get(o.track_id)
                if prev is not None and prev != o.category:
                    self._issue(issues, "tracker", IssueCode.TRACK_CLASS_CHANGED, "track %d changed class from %s to %s" % (o.track_id, prev.value, o.category.value),
                                frame.frame_index)
                    continue
                conf = None
                if not o.is_predicted:
                    d = by_id.get(o.matched_detection_id)
                    if d is None:
                        self._issue(issues, "tracker", IssueCode.UNKNOWN_MATCHED_DETECTION,
                                    "track %d names detection %r which is not a valid detection of this frame" % (o.track_id, o.matched_detection_id), frame.frame_index)
                        continue
                    conf = d.confidence
                track_class.setdefault(o.track_id, o.category)
                tracks.append(TrackRecord(frame.sequence_id, frame.frame_index, frame.timestamp_s, frame.width, frame.height, frame.coordinate_system,
                                          o.track_id, o.category, tuple(float(v) for v in o.bbox_xyxy), conf, o.matched_detection_id, o.is_predicted,
                                          o.tracking_confidence, det_info, trk_info))
        return FrameResult(frame, tuple(dets), tuple(tracks), tuple(issues), True)

    def run(self) -> PerceptionResult:
        frames = list(self.iter_results())
        issues = [i for f in frames for i in f.issues] + list(self._orphan_issues)
        return PerceptionResult(frames, issues, self.source.info, self.detector.info, self.tracker.info)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
