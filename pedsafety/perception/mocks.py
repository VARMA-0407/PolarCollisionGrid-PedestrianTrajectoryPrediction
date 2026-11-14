"""MOCK perception components for tests (Stage 2A).

Everything here is a test double. Each component reports ``ComponentInfo.is_mock=True`` and a name that starts
with ``MOCK``. Their outputs are scripted or computed by a trivial rule; they are NOT a detector, NOT ByteTrack, and
their outputs must never be reported as detection, tracking or performance results. Downstream code enforces this:
the agent-state adapter requires a mock-labelled ``source_dataset`` for mock output, and ``PerformanceSummary`` /
``EvaluationReport`` refuse mock components.
"""

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .detector import Detector
from .sources import FrameSource
from .tracker import Tracker
from .types import ComponentInfo, DetectionRecord, FrameMeta, PerceptionClass, RawDetection, TrackObservation


class ScriptedFrameSource(FrameSource):
    """Yields exactly the (FrameMeta, payload) pairs it was given, in that order (including deliberately bad ones)."""

    def __init__(self, frames: Sequence[Tuple[FrameMeta, Any]], name: str = "MOCK_ScriptedFrameSource"):
        self._frames = list(frames)
        self._info = ComponentInfo("frame_source", name, "mock-1", True)

    @property
    def info(self) -> ComponentInfo:
        return self._info

    def frames(self) -> Iterator[Tuple[FrameMeta, Any]]:
        for item in self._frames:
            yield item


def synthetic_frames(n_frames: int, *, fps: float = 10.0, width: int = 640, height: int = 360, sequence_id: str = "mock_seq",
                     start_frame: int = 0) -> List[Tuple[FrameMeta, Any]]:
    """Deterministic frame metadata only (no pixels): timestamp = frame_index / fps on a MOCK clock."""
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 0:
        raise ValueError("n_frames must be an integer >= 0")
    return [(FrameMeta("mock_source", sequence_id, start_frame + i, (start_frame + i) / fps, width, height, "mock_raster_%dx%d" % (width, height),
                       "mock_clock", fps), None) for i in range(n_frames)]


class MockDetector(Detector):
    """Returns the detections scripted for each frame index (none for unlisted frames)."""

    def __init__(self, script: Mapping[int, Sequence[RawDetection]], *, fail_on_frame: Optional[int] = None, name: str = "MOCK_Detector"):
        self._script = {int(k): tuple(v) for k, v in script.items()}
        self._fail = fail_on_frame
        self._info = ComponentInfo("detector", name, "mock-1", True)
        self.calls: List[int] = []

    @property
    def info(self) -> ComponentInfo:
        return self._info

    def detect(self, frame: FrameMeta, payload: Any) -> Sequence[RawDetection]:
        self.calls.append(frame.frame_index)
        if self._fail is not None and frame.frame_index == self._fail:
            raise RuntimeError("scripted mock detector failure at frame %d" % frame.frame_index)
        return self._script.get(frame.frame_index, ())


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class MockTracker(Tracker):
    """Greedy IoU matcher against each live track's LAST box. Deterministic. NOT ByteTrack.

    A track unmatched for more than ``max_age`` consecutive updates is dropped; a later object gets a NEW id. With
    ``emit_predicted=True`` unmatched live tracks are emitted as predicted-only rows (their last box), which exercises the
    adapter's refusal to treat estimates as observations."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 2, emit_predicted: bool = False, name: str = "MOCK_IoUTracker"):
        self.iou_threshold, self.max_age, self.emit_predicted = iou_threshold, max_age, emit_predicted
        self._info = ComponentInfo("tracker", name, "mock-1", True, (("iou_threshold", str(iou_threshold)), ("max_age", str(max_age))))
        self.reset()

    @property
    def info(self) -> ComponentInfo:
        return self._info

    def reset(self) -> None:
        self._next_id = 1
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, frame: FrameMeta, detections: Sequence[DetectionRecord]) -> Sequence[TrackObservation]:
        pairs = []
        for tid, t in self._tracks.items():
            for d in detections:
                if d.category == t["category"]:
                    v = iou(t["box"], d.bbox_xyxy)
                    if v >= self.iou_threshold:
                        pairs.append((-v, tid, d.detection_id))
        pairs.sort()
        used_t, used_d, matched = set(), set(), {}
        for _, tid, did in pairs:
            if tid in used_t or did in used_d:
                continue
            used_t.add(tid)
            used_d.add(did)
            matched[tid] = did
        by_id = {d.detection_id: d for d in detections}
        out: List[TrackObservation] = []
        for tid in sorted(self._tracks):
            t = self._tracks[tid]
            if tid in matched:
                d = by_id[matched[tid]]
                t["box"], t["missed"] = d.bbox_xyxy, 0
                out.append(TrackObservation(tid, d.category, d.bbox_xyxy, d.detection_id))
            else:
                t["missed"] += 1
                if self.emit_predicted and t["missed"] <= self.max_age:
                    out.append(TrackObservation(tid, t["category"], t["box"], None, is_predicted=True))
        for tid in [k for k, t in self._tracks.items() if t["missed"] > self.max_age]:
            del self._tracks[tid]
        for d in sorted(detections, key=lambda x: x.detection_id):
            if d.detection_id in used_d:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {"box": d.bbox_xyxy, "category": d.category, "missed": 0}
            out.append(TrackObservation(tid, d.category, d.bbox_xyxy, d.detection_id))
        return sorted(out, key=lambda o: o.track_id)
