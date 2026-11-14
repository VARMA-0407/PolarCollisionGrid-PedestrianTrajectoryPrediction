"""ByteTrack tracker wrapper (Phase 3, Task 1C) behind the Stage 2A ``Tracker`` interface.

The association algorithm is the real ByteTrack: ``ultralytics.trackers.byte_tracker.BYTETracker`` (Ultralytics'
implementation of Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022: Kalman
filter, IoU association of high-score detections, a second association of low-score detections with unmatched tracks, a
lost-track buffer). Nothing here re-implements association; this module only converts our validated detections to the
array form that class expects and converts its output back to ``TrackObservation``.

Conversion rules:
* Class is carried through as a numeric index into ``PerceptionClass``; the output row's detection index (``idx``) maps back
  to the matched ``DetectionRecord``, so ``matched_detection_id`` and the semantic category come from the detection itself.
* ByteTrack reports only tracks that were matched in the current frame (lost tracks are kept internally for re-association and
  not emitted), so ``is_predicted`` is always False; a track reappearing after an occlusion keeps its id if it was re-matched
  within ``track_buffer`` frames.
* ByteTrack does not enforce class consistency. ``class_policy='detection'`` reports the matched detection's class (the
  pipeline then flags and drops a class change); ``'first_seen'`` pins each track to its first class and counts the flips.
* ``box_source='detection'`` reports the matched detection's box (an observation, not a filtered estimate); ``'kalman'`` reports
  ByteTrack's filtered box.
* ``track_buffer`` is counted in FRAMES, not seconds.

The tracker is causal and keeps no state beyond the current and earlier frames.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

from .config import TrackerSettings
from .runtime_errors import TrackerInitError
from .tracker import Tracker
from .types import ComponentInfo, DetectionRecord, FrameMeta, PerceptionClass, TrackObservation

BYTETRACK_WRAPPER_VERSION = "bytetrack-wrapper-0.1.0-task1"
_CLASS_ORDER = tuple(PerceptionClass)


class _DetBatch:
    """Minimal results-like object exposing ``conf``, ``cls`` and ``xywh`` (centre x, centre y, w, h) with boolean indexing,
    which is all ``BYTETracker.update`` reads."""

    def __init__(self, conf, cls, xywh):
        self.conf, self.cls, self.xywh = conf, cls, xywh

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_DetBatch":
        return _DetBatch(self.conf[mask], self.cls[mask], self.xywh[mask])


class ByteTrackTracker(Tracker):
    def __init__(self, settings: TrackerSettings):
        try:
            import numpy as np
            import ultralytics
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as exc:
            raise TrackerInitError("ByteTrack needs the 'ultralytics' package (and its 'lap'/'lapx' dependency): %s" % exc)
        self._np = np
        self.settings = settings
        self._ultralytics_version = ultralytics.__version__
        args = SimpleNamespace(track_high_thresh=settings.track_high_thresh, track_low_thresh=settings.track_low_thresh,
                               new_track_thresh=settings.new_track_thresh, track_buffer=settings.track_buffer,
                               match_thresh=settings.match_thresh, fuse_score=settings.fuse_score)
        try:
            self._tracker = BYTETracker(args)
        except Exception as exc:                                    # noqa: BLE001
            raise TrackerInitError("BYTETracker rejected the configuration: %s: %s" % (type(exc).__name__, exc))
        self._first_class: Dict[int, PerceptionClass] = {}
        self.class_flips = 0
        self.frames_updated = 0

    @property
    def info(self) -> ComponentInfo:
        s = self.settings
        return ComponentInfo("tracker", "bytetrack", "%s+ultralytics-%s" % (BYTETRACK_WRAPPER_VERSION, self._ultralytics_version), False, (
            ("implementation", "ultralytics.trackers.byte_tracker.BYTETracker"), ("track_high_thresh", repr(s.track_high_thresh)),
            ("track_low_thresh", repr(s.track_low_thresh)), ("new_track_thresh", repr(s.new_track_thresh)),
            ("track_buffer_frames", str(s.track_buffer)), ("match_thresh", repr(s.match_thresh)), ("fuse_score", str(s.fuse_score)),
            ("box_source", s.box_source), ("class_policy", s.class_policy)))

    def reset(self) -> None:
        self._tracker.reset()
        self._first_class.clear()
        self.class_flips = 0
        self.frames_updated = 0

    def update(self, frame: FrameMeta, detections: Sequence[DetectionRecord]) -> Sequence[TrackObservation]:
        np = self._np
        n = len(detections)
        if n:
            boxes = np.asarray([d.bbox_xyxy for d in detections], dtype=np.float32).reshape(n, 4)
            xywh = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0,
                             boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1).astype(np.float32)
            conf = np.asarray([d.confidence for d in detections], dtype=np.float32)
            cls = np.asarray([_CLASS_ORDER.index(d.category) for d in detections], dtype=np.float32)
        else:
            xywh, conf, cls = np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.float32)
        rows = self._tracker.update(_DetBatch(conf, cls, xywh))     # always called, also for empty frames, so tracks age
        self.frames_updated += 1
        out: List[TrackObservation] = []
        for row in np.asarray(rows).reshape(-1, 8) if np.asarray(rows).size else []:
            tid, idx = int(row[4]), int(row[7])
            if not 0 <= idx < n:
                continue                                             # cannot be matched to a detection of this frame; pipeline never sees it
            det = detections[idx]
            cat = det.category
            first = self._first_class.setdefault(tid, cat)
            if cat != first:
                self.class_flips += 1
                if self.settings.class_policy == "first_seen":
                    cat = first
            box = det.bbox_xyxy if self.settings.box_source == "detection" else tuple(float(v) for v in row[:4])
            out.append(TrackObservation(tid, cat, box, det.detection_id, False, None))
        return out
