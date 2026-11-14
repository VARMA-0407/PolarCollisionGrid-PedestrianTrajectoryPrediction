"""Annotated-video visualisation (Phase 3, Task 1E). Needs OpenCV (imported lazily).

Draws, per frame: tracked boxes with ``ID <track_id> <semantic class> <confidence>`` (one stable colour per track id, so an ID
switch is visible as a colour change), thin grey boxes for detections that were NOT turned into a track (false or low-score
detections, rejected tracks), and a header with the frame index and timestamp. Boxes are image pixels; nothing is projected.
Frames are written incrementally, so an interrupted run still leaves a playable file.
"""

import os
from typing import Any, Dict, Optional, Tuple

from .pipeline import FrameResult
from .runtime_errors import VideoSourceError

_PALETTE = ((230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
            (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40), (128, 0, 0), (170, 255, 195),
            (128, 128, 0), (0, 0, 128))


def track_colour(track_id: int) -> Tuple[int, int, int]:
    """Stable BGR colour for a track id."""
    r, g, b = _PALETTE[track_id % len(_PALETTE)]
    return (b, g, r)


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise VideoSourceError("OpenCV is required for visualisation: %s" % exc)
    return cv2


def draw_frame_result(image: Any, result: FrameResult, *, draw_untracked_detections: bool = True) -> Any:
    """Return an annotated COPY of ``image`` (BGR ndarray); the input is not modified."""
    cv2 = _cv2()
    out = image.copy()
    f = result.frame
    tracked_det_ids = {t.matched_detection_id for t in result.tracks if t.matched_detection_id is not None}
    if draw_untracked_detections:
        for d in result.detections:
            if d.detection_id in tracked_det_ids:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in d.bbox_xyxy)
            cv2.rectangle(out, (x0, y0), (x1, y1), (160, 160, 160), 1)
            cv2.putText(out, "det %s %.2f" % (d.category.value, d.confidence), (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (160, 160, 160), 1, cv2.LINE_AA)
    conf_by_det = {d.detection_id: d.confidence for d in result.detections}
    for t in result.tracks:
        x0, y0, x1, y1 = (int(round(v)) for v in t.bbox_xyxy)
        col = track_colour(t.track_id)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA if not t.is_predicted else cv2.LINE_4)
        conf = t.detection_confidence if t.detection_confidence is not None else conf_by_det.get(t.matched_detection_id)
        label = "ID %d %s%s" % (t.track_id, t.category.value, "" if conf is None else " %.2f" % conf)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 2, y0)
        cv2.rectangle(out, (x0, ty - th - base), (x0 + tw + 2, ty + base - 2), col, -1)
        cv2.putText(out, label, (x0 + 1, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    header = "frame %d  t=%.3fs  tracks=%d  dets=%d%s" % (f.frame_index, f.timestamp_s, len(result.tracks), len(result.detections),
                                                          "" if result.processed else "  [SKIPPED]")
    cv2.rectangle(out, (0, 0), (min(f.width, 8 * len(header) + 12), 18), (0, 0, 0), -1)
    cv2.putText(out, header, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class AnnotatedVideoWriter:
    """Incremental video writer (``mp4v`` in an ``.mp4`` container by default). Use as a context manager."""

    def __init__(self, path: str, fps: float, width: int, height: int, fourcc: str = "mp4v"):
        cv2 = _cv2()
        self.path = os.fspath(path)
        self.width, self.height = int(width), int(height)
        self.frames_written = 0
        self._writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*fourcc), float(fps), (self.width, self.height))
        if not self._writer.isOpened():
            self._writer.release()
            raise VideoSourceError("could not open an annotated-video writer at %s (fourcc %r)" % (self.path, fourcc))
        self.closed = False

    def write(self, image: Any) -> None:
        if self.closed:
            raise VideoSourceError("annotated-video writer is closed")
        h, w = image.shape[:2]
        if (w, h) != (self.width, self.height):
            raise VideoSourceError("frame %dx%d does not match the writer size %dx%d" % (w, h, self.width, self.height))
        self._writer.write(image)
        self.frames_written += 1

    def close(self) -> None:
        if not self.closed:
            self._writer.release()
            self.closed = True

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
