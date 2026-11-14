"""OpenCV video ingestion (Phase 3, Task 1A): a real ``FrameSource`` behind the Stage 2A interface.

* Frames are decoded sequentially, one at a time; the video is never loaded into memory.
* ``frame_index`` counts frames actually delivered (0, 1, 2, ... - strictly increasing). ``timestamp_s`` is
  ``frame_index / fps`` and is labelled ``frame_index_over_fps``: it is exact for constant-frame-rate video and only an
  approximation for variable-frame-rate video (no container PTS is read). The source frame rate is never resampled.
* ``coordinate_system`` names the raster actually decoded (``captured_<W>x<H>``); frames are NOT resized here.
* A failed read ends the stream (OpenCV cannot tell end-of-file from a corrupt frame); the number of frames delivered
  is compared with the container's advisory frame count and the shortfall is exposed in ``stats``.
* The capture is released on exhaustion, on ``close()``, when the generator is closed and on context exit.
* Per-frame decode time is recorded in ``decode_latencies_s`` (monotonic clock) for the performance report.

Requires ``opencv-python`` or ``opencv-python-headless``; imported lazily so importing this module never needs it.
"""

import math
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .runtime_errors import VideoSourceError
from .sources import FrameSource
from .types import ComponentInfo, FrameMeta

VIDEO_SOURCE_VERSION = "opencv-video-source-0.1.0-task1"


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise VideoSourceError("OpenCV is required for video ingestion (pip install opencv-python-headless): %s" % exc)
    return cv2


class OpenCVVideoSource(FrameSource):
    def __init__(self, path: str, *, sequence_id: Optional[str] = None, fps_override: Optional[float] = None,
                 max_frames: Optional[int] = None):
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
            raise VideoSourceError("max_frames must be a positive integer or None, got %r" % (max_frames,))
        if fps_override is not None and not (isinstance(fps_override, (int, float)) and math.isfinite(fps_override) and fps_override > 0):
            raise VideoSourceError("fps_override must be a positive finite number or None, got %r" % (fps_override,))
        p = os.fspath(path)
        if not os.path.isfile(p):
            raise VideoSourceError("video file not found: %s" % p)
        cv2 = _cv2()
        self.released = False
        self._cap = None
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            cap.release()
            raise VideoSourceError("OpenCV could not open the video (unsupported codec or corrupt file): %s" % p)
        self._cv2 = cv2
        self._cap = cap
        self._path = p
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w < 1 or h < 1:
            self.close()
            raise VideoSourceError("video reports an invalid frame size %dx%d: %s" % (w, h, p))
        reported = float(cap.get(cv2.CAP_PROP_FPS))
        if fps_override is not None:
            fps = float(fps_override)
            self._fps_origin = "config_override"
        elif math.isfinite(reported) and reported > 0:
            fps, self._fps_origin = reported, "container"
        else:
            self.close()
            raise VideoSourceError("video reports no usable FPS (%r); set video.fps_override explicitly (frames are never resampled): %s"
                                   % (reported, p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width, self.height, self.fps = w, h, fps
        self.reported_fps = reported if math.isfinite(reported) else None
        self.frame_count = n if n > 0 else None      # advisory only; many containers report 0 or a wrong count
        self.max_frames = max_frames
        self.sequence_id = sequence_id or os.path.splitext(os.path.basename(p))[0]
        self.coordinate_system = "captured_%dx%d" % (w, h)
        self.decode_latencies_s = []                 # type: List[float]
        self.frames_delivered = 0
        self._started = False

    # -------------------------------------------------------------- interface
    @property
    def info(self) -> ComponentInfo:
        return ComponentInfo("frame_source", "opencv-video-source", "%s+opencv-%s" % (VIDEO_SOURCE_VERSION, self._cv2.__version__), False, (
            ("file", os.path.basename(self._path)), ("width", str(self.width)), ("height", str(self.height)), ("fps", repr(self.fps)),
            ("fps_origin", self._fps_origin), ("timestamp_source", "frame_index_over_fps"),
            ("advisory_frame_count", str(self.frame_count)), ("max_frames", str(self.max_frames))))

    @property
    def stats(self) -> Dict[str, Any]:
        short = None
        if self.frame_count is not None and self.max_frames is None:
            short = self.frame_count - self.frames_delivered
        return {"frames_delivered": self.frames_delivered, "advisory_frame_count": self.frame_count,
                "frames_missing_vs_advisory_count": short, "released": self.released}

    def frames(self) -> Iterator[Tuple[FrameMeta, Any]]:
        if self._started:
            raise VideoSourceError("this video source is one-shot and was already consumed; create a new OpenCVVideoSource")
        self._started = True
        try:
            index = 0
            while self.max_frames is None or index < self.max_frames:
                if self.released:
                    return
                t0 = time.perf_counter()
                ok, image = self._cap.read()
                dt = time.perf_counter() - t0
                if not ok or image is None:
                    return                                   # the failed final read is not a decoded frame
                self.decode_latencies_s.append(dt)
                h, w = image.shape[:2]
                if (w, h) != (self.width, self.height):
                    raise VideoSourceError("frame %d is %dx%d but the video was opened as %dx%d; refusing to mix rasters"
                                           % (index, w, h, self.width, self.height))
                meta = FrameMeta(self._path, self.sequence_id, index, index / self.fps, w, h, self.coordinate_system,
                                 "frame_index_over_fps", self.fps)
                self.frames_delivered += 1
                yield meta, image
                index += 1
        finally:
            self.close()

    # -------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if not self.released:
            if self._cap is not None:
                self._cap.release()
            self.released = True

    def __enter__(self) -> "OpenCVVideoSource":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:                                    # noqa: BLE001 - interpreter shutdown
            pass
