"""Locally generated synthetic test video and a replay detector (Phase 3, Task 1; PLUMBING VALIDATION ONLY).

``generate_synthetic_video`` draws moving coloured rectangles (pedestrian-, car-, truck-, bus- and bike-shaped) on a grey
background and writes the video plus a ground-truth JSON. The scene contains: two pedestrians crossing paths (box overlap),
a detector dropout of one pedestrian (temporary occlusion), a low-confidence stretch, a car entering through the frame edge,
and objects that appear and disappear. ``ReplayDetector`` replays the ground-truth boxes as detections.

NOTHING HERE MEASURES DETECTION QUALITY. Rectangles are not road users, YOLO is not run on them, and ``ReplayDetector`` is a
mock (``is_mock=True``); it only lets the real video source, the real ByteTrack wrapper, the pipeline, the exporter and the
visualiser be exercised together without any model weights or downloaded data.
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from .detector import Detector
from .runtime_errors import VideoSourceError
from .types import ComponentInfo, FrameMeta, RawDetection

SYNTHETIC_VERSION = "synthetic-scene-0.1.0-task1"
_COCO = {"person": 0, "bicycle": 1, "car": 2, "bus": 5, "truck": 7}
_COLOUR = {"person": (40, 40, 220), "bicycle": (40, 200, 40), "car": (220, 120, 30), "truck": (30, 200, 220), "bus": (200, 40, 200)}

# (gt_id, label, w, h, x0, y0, vx, vy, first_frame, last_frame)
_OBJECTS = (
    (1, "person", 20, 50, 40, 200, 4.0, 0.0, 0, 119),
    (2, "person", 20, 50, 600, 210, -4.0, 0.0, 0, 119),
    (3, "person", 20, 50, 5, 120, 1.5, 1.0, 30, 100),
    (4, "car", 80, 40, -60, 290, 6.0, 0.0, 0, 119),
    (5, "truck", 110, 50, 500, 60, -3.0, 0.0, 10, 119),
    (6, "bicycle", 30, 40, 300, 330, 2.0, -1.0, 40, 119),
)
_DROPOUT = {3: range(50, 56)}                 # gt 3 undetected for these frames (temporary occlusion / missed detection)
_LOW_CONF = {1: range(20, 26)}                # gt 1 detected with a low score here


def build_ground_truth(width: int = 640, height: int = 360, fps: float = 10.0, n_frames: int = 120) -> Dict[str, Any]:
    """Pure-Python scene description; no OpenCV needed."""
    frames: List[Dict[str, Any]] = []
    for k in range(n_frames):
        objs = []
        for gid, label, w, h, x0, y0, vx, vy, f0, f1 in _OBJECTS:
            if not f0 <= k <= f1:
                continue
            cx, cy = x0 + vx * (k - f0) + w / 2.0, y0 + vy * (k - f0) + h / 2.0
            bx0, by0, bx1, by1 = max(0.0, cx - w / 2.0), max(0.0, cy - h / 2.0), min(float(width), cx + w / 2.0), min(float(height), cy + h / 2.0)
            if bx1 - bx0 < 4.0 or by1 - by0 < 4.0:
                continue                                           # essentially outside the frame
            visible = k not in _DROPOUT.get(gid, ())
            conf = 0.2 if k in _LOW_CONF.get(gid, ()) else 0.9
            objs.append({"gt_id": gid, "label": label, "class_id": _COCO[label], "bbox_xyxy": [bx0, by0, bx1, by1], "visible": visible,
                         "confidence": conf})
        frames.append({"frame_index": k, "objects": objs})
    return {"version": SYNTHETIC_VERSION, "width": width, "height": height, "fps": fps, "n_frames": n_frames, "frames": frames,
            "note": "synthetic rectangles; plumbing validation only, not a detection-quality dataset"}


def generate_synthetic_video(path: str, *, width: int = 640, height: int = 360, fps: float = 10.0, n_frames: int = 120,
                             fourcc: str = "mp4v") -> Dict[str, Any]:
    """Write the video to ``path`` and return the ground truth (the caller may save it with ``save_ground_truth``)."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise VideoSourceError("OpenCV and numpy are required to generate the synthetic video: %s" % exc)
    gt = build_ground_truth(width, height, fps, n_frames)
    writer = cv2.VideoWriter(os.fspath(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (width, height))
    if not writer.isOpened():
        writer.release()
        raise VideoSourceError("could not open a video writer at %s (fourcc %r)" % (path, fourcc))
    try:
        for fr in gt["frames"]:
            img = np.full((height, width, 3), 110, dtype=np.uint8)
            cv2.line(img, (0, height - 20), (width, height - 20), (90, 90, 90), 1)
            for o in fr["objects"]:
                x0, y0, x1, y1 = (int(round(v)) for v in o["bbox_xyxy"])
                cv2.rectangle(img, (x0, y0), (x1, y1), _COLOUR[o["label"]], -1)
                cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), 1)
            writer.write(img)
    finally:
        writer.release()
    return gt


def save_ground_truth(gt: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gt, fh)


def load_ground_truth(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class ReplayDetector(Detector):
    """MOCK detector: returns the ground-truth boxes of the synthetic scene for each frame index (visible objects only)."""

    def __init__(self, gt: Dict[str, Any]):
        self._by_frame = {f["frame_index"]: f["objects"] for f in gt["frames"]}

    @property
    def info(self) -> ComponentInfo:
        return ComponentInfo("detector", "synthetic-replay-detector", SYNTHETIC_VERSION, True, (("source", "ground_truth_replay"),))

    def detect(self, frame: FrameMeta, payload: Any) -> Sequence[RawDetection]:
        out = []
        for o in self._by_frame.get(frame.frame_index, ()):
            if o["visible"]:
                out.append(RawDetection(len(out), o["label"], tuple(o["bbox_xyxy"]), o["confidence"], o["class_id"]))
        return out
