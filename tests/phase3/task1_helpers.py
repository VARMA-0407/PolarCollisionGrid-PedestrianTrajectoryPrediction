"""Shared helpers for the Phase 3 Task 1 tests (no model weights, no downloads, no datasets)."""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from pedsafety.perception import ComponentInfo, DetectionRecord, FrameMeta, PerceptionClass

CI = ComponentInfo("detector", "unit-test-detector", "1", True)


def frame_meta(i: int = 0, w: int = 640, h: int = 480, fps: float = 10.0, seq: str = "seq") -> FrameMeta:
    return FrameMeta("src", seq, i, i / fps, w, h, "captured_%dx%d" % (w, h), "frame_index_over_fps", fps)


def det(i: int, box: Tuple[float, float, float, float], conf: float = 0.9, cat: PerceptionClass = PerceptionClass.PEDESTRIAN,
        label: str = "person", class_id: Optional[int] = 0, frame_index: int = 0) -> DetectionRecord:
    return DetectionRecord("seq", frame_index, frame_index / 10.0, 640, 480, "captured_640x480", i, label, cat, box, conf, False, box, CI, class_id)


def write_test_video(path: str, width: int = 64, height: int = 48, fps: float = 10.0, n: int = 12, fourcc: str = "MJPG") -> None:
    """Tiny locally generated video (MJPG in AVI by default: decodes everywhere, no external codec needed)."""
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (width, height))
    assert w.isOpened(), "test video writer could not be opened"
    for k in range(n):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, 1] = (k * 17) % 256
        cv2.rectangle(img, (k, 5), (k + 10, 25), (255, 255, 255), -1)
        w.write(img)
    w.release()


class _T:
    """Stands in for a torch tensor: exposes ``.cpu().numpy()``."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


class FakeBoxes:
    def __init__(self, xyxy: Sequence, conf: Sequence, cls: Sequence):
        self.xyxy, self.conf, self.cls = _T(np.asarray(xyxy).reshape(-1, 4)), _T(conf), _T(cls)

    def __len__(self):
        return len(self.conf)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYoloModel:
    """Fake of an Ultralytics model: records ``predict`` kwargs and returns canned boxes."""

    def __init__(self, xyxy=(), conf=(), cls=(), names=None):
        self.names = names if names is not None else {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.calls: List[dict] = []
        self._payload = (xyxy, conf, cls)

    def predict(self, image, **kw):
        self.calls.append(dict(kw, _shape=image.shape))
        xyxy, conf, cls = self._payload
        return [FakeResult(FakeBoxes(xyxy, conf, cls) if len(conf) else None)]
