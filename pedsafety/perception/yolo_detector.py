"""Ultralytics YOLO detector wrapper (Phase 3, Task 1B) behind the Stage 2A ``Detector`` interface.

Responsibilities (and only these): load a LOCAL weights file, run inference on one BGR frame, and return
``RawDetection`` objects whose ``label`` is the model's original class name and ``class_id`` its original class index.
Semantic mapping (``person`` -> ``pedestrian`` ...) is NOT done here; it belongs to the configurable ``LabelMap`` applied by
the pipeline. Video decoding and tracking are not touched.

Safety rules:
* Weights are never downloaded: the path must already exist (Ultralytics would otherwise fetch a bare name such as
  ``yolo11n.pt`` from the internet). The Colab notebook downloads and checksums weights explicitly.
* ``device='cuda'`` raises if CUDA is unavailable; ``'auto'`` picks CUDA when present, else CPU. Nothing falls back silently.
* A detector built around an injected ``model`` object is marked ``is_mock=True`` so its numbers can never back a claim.
"""

import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence

from .config import DetectorSettings
from .detector import Detector
from .runtime_errors import DetectorLoadError
from .types import ComponentInfo, FrameMeta, RawDetection

YOLO_WRAPPER_VERSION = "ultralytics-yolo-wrapper-0.1.0-task1"


def resolve_device(requested: str, cuda_available: Optional[bool] = None) -> str:
    """Map 'auto' / 'cpu' / 'cuda' to an Ultralytics device string. Pure and testable via ``cuda_available``."""
    if requested not in ("auto", "cpu", "cuda"):
        raise DetectorLoadError("device must be 'auto', 'cpu' or 'cuda', got %r" % (requested,))
    if requested == "cpu":
        return "cpu"
    if cuda_available is None:
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except ImportError:
            cuda_available = False
    if cuda_available:
        return "cuda:0"
    if requested == "cuda":
        raise DetectorLoadError("device 'cuda' was requested but CUDA is not available (use 'auto' or 'cpu')")
    return "cpu"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class UltralyticsDetector(Detector):
    def __init__(self, settings: DetectorSettings, *, model: Any = None, cuda_available: Optional[bool] = None):
        self.settings = settings
        self.device = resolve_device(settings.device, cuda_available)
        self._injected = model is not None
        if self._injected:
            self._model = model
            self._weights_name, self._weights_sha = "<injected-model>", "n/a"
        else:
            path = os.fspath(settings.model_path)
            if not os.path.isfile(path):
                raise DetectorLoadError("weights file not found: %s (weights are never downloaded automatically; place the file there "
                                        "or set detector.model_path)" % path)
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise DetectorLoadError("the 'ultralytics' package is required for the YOLO detector: %s" % exc)
            try:
                self._model = YOLO(path)
            except Exception as exc:                                    # noqa: BLE001 - surface any loader failure uniformly
                raise DetectorLoadError("could not load weights %s: %s: %s" % (path, type(exc).__name__, exc))
            self._weights_name, self._weights_sha = os.path.basename(path), file_sha256(path)
        names = getattr(self._model, "names", None)
        if not isinstance(names, dict) or not names:
            raise DetectorLoadError("the model exposes no class-name table (model.names); cannot report original class names")
        self.names: Dict[int, str] = {int(k): str(v) for k, v in names.items()}
        if settings.coco_class_ids is not None:
            bad = [i for i in settings.coco_class_ids if i not in self.names]
            if bad:
                raise DetectorLoadError("detector.coco_class_ids %s are not class ids of this model (ids 0..%d)" % (bad, max(self.names)))
        self._info: Optional[ComponentInfo] = None

    @property
    def info(self) -> ComponentInfo:
        if self._info is None:
            s = self.settings
            try:
                import ultralytics
                ver = ultralytics.__version__
            except ImportError:
                ver = "unavailable"
            params = (("weights_file", self._weights_name), ("weights_sha256", self._weights_sha), ("device", self.device), ("conf", repr(s.conf)),
                      ("iou", repr(s.iou)), ("imgsz", str(s.imgsz)), ("max_det", str(s.max_det)), ("half", str(s.half and self.device != "cpu")),
                      ("class_ids", "all" if s.coco_class_ids is None else ",".join(str(i) for i in s.coco_class_ids)),
                      ("model_classes", str(len(self.names))))
            self._info = ComponentInfo("detector", "ultralytics-yolo", "%s+ultralytics-%s" % (YOLO_WRAPPER_VERSION, ver), self._injected, params)
        return self._info

    def detect(self, frame: FrameMeta, payload: Any) -> Sequence[RawDetection]:
        shape = getattr(payload, "shape", None)
        if shape is None or len(shape) != 3 or shape[2] != 3 or shape[0] != frame.height or shape[1] != frame.width:
            raise ValueError("payload must be a %dx%d BGR image array (HxWx3), got shape %r" % (frame.height, frame.width, shape))
        s = self.settings
        kwargs: Dict[str, Any] = dict(conf=s.conf, iou=s.iou, imgsz=s.imgsz, device=self.device, max_det=s.max_det, verbose=False)
        if s.half and self.device != "cpu":
            kwargs["half"] = True                                       # only passed when requested: recent Ultralytics deprecates the argument
        if s.coco_class_ids is not None:
            kwargs["classes"] = list(s.coco_class_ids)
        results = self._model.predict(payload, **kwargs)
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        out: List[RawDetection] = []
        for i in range(len(xyxy)):
            cid = int(cls[i])
            out.append(RawDetection(i, self.names.get(cid, "class_%d" % cid), tuple(float(v) for v in xyxy[i]), float(conf[i]), cid))
        return out

    def warmup(self, width: int, height: int, n: int = 2) -> None:
        """Run ``n`` dummy inferences so one-time initialisation (CUDA context, cuDNN autotune) is excluded from timings."""
        import numpy as np
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        meta = FrameMeta("warmup", "warmup", 0, 0.0, width, height, "warmup", "mock_clock")
        for _ in range(n):
            self.detect(meta, dummy)
