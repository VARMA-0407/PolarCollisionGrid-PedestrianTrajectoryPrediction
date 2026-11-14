"""Central, validated configuration of the perception frontend (Phase 3, Task 1).

Stdlib only (no OpenCV / torch / ultralytics import), JSON on disk. Every threshold, model path and label mapping lives
here or in the JSON file; nothing is hardcoded in the wrappers. Unknown keys are rejected so a typo cannot silently fall
back to a default.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..errors import ConfigError
from .pipeline import PipelineConfig
from .types import LabelMap

CONFIG_VERSION = "perception-config-0.1.0-task1"

# COCO class ids of the five supported semantic classes' detector sources. motorcycle (3) is deliberately absent:
# a motorcycle is not a bicycle, and the semantic mapping is explicit and configurable.
DEFAULT_COCO_CLASS_IDS = (0, 1, 2, 5, 7)
DEFAULT_LABEL_MAP = {"person": "pedestrian", "bicycle": "bike", "car": "car", "bus": "bus", "truck": "truck"}
_DEVICES = ("auto", "cpu", "cuda")


def _num(name: str, v: Any, lo: float, hi: float, lo_open: bool = False) -> float:
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if not ok or (v <= lo if lo_open else v < lo) or v > hi:
        raise ConfigError("%s must be a number in %s%s, %s], got %r" % (name, "(" if lo_open else "[", lo, hi, v))
    return float(v)


def _int(name: str, v: Any, lo: int, hi: Optional[int] = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
        raise ConfigError("%s must be an integer >= %d%s, got %r" % (name, lo, "" if hi is None else " and <= %d" % hi, v))
    return v


def _bool(name: str, v: Any) -> bool:
    if not isinstance(v, bool):
        raise ConfigError("%s must be true or false, got %r" % (name, v))
    return v


def _reject_unknown(section: str, given: Mapping[str, Any], allowed: Tuple[str, ...]) -> None:
    extra = sorted(set(given) - set(allowed))
    if extra:
        raise ConfigError("unknown key(s) in %r: %s (allowed: %s)" % (section, ", ".join(extra), ", ".join(allowed)))


@dataclass(frozen=True)
class VideoSettings:
    """``fps_override`` is only for containers that report no usable FPS; it is never used to resample."""

    fps_override: Optional[float] = None
    max_frames: Optional[int] = None
    sequence_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VideoSettings":
        _reject_unknown("video", d, ("fps_override", "max_frames", "sequence_id"))
        fps = d.get("fps_override")
        mf = d.get("max_frames")
        sid = d.get("sequence_id")
        if sid is not None and not (isinstance(sid, str) and sid.strip()):
            raise ConfigError("video.sequence_id must be a non-empty string or null, got %r" % (sid,))
        return cls(None if fps is None else _num("video.fps_override", fps, 0.0, 1000.0, lo_open=True),
                   None if mf is None else _int("video.max_frames", mf, 1), sid)


@dataclass(frozen=True)
class DetectorSettings:
    model_path: str = "weights/yolo11n.pt"
    device: str = "auto"
    conf: float = 0.10
    iou: float = 0.7
    imgsz: int = 640
    max_det: int = 300
    half: bool = False
    coco_class_ids: Optional[Tuple[int, ...]] = DEFAULT_COCO_CLASS_IDS
    label_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_LABEL_MAP))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DetectorSettings":
        _reject_unknown("detector", d, ("model_path", "device", "conf", "iou", "imgsz", "max_det", "half", "coco_class_ids", "label_map"))
        base = cls()
        mp = d.get("model_path", base.model_path)
        if not (isinstance(mp, str) and mp.strip()):
            raise ConfigError("detector.model_path must be a non-empty string, got %r" % (mp,))
        dev = d.get("device", base.device)
        if dev not in _DEVICES:
            raise ConfigError("detector.device must be one of %s, got %r" % (_DEVICES, dev))
        ids = d.get("coco_class_ids", base.coco_class_ids)
        if ids is not None:
            if not isinstance(ids, (list, tuple)) or not ids:
                raise ConfigError("detector.coco_class_ids must be a non-empty list of integers or null, got %r" % (ids,))
            ids = tuple(_int("detector.coco_class_ids[]", i, 0) for i in ids)
        lm = d.get("label_map", base.label_map)
        if not isinstance(lm, Mapping) or not lm:
            raise ConfigError("detector.label_map must be a non-empty object {detector_label: category}")
        try:
            LabelMap(dict(lm))                                   # validates the targets
        except ValueError as exc:
            raise ConfigError("detector.label_map: %s" % exc)
        return cls(mp, dev, _num("detector.conf", d.get("conf", base.conf), 0.0, 1.0, lo_open=True),
                   _num("detector.iou", d.get("iou", base.iou), 0.0, 1.0, lo_open=True),
                   _int("detector.imgsz", d.get("imgsz", base.imgsz), 32, 8192), _int("detector.max_det", d.get("max_det", base.max_det), 1),
                   _bool("detector.half", d.get("half", base.half)), ids, dict(lm))


@dataclass(frozen=True)
class TrackerSettings:
    """ByteTrack parameters (names follow the Ultralytics ``bytetrack.yaml``). ``track_buffer`` is in FRAMES."""

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    box_source: str = "detection"        # 'detection': report the matched detection's box; 'kalman': the filtered track box
    class_policy: str = "detection"      # 'detection': class of the matched detection; 'first_seen': class the track started with

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrackerSettings":
        _reject_unknown("tracker", d, ("track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh", "fuse_score",
                                       "box_source", "class_policy"))
        bs = d.get("box_source", cls.box_source)
        cp = d.get("class_policy", cls.class_policy)
        if bs not in ("detection", "kalman"):
            raise ConfigError("tracker.box_source must be 'detection' or 'kalman', got %r" % (bs,))
        if cp not in ("detection", "first_seen"):
            raise ConfigError("tracker.class_policy must be 'detection' or 'first_seen', got %r" % (cp,))
        b = cls()
        hi = _num("tracker.track_high_thresh", d.get("track_high_thresh", b.track_high_thresh), 0.0, 1.0)
        lo = _num("tracker.track_low_thresh", d.get("track_low_thresh", b.track_low_thresh), 0.0, 1.0)
        if lo > hi:
            raise ConfigError("tracker.track_low_thresh (%s) must not exceed track_high_thresh (%s)" % (lo, hi))
        return cls(hi, lo, _num("tracker.new_track_thresh", d.get("new_track_thresh", b.new_track_thresh), 0.0, 1.0),
                   _int("tracker.track_buffer", d.get("track_buffer", b.track_buffer), 0),
                   _num("tracker.match_thresh", d.get("match_thresh", b.match_thresh), 0.0, 1.0, lo_open=True),
                   _bool("tracker.fuse_score", d.get("fuse_score", b.fuse_score)), bs, cp)


@dataclass(frozen=True)
class PipelineSettings:
    strict: bool = False
    out_of_frame: str = "reject"
    bbox_tolerance_px: float = 1.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PipelineSettings":
        _reject_unknown("pipeline", d, ("strict", "out_of_frame", "bbox_tolerance_px"))
        b = cls()
        of = d.get("out_of_frame", b.out_of_frame)
        if of not in ("reject", "clip"):
            raise ConfigError("pipeline.out_of_frame must be 'reject' or 'clip', got %r" % (of,))
        return cls(_bool("pipeline.strict", d.get("strict", b.strict)), of,
                   _num("pipeline.bbox_tolerance_px", d.get("bbox_tolerance_px", b.bbox_tolerance_px), 0.0, 1e6))


@dataclass(frozen=True)
class PerceptionConfig:
    video: VideoSettings = field(default_factory=VideoSettings)
    detector: DetectorSettings = field(default_factory=DetectorSettings)
    tracker: TrackerSettings = field(default_factory=TrackerSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PerceptionConfig":
        if not isinstance(d, Mapping):
            raise ConfigError("configuration must be a JSON object")
        _reject_unknown("config", d, ("config_version", "description", "video", "detector", "tracker", "pipeline"))
        return cls(VideoSettings.from_dict(d.get("video", {})), DetectorSettings.from_dict(d.get("detector", {})),
                   TrackerSettings.from_dict(d.get("tracker", {})), PipelineSettings.from_dict(d.get("pipeline", {})))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["config_version"] = CONFIG_VERSION
        if d["detector"]["coco_class_ids"] is not None:
            d["detector"]["coco_class_ids"] = list(d["detector"]["coco_class_ids"])
        return d

    def warnings(self) -> List[str]:
        """Non-fatal consistency notes (recorded in the run manifest)."""
        out: List[str] = []
        if self.detector.conf > self.tracker.track_low_thresh:
            out.append("detector.conf (%s) is above tracker.track_low_thresh (%s): ByteTrack's low-score second association "
                       "can never receive a detection" % (self.detector.conf, self.tracker.track_low_thresh))
        if self.tracker.new_track_thresh < self.tracker.track_high_thresh:
            out.append("tracker.new_track_thresh is below track_high_thresh; new tracks can only start from high-score detections anyway")
        if self.detector.half and self.detector.device == "cpu":
            out.append("detector.half=true has no effect on CPU")
        return out

    def pipeline_config(self) -> PipelineConfig:
        return PipelineConfig(strict=self.pipeline.strict, label_map=LabelMap(dict(self.detector.label_map)),
                              out_of_frame=self.pipeline.out_of_frame, bbox_tolerance_px=self.pipeline.bbox_tolerance_px)


def load_config(path: Optional[Path] = None) -> PerceptionConfig:
    """Load and validate a JSON configuration; ``None`` returns the documented defaults."""
    if path is None:
        return PerceptionConfig()
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError("configuration file not found: %s" % p)
    except (OSError, ValueError) as exc:
        raise ConfigError("cannot read configuration %s: %s" % (p, exc))
    cv = raw.get("config_version") if isinstance(raw, dict) else None
    if cv is not None and cv != CONFIG_VERSION:
        raise ConfigError("config_version %r does not match this code (%r)" % (cv, CONFIG_VERSION))
    return PerceptionConfig.from_dict(raw)
