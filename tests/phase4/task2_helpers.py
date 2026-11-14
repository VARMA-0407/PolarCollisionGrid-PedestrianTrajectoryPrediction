"""Synthetic Phase 3-style agent-state records for the Phase 4 Task 2 tests (no PIE data, no model, no video)."""

import copy
import json
from pathlib import Path

from pedsafety.image_kinematics import ImageKinematicsConfig

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "phase4" / "image_kinematics_default.json"
RASTER = "captured_1920x1080"


def cfg(**over):
    d = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    d.update(over)
    return ImageKinematicsConfig.from_dict(d)


def rec(track="1", frame=0, t=None, fx=100.0, fy=500.0, w=20.0, h=60.0, cls="pedestrian", seq="pie:video_0004", **over):
    """A record whose footpoint (bottom-centre) is (fx, fy). Defaults follow the real Phase 3 field set."""
    t = frame / 30.0 if t is None else t
    b = [fx - w / 2.0, fy - h, fx + w / 2.0, fy]
    r = {"track_id": "%s:%s" % (seq, track), "agent_id": None, "agent_class": cls, "timestamp": t, "frame_index": frame,
         "sequence_id": seq, "scene_id": seq, "bbox_xyxy": b, "bbox_center": [fx, fy - h / 2.0], "footpoint": [fx, fy],
         "detection_confidence": 0.9, "image_x": fx, "image_y": fy, "image_coordinate_system": RASTER, "image_coordinate_unit": "pixel",
         "world_x": None, "world_y": None, "world_coordinate_system": None, "world_coordinate_unit": None, "world_position_valid": False,
         "velocity_x": None, "velocity_y": None, "speed": None, "acceleration_x": None, "acceleration_y": None,
         "kinematic_coordinate_space": None, "kinematic_position_unit": None, "kinematic_time_unit": None, "heading": None,
         "visibility": None, "occlusion_status": None, "tracking_confidence": None, "observation_valid": True,
         "interpolation_flag": False, "source_dataset": "pie", "source_label": cls}
    r.update(over)
    return r


def track(points, track_id="1", **kw):
    """points: iterable of (frame, t, x, y)."""
    return [rec(track=track_id, frame=f, t=t, fx=x, fy=y, **kw) for f, t, x, y in points]


def lines(records):
    return list(range(1, len(records) + 1))


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def deep(records):
    return copy.deepcopy(records)
