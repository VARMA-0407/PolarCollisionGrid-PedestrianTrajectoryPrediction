"""Structured tracked-agent export (Phase 3, Task 1D). Stdlib only.

One JSON object per line, one line per (frame, track). Image space only: boxes are pixels of the raster named in
``coordinate_system`` (top-left origin, y down). There are NO world coordinates, velocities or physical units here; a later
geometry stage derives footpoints from ``bbox_xyxy`` and projects them only with a verified calibration. The schema-valid
agent-state records are produced separately by the existing adapter (``tracks_to_agent_states``) and need a caller-supplied
``scene_id``; this file is the lossless perception-side record that also carries what the agent-state schema cannot.

Row fields: video_id (the sequence id), frame_index, timestamp_s, timestamp_source, track_id, detector_class_id,
detector_class_name (the detector's ORIGINAL label), semantic_class, confidence (matched detection's score, a model score and
not a calibrated probability), bbox_xyxy, bbox_format, image_width, image_height, coordinate_system, coordinate_space,
matched_detection_id, is_predicted, detection_validity, tracking_validity.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .pipeline import FrameResult
from .types import BBOX_CONVENTION

JSONL_SCHEMA_VERSION = "tracked-agent-jsonl-0.1.0-task1"


def track_rows(result: FrameResult) -> List[Dict[str, Any]]:
    """Rows for every validated track of one frame."""
    det = {d.detection_id: d for d in result.detections}
    rows: List[Dict[str, Any]] = []
    for t in result.tracks:
        d = det.get(t.matched_detection_id) if t.matched_detection_id is not None else None
        rows.append({
            "video_id": t.sequence_id, "frame_index": t.frame_index, "timestamp_s": t.timestamp_s,
            "timestamp_source": result.frame.timestamp_source, "track_id": t.track_id,
            "detector_class_id": None if d is None else d.class_id, "detector_class_name": None if d is None else d.label_raw,
            "semantic_class": t.category.value, "confidence": t.detection_confidence, "bbox_xyxy": [float(v) for v in t.bbox_xyxy],
            "bbox_format": BBOX_CONVENTION, "image_width": t.image_width, "image_height": t.image_height,
            "coordinate_system": t.coordinate_system, "coordinate_space": "image_pixels", "matched_detection_id": t.matched_detection_id,
            "is_predicted": t.is_predicted, "detection_validity": d is not None and not t.is_predicted,
            "tracking_validity": not t.is_predicted})
    return rows


def detection_rows(result: FrameResult) -> List[Dict[str, Any]]:
    """Rows for every validated detection of one frame (to inspect false and missed detections)."""
    return [{"video_id": d.sequence_id, "frame_index": d.frame_index, "timestamp_s": d.timestamp_s, "detection_id": d.detection_id,
             "detector_class_id": d.class_id, "detector_class_name": d.label_raw, "semantic_class": d.category.value,
             "confidence": d.confidence, "bbox_xyxy": [float(v) for v in d.bbox_xyxy], "bbox_format": BBOX_CONVENTION,
             "image_width": d.image_width, "image_height": d.image_height, "coordinate_system": d.coordinate_system,
             "coordinate_space": "image_pixels", "clipped": d.clipped} for d in result.detections]


class JsonlWriter:
    """Incremental, flushed JSONL writer (a crash leaves every completed line readable)."""

    def __init__(self, path: str):
        self.path = os.fspath(path)
        self._fh = open(self.path, "w", encoding="utf-8", newline="\n")
        self.rows_written = 0

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        for r in rows:
            self._fh.write(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n")
            self.rows_written += 1
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
