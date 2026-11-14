"""Task 1D/1E: structured tracked-agent export (stdlib) and visualisation (needs OpenCV)."""

import json

import numpy as np
import pytest

from pedsafety.perception import BBOX_CONVENTION, PerceptionPipeline, RawDetection
from pedsafety.perception.export import JsonlWriter, detection_rows, read_jsonl, track_rows
from pedsafety.perception.mocks import MockDetector, MockTracker, ScriptedFrameSource, synthetic_frames

REQUIRED = {"video_id", "frame_index", "timestamp_s", "track_id", "detector_class_id", "detector_class_name", "semantic_class", "confidence",
            "bbox_xyxy", "bbox_format", "image_width", "image_height", "detection_validity", "tracking_validity"}
FORBIDDEN = {"world_x", "world_y", "velocity_x", "velocity_y", "speed", "heading", "world_coordinate_unit"}


def _results(n=4):
    frames = list(synthetic_frames(n, width=320, height=240, sequence_id="vid"))
    script = {k: [RawDetection(0, "person", (20.0 + 2 * k, 30.0, 50.0 + 2 * k, 100.0), 0.87, 0),
                  RawDetection(1, "bicycle", (150.0, 100.0, 190.0, 150.0), 0.6, 1)] for k in range(n)}
    return PerceptionPipeline(ScriptedFrameSource(frames), MockDetector(script), MockTracker()).run()


def test_track_rows_carry_every_required_field_and_no_world_fields():
    res = _results()
    rows = [r for f in res.frames for r in track_rows(f)]
    assert rows
    for r in rows:
        assert REQUIRED <= set(r) and not (FORBIDDEN & set(r))
        assert r["bbox_format"] == BBOX_CONVENTION and r["coordinate_space"] == "image_pixels"
        assert r["detection_validity"] is True and r["tracking_validity"] is True and r["is_predicted"] is False
        assert (r["image_width"], r["image_height"]) == (320, 240) and r["video_id"] == "vid"
    person = [r for r in rows if r["detector_class_name"] == "person"][0]
    assert person["semantic_class"] == "pedestrian" and person["detector_class_id"] == 0 and person["confidence"] == pytest.approx(0.87)
    bike = [r for r in rows if r["detector_class_name"] == "bicycle"][0]
    assert bike["semantic_class"] == "bike" and bike["detector_class_id"] == 1                  # original class preserved beside the mapped one


def test_timestamps_and_frame_indices_are_carried_unchanged():
    res = _results(5)
    rows = [r for f in res.frames for r in track_rows(f)]
    assert sorted({r["frame_index"] for r in rows}) == [0, 1, 2, 3, 4]
    assert all(r["timestamp_s"] == pytest.approx(r["frame_index"] / 10.0) for r in rows)


def test_detection_rows_cover_untracked_detections_too():
    res = _results()
    drows = [r for f in res.frames for r in detection_rows(f)]
    assert len(drows) == 8 and all(r["coordinate_space"] == "image_pixels" and "class_id" not in r for r in drows)


def test_jsonl_writer_is_incremental_valid_and_nan_safe(tmp_path):
    p = str(tmp_path / "o.jsonl")
    with JsonlWriter(p) as w:
        w.write_rows([{"a": 1}, {"b": [1.5, 2]}])
        assert read_jsonl(p) == [{"a": 1}, {"b": [1.5, 2]}]                                     # readable before close (flushed)
        with pytest.raises(ValueError):
            w.write_rows([{"bad": float("nan")}])
    assert w.rows_written == 2
    assert all(json.loads(line) for line in open(p, encoding="utf-8"))


# ---------------------------------------------------------------- visualisation (OpenCV)
cv2 = pytest.importorskip("cv2")


def test_draw_frame_result_annotates_a_copy_and_changes_pixels():
    from pedsafety.perception.visualize import draw_frame_result
    res = _results()
    img = np.full((240, 320, 3), 100, dtype=np.uint8)
    before = img.copy()
    out = draw_frame_result(img, res.frames[1])
    assert np.array_equal(img, before) and out.shape == img.shape and not np.array_equal(out, img)


def test_untracked_detections_are_drawn_differently_and_can_be_hidden():
    from pedsafety.perception.visualize import draw_frame_result
    frames = list(synthetic_frames(1, width=200, height=200))
    script = {0: [RawDetection(0, "person", (10.0, 30.0, 40.0, 90.0), 0.9, 0)]}
    class NoTrack(MockTracker):
        def update(self, frame, detections):
            return []
    res = PerceptionPipeline(ScriptedFrameSource(frames), MockDetector(script), NoTrack()).run()
    img = np.full((200, 200, 3), 100, dtype=np.uint8)
    shown, hidden = draw_frame_result(img, res.frames[0]), draw_frame_result(img, res.frames[0], draw_untracked_detections=False)
    assert not np.array_equal(shown, hidden)


def test_track_colours_are_stable_per_id_and_differ_between_ids():
    from pedsafety.perception.visualize import track_colour
    assert track_colour(5) == track_colour(5) and track_colour(5) != track_colour(6)


def test_incremental_video_writer_writes_frames_and_validates_size(tmp_path):
    from pedsafety.perception.visualize import AnnotatedVideoWriter
    from pedsafety.perception.runtime_errors import VideoSourceError
    from pedsafety.perception.video_source import OpenCVVideoSource
    p = str(tmp_path / "a.avi")
    with AnnotatedVideoWriter(p, 10.0, 64, 48, fourcc="MJPG") as w:
        for k in range(4):
            w.write(np.full((48, 64, 3), 20 * k, dtype=np.uint8))
            assert w.frames_written == k + 1
        with pytest.raises(VideoSourceError):
            w.write(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(VideoSourceError):
        w.write(np.zeros((48, 64, 3), np.uint8))                                                # closed
    with OpenCVVideoSource(p) as s:
        assert len(list(s.frames())) == 4


def test_writer_refuses_an_unusable_path(tmp_path):
    from pedsafety.perception.visualize import AnnotatedVideoWriter
    from pedsafety.perception.runtime_errors import VideoSourceError
    with pytest.raises(VideoSourceError):
        AnnotatedVideoWriter(str(tmp_path / "no_such_dir" / "x.mp4"), 10.0, 64, 48)
