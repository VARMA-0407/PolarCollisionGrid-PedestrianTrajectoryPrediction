"""Task 1C: the REAL ByteTrack (Ultralytics BYTETracker) through the wrapper, on synthetic detections.

No weights, video or dataset is needed: only the tracker library. Skipped when ``ultralytics`` is not installed.
"""

import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("numpy")

from pedsafety.perception import PerceptionClass  # noqa: E402
from pedsafety.perception.bytetrack_tracker import ByteTrackTracker  # noqa: E402
from pedsafety.perception.config import TrackerSettings  # noqa: E402

from task1_helpers import det, frame_meta  # noqa: E402

P, C = PerceptionClass.PEDESTRIAN, PerceptionClass.CAR


def box(x, y=100.0, w=40.0, h=100.0):
    return (float(x), float(y), float(x + w), float(y + h))


def tracker(**kw):
    return ByteTrackTracker(TrackerSettings(**kw))


def step(t, dets, i=0):
    return list(t.update(frame_meta(i), dets))


def ids(obs):
    return sorted(o.track_id for o in obs)


def test_track_creation_and_persistence_across_frames():
    t = tracker()
    seen = set()
    for k in range(10):
        out = step(t, [det(0, box(100 + 3 * k))], k)
        assert len(out) == 1
        seen.add(out[0].track_id)
        assert out[0].matched_detection_id == 0 and out[0].is_predicted is False and out[0].category == P
    assert seen == {1}


def test_new_agent_gets_a_new_id_and_existing_ids_are_unchanged():
    t = tracker()
    for k in range(5):
        step(t, [det(0, box(100 + 3 * k))], k)
    out = []
    for k in range(5, 10):
        out = step(t, [det(0, box(100 + 3 * k)), det(1, box(400, w=30, h=80))], k)
    assert ids(out) == [1, 2]
    by_det = {o.matched_detection_id: o.track_id for o in out}
    assert by_det == {0: 1, 1: 2}


def test_reappearance_after_temporary_occlusion_keeps_the_id():
    t = tracker(track_buffer=30)
    first = None
    for k in range(8):
        out = step(t, [det(0, box(100 + 3 * k))], k)
        first = out[0].track_id
    for k in range(8, 12):                                                     # 4 frames with no detection (occluded)
        assert step(t, [], k) == []
    out = step(t, [det(0, box(100 + 3 * 12))], 12)
    assert [o.track_id for o in out] == [first]


def test_track_lost_beyond_the_buffer_is_not_revived():
    t = tracker(track_buffer=3)
    for k in range(6):
        out = step(t, [det(0, box(100))], k)
    old = out[0].track_id
    for k in range(6, 16):
        step(t, [], k)
    out = []
    for k in range(16, 19):
        out = step(t, [det(0, box(100))], k)
    assert out and out[0].track_id != old


def test_empty_detection_frames_are_accepted_and_age_the_tracks():
    t = tracker()
    assert step(t, [], 0) == []
    step(t, [det(0, box(100))], 1)
    assert step(t, [], 2) == [] and t.frames_updated == 3


def test_multiple_pedestrians_and_vehicles_with_semantic_classes_preserved():
    t = tracker()
    specs = [(P, "person", 0, box(50)), (P, "person", 0, box(150)), (C, "car", 2, box(300, y=200, w=120, h=60)), (C, "car", 2, box(450, y=250, w=110, h=60))]
    out = []
    for k in range(6):
        ds = [det(i, (b[0] + 2 * k, b[1], b[2] + 2 * k, b[3]), cat=c, label=l, class_id=cid) for i, (c, l, cid, b) in enumerate(specs)]
        out = step(t, ds, k)
    assert len(out) == 4 and len(set(o.track_id for o in out)) == 4
    assert {o.matched_detection_id: o.category for o in out} == {0: P, 1: P, 2: C, 3: C}


def test_no_duplicate_active_ids_in_any_frame_for_a_crowd():
    t = tracker()
    for k in range(30):
        ds = [det(i, box(20 + 55 * i + k, y=80 + 7 * i, w=45, h=90)) for i in range(10)]
        out = step(t, ds, k)
        got = [o.track_id for o in out]
        assert len(got) == len(set(got))
        assert len({o.matched_detection_id for o in out}) == len(out)           # one detection feeds at most one track


def test_low_score_detection_keeps_an_established_track_via_the_second_association():
    t = tracker(track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.5)
    for k in range(5):
        out = step(t, [det(0, box(100 + 2 * k), conf=0.9)], k)
    tid = out[0].track_id
    out = step(t, [det(0, box(110), conf=0.2)], 5)                                # between low and high thresholds
    assert [o.track_id for o in out] == [tid]


def test_low_score_detection_alone_does_not_start_a_track():
    t = tracker(track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.5)
    for k in range(5):
        assert step(t, [det(0, box(100), conf=0.2)], k) == []
    assert step(t, [det(0, box(100), conf=0.05)], 5) == []                        # below the low threshold: ignored entirely


def test_class_policy_detection_reports_flips_first_seen_pins_the_class():
    flips = {}
    for pol in ("detection", "first_seen"):
        t = tracker(class_policy=pol)
        for k in range(5):
            out = step(t, [det(0, box(100 + k), cat=P, label="person", class_id=0)], k)
        out = step(t, [det(0, box(106), cat=PerceptionClass.BIKE, label="bicycle", class_id=1)], 5)
        flips[pol] = (out[0].category, t.class_flips)
    assert flips["detection"] == (PerceptionClass.BIKE, 1)
    assert flips["first_seen"] == (P, 1)


def test_box_source_detection_vs_kalman():
    d = det(0, box(100, w=40, h=100))
    for src in ("detection", "kalman"):
        t = tracker(box_source=src)
        for k in range(6):
            out = step(t, [det(0, box(100 + 5 * k))], k)
        b = out[0].bbox_xyxy
        if src == "detection":
            assert b == box(125)
        else:
            assert b != box(125) and abs(b[0] - 125) < 15


def test_reset_starts_a_new_sequence_with_ids_restarting():
    t = tracker()
    for k in range(4):
        out = step(t, [det(0, box(100))], k)
    assert out[0].track_id == 1
    t.reset()
    assert t.class_flips == 0
    out = step(t, [det(0, box(300))], 0)
    assert [o.track_id for o in out] == [1]


def test_two_trackers_do_not_share_ids_or_state():
    a, b = tracker(), tracker()
    for k in range(3):
        step(a, [det(0, box(100)), det(1, box(300))], k)
    out = step(b, [det(0, box(100))], 0)
    assert [o.track_id for o in out] == [1]


def test_thresholds_are_applied_from_settings():
    t = tracker(track_high_thresh=0.8, new_track_thresh=0.8)
    for k in range(4):
        assert step(t, [det(0, box(100), conf=0.6)], k) == []
    t2 = tracker(track_high_thresh=0.3, new_track_thresh=0.3)
    outs = [step(t2, [det(0, box(100), conf=0.6)], k) for k in range(4)]
    assert outs[-1]


def test_tracker_info_documents_the_implementation_and_parameters():
    info = tracker(track_buffer=17).info
    p = dict(info.parameters)
    assert info.kind == "tracker" and info.is_mock is False and "ultralytics-" in info.version
    assert p["implementation"] == "ultralytics.trackers.byte_tracker.BYTETracker" and p["track_buffer_frames"] == "17"


def test_end_to_end_with_the_pipeline_validation_layer():
    """The pipeline owns validation: real ByteTrack output passes the duplicate-id / class / matched-detection checks."""
    from pedsafety.perception import PerceptionPipeline
    from pedsafety.perception.mocks import MockDetector, ScriptedFrameSource, synthetic_frames
    from pedsafety.perception import RawDetection
    frames = list(synthetic_frames(12, width=640, height=360))
    script = {k: [RawDetection(0, "person", (50.0 + 3 * k, 100.0, 90.0 + 3 * k, 200.0), 0.9, 0),
                  RawDetection(1, "car", (300.0, 200.0 + k, 420.0, 260.0 + k), 0.8, 2)] for k in range(12)}
    res = PerceptionPipeline(ScriptedFrameSource(frames), MockDetector(script), ByteTrackTracker(TrackerSettings())).run()
    assert res.issues == []
    ids_seen = {t.track_id for f in res.frames for t in f.tracks}
    assert ids_seen == {1, 2}
