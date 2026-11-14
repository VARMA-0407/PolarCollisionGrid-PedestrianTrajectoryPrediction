"""Stage 2A: perception interfaces, validation, adapter and scaffolding. EVERY component here is a mock
(deterministic test double). Nothing in this file measures or claims real detection, tracking or runtime performance."""

import hashlib
import math
import subprocess
import sys
from pathlib import Path

import pytest

from pedsafety.errors import ConfigError
from pedsafety.perception import (
    AdapterConfig, ComponentInfo, DetectionEvaluator, EvaluationProtocol, EvaluationReport, EvaluationUnavailableError, FrameMeta,
    GroundTruthSequence, IssueCode, LabelMap, PerceptionClass, PerceptionError, PerceptionPipeline, PerformanceRecorder, PipelineConfig, RawDetection,
    Tracker, TrackingEvaluator, TrackObservation, tracks_to_agent_states,
)
from pedsafety.perception.metrics import UnimplementedDetectionEvaluator, UnimplementedTrackingEvaluator
from pedsafety.perception.mocks import MockDetector, MockTracker, ScriptedFrameSource, iou, synthetic_frames
from pedsafety.perception.performance import percentile
from pedsafety.schema import collect_violations

REPO = Path(__file__).resolve().parents[2]
SCHEMA_SHA = "784c8ca9c059ccbf0ddbb8bfcaf42cdfbbd60b33acfc0136c78a00fde46c3c6c"


def det(i, label="person", box=(100.0, 100.0, 140.0, 200.0), conf=0.9):
    return RawDetection(i, label, box, conf)


def run(script, n=4, *, strict=False, tracker=None, frames=None, cfg=None, detector=None):
    detector = detector or MockDetector(script)
    p = PerceptionPipeline(ScriptedFrameSource(frames if frames is not None else synthetic_frames(n)), detector, tracker or MockTracker(),
                           cfg or PipelineConfig(strict=strict))
    return p.run()


def codes(res):
    return [i.code for i in res.issues]


def meta(i, t, **kw):
    d = dict(source_id="s", sequence_id="q", frame_index=i, timestamp_s=t, width=640, height=360, coordinate_system="r", timestamp_source="mock_clock")
    d.update(kw)
    return FrameMeta(**d)


# ------------------------------------------------------------ frames, order, timestamps
def test_ordered_frames_with_gaps_are_accepted_and_timestamps_kept():
    fr = [(meta(0, 0.0), None), (meta(3, 0.3), None), (meta(10, 1.0), None)]
    res = run({}, frames=fr)
    assert [f.frame.frame_index for f in res.frames] == [0, 3, 10] and all(f.processed for f in res.frames) and not res.issues
    assert [f.frame.timestamp_s for f in res.frames] == [0.0, 0.3, 1.0]


@pytest.mark.parametrize("bad,code", [
    (meta(1, 0.1, frame_index=-1), IssueCode.INVALID_FRAME_INDEX),
    (meta(1, float("nan")), IssueCode.INVALID_TIMESTAMP),
    (meta(1, float("inf")), IssueCode.INVALID_TIMESTAMP),
    (meta(1, -0.5), IssueCode.INVALID_TIMESTAMP),
    (meta(1, None), IssueCode.INVALID_TIMESTAMP),                       # missing timestamp
    (meta(1, "0.1"), IssueCode.INVALID_TIMESTAMP),
    (meta(1, 0.1, width=0), IssueCode.INVALID_IMAGE_SIZE),
    (meta(1, 0.1, height=-5), IssueCode.INVALID_IMAGE_SIZE),
    (meta(1, 0.1, coordinate_system=""), IssueCode.INVALID_FRAME_META),
    (meta(1, 0.1, timestamp_source="guess"), IssueCode.INVALID_FRAME_META),
    (meta(1, 0.1, nominal_fps=0.0), IssueCode.INVALID_FRAME_META),
])
def test_invalid_frames_are_skipped_without_calling_detector_or_tracker(bad, code):
    detector = MockDetector({})
    res = run({}, frames=[(meta(0, 0.0), None), (bad, None), (meta(2, 0.2), None)], detector=detector)
    assert codes(res) == [code] and [f.processed for f in res.frames] == [True, False, True]
    assert detector.calls == [0, 2]


def test_out_of_order_and_repeated_frames_and_times():
    fr = [(meta(5, 0.5), None), (meta(5, 0.6), None), (meta(4, 0.7), None), (meta(6, 0.5), None), (meta(7, 0.7), None)]
    res = run({}, frames=fr)
    assert codes(res) == [IssueCode.FRAME_ORDER] * 3 and [f.processed for f in res.frames] == [True, False, False, False, True]


def test_sequence_change_inside_a_run_is_rejected():
    res = run({}, frames=[(meta(0, 0.0), None), (meta(1, 0.1, sequence_id="other"), None)])
    assert codes(res) == [IssueCode.SEQUENCE_CHANGED]


def test_non_framemeta_items_are_reported_not_lost_or_crashed():
    res = run({}, frames=[(meta(0, 0.0), None), ("not a frame", None), (meta(1, 0.1), None)])
    assert codes(res) == [IssueCode.INVALID_FRAME_META] and len(res.frames) == 2


def test_empty_source_gives_empty_result():
    res = run({}, frames=[])
    assert res.frames == [] and res.counters["frames_seen"] == 0


def test_strict_mode_raises_with_structured_issue():
    with pytest.raises(PerceptionError) as ei:
        run({}, frames=[(meta(1, 0.1), None), (meta(1, 0.2), None)], strict=True)
    assert ei.value.issue.code is IssueCode.FRAME_ORDER and ei.value.issue.stage == "source" and ei.value.issue.frame_index == 1


# ------------------------------------------------------------------ detections
def test_empty_frames_still_update_the_tracker():
    tr = MockTracker()
    seen = []
    orig = tr.update
    tr.update = lambda f, d: (seen.append((f.frame_index, len(d))), orig(f, d))[1]
    res = run({0: [det(1)]}, n=3, tracker=tr)
    assert seen == [(0, 1), (1, 0), (2, 0)] and all(f.processed for f in res.frames) and res.frames[1].detections == ()


def test_multiple_detections_one_frame_get_distinct_tracks_and_classes():
    script = {0: [det(1, "person", (10, 10, 40, 90)), det(2, "car", (200, 100, 320, 180)), det(3, "truck", (400, 50, 600, 200)),
                  det(4, "bus", (50, 250, 250, 350)), det(5, "bicycle", (300, 220, 340, 300))]}
    res = run(script, n=1)
    f = res.frames[0]
    assert len(f.detections) == 5 and len({t.track_id for t in f.tracks}) == 5
    assert sorted(d.category.value for d in f.detections) == ["bike", "bus", "car", "pedestrian", "truck"]


@pytest.mark.parametrize("box", [(50, 50, 40, 90), (10, 10, 10, 90), (10, 10, 40, 10), (float("nan"), 0, 5, 5), (0, 0, 5), "abc", None, (0, 0, 5, "x"),
                                 (0, 0, float("inf"), 5)])
def test_invalid_boxes_are_dropped_with_issue(box):
    res = run({0: [det(1, box=box), det(2)]}, n=1)
    assert codes(res) == [IssueCode.INVALID_BBOX] and [d.detection_id for d in res.frames[0].detections] == [2]


@pytest.mark.parametrize("conf", [-0.01, 1.01, float("nan"), float("inf"), None, "0.5", True])
def test_confidence_outside_unit_interval_is_rejected(conf):
    res = run({0: [det(1, conf=conf)]}, n=1)
    assert codes(res) == [IssueCode.INVALID_CONFIDENCE] and res.frames[0].detections == ()


def test_confidence_endpoints_are_accepted():
    res = run({0: [det(1, conf=0.0), det(2, conf=1.0, box=(300, 100, 340, 200))]}, n=1)
    assert not res.issues and [d.confidence for d in res.frames[0].detections] == [0.0, 1.0]


def test_out_of_frame_boxes_reject_by_default_clip_on_request_and_respect_tolerance():
    box = (600.0, 100.0, 700.0, 200.0)                       # frame is 640x360
    res = run({0: [det(1, box=box)]}, n=1)
    assert codes(res) == [IssueCode.BBOX_OUT_OF_FRAME] and res.frames[0].detections == ()
    res = run({0: [det(1, box=box)]}, n=1, cfg=PipelineConfig(out_of_frame="clip"))
    d = res.frames[0].detections[0]
    assert d.clipped and d.bbox_xyxy == (600.0, 100.0, 640.0, 200.0) and d.bbox_raw == box and not res.issues
    res = run({0: [det(1, box=(600.0, 100.0, 645.0, 200.0))]}, n=1, cfg=PipelineConfig(bbox_tolerance_px=5.0))
    assert not res.issues and not res.frames[0].detections[0].clipped                   # within tolerance: kept as given
    res = run({0: [det(1, box=(700.0, 100.0, 800.0, 200.0))]}, n=1, cfg=PipelineConfig(out_of_frame="clip"))
    assert codes(res) == [IssueCode.BBOX_OUT_OF_FRAME]                                  # nothing left after clipping


@pytest.mark.parametrize("label,expected", [("pedestrian", "pedestrian"), ("Person", "pedestrian"), (" PED ", "pedestrian"), ("car", "car"), ("truck", "truck"),
                                            ("bus", "bus"), ("bike", "bike"), ("bicycle", "bike")])
def test_supported_labels_and_aliases(label, expected):
    res = run({0: [det(1, label)]}, n=1)
    assert not res.issues and res.frames[0].detections[0].category.value == expected and res.frames[0].detections[0].label_raw == label


@pytest.mark.parametrize("label", ["motorcycle", "dog", "", "traffic light", None, 7])
def test_unsupported_labels_are_rejected_never_defaulted(label):
    res = run({0: [det(1, label)]}, n=1)
    assert codes(res) == [IssueCode.UNSUPPORTED_CLASS] and res.frames[0].detections == ()


def test_label_map_rejects_unsupported_targets_and_is_explicit():
    with pytest.raises(ValueError):
        LabelMap({"scooter": "motorcycle"})
    assert LabelMap({"x": "bus"}).resolve("X") is PerceptionClass.BUS and LabelMap({"x": "bus"}).resolve("person") is None


def test_duplicate_detection_ids_quarantine_every_copy():
    res = run({0: [det(1), det(1, box=(300, 100, 340, 200)), det(2, box=(500, 100, 540, 200))]}, n=1)
    assert codes(res) == [IssueCode.DUPLICATE_DETECTION_ID] * 2 and [d.detection_id for d in res.frames[0].detections] == [2]


def test_detection_provenance_is_complete_and_marked_mock():
    res = run({1: [det(7, "person", (10, 20, 30, 80), 0.75)]}, n=3)
    d = res.frames[1].detections[0]
    assert (d.sequence_id, d.frame_index, d.timestamp_s, d.image_width, d.image_height, d.detection_id, d.label_raw, d.confidence) == \
           ("mock_seq", 1, 0.1, 640, 360, 7, "person", 0.75)
    assert d.coordinate_system == "mock_raster_640x360" and d.detector.is_mock and d.detector.name.startswith("MOCK") and d.footpoint == (20.0, 80.0)
    assert res.is_mock and all(c.is_mock for c in res.components)


# ----------------------------------------------------------------------- tracking
def moving(n, dx=4.0):
    return {i: [det(1, box=(100 + dx * i, 100, 140 + dx * i, 200))] for i in range(n)}


def test_track_id_persists_while_object_moves():
    res = run(moving(6), n=6)
    ids = [f.tracks[0].track_id for f in res.frames]
    assert ids == [1] * 6 and all(f.tracks[0].matched_detection_id == 1 and not f.tracks[0].is_predicted for f in res.frames)


def test_disappearance_and_reappearance_within_max_age_keeps_the_id():
    script = {0: [det(1)], 1: [det(1)], 4: [det(1)]}                     # absent in frames 2-3 (2 missed updates), max_age=2
    res = run(script, n=5, tracker=MockTracker(max_age=2))
    assert [[t.track_id for t in f.tracks] for f in res.frames] == [[1], [1], [], [], [1]]


def test_reappearance_after_max_age_gets_a_new_id_never_a_reused_one():
    script = {0: [det(1)], 5: [det(1)]}
    res = run(script, n=6, tracker=MockTracker(max_age=1))
    assert [t.track_id for f in res.frames for t in f.tracks] == [1, 2]


def test_different_classes_never_share_a_track():
    res = run({0: [det(1, "person")], 1: [det(1, "car")]}, n=2)
    assert [t.track_id for f in res.frames for t in f.tracks] == [1, 2]


def test_tracker_state_is_reset_per_run_and_runs_are_deterministic():
    tr = MockTracker()
    a = run(moving(4), n=4, tracker=tr)
    b = run(moving(4), n=4, tracker=tr)
    assert [[t.track_id for t in f.tracks] for f in a.frames] == [[t.track_id for t in f.tracks] for f in b.frames] == [[1]] * 4
    assert a.frames == run(moving(4), n=4).frames


def test_iou_helper_sanity():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert math.isclose(iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)


class ScriptedTracker(Tracker):
    """MOCK tracker that returns whatever rows it is told to (to exercise validation of untrusted trackers)."""

    def __init__(self, rows_by_frame):
        self.rows = rows_by_frame
        self._info = ComponentInfo("tracker", "MOCK_ScriptedTracker", "mock-1", True)

    info = property(lambda self: self._info)

    def reset(self):
        pass

    def update(self, frame, detections):
        r = self.rows.get(frame.frame_index, [])
        if isinstance(r, Exception):
            raise r
        return r


def obs(tid, det_id=1, cat=PerceptionClass.PEDESTRIAN, box=(100.0, 100.0, 140.0, 200.0), **kw):
    return TrackObservation(tid, cat, box, det_id, **kw)


@pytest.mark.parametrize("row,code", [
    (obs(0), IssueCode.INVALID_TRACK_ID), (obs(-3), IssueCode.INVALID_TRACK_ID), (obs(True), IssueCode.INVALID_TRACK_ID), (obs(1.5), IssueCode.INVALID_TRACK_ID),
    (obs(1, box=(5, 5, 5, 5)), IssueCode.INVALID_BBOX), (obs(1, det_id=99), IssueCode.UNKNOWN_MATCHED_DETECTION),
    (obs(1, tracking_confidence=2.0), IssueCode.INVALID_CONFIDENCE), (obs(1, det_id=None), IssueCode.INVALID_TRACK),
    (obs(1, det_id=1, is_predicted=True), IssueCode.INVALID_TRACK), ("junk", IssueCode.INVALID_TRACK),
])
def test_invalid_tracker_output_is_dropped_with_issue(row, code):
    res = run({0: [det(1)]}, n=1, tracker=ScriptedTracker({0: [row]}))
    assert codes(res) == [code] and res.frames[0].tracks == ()


def test_duplicate_track_ids_in_a_frame_and_class_changes_are_caught():
    res = run({0: [det(1), det(2, box=(300, 100, 340, 200))]}, n=1, tracker=ScriptedTracker({0: [obs(1, 1), obs(1, 2, box=(300.0, 100.0, 340.0, 200.0))]}))
    assert codes(res) == [IssueCode.DUPLICATE_TRACK_ID] * 2 and res.frames[0].tracks == ()
    tr = ScriptedTracker({0: [obs(1, 1)], 1: [obs(1, 1, cat=PerceptionClass.CAR)]})
    res = run({0: [det(1)], 1: [det(1, "car")]}, n=2, tracker=tr)
    assert codes(res) == [IssueCode.TRACK_CLASS_CHANGED] and len(res.frames[0].tracks) == 1 and res.frames[1].tracks == ()


def test_track_record_carries_matched_detection_confidence_and_provenance():
    res = run({0: [det(5, conf=0.42)]}, n=1, tracker=ScriptedTracker({0: [obs(3, 5, tracking_confidence=0.6)]}))
    t = res.frames[0].tracks[0]
    assert (t.track_id, t.matched_detection_id, t.detection_confidence, t.tracking_confidence) == (3, 5, 0.42, 0.6)
    assert t.detector.is_mock and t.tracker.name == "MOCK_ScriptedTracker" and t.coordinate_system == "mock_raster_640x360"


# ---------------------------------------------------------------- error handling
def test_detector_exception_skips_frame_and_run_continues():
    res = run({0: [det(1)], 1: [det(1)], 2: [det(1)]}, n=3, detector=MockDetector({0: [det(1)], 2: [det(1)]}, fail_on_frame=1))
    assert codes(res) == [IssueCode.DETECTOR_ERROR] and [f.processed for f in res.frames] == [True, False, True]
    assert [t.track_id for f in res.frames for t in f.tracks] == [1, 1]            # identity survives within max_age


def test_tracker_exception_is_reported_and_keeps_detections():
    res = run({0: [det(1)]}, n=1, tracker=ScriptedTracker({0: RuntimeError("boom")}))
    assert codes(res) == [IssueCode.TRACKER_ERROR] and len(res.frames[0].detections) == 1 and not res.frames[0].processed


def test_strict_mode_raises_for_detector_tracker_and_detection_problems():
    with pytest.raises(PerceptionError) as e1:
        run({}, n=2, detector=MockDetector({}, fail_on_frame=0), strict=True)
    assert e1.value.issue.code is IssueCode.DETECTOR_ERROR and e1.value.issue.stage == "detector"
    with pytest.raises(PerceptionError) as e2:
        run({0: [det(1)]}, n=1, tracker=ScriptedTracker({0: RuntimeError("x")}), strict=True)
    assert e2.value.issue.code is IssueCode.TRACKER_ERROR
    with pytest.raises(PerceptionError) as e3:
        run({0: [det(1, conf=7)]}, n=1, strict=True)
    assert e3.value.issue.code is IssueCode.INVALID_CONFIDENCE and "[detector/invalid_confidence]" in str(e3.value)


def test_detector_returning_non_detections_is_handled():
    class Junk(MockDetector):
        def detect(self, frame, payload):
            return ["junk", None, det(1)]
    res = run({}, n=1, detector=Junk({}))
    assert codes(res) == [IssueCode.INVALID_DETECTION] * 2 and len(res.frames[0].detections) == 1


def test_component_kind_is_checked_and_config_validated():
    with pytest.raises(ValueError):
        PerceptionPipeline(ScriptedFrameSource([]), MockTracker(), MockDetector({}))
    with pytest.raises(ValueError):
        PipelineConfig(out_of_frame="crop")
    with pytest.raises(ValueError):
        PipelineConfig(bbox_tolerance_px=-1)
    with pytest.raises(ValueError):
        ComponentInfo("detector", "x", "1", "yes")                                       # is_mock must be an explicit bool


# ------------------------------------------------------------- agent-state adapter
def tracked(n=6):
    script = {i: [det(1, "person", (100 + 4 * i, 100, 140 + 4 * i, 200)), det(2, "car", (300, 150, 420, 230)), det(3, "bus", (50, 250, 250, 350)),
                  det(4, "truck", (450, 50, 600, 150)), det(5, "bike", (20, 20, 60, 80))] for i in range(n)}
    return run(script, n=n)


CFG = AdapterConfig("MOCK_perception", "mock_scene_0")


def test_adapter_records_validate_against_the_unchanged_schema():
    assert hashlib.sha256((REPO / "docs/phase2/schemas/agent_state.schema.json").read_bytes()).hexdigest() == SCHEMA_SHA
    r = tracks_to_agent_states(tracked(), CFG)
    assert r.report.ok and len(r.records) == 30 == len(r.provenance) == len(r.perception_provenance)
    assert all(not collect_violations(rec) for rec in r.records)


def test_adapter_class_identity_and_coordinate_mapping():
    r = tracks_to_agent_states(tracked(2), CFG)
    by_label = {}
    for rec in r.records:
        by_label.setdefault(rec["source_label"], set()).add(rec["agent_class"])
    assert by_label == {"pedestrian": {"pedestrian"}, "car": {"vehicle"}, "truck": {"vehicle"}, "bus": {"vehicle"}, "bike": {"bicyclist"}}
    rec = r.records[0]
    assert rec["track_id"] == "mock_perception:mock_seq:1" and rec["sequence_id"] == "mock_perception:mock_seq" and rec["scene_id"] == "mock_perception:mock_scene_0"
    assert rec["bbox_xyxy"] == [100.0, 100.0, 140.0, 200.0] and rec["footpoint"] == [120.0, 200.0]
    assert (rec["image_x"], rec["image_y"]) == (120.0, 200.0) and rec["image_coordinate_unit"] == "pixel" and rec["image_coordinate_system"] == "mock_raster_640x360"
    assert rec["world_position_valid"] is False and rec["world_x"] is None and rec["world_coordinate_unit"] is None
    assert rec["detection_confidence"] == 0.9 and rec["timestamp"] == 0.0 and rec["frame_index"] == 0
    assert rec["observation_valid"] is True and rec["interpolation_flag"] is False and rec["source_dataset"] == "MOCK_perception"
    assert rec["velocity_x"] is None and rec["kinematic_position_unit"] is None            # no unit contract: nothing invented


def test_adapter_sidecar_preserves_provenance():
    r = tracks_to_agent_states(tracked(2), CFG)
    pp = r.perception_provenance[0]
    assert pp.is_mock and pp.detector.name == "MOCK_Detector" and pp.tracker.name == "MOCK_IoUTracker" and pp.frame_source.name == "MOCK_ScriptedFrameSource"
    assert pp.track_id == 1 and pp.matched_detection_id == 1 and r.provenance[0].raw_track_id == "1"


def test_adapter_requires_explicit_scene_and_mock_labelled_dataset():
    with pytest.raises(ConfigError):
        AdapterConfig("MOCK_x", "")
    with pytest.raises(ConfigError):
        AdapterConfig("", "s")
    with pytest.raises(ConfigError, match="MOCK"):
        tracks_to_agent_states(tracked(1), AdapterConfig("PIE", "scene"))              # mock output cannot masquerade as a real dataset


def test_adapter_drops_and_counts_predicted_only_rows():
    res = run({0: [det(1)]}, n=3, tracker=MockTracker(emit_predicted=True, max_age=5))
    assert sum(1 for f in res.frames for t in f.tracks if t.is_predicted) == 2
    r = tracks_to_agent_states(res, CFG)
    assert r.dropped_predicted == 2 and len(r.records) == 1                              # the estimates are not observations


def test_adapter_sets_tracking_confidence_and_stays_schema_valid():
    res = run({0: [det(1)]}, n=1, tracker=ScriptedTracker({0: [obs(1, 1, tracking_confidence=0.8)]}))
    rec = tracks_to_agent_states(res, CFG).records[0]
    assert rec["tracking_confidence"] == 0.8 and not collect_violations(rec)


def test_adapter_reports_builder_issues_for_clipped_inputs_without_crashing():
    res = run({0: [det(1)]}, n=1)
    assert tracks_to_agent_states(res, CFG).report.ok


def test_adapter_output_feeds_the_existing_stage4_causal_features():
    """Integration: perception records -> Stage 4. Image-space features only, in pixel units, with the real dt."""
    from pedsafety.causal_trajectory import process_causal
    from pedsafety.trajectory import TrajectoryConfig
    r = tracks_to_agent_states(run(moving(6, dx=4.0), n=6), CFG)
    out = process_causal(r.records, r.provenance, None, TrajectoryConfig())
    assert len(out.trajectories) == 1
    t = out.trajectories[0]
    assert t.position_unit == "pixel" and not t.physical and t.velocity_unit == "pixel/s"
    vx = t["vx"]
    assert math.isclose(float(vx[-1]), 4.0 / 0.1, rel_tol=1e-9) and math.isnan(float(vx[0]))      # 4 px per 0.1 s


# --------------------------------------------------------- independence and honesty
def test_perception_package_imports_no_cv_or_ml_runtime():
    code = ("import sys, pedsafety.perception, pedsafety.perception.mocks; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('cv2','ultralytics','torch','torchvision','numpy','lap','filterpy','yolox')); "
            "print(bad)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0 and r.stdout.strip() == "[]", r.stdout + r.stderr


def test_every_concrete_component_in_the_package_is_a_labelled_mock():
    comps = [ScriptedFrameSource([]).info, MockDetector({}).info, MockTracker().info]
    assert all(c.is_mock and c.name.startswith("MOCK") for c in comps)


def test_evaluators_are_interfaces_only_and_refuse_to_invent_numbers():
    res = run({}, n=1)
    gt = GroundTruthSequence("q", "r", (), "unit-test truth")
    proto = EvaluationProtocol(0.5, None, (PerceptionClass.PEDESTRIAN,))
    for ev in (UnimplementedDetectionEvaluator(), UnimplementedTrackingEvaluator()):
        with pytest.raises(EvaluationUnavailableError):
            ev.evaluate(res, [gt], proto)
    with pytest.raises(TypeError):
        DetectionEvaluator()                                                            # abstract
    with pytest.raises(TypeError):
        TrackingEvaluator()
    assert DetectionEvaluator.metrics == ("precision", "recall", "mAP") and TrackingEvaluator.metrics == ("HOTA", "IDF1", "MOTA", "identity_switches")
    with pytest.raises(ValueError):
        GroundTruthSequence("q", "r", (), " ")


def test_reports_refuse_mock_components_and_mixed_metric_families():
    real = ComponentInfo("detector", "SomeRealDetector", "1.0", False)
    mock = MockDetector({}).info
    proto = EvaluationProtocol(0.5, 0.3, (PerceptionClass.CAR,))
    with pytest.raises(EvaluationUnavailableError):
        EvaluationReport("detection", {"precision": 0.5}, (mock,), "ds", proto)
    with pytest.raises(ValueError):
        EvaluationReport("detection", {"HOTA": 0.5}, (real,), "ds", proto)               # tracking metric in a detection report
    with pytest.raises(ValueError):
        EvaluationReport("tracking", {"precision": 0.5}, (real,), "ds", proto)
    with pytest.raises(ValueError):
        EvaluationReport("detection", {}, (), "ds", proto)
    EvaluationReport("detection", {"precision": 0.5}, (real,), "ds", proto)              # construction rules only; the number is a test value


# ------------------------------------------------------------ performance scaffolding
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_percentile_is_nearest_rank():
    assert percentile([1, 2, 3, 4], 50) == 2 and percentile([1, 2, 3, 4], 95) == 4 and percentile([5], 1) == 5 and percentile([3, 1, 2], 100) == 3
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_recorder_arithmetic_with_a_fake_clock_never_supports_claims():
    clock = FakeClock()
    rec = PerformanceRecorder(clock)
    for dt_detect, dt_track in ((0.02, 0.01), (0.04, 0.01)):
        with rec.frame():
            with rec.stage("detect"):
                clock.advance(dt_detect)
            with rec.stage("track"):
                clock.advance(dt_track)
    s = rec.summary([ScriptedFrameSource([]).info], hardware="fake machine")
    assert s.n_frames == 2 and math.isclose(s.elapsed_s, 0.08) and math.isclose(s.processing_fps, 2 / 0.08)
    assert math.isclose(s.stages["detect"].mean_s, 0.03) and math.isclose(s.stages["detect"].max_s, 0.04) and math.isclose(s.end_to_end.mean_s, 0.04)
    assert s.clock_is_real is False and s.any_mock and not s.supports_performance_claims


def test_performance_claims_need_real_clock_real_components_and_declared_hardware():
    real = (ComponentInfo("detector", "SomeRealDetector", "1.0", False),)
    rec = PerformanceRecorder()                                                          # real clock; no real work is timed here
    with rec.frame():
        pass
    assert rec.summary(real, hardware=None).supports_performance_claims is False
    assert rec.summary(real, hardware="  ").supports_performance_claims is False
    assert rec.summary((MockDetector({}).info,), hardware="x").supports_performance_claims is False
    assert PerformanceRecorder().summary(real, hardware="x").supports_performance_claims is False          # zero frames
    # predicate only: a real clock + non-mock components + declared hardware would qualify (nothing is measured by this test)
    assert rec.summary(real, hardware="declared-hardware").supports_performance_claims is True


def test_recorder_integrates_with_the_pipeline_and_tracks_memory_flag():
    clock = FakeClock()
    rec = PerformanceRecorder(clock, track_memory=True)
    p = PerceptionPipeline(ScriptedFrameSource(synthetic_frames(3)), MockDetector({}), MockTracker(), recorder=rec)
    res = p.run()
    s = rec.summary(res.components, hardware="fake")
    assert s.n_frames == 3 and set(s.stages) == {"detect", "track"} and s.peak_python_heap_bytes is not None and not s.supports_performance_claims


def test_a_fake_clock_blocks_claims_even_with_real_components_and_hardware():
    clock = FakeClock()
    rec = PerformanceRecorder(clock)
    with rec.frame():
        clock.advance(0.01)
    s = rec.summary((ComponentInfo("detector", "SomeRealDetector", "1.0", False),), hardware="declared-hardware")
    assert s.clock_is_real is False and s.supports_performance_claims is False
