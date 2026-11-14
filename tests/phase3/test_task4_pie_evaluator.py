"""Task 4: PIE ground-truth detection/tracking evaluator, on small synthetic XML/JSONL fixtures (no PIE data, no model, no video)."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

from pedsafety.errors import ConfigError
from pedsafety.perception.pie_perception_evaluator import (
    Pred, PieEvaluationError, evaluate_detection, evaluate_tracking, evaluation_frames, iou_xyxy, load_predictions, match_boxes, parse_pie_annotations,
    run_evaluation, solve_assignment,
)

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------------------------------------ fixtures
def box_xml(frame, b, outside=0, occluded=0, keyframe=1, attrs=None, raw_coords=None):
    a = "".join('<attribute name="%s">%s</attribute>' % (k, v) for k, v in (attrs or {}).items())
    x0, y0, x1, y1 = raw_coords or b
    return '<box frame="%s" xtl="%s" ytl="%s" xbr="%s" ybr="%s" outside="%d" occluded="%d" keyframe="%d">%s</box>' % (frame, x0, y0, x1, y1, outside, occluded, keyframe, a)


def track_xml(tid, label, boxes):
    return '<track id="%s" label="%s" source="manual">%s</track>' % (tid, label, "".join(boxes))


def write_xml(path, tracks, name="video_0004"):
    path.write_text('<?xml version="1.0"?><annotations><version>1.1</version><meta><task><name>%s</name></task></meta>%s</annotations>' % (name, "".join(tracks)),
                    encoding="utf-8")
    return path


def ped(tid, pid, frames, b=(100, 100, 140, 200), **kw):
    return track_xml(tid, "pedestrian", [box_xml(f, b, attrs={"id": pid, "occlusion": kw.get("occlusion", 0), "action": "standing"}, occluded=kw.get("occluded", 0)) for f in frames])


def veh(tid, vid, vtype, frames, b=(300, 300, 400, 360)):
    return track_xml(tid, "vehicle", [box_xml(f, b, attrs={"id": vid, "type": vtype}) for f in frames])


def det_row(frame, cls, b, conf=0.9, did=0, vid="video_0004"):
    return {"video_id": vid, "frame_index": frame, "timestamp_s": frame / 30.0, "detection_id": did, "detector_class_id": 0, "detector_class_name": cls,
            "semantic_class": cls, "confidence": conf, "bbox_xyxy": list(b), "bbox_format": "xyxy_image_pixels_top_left_origin_y_down",
            "image_width": 1920, "image_height": 1080, "coordinate_system": "captured_1920x1080", "coordinate_space": "image_pixels", "clipped": False}


def trk_row(frame, cls, b, tid, conf=0.9, predicted=False, valid=True, vid="video_0004"):
    return {"video_id": vid, "frame_index": frame, "timestamp_s": frame / 30.0, "timestamp_source": "frame_index_over_fps", "track_id": tid, "detector_class_id": 0,
            "detector_class_name": cls, "semantic_class": cls, "confidence": conf, "bbox_xyxy": list(b), "bbox_format": "xyxy_image_pixels_top_left_origin_y_down",
            "image_width": 1920, "image_height": 1080, "coordinate_system": "captured_1920x1080", "coordinate_space": "image_pixels", "matched_detection_id": 0,
            "is_predicted": predicted, "detection_validity": not predicted, "tracking_validity": valid}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def preds_by_frame(rows):
    out = {}
    for r in rows:
        out.setdefault(r["frame_index"], []).append(Pred(r["frame_index"], r["semantic_class"], tuple(r["bbox_xyxy"]), r["confidence"], r.get("track_id", r.get("detection_id")),
                                                          r.get("is_predicted", False), r.get("tracking_validity", True)))
    return out


B = (100.0, 100.0, 140.0, 200.0)


# ---------------------------------------------------------------------------------------------------------- parsing
def test_pedestrian_boxes_are_parsed_with_frame_box_and_attributes(tmp_path):
    gt = parse_pie_annotations(write_xml(tmp_path / "a.xml", [ped(1, "1_1_1b", [0, 1, 2])]))
    assert [b.frame for b in gt.boxes] == [0, 1, 2]
    b = gt.boxes[0]
    assert (b.label, b.eval_class, b.bbox, b.ident) == ("pedestrian", "pedestrian", B, "1_1_1b") and dict(b.attrs)["action"] == "standing"
    assert gt.frame_range == (0, 2) and gt.counters["supported_gt_boxes"] == 3


def test_vehicle_subtype_mapping_and_train_is_counted_not_supported(tmp_path):
    gt = parse_pie_annotations(write_xml(tmp_path / "a.xml", [veh(1, "v1", "car", [0]), veh(2, "v2", "bicycle", [0], (10, 10, 40, 60)), veh(3, "v3", "train", [0, 1])]))
    cls = sorted((b.ident, b.eval_class) for b in gt.boxes)
    assert cls == [("v1", "car"), ("v2", "bike"), ("v3", "unsupported_train"), ("v3", "unsupported_train")]
    assert gt.unsupported_counts == {"train": 2} and gt.counters["supported_gt_boxes"] == 2 and gt.vehicle_type_counts == {"bicycle": 1, "car": 1, "train": 2}


def test_unknown_vehicle_type_is_unsupported_and_never_defaulted(tmp_path):
    gt = parse_pie_annotations(write_xml(tmp_path / "a.xml", [veh(1, "v1", "bus", [0]), veh(2, "v2", "", [0])]))
    assert gt.counters["supported_gt_boxes"] == 0 and gt.unsupported_counts == {"<missing>": 1, "bus": 1}


def test_per_box_id_attribute_is_the_identity_not_the_cvat_track_id(tmp_path):
    t = track_xml(7, "pedestrian", [box_xml(0, B, attrs={"id": "A"}), box_xml(1, B, attrs={"id": "B"})])
    gt = parse_pie_annotations(write_xml(tmp_path / "a.xml", [t]))
    assert [b.ident for b in gt.boxes] == ["A", "B"] and all(b.cvat_track_id == "7" for b in gt.boxes)
    assert gt.id_report["cvat_tracks_with_multiple_ids"] == 1 and gt.id_report["ok"] is False          # inconsistency is reported, not hidden


def test_missing_shared_and_multi_class_identities_are_reported(tmp_path):
    xml = write_xml(tmp_path / "a.xml", [track_xml(1, "pedestrian", [box_xml(0, B, attrs={"action": "x"})]), ped(2, "P", [0], (300, 300, 340, 400)),
                                         ped(3, "P", [1], (300, 300, 340, 400)), veh(4, "P", "car", [2]),
                                         track_xml(5, "pedestrian", [box_xml(5, B, attrs={"id": "Q"}), box_xml(5, (500, 500, 540, 600), attrs={"id": "Q"})])])
    r = parse_pie_annotations(xml).id_report
    assert r["boxes_missing_id"] == 1 and r["ids_spanning_multiple_cvat_tracks"] == 1 and r["ids_with_multiple_classes"] == 1 and r["identity_duplicates_within_a_frame"] == 1


def test_outside_boxes_are_ignored_and_counted(tmp_path):
    t = track_xml(1, "pedestrian", [box_xml(0, B, attrs={"id": "A"}), box_xml(1, B, outside=1, attrs={"id": "A"})])
    gt = parse_pie_annotations(write_xml(tmp_path / "a.xml", [t]))
    assert [b.frame for b in gt.boxes] == [0] and gt.counters["ignored_outside"] == 1 and gt.raw_counts_by_label == {"pedestrian": 2}


def test_partially_occluded_boxes_are_retained_and_fully_occluded_follows_the_flag(tmp_path):
    xml = write_xml(tmp_path / "a.xml", [ped(1, "A", [0], occlusion=0), ped(2, "B", [0], occlusion=1, b=(300, 100, 340, 200)), ped(3, "C", [0], occlusion=2, b=(500, 100, 540, 200)),
                                         veh(4, "v", "car", [0])])
    g = parse_pie_annotations(xml)
    assert sorted(b.occlusion_level for b in g.boxes if b.eval_class == "pedestrian") == ["full", "none", "partial"]
    assert g.counters["supported_partially_occluded_retained"] == 1 and g.counters["supported_fully_occluded"] == 1
    g2 = parse_pie_annotations(xml, exclude_fully_occluded=True)
    assert g2.counters["supported_gt_boxes"] == g.counters["supported_gt_boxes"] - 1 and g2.counters["excluded_fully_occluded"] == 1


def test_ignored_labels_are_counted_and_not_supported(tmp_path):
    xml = write_xml(tmp_path / "a.xml", [track_xml(1, "traffic_light", [box_xml(0, B)]), track_xml(2, "sign", [box_xml(0, B)]), track_xml(3, "crosswalk", [box_xml(0, B)]), ped(4, "A", [0])])
    g = parse_pie_annotations(xml)
    assert g.nonoutside_counts_by_label == {"crosswalk": 1, "pedestrian": 1, "sign": 1, "traffic_light": 1} and g.counters["supported_gt_boxes"] == 1
    assert g.annotated_frames_any == [0]


def test_malformed_and_zero_area_boxes_are_discarded_with_counters(tmp_path):
    t = track_xml(1, "pedestrian", [box_xml(0, B, attrs={"id": "A"}), box_xml(1, B, attrs={"id": "A"}, raw_coords=(10, 10, 10, 50)),
                                    box_xml(2, B, attrs={"id": "A"}, raw_coords=(50, 10, 20, 50)), box_xml("x", B, attrs={"id": "A"}),
                                    box_xml(4, B, attrs={"id": "A"}, raw_coords=("a", 10, 20, 50)), box_xml(-3, B, attrs={"id": "A"}),
                                    box_xml(6, B, attrs={"id": "A"}, raw_coords=("nan", 10, 20, 50))])
    g = parse_pie_annotations(write_xml(tmp_path / "a.xml", [t]))
    assert [b.frame for b in g.boxes] == [0] and g.counters["discarded_zero_or_negative_area"] == 2 and g.counters["discarded_malformed"] == 4
    assert len(g.discarded_examples) == 6


def test_gt_boxes_are_clipped_to_the_image_and_fully_outside_ones_dropped(tmp_path):
    t = track_xml(1, "pedestrian", [box_xml(0, (0, 0, 1, 1), raw_coords=(-10, 100, 40, 200), attrs={"id": "A"}), box_xml(1, B, raw_coords=(2000, 100, 2100, 200), attrs={"id": "A"})])
    g = parse_pie_annotations(write_xml(tmp_path / "a.xml", [t]), image_size=(1920, 1080))
    assert [b.bbox for b in g.boxes] == [(0.0, 100.0, 40.0, 200.0)] and g.counters["clipped_to_image"] == 1 and g.counters["discarded_outside_image"] == 1


def test_missing_and_broken_xml_raise_clear_errors(tmp_path):
    with pytest.raises(PieEvaluationError, match="not found"):
        parse_pie_annotations(tmp_path / "nope.xml")
    bad = tmp_path / "bad.xml"
    bad.write_text("<annotations><track", encoding="utf-8")
    with pytest.raises(PieEvaluationError, match="cannot parse"):
        parse_pie_annotations(bad)


# ------------------------------------------------------------------------------------------------------- geometry
def test_iou_values():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou_xyxy((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0                                       # touching edges
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou_xyxy((0, 0, 10, 10), (2, 2, 8, 8)) == pytest.approx(0.36)                           # containment
    assert iou_xyxy((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0                                          # degenerate


def test_matching_is_one_to_one_class_aware_and_maximises_pairs_then_iou():
    gt = [("pedestrian", (0, 0, 10, 10))]
    pr = [("pedestrian", (0, 0, 10, 10)), ("pedestrian", (1, 0, 11, 10))]
    assert match_boxes(gt, pr, 0.5) == [(0, 0, 1.0)]                                              # the better of two overlapping predictions
    assert match_boxes(gt, [("car", (0, 0, 10, 10))], 0.5) == []                                   # class-aware
    # two GT, two predictions where taking the single best pair would lose a pair
    gt2 = [("car", (0, 0, 10, 10)), ("car", (6, 0, 16, 10))]
    pr2 = [("car", (3, 0, 13, 10)), ("car", (0, 0, 10, 10))]
    pairs = match_boxes(gt2, pr2, 0.3)
    assert len(pairs) == 2 and len({a for a, _, _ in pairs}) == 2 and len({b for _, b, _ in pairs}) == 2
    assert match_boxes(gt, [("pedestrian", (0, 0, 10, 4))], 0.5) == []                             # below threshold


def test_solver_python_fallback_matches_scipy():
    pytest.importorskip("scipy")
    rnd = random.Random(7)
    for n, m in [(1, 1), (2, 3), (3, 2), (5, 5), (4, 7), (7, 4)]:
        c = [[round(rnd.random(), 3) for _ in range(m)] for _ in range(n)]
        a, b = solve_assignment(c, prefer_scipy=True), solve_assignment(c, prefer_scipy=False)
        assert len(a) == len(b) == min(n, m)
        assert sum(c[i][j] for i, j in a) == pytest.approx(sum(c[i][j] for i, j in b))


# ------------------------------------------------------------------------------------------- detection metrics
def _gt(tmp_path, tracks):
    return parse_pie_annotations(write_xml(tmp_path / "gt.xml", tracks))


def test_precision_recall_f1_counts(tmp_path):
    # frame 0: 5 GT pedestrians; 2 detected exactly, 1 extra detection somewhere else -> TP 2, FP 1, FN 3
    boxes = [(100 + 100 * i, 100, 140 + 100 * i, 200) for i in range(5)]
    gt = _gt(tmp_path, [ped(i + 1, "P%d" % i, [0], boxes[i]) for i in range(5)])
    rows = [det_row(0, "pedestrian", boxes[0], did=0), det_row(0, "pedestrian", boxes[1], did=1), det_row(0, "pedestrian", (1000, 500, 1040, 600), did=2)]
    d = evaluate_detection(gt, preds_by_frame(rows), [0], 0.5)
    o = d["overall_supported_classes"]
    assert (o["tp"], o["fp"], o["fn"]) == (2, 1, 3)
    assert o["precision"] == pytest.approx(2 / 3) and o["recall"] == pytest.approx(2 / 5) and o["f1"] == pytest.approx(0.5) and o["mean_iou_matched"] == pytest.approx(1.0)
    assert d["per_class"]["pedestrian"]["tp"] == 2 and d["per_class"]["car"]["tp"] == 0 and d["per_class"]["car"]["precision"] is None
    assert d["counts"]["supported_gt_boxes"] == 5 and d["counts"]["supported_class_predictions_considered"] == 3 and d["counts"]["frames_with_applicable_gt"] == 1


def test_iou_thresholds_change_the_outcome(tmp_path):
    gt = _gt(tmp_path, [ped(1, "A", [0], (0, 0, 100, 100))])
    rows = [det_row(0, "pedestrian", (0, 0, 100, 65))]                                              # IoU 0.65
    p = preds_by_frame(rows)
    assert evaluate_detection(gt, p, [0], 0.5)["overall_supported_classes"]["tp"] == 1
    assert evaluate_detection(gt, p, [0], 0.75)["overall_supported_classes"]["tp"] == 0


def test_class_mismatch_is_fp_and_fn_and_shown_in_diagnostics(tmp_path):
    gt = _gt(tmp_path, [veh(1, "v", "car", [0])])
    d = evaluate_detection(gt, preds_by_frame([det_row(0, "bike", (300, 300, 400, 360))]), [0], 0.5)
    assert (d["overall_supported_classes"]["tp"], d["overall_supported_classes"]["fp"], d["overall_supported_classes"]["fn"]) == (0, 1, 1)
    assert d["diagnostics"]["class_mismatch_pairs_iou_ge_threshold(leftovers)"] == {"pred:bike->gt:car": 1}


def test_truck_and_bus_predictions_are_out_of_scope_not_false_positives(tmp_path):
    gt = _gt(tmp_path, [veh(1, "v", "car", [0])])
    rows = [det_row(0, "truck", (300, 300, 400, 360), did=0), det_row(0, "bus", (900, 300, 1000, 360), did=1)]
    d = evaluate_detection(gt, preds_by_frame(rows), [0], 0.5)
    assert d["overall_supported_classes"]["fp"] == 0 and d["overall_supported_classes"]["fn"] == 1
    assert d["counts"]["out_of_scope_predictions_by_class(truck_bus)"] == {"bus": 1, "truck": 1}
    assert d["diagnostics"]["class_mismatch_pairs_iou_ge_threshold(leftovers)"] == {"pred:truck->gt:car": 1}


def test_train_gt_is_never_evaluated_and_predictions_on_it_are_diagnosed(tmp_path):
    gt = _gt(tmp_path, [veh(1, "t", "train", [0], (300, 300, 800, 360))])
    d = evaluate_detection(gt, preds_by_frame([det_row(0, "car", (300, 300, 800, 360))]), [0], 0.5)
    assert d["overall_supported_classes"]["fn"] == 0 and d["counts"]["supported_gt_boxes"] == 0                      # train is not a missed car
    assert d["overall_supported_classes"]["fp"] == 1                                                                # but a car on a train is a car false positive
    assert d["diagnostics"]["unmatched_predictions_overlapping_unsupported_train_gt"] == {"car": 1}


def test_recall_is_stratified_by_occlusion_level(tmp_path):
    gt = _gt(tmp_path, [ped(1, "A", [0], occlusion=0), ped(2, "B", [0], occlusion=2, b=(500, 100, 540, 200))])
    d = evaluate_detection(gt, preds_by_frame([det_row(0, "pedestrian", B)]), [0], 0.5)
    assert d["recall_by_gt_occlusion_level"] == {"full": {"gt": 1, "matched": 0, "recall": 0.0}, "none": {"gt": 1, "matched": 1, "recall": 1.0}}


def test_empty_evaluation_returns_nulls_not_invented_numbers(tmp_path):
    gt = _gt(tmp_path, [ped(1, "A", [0])])
    o = evaluate_detection(gt, {}, [], 0.5)["overall_supported_classes"]
    assert (o["tp"], o["fp"], o["fn"]) == (0, 0, 0) and o["precision"] is None and o["recall"] is None and o["f1"] is None and o["mean_iou_matched"] is None


# ------------------------------------------------------------------------------------ frames, alignment, no-GT frames
def test_predictions_in_frames_without_annotations_are_not_false_positives(tmp_path):
    gt = _gt(tmp_path, [ped(1, "A", [0, 10])])
    rows = [det_row(0, "pedestrian", B), det_row(5, "pedestrian", (900, 900, 940, 1000)), det_row(10, "pedestrian", B)]
    fs = evaluation_frames(gt, "annotated_any", (0, 10))
    assert fs["frames"] == [0, 10]
    o = evaluate_detection(gt, preds_by_frame(rows), fs["frames"], 0.5)["overall_supported_classes"]
    assert (o["tp"], o["fp"], o["fn"]) == (2, 0, 0)                                                 # frame 5 (no GT frame) is never scored


def test_frames_outside_the_prediction_coverage_are_dropped_not_counted_as_misses(tmp_path):
    gt = _gt(tmp_path, [ped(1, "A", [0, 10, 20, 30])])
    fs = evaluation_frames(gt, "annotated_any", (0, 20))
    assert fs["frames"] == [0, 10, 20] and fs["dropped_outside_prediction_coverage"] == 1
    assert evaluation_frames(gt, "annotated_any", None)["frames"] == []


def test_supported_gt_only_policy_is_stricter_than_annotated_any(tmp_path):
    gt = _gt(tmp_path, [ped(1, "A", [0]), track_xml(2, "traffic_light", [box_xml(3, B)])])
    assert evaluation_frames(gt, "annotated_any", (0, 9))["frames"] == [0, 3] and evaluation_frames(gt, "supported_gt_only", (0, 9))["frames"] == [0]
    with pytest.raises(ConfigError):
        evaluation_frames(gt, "everything", (0, 9))


def test_frame_alignment_offset_is_applied_when_loading(tmp_path):
    rows = [det_row(11, "pedestrian", B)]
    p = write_jsonl(tmp_path / "d.jsonl", rows)
    assert list(load_predictions(p, video_id="video_0004", tracked=False).by_frame) == [11]
    assert list(load_predictions(p, video_id="video_0004", tracked=False, frame_offset=-1).by_frame) == [10]


def test_prediction_loader_filters_video_id_confidence_and_counts_bad_rows(tmp_path):
    rows = [det_row(0, "pedestrian", B), det_row(0, "pedestrian", B, vid="other"), det_row(1, "pedestrian", B, conf=0.05), det_row(2, "pedestrian", (5, 5, 5, 9))]
    p = tmp_path / "d.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n", encoding="utf-8")
    ld = load_predictions(p, video_id="video_0004", tracked=False, min_confidence=0.1)
    assert ld.counters["rows_loaded"] == 1 and ld.counters["rows_other_video_id"] == 1 and ld.counters["rows_below_min_confidence"] == 1
    assert ld.counters["rows_invalid_box"] == 1 and ld.counters["rows_malformed"] == 1 and ld.image_size == (1920, 1080)
    only_other = write_jsonl(tmp_path / "o.jsonl", [det_row(0, "pedestrian", B, vid="other")])
    with pytest.raises(PieEvaluationError, match="no usable rows"):
        load_predictions(only_other, video_id="video_0004", tracked=False)


# ----------------------------------------------------------------------------------------------- tracking metrics
def _track_scenario(tmp_path, gt_frames, pred_spans, b=B):
    gt = _gt(tmp_path, [ped(1, "A", gt_frames, b)])
    rows = [trk_row(f, "pedestrian", b, tid) for tid, (lo, hi) in pred_spans.items() for f in range(lo, hi + 1)]
    return gt, preds_by_frame(rows)


def test_idf1_uses_one_global_identity_assignment_not_framewise_matches(tmp_path):
    gt, rows = _track_scenario(tmp_path, range(10), {1: (0, 6), 2: (7, 9)})
    m = evaluate_tracking(gt, rows, list(range(10)), 0.5)
    assert m["matches"] == 10 and m["false_positives"] == 0 and m["false_negatives"] == 0                      # frame-wise everything matches ...
    assert (m["idtp"], m["idfp"], m["idfn"]) == (7, 3, 3)                                                       # ... but only one identity can be credited
    assert m["id_precision"] == pytest.approx(0.7) and m["id_recall"] == pytest.approx(0.7) and m["idf1"] == pytest.approx(0.7)
    assert m["id_switches"] == 1 and m["mota"] == pytest.approx(0.9) and m["motp_mean_iou"] == pytest.approx(1.0)
    assert m["gt_identities"] == 1 and m["predicted_identities"] == 2 and m["matched_identities_in_global_assignment"] == 1


def test_perfect_tracking_scores_one(tmp_path):
    gt, rows = _track_scenario(tmp_path, range(5), {9: (0, 4)})
    m = evaluate_tracking(gt, rows, list(range(5)), 0.5)
    assert m["idf1"] == 1.0 and m["mota"] == 1.0 and m["id_switches"] == 0 and m["gt_identities_with_any_frame_match"] == 1


def test_id_switch_persists_across_unmatched_frames_and_gaps(tmp_path):
    gt, rows = _track_scenario(tmp_path, range(6), {1: (0, 1), 2: (4, 5)})                                       # frames 2-3 have no prediction
    m = evaluate_tracking(gt, rows, list(range(6)), 0.5)
    assert m["id_switches"] == 1 and m["false_negatives"] == 2 and m["false_positives"] == 0 and m["mota"] == pytest.approx(1 - 3 / 6)


def test_annotation_gaps_do_not_create_extra_switches_or_misses(tmp_path):
    gt, rows = _track_scenario(tmp_path, [0, 1, 10, 11], {1: (0, 11)})                                           # GT annotated at 4 frames only
    m = evaluate_tracking(gt, rows, [0, 1, 10, 11], 0.5)
    assert m["id_switches"] == 0 and m["false_negatives"] == 0 and m["false_positives"] == 0 and m["gt_boxes"] == 4 and m["pred_boxes"] == 4


def test_two_identities_swapping_predicted_ids_are_counted(tmp_path):
    ba, bb = (100, 100, 140, 200), (400, 100, 440, 200)
    gt = _gt(tmp_path, [ped(1, "A", range(6), ba), ped(2, "B", range(6), bb)])
    rows = []
    for f in range(6):
        swap = f >= 3
        rows += [trk_row(f, "pedestrian", ba, 2 if swap else 1), trk_row(f, "pedestrian", bb, 1 if swap else 2)]
    m = evaluate_tracking(gt, preds_by_frame(rows), list(range(6)), 0.5)
    assert m["id_switches"] == 2 and m["idtp"] == 6 and m["idf1"] == pytest.approx(0.5)


def test_tracking_is_class_aware_and_ignores_out_of_scope_rows(tmp_path):
    gt = _gt(tmp_path, [veh(1, "v", "car", [0, 1])])
    rows = preds_by_frame([trk_row(0, "bike", (300, 300, 400, 360), 1), trk_row(1, "truck", (300, 300, 400, 360), 2)])
    m = evaluate_tracking(gt, rows, [0, 1], 0.5)
    assert m["matches"] == 0 and m["false_negatives"] == 2 and m["false_positives"] == 1 and m["excluded"]["out_of_scope_class_rows(truck_bus_etc)"] == 1


def test_gt_boxes_without_id_or_with_duplicate_identity_are_excluded_from_tracking_only(tmp_path):
    no_id = track_xml(1, "pedestrian", [box_xml(0, B, attrs={"action": "x"})])
    dup = track_xml(2, "pedestrian", [box_xml(0, (500, 100, 540, 200), attrs={"id": "Q"}), box_xml(0, (700, 100, 740, 200), attrs={"id": "Q"})])
    gt = _gt(tmp_path, [no_id, dup])
    m = evaluate_tracking(gt, {}, [0], 0.5)
    assert m["excluded"]["gt_boxes_without_id"] == 1 and m["excluded"]["gt_duplicate_identity_in_frame"] == 1 and m["gt_boxes"] == 1
    d = evaluate_detection(gt, {}, [0], 0.5)
    assert d["counts"]["supported_gt_boxes"] == 3                                                                 # all three still count for detection


def test_tracking_with_no_gt_returns_nulls_with_an_explanation(tmp_path):
    gt = _gt(tmp_path, [track_xml(1, "sign", [box_xml(0, B)])])
    m = evaluate_tracking(gt, preds_by_frame([trk_row(0, "pedestrian", B, 1)]), [0], 0.5)
    assert m["mota"] is None and m["idf1"] is None and m["id_recall"] is None and m["id_precision"] == 0.0 and m["notes"]


def test_tracking_validity_and_is_predicted_rows_are_filtered_and_counted_not_dropped_silently(tmp_path):
    gt = _gt(tmp_path, [ped(1, "A", [0, 1])])
    rows = [trk_row(0, "pedestrian", B, 1), trk_row(1, "pedestrian", B, 1, predicted=True, valid=False)]
    p = write_jsonl(tmp_path / "t.jsonl", rows)
    out = run_evaluation(_xml_for(tmp_path), write_jsonl(tmp_path / "d.jsonl", [det_row(0, "pedestrian", B)]), p, tmp_path / "out")
    t = out["tracking"]
    assert t["predictions"]["rows_is_predicted_true"] == 1 and t["predictions"]["rows_tracking_validity_false"] == 1 and t["predictions"]["rows_loaded"] == 2
    prim = t["results"]["primary"]["variants"]
    assert prim["primary_tracking_validity_true"]["0.50"]["pred_boxes"] == 1                                        # invalid/predicted row excluded from the primary variant
    assert prim["diagnostic_all_rows"]["0.50"]["pred_boxes"] == 2                                                   # but visible in the diagnostic variant


def _xml_for(tmp_path):
    return write_xml(tmp_path / "pie.xml", [ped(1, "A", [0, 1])])


# ------------------------------------------------------------------------------------------- end to end + determinism
def _make_inputs(tmp_path):
    xml = write_xml(tmp_path / "pie.xml", [ped(1, "A", range(0, 6)), veh(2, "V", "car", range(0, 6)), veh(3, "T", "train", range(0, 6), (900, 100, 1500, 300)),
                                           track_xml(4, "traffic_light", [box_xml(0, B)])])
    drows = [det_row(f, "pedestrian", B, did=0) for f in range(6)] + [det_row(f, "car", (300, 300, 400, 360), did=1) for f in range(6)]
    trows = [trk_row(f, "pedestrian", B, 1) for f in range(6)] + [trk_row(f, "car", (300, 300, 400, 360), 2) for f in range(6)]
    return xml, write_jsonl(tmp_path / "detections.jsonl", drows), write_jsonl(tmp_path / "tracked_agents.jsonl", trows)


def test_end_to_end_writes_the_four_reports_with_a_complete_manifest(tmp_path):
    xml, d, t = _make_inputs(tmp_path)
    out = tmp_path / "out"
    r = run_evaluation(xml, d, t, out)
    assert sorted(p.name for p in out.iterdir()) == ["pie_detection_metrics.json", "pie_evaluation_manifest.json", "pie_evaluation_summary.md", "pie_tracking_metrics.json"]
    m = json.loads((out / "pie_evaluation_manifest.json").read_text())
    for k in ("evaluator_version", "created_utc", "inputs", "video_id", "annotation_frame_range", "evaluated_frame_count", "iou_thresholds", "class_mapping",
              "occlusion_policy", "unsupported_class_policy", "matching_method", "software", "limitations"):
        assert k in m
    assert m["video_id"] == "video_0004" and m["iou_thresholds"] == [0.5, 0.75] and m["annotation_frame_range"] == [0, 5]
    assert all(len(v["sha256"]) == 64 for k, v in m["inputs"].items() if v) and m["class_mapping"]["vehicle:type=train"] == "unsupported_train"
    det = json.loads((out / "pie_detection_metrics.json").read_text())
    o = det["results"]["primary"]["by_iou_threshold"]["0.50"]["overall_supported_classes"]
    assert (o["tp"], o["fp"], o["fn"]) == (12, 0, 0) and det["ground_truth"]["unsupported_gt_counts"] == {"train": 6}
    trk = json.loads((out / "pie_tracking_metrics.json").read_text())
    tm = trk["results"]["primary"]["variants"]["primary_tracking_validity_true"]["0.75"]
    assert tm["idf1"] == 1.0 and tm["mota"] == 1.0 and tm["gt_identities"] == 2
    md = (out / "pie_evaluation_summary.md").read_text()
    assert "single-video" in md and "production-ready" in md and "unsupported_train" in md and "Task 4 exit criteria" in md and "mAP" in md
    assert r["output_dir"] == str(out)


def test_results_are_deterministic_across_repeated_runs(tmp_path):
    xml, d, t = _make_inputs(tmp_path)
    run_evaluation(xml, d, t, tmp_path / "o1")
    run_evaluation(xml, d, t, tmp_path / "o2")
    for name in ("pie_detection_metrics.json", "pie_tracking_metrics.json", "pie_evaluation_summary.md"):
        assert (tmp_path / "o1" / name).read_bytes() == (tmp_path / "o2" / name).read_bytes()
    m1, m2 = (json.loads((tmp_path / o / "pie_evaluation_manifest.json").read_text()) for o in ("o1", "o2"))
    m1.pop("created_utc"); m2.pop("created_utc")
    assert m1 == m2


def test_pure_python_solver_gives_the_same_reports_as_scipy(tmp_path):
    pytest.importorskip("scipy")
    xml, d, t = _make_inputs(tmp_path)
    run_evaluation(xml, d, t, tmp_path / "a", prefer_scipy=True)
    run_evaluation(xml, d, t, tmp_path / "b", prefer_scipy=False)
    for name in ("pie_detection_metrics.json", "pie_tracking_metrics.json"):
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()


def test_invalid_configuration_and_unsafe_output_directory_are_rejected(tmp_path):
    xml, d, t = _make_inputs(tmp_path)
    for bad in ([], [0.0], [1.5], [-0.5]):
        with pytest.raises(ConfigError):
            run_evaluation(xml, d, t, tmp_path / "x", iou_thresholds=bad)
    with pytest.raises(ConfigError):
        run_evaluation(xml, d, t, tmp_path / "x", frame_policy="all")
    used = tmp_path / "used"
    used.mkdir()
    (used / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ConfigError, match="refusing to overwrite"):
        run_evaluation(xml, d, t, used)
    assert (used / "keep.txt").read_text() == "keep"
    with pytest.raises(PieEvaluationError):
        run_evaluation(tmp_path / "missing.xml", d, t, tmp_path / "y")


def test_prediction_coverage_from_the_run_manifest_limits_the_frames(tmp_path):
    xml, d, t = _make_inputs(tmp_path)
    mf = tmp_path / "run_manifest.json"
    mf.write_text(json.dumps({"results": {"first_frame_index": 0, "last_frame_index": 2, "frames_seen": 3}, "components": {"detector": {"name": "x", "parameters": {"weights_sha256": "ab"}}}}), encoding="utf-8")
    # rows exist up to frame 5, so the manifest range must NOT shrink coverage below the rows; union is used
    r = run_evaluation(xml, d, t, tmp_path / "o", run_manifest=mf)
    assert r["manifest"]["prediction_run"]["frames_seen"] == 3 and r["detection"]["results"]["primary"]["frames"]["count"] == 6


def test_frame_alignment_sensitivity_flags_a_shifted_prediction_set(tmp_path):
    xml = write_xml(tmp_path / "pie.xml", [ped(1, "A", [0, 1, 2, 3, 4, 5], B)])
    moving = lambda f: (100 + 30 * f, 100, 140 + 30 * f, 200)                                                  # the pedestrian moves, so a shift breaks the match
    xml = write_xml(tmp_path / "pie.xml", [track_xml(1, "pedestrian", [box_xml(f, moving(f), attrs={"id": "A"}) for f in range(6)])])
    d = write_jsonl(tmp_path / "d.jsonl", [det_row(f + 1, "pedestrian", moving(f)) for f in range(6)])           # predictions are one frame late
    t = write_jsonl(tmp_path / "t.jsonl", [trk_row(f + 1, "pedestrian", moving(f), 1) for f in range(6)])
    r = run_evaluation(xml, d, t, tmp_path / "o")
    al = r["detection"]["results"]["frame_alignment_sensitivity_at_lowest_iou_threshold"]["by_offset"]
    assert al["-1"]["f1"] > al["+0"]["f1"]


def test_cli_exit_codes_and_success(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("evaluate_pie_perception", str(REPO / "scripts" / "evaluate_pie_perception.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    xml, d, t = _make_inputs(tmp_path)
    assert cli.main(["--annotations", str(xml), "--detections", str(d), "--tracked-agents", str(t), "--output-dir", str(tmp_path / "out")]) == 0
    assert "IoU 0.50" in capsys.readouterr().out and (tmp_path / "out" / "pie_evaluation_summary.md").exists()
    assert cli.main(["--annotations", str(xml), "--detections", str(d), "--tracked-agents", str(t), "--output-dir", str(tmp_path / "out")]) == 2        # not empty now
    assert cli.main(["--annotations", str(tmp_path / "no.xml"), "--detections", str(d), "--tracked-agents", str(t), "--output-dir", str(tmp_path / "o2")]) == 2
    assert cli.main(["--annotations", str(xml), "--detections", str(d), "--tracked-agents", str(t), "--output-dir", str(tmp_path / "o3"), "--video-id", "other"]) == 2
    assert "no usable rows" in capsys.readouterr().err
