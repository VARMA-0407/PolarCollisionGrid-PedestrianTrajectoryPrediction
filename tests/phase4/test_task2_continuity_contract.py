"""Track continuity, gaps, duplicates, ordering, identity scope and the Phase 4 record contract."""

import pytest

from pedsafety.errors import ConfigError
from pedsafety.image_kinematics import process_causal_records, validate_phase3_record
from pedsafety.track_continuity import continuity_report
from task2_helpers import cfg, deep, lines, rec, track


def run_all(records, **over):
    c = cfg(**over)
    res = process_causal_records(records, lines(records), c)
    return res, continuity_report(records, res, c)


def frames_track(frames, dx=3.0, track_id="1", **kw):
    return track([(f, f / 30.0, 100.0 + dx * f, 500.0) for f in frames], track_id=track_id, **kw)


def test_consecutive_observations_have_no_gaps():
    res, rep = run_all(frames_track(range(6)))
    assert rep["summary"]["n_gaps"] == 0 and rep["per_track"][0]["missing_frames_total"] == 0
    assert all(r["frame_gap_before"] in (None, 0) for r in res.rows)


def test_one_frame_gap_breaks_derivatives_and_is_reported():
    res, rep = run_all(frames_track([0, 1, 3, 4]))
    r = res.rows
    assert r[2]["velocity_state"] == "gap_before" and r[2]["velocity_x"] is None and r[2]["frame_gap_before"] == 1
    assert r[3]["velocity_state"] == "valid" and r[3]["acceleration_state"] == "previous_velocity_unavailable"
    g = rep["per_track"][0]["gaps"]
    assert g == [{"after_frame": 1, "before_frame": 3, "missing_frames": 1, "elapsed_s": pytest.approx(2 / 30.0)}]
    assert rep["summary"]["gap_missing_frames_histogram"] == {"1": 1}


def test_long_gap():
    res, rep = run_all(frames_track([0, 1, 100, 101]))
    assert res.rows[2]["velocity_state"] == "gap_before" and res.rows[2]["frame_gap_before"] == 98
    assert rep["per_track"][0]["max_gap_frames"] == 98
    assert rep["per_track"][0]["gaps"][0]["elapsed_s"] == pytest.approx(99 / 30.0)


def test_gap_convention_separates_gap_boundaries_from_track_span():
    """Regression for the Task 2 review: the reported longest gap must use the observations on either side of the gap
    (missing = frame difference - 1), not the track's first/last frames. Layout mirrors PIE track 8431: observed
    3994-4009, then 4111, 4114, 4115 (gaps of 101 and 2 frames; track span 3994-4115)."""
    frames = list(range(3994, 4010)) + [4111, 4114, 4115]
    res, rep = run_all(frames_track(frames, dx=0.01))           # small step keeps every footpoint inside the 1920-px raster
    assert all(r["position_state"] == "ok" for r in res.rows)
    t = rep["per_track"][0]
    assert (t["first_frame"], t["last_frame"], t["span_frames"]) == (3994, 4115, 122)
    assert t["gaps"][0]["after_frame"] == 4009 and t["gaps"][0]["before_frame"] == 4111
    assert t["gaps"][0]["missing_frames"] == 101                                   # 4111 - 4009 - 1, not 4115 - 3994
    assert t["gaps"][0]["elapsed_s"] == pytest.approx(102 / 30.0)
    assert t["max_gap_frames"] == 101 and t["missing_frames_total"] == 103 and t["n_gaps"] == 2
    assert rep["summary"]["gap_missing_frames_histogram"] == {"2": 1, "101": 1}
    gap_rows = [r for r in res.rows if r["velocity_state"] == "gap_before"]
    assert [(r["frame_index"], r["frame_gap_before"]) for r in gap_rows] == [(4111, 101), (4114, 2)]


def test_gap_bridging_is_refused_by_configuration():
    with pytest.raises(ConfigError, match="max_derivative_gap_frames must be 0"):
        cfg(max_derivative_gap_frames=1)


def test_duplicate_timestamp_with_new_frame():
    recs = frames_track([0, 1, 2, 3])
    recs[2]["timestamp"] = recs[1]["timestamp"]
    res, rep = run_all(recs)
    assert res.rows[2]["velocity_state"] == "rejected_timestamp_not_monotonic"
    assert rep["per_track"][0]["n_timestamp_not_monotonic"] == 1


def test_duplicate_frame_observation_quarantines_all_copies():
    recs = frames_track([0, 1, 2])
    recs.insert(2, deep([recs[1]])[0])
    res, rep = run_all(recs)
    states = [r["velocity_state"] for r in res.rows]
    assert states == ["insufficient_samples", "rejected_duplicate_observation", "rejected_duplicate_observation", "gap_before"]
    assert rep["per_track"][0]["n_duplicate_rows"] == 2


def test_non_monotonic_frame_index():
    res, rep = run_all(frames_track([0, 1, 3, 2, 4]))
    s = [r["velocity_state"] for r in res.rows]
    assert s == ["insufficient_samples", "valid", "gap_before", "rejected_out_of_order_frame", "valid"]
    assert rep["per_track"][0]["n_out_of_order_rows"] == 1


def test_short_track():
    res, rep = run_all(frames_track([5]) + frames_track(range(10), track_id="2"))
    assert res.rows[0]["velocity_state"] == "insufficient_samples"
    short = {d["valid_positions_lt"]: d["n_tracks"] for d in rep["summary"]["short_tracks"]}
    assert short[2] == 1 and short[10] == 1 and short[30] == 2


def test_separate_sequences_are_independent():
    a = frames_track(range(3), seq="pie:video_0004", dx=3.0)
    b = frames_track(range(3), seq="pie:video_0005", dx=6.0)
    recs = [a[0], b[0], a[1], b[1], a[2], b[2]]
    res, rep = run_all(recs)
    assert res.rows[4]["velocity_x"] == pytest.approx(90.0) and res.rows[5]["velocity_x"] == pytest.approx(180.0)
    assert rep["summary"]["n_sequences"] == 2 and rep["summary"]["n_tracks"] == 2


def test_same_track_id_in_two_sequences_is_never_joined():
    recs = frames_track(range(3))
    recs[1]["sequence_id"] = "pie:video_0099"                    # same track_id string, different sequence
    res, rep = run_all(recs)
    assert all(r["velocity_state"] == "rejected_track_sequence_conflict" for r in res.rows)
    assert rep["summary"]["grouping_and_stream_issues"]["track_sequence_conflict"] == 3


def test_separate_track_ids_interleaved():
    a, b = frames_track(range(3), track_id="1", dx=3.0), frames_track(range(3), track_id="2", dx=-3.0)
    res, _ = run_all([a[0], b[0], a[1], b[1], a[2], b[2]])
    assert res.rows[4]["velocity_x"] == pytest.approx(90.0) and res.rows[5]["velocity_x"] == pytest.approx(-90.0)


def test_fragment_is_reported_not_merged():
    a = frames_track(range(5), track_id="7")
    b = track([(f, f / 30.0, 100.0 + 3.0 * f, 500.0) for f in range(7, 10)], track_id="8")
    res, rep = run_all(a + b)
    assert {r["track_id"] for r in res.rows} == {"pie:video_0004:7", "pie:video_0004:8"}          # ids preserved
    assert res.rows[5]["velocity_state"] == "insufficient_samples"                                # nothing carried across
    fp = rep["anomaly_indicators"]["possible_fragment_pairs"]
    assert fp["items"][0]["earlier_track_id"] == "pie:video_0004:7" and fp["items"][0]["frame_gap"] == 3
    assert "not merged" in fp["items"][0]["label"]
    assert {d["max_gap_frames"]: d["n_pairs"] for d in fp["table"]} == {1: 0, 5: 1, 15: 1, 30: 1}
    assert rep["anomaly_indicators"]["confirmed_id_switches"] is None


def test_fragment_candidates_require_same_class_and_proximity():
    a = frames_track(range(5), track_id="7")
    car = track([(f, f / 30.0, 100.0 + 3.0 * f, 500.0) for f in range(7, 10)], track_id="8", cls="car")        # same place, other class
    far = track([(f, f / 30.0, 900.0 + 3.0 * f, 500.0) for f in range(7, 10)], track_id="9")                    # same class, 13 box heights away
    _, rep = run_all(a + car + far)
    assert rep["anomaly_indicators"]["possible_fragment_pairs"]["items"] == []


def test_suspicious_jump_is_an_indicator_only():
    recs = frames_track(range(4))
    recs[2] = rec(track="1", frame=2, fx=400.0, fy=500.0, h=150.0)            # 294 px jump, box height 60 -> 150
    recs[3] = rec(track="1", frame=3, fx=403.0, fy=500.0, h=150.0)
    res, rep = run_all(recs)
    sj = rep["anomaly_indicators"]["suspicious_jumps_consecutive_frames"]
    d = {x["displacement_over_box_height_gt"]: x["n_pairs"] for x in sj["displacement_table"]}
    assert d[2.0] == 1 and d[0.25] == 1                                       # 294 / 105 = 2.8
    assert sj["possible_id_switch_indicators"]["n_pairs"] == 1
    assert res.rows[2]["velocity_state"] == "valid"                            # the data are not altered
    assert rep["anomaly_indicators"]["confirmed_id_switches"] is None


@pytest.mark.parametrize("field,value,reason", [
    ("image_coordinate_system", "captured_640x360", "coordinate_system_mismatch"),
    ("image_coordinate_unit", "m", "image_unit_not_pixel"),
    ("world_x", 3.0, "world_fields_not_null_or_world_valid"),
    ("world_position_valid", True, "world_fields_not_null_or_world_valid"),
    ("observation_valid", False, "observation_valid_not_true"),
    ("interpolation_flag", True, "interpolated_input_not_supported"),
    ("bbox_xyxy", [110.0, 440.0, 90.0, 500.0], "invalid_bbox"),
    ("bbox_xyxy", [1900.0, 440.0, 1930.0, 500.0], "bbox_outside_image"),
    ("footpoint", [101.0, 500.0], "footpoint_missing_or_not_bottom_centre"),
    ("image_y", 499.0, "image_xy_missing_or_not_footpoint"),
    ("image_x", None, "image_xy_missing_or_not_footpoint"),
])
def test_contract_violations_are_marked_not_repaired(field, value, reason):
    r = rec()
    r[field] = value
    c = validate_phase3_record(r, cfg())
    assert c.placeable and c.position is None and c.reason == reason


@pytest.mark.parametrize("field,value", [("track_id", ""), ("sequence_id", None), ("frame_index", -1), ("frame_index", 1.0),
                                         ("frame_index", True), ("timestamp", None), ("timestamp", float("inf"))])
def test_unplaceable_records_are_quarantined(field, value):
    r = rec()
    r[field] = value
    res = process_causal_records([r], [1], cfg())
    assert res.rows == [] and len(res.quarantined) == 1


def test_border_touching_box_is_flagged():
    c = validate_phase3_record(rec(fy=1080.0), cfg())
    assert c.position is not None and c.touches_border is True
    assert validate_phase3_record(rec(), cfg()).touches_border is False
