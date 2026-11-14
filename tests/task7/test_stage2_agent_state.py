import dataclasses
import json
import math
from pathlib import Path

import pytest

import pedsafety.agent_state as agent_state_module
import pedsafety.validation as validation_module
from pedsafety.agent_state import (
    AgentStateConfig, DEFAULT_CLASS_MAP, VelocityContract, build_agent_states, classify_label,
)
from pedsafety.errors import ConfigError, PedsafetyError
from pedsafety.io.base import RawObservation
from pedsafety.io.csv_adapter import CsvAnnotationAdapter
from pedsafety.io.synthetic import generate_scene, synthetic_csv_config, write_scene_csv
from pedsafety.schema import collect_violations, schema_sha256
from pedsafety.validation import (
    BoundingBoxError, ObservationIssueCode as C, ObservationValidationError,
    bbox_center, bbox_footpoint, bbox_problem, validate_bbox_xyxy,
)


def obs(**kw):
    base = dict(
        source_dataset="TESTSET", sequence_id="s1", scene_id="sc1", track_id="t1", frame_index=0,
        timestamp=0.0, source_label="pedestrian", bbox_xyxy=(10.0, 20.0, 30.0, 60.0), pos_x=20.0, pos_y=60.0,
        position_semantics="image_pixel", position_unit="pixel", position_coordinate_system="camera_image",
        source_line=2,
    )
    base.update(kw)
    return RawObservation(**base)


def codes(result):
    return [i.code for i in result.report.issues]


def one(o, cfg=None):
    return build_agent_states([o], cfg)


# ------------------------------------------------------- 1-3 valid records, classes
def test_valid_pedestrian_record_is_schema_valid():
    r = one(obs())
    assert r.report.ok and len(r.records) == 1
    assert collect_violations(r.records[0]) == []
    rec = r.records[0]
    assert (rec["agent_class"], rec["source_label"], rec["observation_valid"], rec["interpolation_flag"]) == \
        ("pedestrian", "pedestrian", True, False)


def test_valid_vehicle_record_is_schema_valid():
    r = one(obs(source_label="car", bbox_xyxy=(0.0, 0.0, 90.0, 60.0), pos_x=45.0, pos_y=60.0))
    assert collect_violations(r.records[0]) == [] and r.records[0]["agent_class"] == "vehicle"


@pytest.mark.parametrize("label,expected", [
    ("pedestrian", "pedestrian"), ("car", "vehicle"), ("truck", "vehicle"), ("bus", "vehicle"),
    ("bike", "bicyclist"), ("bicycle", "bicyclist"), ("motorcycle", "motorcyclist"),
])
def test_each_supported_class_maps_explicitly(label, expected):
    r = one(obs(source_label=label))
    assert r.records[0]["agent_class"] == expected and r.records[0]["source_label"] == label
    assert r.provenance[0].mapped_agent_class == expected


def test_label_case_and_space_normalised_for_lookup_but_source_label_preserved():
    r = one(obs(source_label=" Car "))
    assert r.records[0]["agent_class"] == "vehicle" and r.records[0]["source_label"] == " Car "


# ----------------------------------------------------------- 4 unknown labels
def test_unknown_label_is_not_silently_mapped():
    r = one(obs(source_label="scooter"))
    assert r.records == [] and codes(r) == [C.UNMAPPED_CLASS]
    assert classify_label("scooter", AgentStateConfig()) is None


def test_missing_label_is_an_issue_not_a_default():
    for label in (None, "", "  "):
        r = one(obs(source_label=label))
        assert r.records == [] and codes(r) == [C.MISSING_CLASS_LABEL]


def test_custom_class_map_is_explicit_and_validated_against_schema_enum():
    cfg = AgentStateConfig(class_map={"scooter": "other", "ped": "pedestrian"})
    assert one(obs(source_label="scooter"), cfg).records[0]["agent_class"] == "other"
    assert one(obs(source_label="car"), cfg).records == []  # default map is replaced, not merged
    with pytest.raises(ConfigError, match="not a schema agent_class"):
        AgentStateConfig(class_map={"car": "truck"})
    with pytest.raises(ConfigError, match="conflicting"):
        AgentStateConfig(class_map={"Car": "vehicle", "car ": "other"})
    with pytest.raises(ConfigError, match="non-empty"):
        AgentStateConfig(class_map={" ": "other"})


# ------------------------------------------------------------ 5-6 box features
def test_bbox_center_and_footpoint_are_correct():
    rec = one(obs(bbox_xyxy=(10.0, 20.0, 30.0, 60.0))).records[0]
    assert rec["bbox_center"] == [20.0, 40.0] and rec["footpoint"] == [20.0, 60.0]
    rec = one(obs(bbox_xyxy=(0.0, 0.0, 5.0, 10.0), pos_x=2.5, pos_y=10.0)).records[0]
    assert rec["bbox_center"] == [2.5, 5.0] and rec["footpoint"] == [2.5, 10.0]
    assert bbox_center((1.0, 2.0, 4.0, 8.0)) == (2.5, 5.0) and bbox_footpoint((1.0, 2.0, 4.0, 8.0)) == (2.5, 8.0)


def test_missing_bbox_is_not_fabricated():
    rec = one(obs(bbox_xyxy=None)).records[0]
    assert rec["bbox_xyxy"] is rec["bbox_center"] is rec["footpoint"] is None
    assert (rec["image_x"], rec["image_y"]) == (20.0, 60.0)
    assert collect_violations(rec) == []


# ------------------------------------------------------- 7-9 invalid boxes
@pytest.mark.parametrize("bbox,code", [
    ((30.0, 20.0, 10.0, 60.0), C.BBOX_INVERTED),
    ((10.0, 60.0, 30.0, 20.0), C.BBOX_INVERTED),
    ((10.0, 20.0, 10.0, 60.0), C.BBOX_DEGENERATE),
    ((10.0, 20.0, 30.0, 20.0), C.BBOX_DEGENERATE),
    ((float("nan"), 20.0, 30.0, 60.0), C.BBOX_NON_FINITE),
    ((10.0, 20.0, float("inf"), 60.0), C.BBOX_NON_FINITE),
    ((10.0, float("-inf"), 30.0, 60.0), C.BBOX_NON_FINITE),
    ((10.0, 20.0, 30.0), C.BBOX_MALFORMED),
    ((10.0, 20.0, 30.0, 60.0, 1.0), C.BBOX_MALFORMED),
    (("a", 20.0, 30.0, 60.0), C.BBOX_MALFORMED),
    ((True, 20.0, 30.0, 60.0), C.BBOX_MALFORMED),
    ("1234", C.BBOX_MALFORMED),
    (5, C.BBOX_MALFORMED),
])
def test_invalid_boxes_rejected_never_reordered(bbox, code):
    r = one(obs(bbox_xyxy=bbox))
    assert r.records == [] and codes(r) == [code]
    assert bbox_problem(bbox)[0] == code
    with pytest.raises(BoundingBoxError) as ei:
        validate_bbox_xyxy(bbox)
    assert isinstance(ei.value, ValueError) and ei.value.code == code


def test_valid_box_helper_returns_floats():
    assert validate_bbox_xyxy([1, 2, 3, 4]) == (1.0, 2.0, 3.0, 4.0)


# -------------------------------------------------------------- 10 confidence
@pytest.mark.parametrize("conf", [1.5, -0.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_confidence_rejected(conf):
    r = one(obs(detection_confidence=conf))
    assert r.records == [] and codes(r) == [C.INVALID_CONFIDENCE]


@pytest.mark.parametrize("conf", [0.0, 1.0, 0.37, None])
def test_valid_confidence_boundaries(conf):
    assert one(obs(detection_confidence=conf)).records[0]["detection_confidence"] == conf


# ------------------------------------------- scalars: frame, timestamp, identifiers
@pytest.mark.parametrize("kw,code", [
    (dict(frame_index=-1), C.INVALID_FRAME_INDEX),
    (dict(frame_index=1.5), C.INVALID_FRAME_INDEX),
    (dict(frame_index=True), C.INVALID_FRAME_INDEX),
    (dict(timestamp=float("nan")), C.INVALID_TIMESTAMP),
    (dict(timestamp=float("inf")), C.INVALID_TIMESTAMP),
    (dict(timestamp=-0.5), C.INVALID_TIMESTAMP),
    (dict(timestamp="1.0"), C.INVALID_TIMESTAMP),
    (dict(track_id=""), C.INVALID_IDENTIFIER),
    (dict(sequence_id="  "), C.INVALID_IDENTIFIER),
    (dict(scene_id=None), C.INVALID_IDENTIFIER),
    (dict(source_dataset=""), C.INVALID_IDENTIFIER),
    (dict(pos_x=float("nan")), C.NON_FINITE_VALUE),
    (dict(pos_y=None), C.INCOMPLETE_POSITION),
])
def test_scalar_validation(kw, code):
    r = one(obs(**kw))
    assert r.records == [] and code in codes(r)


# ------------------------------------------------- 11-12 identity policy
def test_identity_is_sequence_scoped_and_deterministic():
    a = one(obs()).records[0]
    b = one(obs()).records[0]
    assert a == b
    assert a["track_id"] == "testset:s1:t1" and a["sequence_id"] == "testset:s1" and a["scene_id"] == "testset:sc1"
    assert a["agent_id"] is None and a["source_dataset"] == "TESTSET"


def test_agent_id_policy_source_track_id_is_opt_in():
    rec = one(obs(), AgentStateConfig(agent_id_policy="source_track_id")).records[0]
    assert rec["agent_id"] == "t1" and rec["track_id"] == "testset:s1:t1"
    with pytest.raises(ConfigError, match="agent_id_policy"):
        AgentStateConfig(agent_id_policy="global")


def test_same_track_id_in_different_sequences_is_distinct_and_not_a_duplicate():
    r = build_agent_states([obs(sequence_id="s1"), obs(sequence_id="s2", scene_id="sc2")])
    assert r.report.ok and len(r.records) == 2
    assert r.records[0]["track_id"] != r.records[1]["track_id"]


def test_same_track_in_different_datasets_is_distinct():
    r = build_agent_states([obs(source_dataset="A"), obs(source_dataset="B")])
    assert r.report.ok and {x["track_id"] for x in r.records} == {"a:s1:t1", "b:s1:t1"}


def test_canonical_track_id_collision_is_detected():
    r = build_agent_states([obs(sequence_id="a:b", track_id="c"), obs(sequence_id="a", track_id="b:c")])
    assert r.records == [] and codes(r) == [C.TRACK_ID_COLLISION] * 2


# --------------------------------------------------- 13 duplicates
def test_duplicate_detected_by_identity_not_by_coordinates():
    same_coords = build_agent_states([obs(), obs(source_line=3)])
    assert same_coords.records == [] and codes(same_coords) == [C.DUPLICATE_OBSERVATION] * 2
    other_coords = build_agent_states([obs(), obs(bbox_xyxy=(1.0, 1.0, 2.0, 2.0), pos_x=1.5, pos_y=2.0)])
    assert other_coords.records == [] and codes(other_coords) == [C.DUPLICATE_OBSERVATION] * 2
    # identical coordinates on different frames are NOT duplicates
    assert build_agent_states([obs(frame_index=0), obs(frame_index=1, timestamp=0.1)]).report.ok


def test_duplicate_quarantines_all_members_and_keeps_other_tracks():
    r = build_agent_states([obs(track_id="a"), obs(track_id="b"), obs(track_id="a")])
    assert [x["track_id"] for x in r.records] == ["testset:s1:b"]
    issue = r.report.issues[0]
    assert issue.input_index == 0 and dict(issue.details)["members"] == "[0, 2]"


# ----------------------------------------- 14 ordering without sorting
def test_out_of_order_frames_detected_and_not_sorted():
    inputs = [obs(frame_index=0, timestamp=0.0), obs(frame_index=2, timestamp=0.2), obs(frame_index=1, timestamp=0.1)]
    before = list(inputs)
    r = build_agent_states(inputs)
    assert inputs == before  # input untouched
    assert [x["frame_index"] for x in r.records] == [0, 2]  # input order kept, offender quarantined
    assert codes(r) == [C.OUT_OF_ORDER_FRAME] and r.report.issues[0].input_index == 2


def test_fully_reversed_input_is_flagged_not_repaired():
    r = build_agent_states([obs(frame_index=f, timestamp=f / 10) for f in (3, 2, 1, 0)])
    assert [x["frame_index"] for x in r.records] == [3] and codes(r) == [C.OUT_OF_ORDER_FRAME] * 3


def test_timestamp_must_increase_with_frame():
    r = build_agent_states([obs(frame_index=0, timestamp=1.0), obs(frame_index=1, timestamp=1.0)])
    assert [x["frame_index"] for x in r.records] == [0] and codes(r) == [C.TIMESTAMP_NOT_MONOTONIC]
    r = build_agent_states([obs(frame_index=0, timestamp=1.0), obs(frame_index=1, timestamp=0.5)])
    assert codes(r) == [C.TIMESTAMP_NOT_MONOTONIC]


def test_frame_gaps_are_allowed_and_tracks_are_independent():
    r = build_agent_states([obs(frame_index=0), obs(frame_index=5, timestamp=0.5),
                            obs(track_id="b", frame_index=2, timestamp=0.2)])
    assert r.report.ok and len(r.records) == 3


# --------------------------------------------- 15-17 coordinate metadata
def test_image_coordinate_metadata_preserved():
    rec = one(obs()).records[0]
    assert (rec["image_coordinate_system"], rec["image_coordinate_unit"]) == ("camera_image", "pixel")
    assert (rec["image_x"], rec["image_y"]) == (20.0, 60.0)
    assert one(obs()).provenance[0].position_coordinate_system == "camera_image"


def test_image_geometry_without_a_named_system_is_an_issue_not_invented():
    r = one(obs(position_coordinate_system=None))
    assert r.records == [] and codes(r) == [C.MISSING_COORDINATE_SYSTEM]


def test_unverified_planar_gets_no_unit_and_is_not_promoted():
    r = one(obs(bbox_xyxy=None, position_semantics="unverified_planar", position_unit=None,
                position_coordinate_system="hbs_local", pos_x=49.9, pos_y=41.1))
    rec = r.records[0]
    assert r.report.ok and collect_violations(rec) == []
    for k in ("image_x", "image_y", "image_coordinate_system", "image_coordinate_unit", "world_x", "world_y",
              "world_coordinate_system", "world_coordinate_unit"):
        assert rec[k] is None, k
    assert rec["world_position_valid"] is False
    p = r.provenance[0]
    assert (p.withheld_position_status, p.withheld_position_xy, p.position_unit) == \
        ("unverified_planar_not_in_record", (49.9, 41.1), None)
    assert r.report.counters["unverified_planar_position_withheld"] == 1


def test_unverified_planar_with_a_unit_is_rejected():
    r = one(obs(bbox_xyxy=None, position_semantics="unverified_planar", position_unit="m"))
    assert r.records == [] and codes(r) == [C.COORDINATE_DECLARATION_INVALID]


def test_declared_world_coordinates_do_not_imply_verified_calibration():
    r = one(obs(bbox_xyxy=None, position_semantics="world_coordinate", position_unit="m",
                position_coordinate_system="site_map", pos_x=24.7, pos_y=11.2))
    rec = r.records[0]
    assert collect_violations(rec) == []
    assert rec["world_position_valid"] is False
    assert rec["world_x"] is None and rec["world_y"] is None
    assert rec["world_coordinate_unit"] is None and rec["world_coordinate_system"] is None
    p = r.provenance[0]
    assert (p.withheld_position_status, p.withheld_position_xy, p.position_unit, p.position_coordinate_system) == \
        ("declared_world_not_calibration_verified", (24.7, 11.2), "m", "site_map")
    assert r.report.counters["declared_world_position_withheld"] == 1


def test_world_declaration_requires_physical_unit_and_system_at_observation_level():
    for kw in (dict(position_unit="pixel"), dict(position_unit=None), dict(position_coordinate_system=None)):
        base = dict(bbox_xyxy=None, position_semantics="world_coordinate", position_unit="m",
                    position_coordinate_system="site_map")
        base.update(kw)
        r = one(obs(**base))
        assert r.records == [] and codes(r) == [C.COORDINATE_DECLARATION_INVALID]


def test_image_pixel_declared_with_physical_unit_is_rejected():
    r = one(obs(position_unit="m"))
    assert r.records == [] and C.COORDINATE_DECLARATION_INVALID in codes(r)


def test_bbox_with_non_image_semantics_is_rejected():
    for sem, unit, cs in (("world_coordinate", "m", "site"), ("unverified_planar", None, None)):
        r = one(obs(position_semantics=sem, position_unit=unit, position_coordinate_system=cs))
        assert r.records == [] and codes(r) == [C.BBOX_SEMANTICS_MISMATCH]


def test_geometry_without_declared_semantics_is_rejected():
    r = one(obs(position_semantics=None, position_unit=None, position_coordinate_system=None))
    assert r.records == [] and C.COORDINATE_DECLARATION_INVALID in codes(r)


def test_observation_without_any_geometry_is_valid_and_has_no_image_metadata():
    rec = one(obs(bbox_xyxy=None, pos_x=None, pos_y=None, position_semantics=None, position_unit=None,
                  position_coordinate_system=None)).records[0]
    assert collect_violations(rec) == []
    assert rec["image_coordinate_system"] is None and rec["image_coordinate_unit"] is None


# -------------------------------------------------- 18 velocity
def test_raw_velocity_not_emitted_without_unit_contract():
    r = one(obs(vel_x=3.0, vel_y=-1.0))
    rec = r.records[0]
    for k in ("velocity_x", "velocity_y", "speed", "acceleration_x", "acceleration_y",
              "kinematic_coordinate_space", "kinematic_position_unit", "kinematic_time_unit", "heading"):
        assert rec[k] is None, k
    p = r.provenance[0]
    assert (p.raw_velocity, p.velocity_status) == ((3.0, -1.0), "withheld_no_unit_contract")
    assert r.report.counters["raw_velocity_withheld"] == 1


def test_velocity_emitted_only_under_declared_image_contract_and_schema_valid():
    cfg = AgentStateConfig(velocity_contract=VelocityContract())
    r = one(obs(vel_x=3.0, vel_y=-1.0), cfg)
    rec = r.records[0]
    assert (rec["velocity_x"], rec["velocity_y"]) == (3.0, -1.0)
    assert (rec["kinematic_coordinate_space"], rec["kinematic_position_unit"], rec["kinematic_time_unit"]) == \
        ("image", "pixel", "s")
    assert rec["speed"] is None and rec["acceleration_x"] is None  # nothing is estimated in Stage 2
    assert collect_violations(rec) == [] and r.provenance[0].velocity_status == "emitted_under_declared_contract"


def test_world_or_other_velocity_contracts_are_refused():
    with pytest.raises(ConfigError, match="Stage 3"):
        VelocityContract(space="world", position_unit="m")
    with pytest.raises(ConfigError, match="only an image-space"):
        VelocityContract(position_unit="m")
    with pytest.raises(ConfigError, match="only an image-space"):
        VelocityContract(time_unit="ms")


def test_image_contract_on_non_image_observation_and_partial_velocity_rejected():
    cfg = AgentStateConfig(velocity_contract=VelocityContract())
    r = one(obs(bbox_xyxy=None, position_semantics="unverified_planar", position_unit=None, vel_x=1.0, vel_y=1.0), cfg)
    assert r.records == [] and codes(r) == [C.VELOCITY_CONTRACT_MISMATCH]
    assert codes(one(obs(vel_x=1.0))) == [C.INCOMPLETE_VELOCITY]
    assert codes(one(obs(vel_x=float("nan"), vel_y=1.0))) == [C.NON_FINITE_VALUE]


# --------------------------------------------- 19-20 strict vs lenient
def test_strict_mode_raises_informative_error_with_full_report():
    batch = [obs(), obs(track_id="b", source_label="scooter"), obs(track_id="c", bbox_xyxy=(5.0, 5.0, 1.0, 9.0))]
    with pytest.raises(ObservationValidationError) as ei:
        build_agent_states(batch, AgentStateConfig(strict=True))
    msg = str(ei.value)
    assert "unmapped_class" in msg and "bbox_inverted" in msg and "2 quarantined" in msg
    assert isinstance(ei.value, PedsafetyError) and len(ei.value.report.issues) == 2


def test_strict_mode_passes_clean_input():
    r = build_agent_states([obs()], AgentStateConfig(strict=True))
    assert r.report.ok and len(r.records) == 1


def test_lenient_mode_returns_valid_records_and_structured_issues():
    batch = [obs(), obs(track_id="b", source_label="scooter"), obs(track_id="c", frame_index=-1)]
    r = build_agent_states(batch)
    assert [x["track_id"] for x in r.records] == ["testset:s1:t1"]
    assert r.report.n_input == 3 and r.report.n_records == 1 and r.report.n_quarantined == 2
    assert r.report.quarantined_indices == [1, 2]
    assert r.report.counts_by_code() == {"invalid_frame_index": 1, "unmapped_class": 1}
    iss = r.report.issues[0]
    assert (iss.input_index, iss.source_line, iss.track_id, iss.field) == (1, 2, "b", "source_label")
    json.dumps(r.report.to_dict())  # JSON-serialisable, no evidence dropped
    assert len(r.report.to_dict()["issues"]) == 2


def test_multiple_issues_per_observation_all_reported():
    r = one(obs(source_label="scooter", detection_confidence=2.0, bbox_xyxy=(3.0, 3.0, 1.0, 1.0)))
    assert set(codes(r)) == {C.UNMAPPED_CLASS, C.INVALID_CONFIDENCE, C.BBOX_INVERTED}


# ------------------------------------- 21 + determinism + Stage 1 integration
def test_every_record_from_synthetic_scene_is_schema_valid_and_output_is_deterministic(tmp_path):
    scene = generate_scene(seed=7, position_noise_px=1.0)
    path = write_scene_csv(scene, tmp_path / "s.csv")
    parsed = CsvAnnotationAdapter(path, synthetic_csv_config()).parse()
    assert parsed.issues == []
    a = build_agent_states(parsed.observations)
    b = build_agent_states(parsed.observations)
    assert a.report.ok and len(a.records) == len(scene.observations)
    assert all(collect_violations(r) == [] for r in a.records)
    assert a.records == b.records and a.provenance == b.provenance
    assert json.dumps(a.report.to_dict(), sort_keys=True) == json.dumps(b.report.to_dict(), sort_keys=True)
    # output order == input order
    assert [(r["track_id"], r["frame_index"]) for r in a.records] == \
        [("synthetic:%s:%s" % (scene.sequence_id, o.track_id), o.frame_index) for o in parsed.observations]
    assert {r["agent_class"] for r in a.records} == {"pedestrian", "vehicle"}
    assert a.report.counters["raw_velocity_withheld"] == len(a.records)
    assert all(r["velocity_x"] is None for r in a.records)


def test_provenance_sidecar_contents():
    r = one(obs())
    p = r.provenance[0]
    assert (p.state_origin, p.raw_track_id, p.raw_sequence_id, p.source_line, p.input_index) == ("observed", "t1", "s1", 2, 0)
    assert p.schema_sha256 == schema_sha256()


def test_shuffled_input_is_not_sorted_back():
    scene = generate_scene(n_frames=6)
    items = [o for o in scene.observations if o.track_id == "ped_const"]
    shuffled = [items[i] for i in (0, 2, 1, 3, 5, 4)]
    r = build_agent_states(shuffled)
    assert [x["frame_index"] for x in r.records] == [0, 2, 3, 5]
    assert [i.input_index for i in r.report.issues] == [2, 5]


def test_empty_input():
    r = build_agent_states([])
    assert r.records == [] and r.report.ok and r.report.n_input == 0


# --------------------------------------------- source hygiene
def test_production_code_does_not_import_tests_or_pcg_code():
    for mod in (agent_state_module, validation_module):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for banned in ("tests.phase2", "bounding_box_validation", "import utils", "from utils", "import model", "import torch"):
            assert banned not in src, (mod.__name__, banned)
