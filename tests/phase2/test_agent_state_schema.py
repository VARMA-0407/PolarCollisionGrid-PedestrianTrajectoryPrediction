import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from bounding_box_validation import validate_bbox_xyxy


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "phase2"
    / "schemas"
    / "agent_state.schema.json"
)


def load_validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def base_record():
    return {
        "track_id": "seq-1:agent-1",
        "agent_class": "pedestrian",
        "timestamp": 1.0,
        "frame_index": 2,
        "sequence_id": "seq-1",
        "scene_id": "scene-1",
        "world_position_valid": False,
        "observation_valid": True,
        "interpolation_flag": False,
        "source_dataset": "PIE",
    }


def image_pedestrian():
    record = base_record()
    record.update(
        {
            "bbox_xyxy": [10.0, 20.0, 30.0, 60.0],
            "bbox_center": [20.0, 40.0],
            "footpoint": [20.0, 60.0],
            "image_coordinate_system": "camera_image",
            "image_coordinate_unit": "pixel",
            "detection_confidence": 0.9,
        }
    )
    return record


def image_kinematic_pedestrian():
    record = image_pedestrian()
    record.update(
        {
            "velocity_x": 2.0,
            "velocity_y": 0.0,
            "speed": 2.0,
            "kinematic_coordinate_space": "image",
            "kinematic_position_unit": "pixel",
            "kinematic_time_unit": "s",
        }
    )
    return record


def all_null_kinematics():
    record = base_record()
    record.update(
        {
            "velocity_x": None,
            "velocity_y": None,
            "speed": None,
            "acceleration_x": None,
            "acceleration_y": None,
        }
    )
    return record


def world_vehicle():
    record = base_record()
    record.update(
        {
            "track_id": "ind:recording-1:agent-2",
            "agent_class": "vehicle",
            "world_position_valid": True,
            "world_x": 1.0,
            "world_y": 2.0,
            "world_coordinate_system": "ind_local_map",
            "world_coordinate_unit": "m",
            "velocity_x": 1.5,
            "velocity_y": 0.0,
            "speed": 1.5,
            "kinematic_coordinate_space": "world",
            "kinematic_position_unit": "m",
            "kinematic_time_unit": "s",
        }
    )
    return record


@pytest.mark.parametrize(
    "record",
    [
        image_pedestrian(),
        image_kinematic_pedestrian(),
        world_vehicle(),
        base_record(),
        all_null_kinematics(),
        {**base_record(), "bbox_xyxy": None, "footpoint": None},
    ],
)
def test_positive_agent_state_records_validate(record):
    load_validator().validate(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_x", None),
        ("bbox_xyxy", [1.0, 2.0, 3.0]),
        ("footpoint", [1.0]),
        ("detection_confidence", 1.1),
    ],
)
def test_invalid_agent_state_records_fail(field, value):
    record = world_vehicle() if field == "world_x" else image_pedestrian()
    record[field] = value
    with pytest.raises(ValidationError):
        load_validator().validate(record)


def test_undeclared_property_fails():
    record = base_record()
    record["unexpected"] = True
    with pytest.raises(ValidationError):
        load_validator().validate(record)


def test_missing_required_identifier_fails():
    record = base_record()
    del record["track_id"]
    with pytest.raises(ValidationError):
        load_validator().validate(record)


@pytest.mark.parametrize(
    "record",
    [
        {**world_vehicle(), "world_y": None},
        {**base_record(), "world_x": 1.0, "world_y": 2.0},
        {**base_record(), "image_x": 10.0},
        {**image_pedestrian(), "image_coordinate_unit": "m"},
        {
            **world_vehicle(),
            "kinematic_position_unit": "pixel",
        },
        {
            **image_kinematic_pedestrian(),
            "kinematic_position_unit": "m",
        },
        {
            **image_pedestrian(),
            "velocity_x": 2.0,
            "kinematic_coordinate_space": None,
            "kinematic_position_unit": None,
            "kinematic_time_unit": None,
        },
    ],
)
def test_invalid_coordinate_and_kinematic_contracts_fail(record):
    with pytest.raises(ValidationError):
        load_validator().validate(record)


def test_bbox_ordering_accepts_valid_coordinates():
    assert validate_bbox_xyxy([10.0, 20.0, 30.0, 60.0]) is True


@pytest.mark.parametrize(
    "bbox",
    [
        [30.0, 20.0, 10.0, 60.0],
        [10.0, 60.0, 30.0, 20.0],
    ],
)
def test_bbox_ordering_rejects_inverted_coordinates(bbox):
    with pytest.raises(ValueError):
        validate_bbox_xyxy(bbox)
