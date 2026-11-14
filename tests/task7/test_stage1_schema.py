import json
import re

import pytest

from pedsafety.errors import SchemaLoadError, SchemaViolation
from pedsafety.schema import (
    DEFAULT_SCHEMA_PATH,
    collect_violations,
    find_non_finite,
    load_schema,
    schema_property_names,
    schema_represents_causal_estimate,
    schema_sha256,
    validate_record,
    validate_records,
)


def base():
    return {
        "track_id": "s:1", "agent_class": "pedestrian", "timestamp": 1.0, "frame_index": 2,
        "sequence_id": "s", "scene_id": "sc", "world_position_valid": False,
        "observation_valid": True, "interpolation_flag": False, "source_dataset": "SYNTHETIC",
    }


def image_record():
    r = base()
    r.update(bbox_xyxy=[1.0, 2.0, 3.0, 4.0], image_coordinate_system="camera_image",
             image_coordinate_unit="pixel")
    return r


def test_schema_file_exists_and_loads():
    assert DEFAULT_SCHEMA_PATH.is_file()
    schema = load_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    assert "track_id" in schema_property_names()


def test_schema_sha256_is_stable_hex():
    h = schema_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", h) and h == schema_sha256()


def test_load_schema_returns_copy():
    s = load_schema()
    s["properties"].clear()
    assert load_schema()["properties"]


@pytest.mark.parametrize("make", [base, image_record])
def test_valid_records_pass(make):
    validate_record(make())
    assert collect_violations(make()) == []


def test_violation_lists_every_problem_with_field_paths():
    r = image_record()
    r["detection_confidence"] = 1.5
    r["bbox_xyxy"] = [1.0, 2.0]
    r["bogus"] = 1
    with pytest.raises(SchemaViolation) as ei:
        validate_record(r, record_index=7)
    msg = str(ei.value)
    paths = {v.path for v in ei.value.violations}
    assert "record 7" in msg
    assert {"detection_confidence", "bbox_xyxy"} <= paths
    assert len(ei.value.violations) >= 3


def test_missing_required_field_reported():
    r = base()
    del r["track_id"]
    assert any("track_id" in v.message for v in collect_violations(r))


def test_conditional_contract_enforced():
    r = base()
    r["image_x"] = 3.0  # image values without coordinate system/unit
    assert collect_violations(r)


def test_non_dict_record_rejected():
    assert collect_violations([1, 2])[0].path == "<record>"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_rejected(bad):
    r = base()
    r["timestamp"] = bad
    v = collect_violations(r)
    assert any(x.path == "timestamp" and "non-finite" in x.message for x in v)
    assert find_non_finite({"a": [1.0, bad]}) == ["a[1]"]


def test_validate_records_batch_non_raising():
    bad = validate_records([base(), {**base(), "frame_index": -1}, base()])
    assert [i for i, _ in bad] == [1]


def test_schema_has_no_causal_estimate_field_so_provenance_sidecar_needed():
    """Safeguard 2: the approved schema cannot mark causal estimates."""
    assert schema_represents_causal_estimate() is False
    props = load_schema()["properties"]
    assert props["interpolation_flag"]["type"] == "boolean"  # offline interpolation only
    assert load_schema()["additionalProperties"] is False    # cannot smuggle extra fields in


def test_schema_load_errors(tmp_path):
    with pytest.raises(SchemaLoadError, match="cannot read"):
        load_schema(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SchemaLoadError, match="not valid"):
        load_schema(bad)
    inv = tmp_path / "inv.json"
    inv.write_text(json.dumps({"type": 12}), encoding="utf-8")
    with pytest.raises(SchemaLoadError, match="Draft 2020-12"):
        load_schema(inv)
    arr = tmp_path / "arr.json"
    arr.write_text("[]", encoding="utf-8")
    with pytest.raises(SchemaLoadError, match="JSON object"):
        load_schema(arr)
