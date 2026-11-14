import pytest

from pedsafety.errors import ConfigError, ParseError
from pedsafety.io.base import AnnotationAdapter, IssueCode, ParseResult
from pedsafety.io.csv_adapter import CsvAdapterConfig, CsvAnnotationAdapter

COLS = {"track_id": "id", "frame_index": "f", "timestamp": "t", "label": "cls",
        "x": "px", "y": "py", "confidence": "c"}
HEADER = "id,f,t,cls,px,py,c,note\n"


def cfg(**kw):
    base = dict(source_dataset="TEST", columns=dict(COLS), sequence_id="seq", scene_id="scn",
                position_semantics="image_pixel", position_unit="pixel")
    base.update(kw)
    return CsvAdapterConfig(**base)


def write(tmp_path, body, header=HEADER, name="a.csv"):
    p = tmp_path / name
    p.write_text(header + body, encoding="utf-8")
    return p


def parse(tmp_path, body, **kw):
    return CsvAnnotationAdapter(write(tmp_path, body), cfg(**kw)).parse()


# ---------------------------------------------------------------- happy path
def test_basic_parse_preserves_order_and_declares_semantics(tmp_path):
    r = parse(tmp_path, "b,1,0.1,ped,5,6,0.5,x\na,0,0.0,car,1,2,,y\n")
    assert [o.track_id for o in r.observations] == ["b", "a"]
    o = r.observations[0]
    assert (o.sequence_id, o.scene_id, o.source_dataset) == ("seq", "scn", "TEST")
    assert (o.frame_index, o.timestamp, o.pos_x, o.pos_y) == (1, 0.1, 5.0, 6.0)
    assert (o.position_semantics, o.position_unit) == ("image_pixel", "pixel")
    assert o.detection_confidence == 0.5 and r.observations[1].detection_confidence is None
    assert o.extras == (("note", "x"),) and o.source_line == 2
    assert r.issues == [] and r.rows_read == 2


def test_timestamp_from_fps_and_ms_conversion(tmp_path):
    cols = {k: v for k, v in COLS.items() if k != "timestamp"}
    r = parse(tmp_path, "a,5,0.0,ped,1,2,,n\n", columns=cols, fps=10.0)
    assert r.observations[0].timestamp == 0.5
    r = parse(tmp_path, "a,5,1500,ped,1,2,,n\n", timestamp_unit="ms")
    assert r.observations[0].timestamp == 1.5


def test_column_ids_and_bom_and_delimiter(tmp_path):
    p = tmp_path / "b.csv"
    p.write_bytes("﻿id;f;t;s;q\nz;0;0.0;S1;S2\n".encode("utf-8"))
    c = CsvAdapterConfig("T", {"track_id": "id", "frame_index": "f", "timestamp": "t",
                               "sequence_id": "s", "scene_id": "q"}, delimiter=";")
    o = CsvAnnotationAdapter(p, c).parse().observations[0]
    assert (o.sequence_id, o.scene_id, o.position_semantics, o.pos_x) == ("S1", "S2", None, None)


def test_bbox_passed_through_unchecked(tmp_path):
    cols = dict(COLS, bbox_x_min="a", bbox_y_min="b", bbox_x_max="c2", bbox_y_max="d")
    p = tmp_path / "c.csv"
    p.write_text(HEADER.rstrip() + ",a,b,c2,d\nk,0,0.0,p,1,2,,n,30,20,10,60\n", encoding="utf-8")
    o = CsvAnnotationAdapter(p, cfg(columns=cols)).parse().observations[0]
    assert o.bbox_xyxy == (30.0, 20.0, 10.0, 60.0)  # inverted: Stage 2 rejects it, not the parser


# ----------------------------------------------------------- config errors
@pytest.mark.parametrize("kw,match", [
    (dict(columns={"frame_index": "f"}), "track_id"),
    (dict(columns=dict(COLS, bogus="z")), "unknown canonical"),
    (dict(columns=dict(COLS, sequence_id="s")), "exactly one"),
    (dict(sequence_id=None), "exactly one"),
    (dict(timestamp_unit="min"), "timestamp_unit"),
    (dict(columns={k: v for k, v in COLS.items() if k != "timestamp"}), "fps is required"),
    (dict(fps=-1.0), "fps"),
    (dict(columns={k: v for k, v in COLS.items() if k != "y"}), "both 'x' and 'y'"),
    (dict(columns=dict(COLS, bbox_x_min="a")), "bbox needs all"),
    (dict(position_semantics=None), "position_semantics"),
    (dict(position_unit="furlong"), "position_unit"),
    (dict(delimiter=";;"), "delimiter"),
    (dict(source_dataset=" "), "source_dataset"),
])
def test_invalid_config(kw, match):
    with pytest.raises(ConfigError, match=match):
        cfg(**kw)


# ------------------------------------- correction 1: semantics vs unit rules
@pytest.mark.parametrize("sem,unit,cs", [
    ("image_pixel", "pixel", None),
    ("image_pixel", "pixel", "camera_image"),
    ("world_coordinate", "m", "site_map"),
    ("world_coordinate", "cm", "site_map"),
    ("world_coordinate", "mm", "site_map"),
    ("world_coordinate", "ft", "site_map"),
    ("unverified_planar", None, None),
    ("unverified_planar", None, "hbs_local"),
])
def test_valid_semantics_unit_combinations(sem, unit, cs):
    c = cfg(position_semantics=sem, position_unit=unit, position_coordinate_system=cs)
    assert (c.position_semantics, c.position_unit, c.position_coordinate_system) == (sem, unit, cs)


@pytest.mark.parametrize("sem,unit,cs,match", [
    ("image_pixel", "m", None, "requires position_unit 'pixel'"),
    ("image_pixel", "cm", None, "requires position_unit 'pixel'"),
    ("image_pixel", None, None, "requires position_unit 'pixel'"),
    ("world_coordinate", "pixel", "site_map", "physical position_unit"),
    ("world_coordinate", None, "site_map", "physical position_unit"),
    ("world_coordinate", "m", None, "naming the world frame"),
    ("world_coordinate", "m", "   ", "non-empty"),
    ("unverified_planar", "pixel", None, "must not be given a position_unit"),
    ("unverified_planar", "m", None, "must not be given a position_unit"),
    ("unverified_planar", "ft", "hbs_local", "must not be given a position_unit"),
    ("metric", "m", "site_map", "is not one of"),
    ("image_pixel", "furlong", None, "position_unit"),
    (None, "pixel", None, "position_semantics is not declared"),
    ("", None, None, "position_semantics is not declared"),
])
def test_incompatible_semantics_unit_combinations_rejected(sem, unit, cs, match):
    with pytest.raises(ConfigError, match=match):
        cfg(position_semantics=sem, position_unit=unit, position_coordinate_system=cs)


def test_bbox_only_mapping_still_requires_declared_semantics():
    cols = {k: v for k, v in COLS.items() if k not in ("x", "y")}
    cols.update(bbox_x_min="a", bbox_y_min="b", bbox_x_max="c2", bbox_y_max="d")
    with pytest.raises(ConfigError, match="position_semantics is not declared"):
        cfg(columns=cols, position_semantics=None, position_unit=None)
    cfg(columns=cols)  # image_pixel/pixel default is accepted


@pytest.mark.parametrize("kw", [
    dict(position_semantics="image_pixel", position_unit=None),
    dict(position_semantics=None, position_unit="pixel"),
    dict(position_semantics=None, position_unit=None, position_coordinate_system="x"),
])
def test_position_declarations_without_position_columns_rejected(kw):
    cols = {k: v for k, v in COLS.items() if k not in ("x", "y")}
    with pytest.raises(ConfigError, match="no position columns"):
        cfg(columns=cols, **kw)
    cfg(columns=cols, position_semantics=None, position_unit=None)  # nothing declared: fine


def test_declared_semantics_pass_through_to_observations(tmp_path):
    row = "a,0,0.0,p,1,2,,n\n"
    w = parse(tmp_path, row, position_semantics="world_coordinate", position_unit="m",
              position_coordinate_system="site_map").observations[0]
    assert (w.position_semantics, w.position_unit, w.position_coordinate_system) == ("world_coordinate", "m", "site_map")
    u = parse(tmp_path, row, position_semantics="unverified_planar", position_unit=None).observations[0]
    assert (u.position_semantics, u.position_unit) == ("unverified_planar", None)
    i = parse(tmp_path, row).observations[0]
    assert (i.position_semantics, i.position_unit, i.position_coordinate_system) == ("image_pixel", "pixel", None)


def test_unverified_values_never_receive_a_unit(tmp_path):
    for o in parse(tmp_path, "a,0,0.0,p,1,2,,n\nb,1,0.1,p,3,4,,n\n",
                   position_semantics="unverified_planar", position_unit=None).observations:
        assert o.position_unit is None


# ---------------------------------- correction 2: duplicate column mappings
def test_two_canonical_fields_mapped_to_same_header_rejected():
    with pytest.raises(ConfigError) as ei:
        cfg(columns=dict(COLS, frame_index="id"))  # track_id and frame_index both -> "id"
    msg = str(ei.value)
    assert "'id'" in msg and "frame_index" in msg and "track_id" in msg


def test_all_conflicting_headers_are_listed():
    cols = dict(COLS, frame_index="id", y="px")  # conflicts: id (track_id, frame_index), px (x, y)
    with pytest.raises(ConfigError) as ei:
        cfg(columns=cols)
    msg = str(ei.value)
    for token in ("'id'", "'px'", "frame_index", "track_id", "'x'", "'y'"):
        assert token in msg


def test_three_way_conflict_and_sequence_column_conflict():
    with pytest.raises(ConfigError, match="sequence_id"):
        CsvAdapterConfig("T", {"track_id": "id", "frame_index": "f", "timestamp": "t",
                               "sequence_id": "id", "scene_id": "q"})
    with pytest.raises(ConfigError) as ei:
        cfg(columns=dict(COLS, frame_index="id", timestamp="id"))
    assert all(f in str(ei.value) for f in ("frame_index", "timestamp", "track_id"))


@pytest.mark.parametrize("bad", ["", "   ", None, 5])
def test_empty_or_non_string_header_name_rejected(bad):
    with pytest.raises(ConfigError, match="non-empty string"):
        cfg(columns=dict(COLS, label=bad))


def test_distinct_headers_still_accepted():
    cfg(columns=dict(COLS, label="other"))


def test_missing_mapped_column_lists_available(tmp_path):
    p = write(tmp_path, "a,0,0.0,p,1,2,\n", header="id,f,t,cls,px,py,zzz\n")
    with pytest.raises(ConfigError) as ei:
        CsvAnnotationAdapter(p, cfg()).parse()
    assert "'c'" in str(ei.value) and "available columns" in str(ei.value)


def test_duplicate_header_and_empty_and_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="duplicate header"):
        CsvAnnotationAdapter(write(tmp_path, "", header="id,f,t,cls,px,py,c,c\n"), cfg()).parse()
    empty = tmp_path / "e.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        CsvAnnotationAdapter(empty, cfg()).parse()
    with pytest.raises(ConfigError, match="cannot open"):
        CsvAnnotationAdapter(tmp_path / "none.csv", cfg()).parse()


def test_non_utf8_file(tmp_path):
    p = tmp_path / "l.csv"
    p.write_bytes(HEADER.encode() + b"a,0,0.0,\xff\xfe,1,2,,n\n")
    with pytest.raises(ConfigError, match="UTF-8"):
        CsvAnnotationAdapter(p, cfg()).parse()


# ------------------------------------------------------------- row errors
@pytest.mark.parametrize("row,col,code", [
    ("a,0,abc,p,1,2,", "t", IssueCode.INVALID_NUMBER),
    ("a,0,0.0,p,1_0,2,", "px", IssueCode.INVALID_NUMBER),
    ("a,0,nan,p,1,2,", "t", IssueCode.NON_FINITE),
    ("a,0,0.0,p,inf,2,", "px", IssueCode.NON_FINITE),
    ("a,0,,p,1,2,", "t", IssueCode.MISSING_VALUE),
    (",0,0.0,p,1,2,", "id", IssueCode.MISSING_VALUE),
    ("a,1.5,0.0,p,1,2,", "f", IssueCode.INVALID_FRAME_INDEX),
    ("a,-1,0.0,p,1,2,", "f", IssueCode.INVALID_FRAME_INDEX),
    ("a,0,0.0,p,1,2,1.5", "c", IssueCode.OUT_OF_RANGE),
    ("a,0,0.0,p,1,,", "px", IssueCode.MISSING_VALUE),
])
def test_bad_values_become_issues_with_line_and_column(tmp_path, row, col, code):
    r = parse(tmp_path, "ok,0,0.0,p,1,2,,n\n" + row + ",n\n")
    assert len(r.observations) == 1 and r.observations[0].track_id == "ok"
    assert r.rows_rejected == 1
    issue = r.issues[0]
    assert (issue.code, issue.column, issue.source_line) == (code, col, 3)
    assert "line 3" in str(issue)


def test_ragged_rows(tmp_path):
    r = parse(tmp_path, "a,0,0.0,p,1,2,,n,EXTRA\nb,0,0.0,p\n")
    assert [i.code for i in r.issues] == [IssueCode.ROW_TOO_LONG, IssueCode.ROW_TOO_SHORT]
    assert r.observations == []


def test_multiple_issues_in_one_row_all_reported(tmp_path):
    r = parse(tmp_path, "a,x,y,p,1,2,,n\n")
    assert len(r.issues) == 2 and r.rows_rejected == 1


def test_strict_mode_raises_on_first_bad_row(tmp_path):
    p = write(tmp_path, "a,0,0.0,p,1,2,,n\nb,0,zzz,p,1,2,,n\n")
    with pytest.raises(ParseError) as ei:
        CsvAnnotationAdapter(p, cfg(strict=True)).parse()
    assert ei.value.issues[0].source_line == 3


def test_raise_if_issues(tmp_path):
    r = parse(tmp_path, "a,0,zzz,p,1,2,,n\n")
    with pytest.raises(ParseError):
        r.raise_if_issues()
    ParseResult().raise_if_issues()


def test_adapter_does_not_reorder_or_drop_duplicates(tmp_path):
    r = parse(tmp_path, "a,5,0.5,p,1,2,,n\na,5,0.5,p,9,9,,n\na,2,0.2,p,3,3,,n\n")
    assert [(o.frame_index, o.pos_x) for o in r.observations] == [(5, 1.0), (5, 9.0), (2, 3.0)]


# -------------------------------------------------------------- interface
def test_adapter_base_is_abstract_and_extensible():
    with pytest.raises(TypeError):
        AnnotationAdapter()

    class Dummy(AnnotationAdapter):
        name = "dummy"

        def parse(self):
            return ParseResult()

    assert Dummy().parse().observations == []
