"""Input contracts and provenance joins, end to end: synthetic Phase 3 records -> REAL Task 2 runner -> Task 3 adapter."""

import json
import math

import pytest

from pedsafety.interaction.config import InteractionContractError
from pedsafety.interaction.input_adapter import load_inputs
from pedsafety.interaction.nodes import NODE_FLOAT_FEATURES, build_node_table
from task2_helpers import rec
from task3_helpers import cfg3, make_task2_run, read_rows, resync, scene_records, sha, write_rows


@pytest.fixture()
def t2(tmp_path):
    return make_task2_run(tmp_path, scene_records())


def expect(code, causal, manifest, p3, **cfg_over):
    with pytest.raises(InteractionContractError) as ei:
        load_inputs(causal, manifest, p3, cfg3(**cfg_over))
    assert ei.value.code == code, str(ei.value)


def test_clean_inputs_pass_and_counts(t2):
    causal, manifest, p3 = t2
    res = load_inputs(causal, manifest, p3, cfg3())
    assert res.counts["causal_rows"] == 16 and res.counts["frame_gap_before_rows_verified"] == 16 and res.counts["frames"] == 4
    assert res.counts["observation_types"] == {"observed": 16}
    nt = build_node_table(res.rows)
    assert nt.n == 16 and sorted(set(nt.node_type.tolist())) == [0, 1, 2]
    car = [k for k in range(nt.n) if nt.track_ids[nt.track_index[k]].endswith(":3")]
    assert all(nt.border[k] for k in car) and not any(nt.border[k] for k in range(nt.n) if k not in car)     # inherited flag
    assert all(nt.features[k, NODE_FLOAT_FEATURES.index("box_h_px")] == 200.0 for k in car)                 # joined Phase 3 height


def test_physical_line_semantics_with_blank_line(tmp_path):
    recs = scene_records()[:8]
    raw = [json.dumps(r) for r in recs[:3]] + [""] + [json.dumps(r) for r in recs[3:]]       # blank physical line 4
    causal, manifest, p3 = make_task2_run(tmp_path, [], raw_lines=raw)
    rows = read_rows(causal)
    assert [r["source_line"] for r in rows] == [1, 2, 3, 5, 6, 7, 8, 9]                       # one-based PHYSICAL numbering
    res = load_inputs(causal, manifest, p3, cfg3())
    assert res.counts["phase3_blank_lines"] == 1
    for r in res.rows:                                                                      # each join hit the right record
        src = recs[r.source_line - 1 if r.source_line < 4 else r.source_line - 2]
        assert (r.track_id, r.frame_index) == (src["track_id"], src["frame_index"]) and list(r.bbox_xyxy) == src["bbox_xyxy"]
    rows[0]["source_line"] = 4                                                              # point a row at the blank line
    write_rows(causal, rows)
    resync(manifest, causal, p3)
    expect("SOURCE_LINE_BLANK", causal, manifest, p3)


def test_source_line_out_of_range_and_duplicate(t2):
    causal, manifest, p3 = t2
    rows = read_rows(causal)
    bad = [dict(r) for r in rows]
    bad[0]["source_line"] = 999
    write_rows(causal, bad)
    resync(manifest, causal, p3)
    expect("SOURCE_LINE_OUT_OF_RANGE", causal, manifest, p3)
    bad = [dict(r) for r in rows]
    bad[1]["source_line"] = bad[0]["source_line"]
    write_rows(causal, bad)
    resync(manifest, causal, p3)
    expect("SOURCE_LINE_DUPLICATE", causal, manifest, p3)


def test_no_row_order_fallback(t2):
    causal, manifest, p3 = t2
    lines = p3.read_text(encoding="utf-8").splitlines()
    lines[1], lines[3] = lines[3], lines[1]                                                # swap two physical lines
    p3.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    resync(manifest, causal, p3)
    expect("SOURCE_LINE_IDENTITY_MISMATCH", causal, manifest, p3)


@pytest.mark.parametrize("field,value", [("track_id", "pie:video_0004:99"), ("frame_index", 7), ("timestamp", 9.5)])
def test_identity_mismatch(t2, field, value):
    causal, manifest, p3 = t2
    lines = [json.loads(l) for l in p3.read_text(encoding="utf-8").splitlines()]
    lines[2][field] = value
    p3.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    resync(manifest, causal, p3)
    expect("SOURCE_LINE_IDENTITY_MISMATCH", causal, manifest, p3)


def test_malformed_phase3_line_fails(tmp_path):
    raw = [json.dumps(r) for r in scene_records()[:6]] + ["{not json"]
    causal, manifest, p3 = make_task2_run(tmp_path, [], raw_lines=raw)                     # Task 2 quarantines the malformed line
    expect("PHASE3_MALFORMED_LINE", causal, manifest, p3)


@pytest.mark.parametrize("bbox,code", [([float("nan"), 440.0, 110.0, 500.0], "BBOX_NON_FINITE"), ([90.0, 500.0, 110.0, 500.0], "BBOX_NON_POSITIVE"),
                                       ([90.0, 440.0, 110.0], "BBOX_NON_FINITE")])
def test_bbox_validity(t2, bbox, code):
    causal, manifest, p3 = t2
    lines = [json.loads(l) for l in p3.read_text(encoding="utf-8").splitlines()]
    lines[0]["bbox_xyxy"] = bbox
    p3.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    resync(manifest, causal, p3)
    expect(code, causal, manifest, p3)


def test_frame_timestamp_inconsistency(t2):
    causal, manifest, p3 = t2
    rows = read_rows(causal)
    lines = [json.loads(l) for l in p3.read_text(encoding="utf-8").splitlines()]
    rows[1]["timestamp"] = rows[1]["timestamp"] + 0.001                                     # same frame as rows[0], other time
    lines[rows[1]["source_line"] - 1]["timestamp"] = rows[1]["timestamp"]                   # keep the join identity consistent
    write_rows(causal, rows)
    p3.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    resync(manifest, causal, p3)
    expect("FRAME_TIMESTAMP_INCONSISTENT", causal, manifest, p3)


def test_null_border_flag_on_observed_row(t2):
    causal, manifest, p3 = t2
    rows = read_rows(causal)
    rows[0]["bbox_touches_border"] = None
    write_rows(causal, rows)
    resync(manifest, causal, p3)
    expect("BORDER_FLAG_INVALID", causal, manifest, p3)


@pytest.mark.parametrize("field,value,code", [("view", "OFFLINE_NON_CAUSAL", "OFFLINE_ROW_REFUSED"),
                                              ("velocity_method", "central_difference_actual_dt", "OFFLINE_ROW_REFUSED"),
                                              ("world_x", 12.0, "WORLD_FIELDS_POPULATED"), ("world_position_valid", True, "WORLD_FIELDS_POPULATED"),
                                              ("position_unit", "m", "CAUSAL_ROW_CONTRACT"), ("velocity_unit", "m/s", "CAUSAL_ROW_CONTRACT"),
                                              ("coordinate_space", "world", "CAUSAL_ROW_CONTRACT"),
                                              ("coordinate_system", "captured_640x360", "COORDINATE_SYSTEM_MISMATCH")])
def test_row_contract_rejections(t2, field, value, code):
    causal, manifest, p3 = t2
    rows = read_rows(causal)
    rows[3][field] = value
    write_rows(causal, rows)
    resync(manifest, causal, p3)
    expect(code, causal, manifest, p3)


def test_schema_violation_rejected(t2):
    causal, manifest, p3 = t2
    rows = read_rows(causal)
    rows[5]["speed"] = -1.0                                                                 # passes explicit checks, fails the schema
    write_rows(causal, rows)
    resync(manifest, causal, p3)
    expect("SIDECAR_SCHEMA_VIOLATION", causal, manifest, p3)


def test_offline_file_refused_by_name(t2, tmp_path):
    causal, manifest, p3 = t2
    off = causal.parent / "offline_kinematics.jsonl"
    expect("OFFLINE_INPUT_REFUSED", off, manifest, p3)


def test_hash_mismatches(t2):
    causal, manifest, p3 = t2
    rows = read_rows(causal)
    rows[0]["speed"] = rows[0]["speed"]
    write_rows(causal, rows[:-1])                                                           # changed file, manifest not resynced
    expect("CAUSAL_HASH_MISMATCH", causal, manifest, p3)
    resync(manifest, causal, p3)
    p3.write_text(p3.read_text(encoding="utf-8") + "\n", encoding="utf-8")                  # Phase 3 file changed afterwards
    expect("PHASE3_HASH_MISMATCH", causal, manifest, p3)


def test_frame_gap_semantics_verified(t2):
    causal, manifest, p3 = t2
    rows = read_rows(causal)
    k = next(i for i, r in enumerate(rows) if r["frame_gap_before"] == 0)
    rows[k]["frame_gap_before"] = 3
    write_rows(causal, rows)
    resync(manifest, causal, p3)
    expect("FRAME_GAP_SEMANTICS", causal, manifest, p3)


def test_invalid_rejected_and_quarantined_rows_counted_not_nodes(tmp_path):
    recs = scene_records()[:8]
    bad = rec(track="5", frame=1, fx=300.0, fy=500.0)
    bad["observation_valid"] = False                                                       # Task 2: invalid row (position unavailable)
    dup1, dup2 = rec(track="6", frame=2, fx=350.0, fy=500.0), rec(track="6", frame=2, fx=351.0, fy=500.0)   # Task 2: both rejected
    unplaceable = rec(track="7", frame=1, fx=400.0, fy=500.0)
    unplaceable["track_id"] = ""                                                            # Task 2: quarantined
    causal, manifest, p3 = make_task2_run(tmp_path, recs + [bad, dup1, dup2, unplaceable])
    res = load_inputs(causal, manifest, p3, cfg3())
    assert res.counts["observation_types"] == {"invalid": 1, "observed": 8, "rejected": 2}
    assert res.counts["invalid_rows_by_position_state"] == {"observation_valid_not_true": 1}
    assert res.counts["rejected_rows_by_state"] == {"rejected_duplicate_observation": 2}
    assert res.counts["task2_quarantined_rows"] == 1
    nt = build_node_table(res.rows)
    assert nt.n == 8 and not any(t.endswith((":5", ":6")) for t in nt.track_ids)


def test_task2_config_requirements(t2):
    causal, manifest, p3 = t2
    cfgp = manifest.parent / "config_resolved.json"
    d = json.loads(cfgp.read_text(encoding="utf-8"))
    d["smoothing"] = {"kind": "time_aware_ema", "alpha": None, "tau_s": 0.2}
    cfgp.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    m["outputs"]["config_resolved.json"]["sha256"] = sha(cfgp)
    manifest.write_text(json.dumps(m), encoding="utf-8")
    expect("TASK2_SMOOTHING", causal, manifest, p3)
