"""End-to-end runner and CLI: outputs, schema, units, provenance, preservation, determinism and output policy."""

import hashlib
import json
import subprocess
import sys

import pytest

from pedsafety.image_kinematics import process_causal_records
from pedsafety.image_kinematics_run import load_sidecar_validator, run_image_kinematics
from pedsafety.output_dir import OutputDirectoryError
from task2_helpers import DEFAULT_CONFIG, REPO, cfg, deep, lines, rec, track, write_jsonl

CANONICAL_SCHEMA_SHA256 = "784c8ca9c059ccbf0ddbb8bfcaf42cdfbbd60b33acfc0136c78a00fde46c3c6c"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sample_records():
    a = track([(f, f / 30.0, 100.0 + 3 * f, 500.0) for f in (0, 1, 2, 4, 5, 6, 7)], track_id="1")
    b = track([(f, f / 30.0, 900.0, 300.0 - 2 * f) for f in range(5)], track_id="2", cls="car")
    bad = rec(track="3", frame=1)
    bad["observation_valid"] = False
    return a + b + [bad, rec(track="4", frame=0, fy=1080.0)]


@pytest.fixture()
def run_dir(tmp_path):
    src = write_jsonl(tmp_path / "agent_states.jsonl", sample_records())
    with open(src, "a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    out = tmp_path / "out"
    before = sha(src)
    m = run_image_kinematics(src, cfg(), out, config_path=str(DEFAULT_CONFIG))
    return src, out, m, before


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_outputs_exist_parse_and_validate(run_dir):
    src, out, m, _ = run_dir
    names = {"causal_kinematics.jsonl", "offline_kinematics.jsonl", "quarantine.jsonl", "track_continuity_report.json",
             "kinematic_quality_report.json", "config_resolved.json", "processing_log.txt", "run_manifest.json"}
    assert {p.name for p in out.iterdir()} == names
    v = load_sidecar_validator()
    causal, off = read_jsonl(out / "causal_kinematics.jsonl"), read_jsonl(out / "offline_kinematics.jsonl")
    assert len(causal) == 14 and len(off) == 14            # 14 placeable records, all accepted (one has an invalid position)
    for r in causal + off:
        assert not list(v.iter_errors(r))
    assert all(r["view"] == "CAUSAL" for r in causal) and all(r["view"] == "OFFLINE_NON_CAUSAL" for r in off)
    q = read_jsonl(out / "quarantine.jsonl")
    assert len(q) == 1 and q[0]["reason"].startswith("malformed_json") and q[0]["source_line"] == 15


def test_units_and_world_fields(run_dir):
    _, out, m, _ = run_dir
    for r in read_jsonl(out / "causal_kinematics.jsonl"):
        assert (r["position_unit"], r["velocity_unit"], r["acceleration_unit"], r["heading_unit"], r["time_unit"]) == \
               ("pixel", "pixel/s", "pixel/s^2", "rad", "s")
        assert r["coordinate_space"] == "image" and r["world_x"] is None and r["world_y"] is None and r["world_position_valid"] is False
    assert m["claims"]["metric_world_coordinates"] is False and m["claims"]["physical_speed"] is False


def test_manifest_provenance_is_complete_and_hashes_match(run_dir):
    src, out, m, before = run_dir
    for k in ("manifest_version", "processing_version", "engine", "sidecar_schema", "started_utc", "finished_utc", "environment", "input",
              "config", "modes", "conventions", "policies", "counts", "outputs", "claims", "limitations"):
        assert k in m, k
    assert m["input"]["sha256_before"] == m["input"]["sha256_after"] == before == sha(src)
    assert m["config"]["sha256_resolved"] == cfg().sha256() and m["modes"]["offline_rows_in_causal_output"] is False
    for k in ("timestamp", "ordering", "gaps", "smoothing", "interpolation", "provenance"):
        assert m["policies"][k]
    for name, info in m["outputs"].items():
        assert sha(out / name) == info["sha256"]
    on_disk = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["counts"] == m["counts"]


def test_quality_and_continuity_counts(run_dir):
    _, out, m, _ = run_dir
    q = json.loads((out / "kinematic_quality_report.json").read_text(encoding="utf-8"))
    # track 1: frames 0,1,2,4,5,6,7 -> valid velocity at 1,2,5,6,7 (4 is after a gap) = 5 ; track 2: 4 ; others 0
    assert q["causal_raw"]["valid_velocity"] == 9 and m["counts"]["valid_velocity"] == 9
    assert q["valid_position_rows"] == 13 and q["invalid_position_rows"] == 1 and q["bbox_touches_border_rows"] == 1
    c = json.loads((out / "track_continuity_report.json").read_text(encoding="utf-8"))
    assert c["summary"]["n_tracks"] == 4 and c["summary"]["gap_missing_frames_histogram"] == {"1": 1}


def test_source_records_are_not_mutated():
    recs = sample_records()
    copy_ = deep(recs)
    process_causal_records(recs, lines(recs), cfg())
    assert recs == copy_


def test_deterministic_outputs(tmp_path):
    src = write_jsonl(tmp_path / "in.jsonl", sample_records())
    m1 = run_image_kinematics(src, cfg(), tmp_path / "o1")
    m2 = run_image_kinematics(src, cfg(), tmp_path / "o2")
    for name in ("causal_kinematics.jsonl", "offline_kinematics.jsonl", "quarantine.jsonl", "track_continuity_report.json",
                 "kinematic_quality_report.json", "config_resolved.json"):
        assert m1["outputs"][name]["sha256"] == m2["outputs"][name]["sha256"], name


def test_output_directory_policy(tmp_path):
    src = write_jsonl(tmp_path / "in.jsonl", sample_records())
    out = tmp_path / "busy"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(OutputDirectoryError):
        run_image_kinematics(src, cfg(), out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_canonical_schema_untouched():
    assert sha(REPO / "docs" / "phase2" / "schemas" / "agent_state.schema.json") == CANONICAL_SCHEMA_SHA256


def test_cli_exit_codes(tmp_path):
    src = write_jsonl(tmp_path / "in.jsonl", sample_records())
    script = str(REPO / "scripts" / "run_image_kinematics.py")
    ok = subprocess.run([sys.executable, script, "--input", str(src), "--config", str(DEFAULT_CONFIG), "--out", str(tmp_path / "o")],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert (tmp_path / "o" / "run_manifest.json").is_file()
    bad_cfg = tmp_path / "bad.json"
    d = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    d["max_derivative_gap_frames"] = 2
    bad_cfg.write_text(json.dumps(d))
    r = subprocess.run([sys.executable, script, "--input", str(src), "--config", str(bad_cfg), "--out", str(tmp_path / "o2")],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "max_derivative_gap_frames" in r.stderr and not (tmp_path / "o2").exists()
    r = subprocess.run([sys.executable, script, "--input", str(src), "--config", str(DEFAULT_CONFIG), "--out", str(tmp_path / "o")],
                       capture_output=True, text=True)
    assert r.returncode == 2                                     # non-empty output directory refused


def test_unknown_config_key_rejected():
    from pedsafety.errors import ConfigError
    d = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    d["max_gap"] = 3
    from pedsafety.image_kinematics import ImageKinematicsConfig
    with pytest.raises(ConfigError, match="unknown configuration keys"):
        ImageKinematicsConfig.from_dict(d)
