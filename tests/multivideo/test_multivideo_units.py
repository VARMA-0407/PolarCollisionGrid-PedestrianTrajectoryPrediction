"""UNIT tests: inventory, official-split discovery, frozen recording-level split, experiment lock / stage tracker, Phase 3 batch wrapper."""

import json
import shutil
import subprocess

import pytest

from mv_helpers import make_pie_tree, write_phase3_dir
from pedsafety.multivideo.common import MARKER, StageError, check_marker, publish_dir, read_json, write_marker
from pedsafety.multivideo.experiment import StageTracker, create_or_resume_experiment
from pedsafety.multivideo.inventory import extract_annotation_archives, inventory_pie
from pedsafety.multivideo.perception_batch import perception_stage_config, run_perception_batch
from pedsafety.multivideo.split import (PIE_TOOLKIT_REFERENCE_SPLIT, build_split_manifest, check_split_integrity, freeze_split, load_frozen_split,
                                        parse_official_split, resolve_official_split)

REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
quiet = lambda s: None                                                                                  # noqa: E731


# ------------------------------------------------------------------------------------------------------------------- inventory
def test_inventory_complete_tree_counts_hashes_and_annotations(tmp_path):
    pie = make_pie_tree(tmp_path)
    inv = inventory_pie(pie, hash_cache_path=tmp_path / "c.json", progress=quiet)
    assert inv["status"] == "COMPLETE_WITH_WARNINGS"                                    # counts differ from the PIE documentation (informational warning)
    assert inv["n_videos"] == 7 and inv["sets"]["set01"]["n_videos"] == 2
    assert all(len(v["sha256"]) == 64 for v in inv["videos"]) and inv["annotations"]["n_videos_with_annt_xml"] == 7
    assert {v["key"] for v in inv["videos"]} >= {"set01_video_0001", "set01_video_0002", "set02_video_0001"}   # same stem in different sets stays distinct
    cache = read_json(tmp_path / "c.json")
    assert len(cache) == 7


def test_inventory_reports_missing_sets_duplicates_and_missing_annotations(tmp_path):
    pie = make_pie_tree(tmp_path, {"set01": 1, "set02": 1}, with_annotations=False)
    shutil.copyfile(pie / "PIE_clips/set01/video_0001.mp4", pie / "PIE_clips/set02/video_0002.mp4")
    inv = inventory_pie(pie, progress=quiet)
    assert inv["status"] == "MISSING_REQUIRED_DATA"
    txt = " ".join(inv["problems"])
    assert "set directory missing" in txt and "byte-identical duplicate" in txt and "annotation directory missing" in txt
    inv2 = inventory_pie(pie, progress=quiet, require_annotations=False)
    assert not any("annotation" in p for p in inv2["problems"])
    with pytest.raises(StageError):
        inventory_pie(tmp_path / "nothing", progress=quiet)


def test_annotation_archive_extraction_is_safe_and_never_overwrites(tmp_path):
    import zipfile
    z = tmp_path / "annotations.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("annotations/set01/video_0001_annt.xml", "<a/>")
    r = extract_annotation_archives([z], tmp_path / "out")
    assert r[str(z)]["extracted"] and (tmp_path / "out/annotations/annotations/set01/video_0001_annt.xml").is_file()
    assert extract_annotation_archives([z], tmp_path / "out")[str(z)]["extracted"] is False
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.xml", "x")
    with pytest.raises(StageError):
        extract_annotation_archives([bad], tmp_path / "out2")


# ------------------------------------------------------------------------------------------------------------------- split
def test_official_split_is_parsed_from_source_without_execution(tmp_path):
    pie = make_pie_tree(tmp_path)
    p = pie / "utilities" / "pie_data.py"
    p.write_text(p.read_text() + "\nraise SystemExit('this file must never be executed')\n")
    assert parse_official_split(p) == PIE_TOOLKIT_REFERENCE_SPLIT
    r = resolve_official_split([pie])
    assert r["status"] == "VERIFIED_FROM_SOURCE" and r["matches_transcribed_reference"]


def test_missing_pie_data_blocks_unless_reference_is_explicitly_accepted(tmp_path):
    pie = make_pie_tree(tmp_path, with_pie_data=False)
    with pytest.raises(StageError) as e:
        resolve_official_split([pie])
    assert e.value.code == "OFFICIAL_SPLIT_UNVERIFIED"
    r = resolve_official_split([pie], allow_unverified_reference=True)
    assert r["status"] == "REFERENCE_NOT_VERIFIED_AGAINST_SOURCE"


def test_split_manifest_is_recording_level_frozen_and_tamper_evident(tmp_path):
    pie = make_pie_tree(tmp_path)
    inv = inventory_pie(pie, progress=quiet)
    m = build_split_manifest(inv, resolve_official_split([pie]))
    integ = check_split_integrity(m, inv)
    assert integ["ok"] and integ["videos_per_partition"] == {"train": 4, "calibration": 0, "validation": 2, "test": 1}
    assert not integ["phase9_independent_evaluation_ready"]                              # annotations / identity mapping unverified -> Phase 9 gate stays blocked
    f = freeze_split(m, tmp_path / "splits")
    assert f["manifest"]["frozen"] and not f["reused"]
    assert freeze_split(m, tmp_path / "splits")["reused"]                               # idempotent, never rewritten
    assert load_frozen_split(f["path"])["split_id"] == m["split_id"]
    t = read_json(f["path"])
    t["partitions"]["train"].append("set03")
    (tmp_path / "t.json").write_text(json.dumps(t))
    with pytest.raises(StageError) as e:
        load_frozen_split(tmp_path / "t.json")
    assert e.value.code == "SPLIT_TAMPERED"


def test_calibration_scheme_and_leak_detection(tmp_path):
    pie = make_pie_tree(tmp_path)
    inv = inventory_pie(pie, progress=quiet)
    off = resolve_official_split([pie])
    m = build_split_manifest(inv, off, scheme="PIE_OFFICIAL_CALIBRATION_FROM_VAL")
    assert m["partitions"]["calibration"] == ["set06"] and m["partitions"]["validation"] == ["set05"] and m["deviations_from_official"]
    with pytest.raises(StageError):
        build_split_manifest(inv, off, scheme="PIE_OFFICIAL_CALIBRATION_FROM_VAL", calibration_sets=["set05", "set06"])
    bad = json.loads(json.dumps(m))
    bad["groups"]["set01"]["videos"].append("set03_video_0001")                         # a clip of another recording placed in a train group
    bad["partitions"]["test"].append("set01")
    probs = " ".join(check_split_integrity(bad, inv)["problems"])
    assert "cross-recording leak" in probs and "in both" in probs


# ------------------------------------------------------------------------------------------------------------------- experiment
def test_experiment_lock_refuses_changed_configuration_and_commit(tmp_path):
    e = create_or_resume_experiment(tmp_path, "", {"a": 1}, {"commit": "c1"})
    eid = e["experiment_id"]
    with pytest.raises(StageError) as x:
        create_or_resume_experiment(tmp_path, eid, {"a": 2}, {"commit": "c1"})
    assert x.value.code == "LOCKED_CONFIG_CHANGED"
    with pytest.raises(StageError) as x:
        create_or_resume_experiment(tmp_path, eid, {"a": 1}, {"commit": "c2"})
    assert x.value.code == "CODE_COMMIT_CHANGED"
    r = create_or_resume_experiment(tmp_path, eid, {"a": 1}, {"commit": "c2"}, allow_commit_change=True)
    assert r["resumed"] and r["lock"]["commit_changes_on_resume"]
    with pytest.raises(StageError):
        create_or_resume_experiment(tmp_path, "does_not_exist", {"a": 1}, {"commit": "c1"})


def test_stage_tracker_records_failure_and_blocks_dependants(tmp_path):
    t = StageTracker(tmp_path / "s.json")
    with t.stage("A") as box:
        box["detail"]["x"] = 1
    with t.stage("B"):
        raise ValueError("boom")
    assert t.status("A") == "COMPLETED" and t.status("B") == "FAILED"
    assert t.require("C", ["A"]) and not t.require("D", ["A", "B"])
    assert t.status("D") == "BLOCKED_UPSTREAM"
    assert StageTracker(tmp_path / "s.json").status("B") == "FAILED"                     # persisted


# ------------------------------------------------------------------------------------------------------------------- markers / publish
def test_publish_and_marker_states(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("hello")
    files = publish_dir(src, tmp_path / "dst" / "x", tmp_path / "q")
    write_marker(tmp_path / "dst" / "x", stage="S", key="k", config_sha256="c", files=files)
    assert check_marker(tmp_path / "dst" / "x", stage="S", config_sha256="c", verify="sha256")[0] == "COMPLETE"
    assert check_marker(tmp_path / "dst" / "x", stage="S", config_sha256="other")[0] == "STALE"
    (tmp_path / "dst" / "x" / "a.txt").write_text("HELLO")
    assert check_marker(tmp_path / "dst" / "x", stage="S", config_sha256="c", verify="sha256")[0] == "CORRUPT"
    assert check_marker(tmp_path / "nope", stage="S", config_sha256="c")[0] == "ABSENT"


# ------------------------------------------------------------------------------------------------------------------- Phase 3 batch wrapper
def test_perception_batch_wrapper_runs_cli_per_video_and_resumes(tmp_path, monkeypatch):
    """The CLI subprocess is replaced by a fake that writes a synthetic Phase 3 run folder (no YOLO / OpenCV here); the wrapper logic -
    per-video config with a unique sequence id, contract validation, gzip of detections, verified publication, marker, resume - is real."""
    pie = make_pie_tree(tmp_path, {"set01": 1, "set02": 1})
    inv = inventory_pie(pie, progress=quiet)
    calls = []

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        calls.append(cmd)
        a = dict(zip(cmd[::1], cmd[1::1]))
        cfg = json.loads(open(a["--config"]).read())
        key = cfg["video"]["sequence_id"]
        assert a["--scene-id"] == "pie_" + key and a["--source-dataset"] == "PIE" and a["--device"] == "cpu"
        run = __import__("pathlib").Path(a["--output-root"]) / ("20990101T000000Z_" + key)
        write_phase3_dir(run, key)
        (run / "detections.jsonl").write_text('{"x": 1}\n' * 5)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"PK" + b"0" * 100)
    sc = perception_stage_config(REPO / "configs/phase3/perception_default.json", weights_sha256="w", device="cpu", max_frames=None, save_video=False,
                                 write_detections=True, compress_detections=True)
    kw = dict(base_config_path=REPO / "configs/phase3/perception_default.json", weights_path=weights, weights_sha256="w", device="cpu", phase3_root=tmp_path / "p3",
              eval_root=tmp_path / "p3eval", work_root=tmp_path / "work", quarantine_root=tmp_path / "q", stage_cfg=sc, run_pie_evaluation=False)
    r = run_perception_batch(REPO, inv["videos"], progress=quiet, **kw)
    assert r["all_complete"] and len(calls) == 2
    d = tmp_path / "p3" / "set01" / "set01_video_0001"
    assert (d / "detections.jsonl.gz").is_file() and not (d / "detections.jsonl").exists() and (d / MARKER).is_file()
    assert read_json(d / MARKER)["detail"]["sequence_id"] == "set01_video_0001"
    r2 = run_perception_batch(REPO, inv["videos"], progress=quiet, **kw)
    assert len(calls) == 2 and set(r2["by_status"]) == {"SKIPPED_ALREADY_COMPLETE"}
    r3 = run_perception_batch(REPO, inv["videos"], progress=quiet, reuse_roots=[tmp_path / "p3"], **dict(kw, phase3_root=tmp_path / "other"))
    assert set(r3["by_status"]) == {"REUSED_VERIFIED"} and len(calls) == 2
