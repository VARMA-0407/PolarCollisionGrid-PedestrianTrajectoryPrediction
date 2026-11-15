"""P5.3 end-to-end: manifest generation, determinism, input/output hash consistency, CLI scripts, output policy."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from pedsafety.image_kinematics_run import sha256_file
from pedsafety.output_dir import OutputDirectoryError
from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.runner import generate_windows
from phase5_helpers import CONFIG, wcfg

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import phase5_generate_windows  # noqa: E402
import phase5_validate_windows  # noqa: E402

REQUIRED_MANIFEST_KEYS = ("config", "sources", "outputs", "git", "counts", "coordinates", "target_source", "sampling", "label", "claims",
                          "limitations", "validation", "dependency_and_overlap", "versions")


@pytest.fixture(scope="module")
def gen(scene_run_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("p5gen") / "run_a"
    m = generate_windows(scene_run_dir, wcfg(), out, config_path=str(CONFIG), command="test")
    return out, m


def test_manifest_generation(gen, scene_run_dir):
    out, m = gen
    for k in REQUIRED_MANIFEST_KEYS:
        assert k in m
    assert m["label"] == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT"
    assert all(v is False for v in m["claims"].values() if isinstance(v, bool))
    c = m["counts"]
    assert c["candidate_windows"] == c["accepted_windows"] + c["rejected_windows"]
    assert sum(c["rejection_reasons"].values()) == c["rejected_windows"]
    assert c["accepted_windows"] == 20 and c["unique_pedestrian_target_tracks_accepted"] == 2
    assert c["video_count"] == 1 and c["scene_group_count"] == 1
    assert m["target_source"]["mode"] == "PILOT_SMOKE"
    assert m["target_source"]["annotated_target"]["status"] == "NOT_IMPLEMENTED_FUTURE_EXTENSION"
    assert m["coordinates"]["coordinate_system"] == "captured_1920x1080" and m["coordinates"]["world_position_valid"] is False
    assert m["sampling"]["history_length"] == 6 and m["sampling"]["prediction_length"] == 15
    assert m["dependency_and_overlap"]["dependency_interval_length_frames"] == 62
    assert m["input_source"]["offline_kinematics_used"] is False
    assert m["validation"]["ok"]
    assert m["sources"]["task3_run_manifest"]["sha256"] == sha256_file(Path(scene_run_dir) / "run_manifest.json")
    assert set(m["outputs"]) == {"windows.npz", "graph_bank.npz", "rejected_candidates.npz", "smoke_manifest.json"}
    smoke = json.loads((out / "smoke_manifest.json").read_text(encoding="utf-8"))
    assert smoke["label"] == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" and smoke["split_created"] is False
    assert list(smoke["partitions"]) == ["pilot_development"]
    assert smoke["formal_split_requirements"]["minimum_purge_gap_frames_for_chronological_blocks"] == 62
    assert "git" in m and "working_tree_dirty" in m["git"]


def test_deterministic_generation(gen, scene_run_dir, tmp_path):
    out, m = gen
    m2 = generate_windows(scene_run_dir, wcfg(), tmp_path / "run_b", config_path=str(CONFIG), command="test")
    for name in ("windows.npz", "graph_bank.npz", "rejected_candidates.npz", "smoke_manifest.json"):
        assert m["outputs"][name]["sha256"] == m2["outputs"][name]["sha256"], name
    assert m["counts"] == m2["counts"]


def test_input_output_hash_consistency_and_cli_validation(gen, tmp_path, capsys):
    out, m = gen
    for name, rec in m["outputs"].items():
        assert sha256_file(out / name) == rec["sha256"]
    rep = phase5_validate_windows.validate_run(out)
    assert rep["ok"], [k for k, v in rep["checks"].items() if not v["pass"]]
    bad = tmp_path / "tampered"
    shutil.copytree(out, bad)
    p = bad / "rejected_candidates.npz"
    p.write_bytes(p.read_bytes() + b"\0")
    rep2 = phase5_validate_windows.validate_run(bad)
    assert not rep2["ok"] and not rep2["checks"]["output_hashes_match_manifest"]["pass"]


def test_output_directory_never_overwritten(gen, scene_run_dir):
    out, _ = gen
    with pytest.raises(OutputDirectoryError):
        generate_windows(scene_run_dir, wcfg(), out)


def test_annotated_target_mode_refused_end_to_end(scene_run_dir, tmp_path):
    with pytest.raises(WindowContractError, match="TARGET_SOURCE_NOT_IMPLEMENTED"):
        generate_windows(scene_run_dir, wcfg(target_source_mode="ANNOTATED_TARGET"), tmp_path / "ann")
    assert not (tmp_path / "ann").exists()


def test_upstream_hash_mismatch_detected(scene_run_dir, tmp_path):
    m = json.loads((Path(scene_run_dir) / "run_manifest.json").read_text(encoding="utf-8"))
    p3 = Path(m["inputs"]["phase3_agent_states"]["path"])
    original = p3.read_bytes()
    try:
        p3.write_bytes(original + b"\n")
        with pytest.raises(WindowContractError, match="UPSTREAM_HASH_MISMATCH"):
            generate_windows(scene_run_dir, wcfg(), tmp_path / "up")
    finally:
        p3.write_bytes(original)
    assert sha256_file(p3) == m["inputs"]["phase3_agent_states"]["sha256"]


def test_generate_cli_exit_codes(scene_run_dir, tmp_path, capsys):
    rc = phase5_generate_windows.main(["--task3", str(scene_run_dir), "--config", str(CONFIG), "--out", str(tmp_path / "cli")])
    assert rc == 0 and "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" in capsys.readouterr().out
    rc = phase5_generate_windows.main(["--task3", str(scene_run_dir), "--config", str(CONFIG), "--out", str(tmp_path / "cli")])
    assert rc == 2
    assert phase5_validate_windows.main(["--run", str(tmp_path / "cli")]) == 0
