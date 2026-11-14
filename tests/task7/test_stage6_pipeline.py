"""End-to-end synthetic tests through the same entry point the CLI uses."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pedsafety import pipeline
from pedsafety.cli import main
from pedsafety.normalization import NormalizationState
from pedsafety.pipeline_config import load_config
from pedsafety.schema import schema_sha256
from stage6_fixtures import calibrated_setup, load_cfg, save_cfg, synthetic

PARTS = ("train", "validation", "test")
REPO = Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def jl(p):
    return json.loads(open(p, encoding="utf-8").read())


def err(out):
    return jl(out / "error.json")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    csv_path, cfg_path = synthetic(d)
    before = sha(csv_path)
    rc1 = main(["build", str(cfg_path), "--out", str(d / "o1")])
    rc2 = main(["build", str(cfg_path), "--out", str(d / "o2")])
    return {"dir": d, "csv": csv_path, "cfg": cfg_path, "before": before, "rc": (rc1, rc2), "o1": d / "o1", "o2": d / "o2"}


def test_build_succeeds_and_input_is_untouched(run):
    assert run["rc"] == (0, 0)
    assert sha(run["csv"]) == run["before"]


def test_expected_outputs_exist(run):
    names = {p.name for p in run["o1"].iterdir()}
    required = ["run_manifest.json", "config_resolved.json", "parse_report.json", "validation_report.json", "split_plan.json",
                "split_report.json", "normalization.json"]
    required += ["windows_%s.npz" % p for p in PARTS] + ["windows_%s_index.json" % p for p in PARTS]
    for n in required:
        assert n in names
    assert "error.json" not in names


def test_manifest_records_provenance_correctly(run):
    m = jl(run["o1"] / "run_manifest.json")
    cfg = load_config(run["cfg"])
    assert m["input"]["sha256"] == run["before"] and m["input"]["read_only"] is True
    assert m["schema_sha256"] == schema_sha256()
    assert m["config"]["sha256"] == cfg.sha256()
    assert m["status"] == "completed" and m["warnings"] == []
    assert m["counts"]["rows_rejected"] == 0 and m["counts"]["records_quarantined"] == 0
    assert m["calibration"] is None
    for name, digest in m["outputs"].items():
        assert sha(run["o1"] / name) == digest, name
    plan = jl(run["o1"] / "split_plan.json")
    recomputed = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert m["split"]["plan_sha256"] == recomputed
    assert all(m["split"]["windows"][p] > 0 for p in PARTS)


def test_claims_do_not_overreach(run):
    c = jl(run["o1"] / "run_manifest.json")["claims"]
    assert c["any_window_physical"] is False and c["calibration_supplied"] is False
    assert c["software_eligibility_is_physical_verification"] is False
    assert {i["position_unit"] for i in c["coordinate_identities_in_windows"]} == {"pixel"}


def test_two_runs_are_reproducible(run):
    a, b = run["o1"], run["o2"]
    for p in sorted(a.glob("*.json")):
        assert p.read_bytes() == (b / p.name).read_bytes(), p.name
    assert jl(a / "run_manifest.json")["output_array_digests"] == jl(b / "run_manifest.json")["output_array_digests"]
    for part in PARTS:
        x, y = np.load(a / ("windows_%s.npz" % part)), np.load(b / ("windows_%s.npz" % part))
        assert sorted(x.files) == sorted(y.files)
        for k in x.files:
            assert np.array_equal(x[k], y[k], equal_nan=True), k


def test_seed_change_changes_the_plan_hash(run, tmp_path):
    d = load_cfg(run["cfg"])
    d["input"]["csv"] = str(run["csv"])
    d["split"]["seed"] = 8
    p = save_cfg(tmp_path / "seed8.json", d)
    assert main(["build", str(p), "--out", str(tmp_path / "o")]) == 0
    assert jl(tmp_path / "o" / "run_manifest.json")["split"]["plan_sha256"] != jl(run["o1"] / "run_manifest.json")["split"]["plan_sha256"]


def test_windows_never_cross_partitions_and_targets_follow_history(run):
    seen = {}
    for part in PARTS:
        idx = jl(run["o1"] / ("windows_%s_index.json" % part))
        arr = np.load(run["o1"] / ("windows_%s.npz" % part))
        assert idx["n_windows"] == len(idx["windows"]) == arr["target_xy"].shape[0] > 0
        for w in idx["windows"]:
            assert w["partition"] == part
            for key in (("group", w["group_id"]), ("seq", w["raw_sequence_id"]), ("scene", w["raw_scene_id"])):
                assert seen.setdefault(key, part) == part, key
        assert (arr["target_frame_index"][:, 0] > arr["history_frame_index"][:, -1]).all()
        assert (arr["target_timestamp"][:, 0] > arr["history_timestamp"][:, -1]).all()


def test_normalization_is_linked_to_the_plan(run):
    state = NormalizationState.load(run["o1"] / "normalization.json")
    m = jl(run["o1"] / "run_manifest.json")
    assert m["normalization_sha256"] == state.sha256()
    assert state.to_dict()["fitted_on"]["split_plan_sha256"] == m["split"]["plan_sha256"]


def test_ingest_and_features_commands(run, tmp_path):
    assert main(["ingest", str(run["cfg"]), "--out", str(tmp_path / "i")]) == 0
    assert main(["features", str(run["cfg"]), "--out", str(tmp_path / "f")]) == 0
    f = jl(tmp_path / "f" / "features_summary.json")
    assert f["partitioned"] is False and "descriptive only" in f["use"] and f["n_tracks"] > 0
    assert not (tmp_path / "f" / "windows_train.npz").exists()


def test_validate_config_command_and_module_entry_point(run):
    assert main(["validate-config", str(run["cfg"])]) == 0
    r = subprocess.run([sys.executable, "-m", "pedsafety", "validate-config", str(run["cfg"])], capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0 and "config OK" in r.stdout


# ------------------------------------------------------------ error reporting
def test_missing_input_exit_3_with_error_json(run, tmp_path):
    d = load_cfg(run["cfg"])
    d["input"]["csv"] = str(tmp_path / "gone.csv")
    assert main(["build", str(save_cfg(tmp_path / "c.json", d)), "--out", str(tmp_path / "o")]) == 3
    assert err(tmp_path / "o")["code"] == "FILE_NOT_FOUND"


def test_single_scene_split_is_refused_exit_4(tmp_path):
    _, cfg = synthetic(tmp_path, n_scenes=1)
    assert main(["build", str(cfg), "--out", str(tmp_path / "o")]) == 4
    e = err(tmp_path / "o")
    assert e["stage"] == "split" and e["code"] == "INSUFFICIENT_GROUPS" and e["status"] == "failed"
    assert not (tmp_path / "o" / "run_manifest.json").exists()


def test_missing_grouping_is_a_config_error_exit_2(run, tmp_path):
    d = load_cfg(run["cfg"])
    d["input"]["csv"] = str(run["csv"])
    d.pop("grouping")
    assert main(["build", str(save_cfg(tmp_path / "c.json", d)), "--out", str(tmp_path / "o")]) == 2
    assert "no default grouping" in err(tmp_path / "o")["message"]


def test_bad_config_json_exit_2(tmp_path):
    (tmp_path / "c.json").write_text("{", encoding="utf-8")
    assert main(["build", str(tmp_path / "c.json"), "--out", str(tmp_path / "o")]) == 2


def test_offline_view_requires_explicit_consent(run, tmp_path):
    d = load_cfg(run["cfg"])
    d["input"]["csv"] = str(run["csv"])
    d["view"] = "offline"
    assert main(["build", str(save_cfg(tmp_path / "c.json", d)), "--out", str(tmp_path / "o")]) == 2
    d["windows"]["allow_offline_view"] = True
    assert main(["build", str(save_cfg(tmp_path / "c2.json", d)), "--out", str(tmp_path / "o2")]) == 0
    idx = jl(tmp_path / "o2" / "windows_train_index.json")
    assert all(w["history_may_use_future"] for w in idx["windows"])


def corrupt(csv_path, tmp_path):
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    cols = lines[0].split(",")
    row = lines[5].split(",")
    row[cols.index("foot_x")] = "not_a_number"
    lines[5] = ",".join(row)
    bad = tmp_path / "bad.csv"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bad


def test_bad_row_lenient_reports_line_strict_fails(run, tmp_path):
    bad = corrupt(run["csv"], tmp_path)
    cfg = str(run["cfg"])
    assert main(["build", cfg, "--input", str(bad), "--out", str(tmp_path / "lenient")]) == 0
    m = jl(tmp_path / "lenient" / "run_manifest.json")
    assert m["status"] == "completed_with_warnings" and m["counts"]["rows_rejected"] == 1
    pr = jl(tmp_path / "lenient" / "parse_report.json")
    assert pr["issues"]["items"][0]["source_line"] == 6 and pr["issues"]["items"][0]["column"] == "foot_x"
    assert m["input"]["sha256"] == sha(bad)
    assert main(["build", cfg, "--input", str(bad), "--strict", "--out", str(tmp_path / "strict")]) == 3
    assert err(tmp_path / "strict")["code"] == "PARSE_ERROR"
    assert main(["ingest", cfg, "--input", str(bad), "--fail-on-issues", "--out", str(tmp_path / "foi")]) == 3
    assert err(tmp_path / "foi")["code"] == "ISSUES_PRESENT"


def test_unknown_figure_name_exit_2(run, tmp_path):
    assert main(["plot", str(run["cfg"]), "--which", "nonsense", "--out", str(tmp_path / "p")]) == 2


def test_unexpected_exception_maps_to_internal_error():
    info = pipeline.classify_error(RuntimeError("boom"), "build")
    assert info["exit_code"] == pipeline.EXIT_INTERNAL and info["code"] == "INTERNAL_ERROR"


# ------------------------------------------------------------------ calibration
def test_unverified_projection_stays_non_physical(tmp_path):
    _, cfg = calibrated_setup(tmp_path, mode="projective_unverified", promote=False)
    assert main(["build", str(cfg), "--out", str(tmp_path / "o")]) == 0
    m = jl(tmp_path / "o" / "run_manifest.json")
    ids = m["claims"]["coordinate_identities_in_windows"]
    assert {i["coordinate_mode"] for i in ids} == {"projective_unverified"} and not any(i["physical"] for i in ids)
    assert m["calibration"]["physically_verified_by_software"] is False
    pr = jl(tmp_path / "o" / "projection_report.json")
    assert pr["eligibility"]["physical_verification_established_by_software"] is False and "NOT independent" in pr["note"]


def test_eligible_manifest_is_labelled_software_checks_only(tmp_path):
    _, cfg = calibrated_setup(tmp_path, mode="metric_manifest_eligible", promote=True)
    assert main(["build", str(cfg), "--out", str(tmp_path / "o")]) == 0
    m = jl(tmp_path / "o" / "run_manifest.json")
    c = m["claims"]
    assert c["any_window_physical"] is True and c["physical_basis"] == ["software_manifest_checks_only"]
    assert c["software_eligibility_is_physical_verification"] is False
    assert m["calibration"]["eligible_by_software_checks"] is True and m["calibration"]["physically_verified_by_software"] is False


def test_metric_mode_without_promotion_never_yields_metric_windows(tmp_path):
    _, cfg = calibrated_setup(tmp_path, mode="metric_manifest_eligible", promote=False)
    assert main(["build", str(cfg), "--out", str(tmp_path / "o")]) == 3
    e = err(tmp_path / "o")
    assert e["stage"] == "normalization" and e["code"] == "NO_TRAINING_WINDOWS"
    assert not list((tmp_path / "o").glob("windows_*"))          # nothing written as if it were a valid dataset


def test_unusable_calibration_raster_exit_5(tmp_path):
    _, cfg = calibrated_setup(tmp_path, mode="projective_unverified", promote=False)
    d = load_cfg(cfg)
    d["calibration"]["observed_raster"]["width"] = 1280
    save_cfg(cfg, d)
    assert main(["build", str(cfg), "--out", str(tmp_path / "o")]) == 5
    assert err(tmp_path / "o")["stage"] == "calibration"


def test_input_changing_during_the_run_is_detected(run, tmp_path, monkeypatch):
    calls = {"n": 0}
    real = pipeline.file_sha256

    def flaky(path):
        calls["n"] += 1
        return real(path) if calls["n"] == 1 else "0" * 64

    monkeypatch.setattr(pipeline, "file_sha256", flaky)
    assert main(["ingest", str(run["cfg"]), "--out", str(tmp_path / "o")]) == 3
    assert err(tmp_path / "o")["code"] == "INPUT_CHANGED_DURING_RUN"
