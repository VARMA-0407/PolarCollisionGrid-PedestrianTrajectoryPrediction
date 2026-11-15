"""P5.4 end-to-end on the synthetic scene: P5.3 windows -> baselines -> evaluation; determinism, recomputation, causality."""

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from pedsafety.image_kinematics_run import sha256_file
from pedsafety.output_dir import OutputDirectoryError
from pedsafety.phase5.baselines import build_history_input, make_predictor
from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.evaluation.experiment import load_baseline_config, run_baseline_experiment, verify_window_run
from pedsafety.phase5.runner import generate_windows
from pedsafety.phase5.task3_source import NODE_FEATURES
from phase5_helpers import CONFIG, wcfg

REPO = Path(__file__).resolve().parents[2]
BCFG = REPO / "configs" / "phase5" / "baseline_evaluation_default.json"
sys.path.insert(0, str(REPO / "scripts"))
import phase5_evaluate_predictions  # noqa: E402
import phase5_run_baselines  # noqa: E402
import phase5_validate_evaluation  # noqa: E402

DATA = ("predictions_B0_zero_velocity.npz", "predictions_B1_last_valid_causal_velocity.npz", "predictions_B2_history_linear_regression.npz",
        "evaluation_B0_zero_velocity.npz", "evaluation_B1_last_valid_causal_velocity.npz", "evaluation_B2_history_linear_regression.npz",
        "evaluation_results.json", "evaluation_report.md")


@pytest.fixture(scope="module")
def windows_run(scene_run_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("p54win") / "windows"
    generate_windows(scene_run_dir, wcfg(), out, config_path=str(CONFIG))
    return out


@pytest.fixture(scope="module")
def run_a(windows_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("p54run") / "a"
    return out, run_baseline_experiment(windows_run, BCFG, out, command="test")


def test_manifest_and_labels(run_a):
    out, m = run_a
    assert m["label"] == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT"
    assert set(m["data_artifacts"]) == set(DATA)
    assert m["counts"]["windows"] == 20 and m["counts"]["unique_target_tracks"] == 2
    for b in m["counts"]["per_baseline"].values():
        assert b["valid_predictions"] + b["invalid_predictions"] == 20
    assert m["inputs"]["offline_kinematics_used"] is False
    assert "git" in m["run_metadata"] and "peak_memory_bytes" in m["run_metadata"]
    assert all(v is False for v in m["claims"].values() if isinstance(v, bool))
    res = json.loads((out / "evaluation_results.json").read_text(encoding="utf-8"))
    assert res["label"] == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" and "started_utc" not in json.dumps(res)
    assert "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" in (out / "evaluation_report.md").read_text(encoding="utf-8")
    assert res["baselines"]["B0_zero_velocity"]["summary"]["miss_rate"]["status"] == "NOT_CONFIGURED"


def test_predictions_saved_without_targets(run_a):
    out, _ = run_a
    a = np.load(out / "predictions_B2_history_linear_regression.npz")
    assert not any("target_future" in k or k.startswith("future") for k in a.files)
    assert a["positions_uv_px"].shape == (20, 15, 2)


def test_two_identical_runs_have_identical_data_hashes(run_a, windows_run, tmp_path):
    _, m1 = run_a
    m2 = run_baseline_experiment(windows_run, BCFG, tmp_path / "b")
    for k in DATA:
        assert m1["data_artifacts"][k]["sha256"] == m2["data_artifacts"][k]["sha256"], k


def test_independent_validation_and_tamper_detection(run_a, tmp_path):
    out, _ = run_a
    rep = phase5_validate_evaluation.validate_run(out)
    assert rep["ok"], [k for k, v in rep["checks"].items() if not v["pass"]]
    bad = tmp_path / "tampered"
    shutil.copytree(out, bad)
    p = bad / "evaluation_B1_last_valid_causal_velocity.npz"
    p.write_bytes(p.read_bytes() + b"\0")
    assert not phase5_validate_evaluation.validate_run(bad)["checks"]["data_artifact_hashes_match_manifest"]["pass"]


def test_results_recomputed_from_saved_predictions(run_a, windows_run, tmp_path):
    out, _ = run_a
    _, s = phase5_evaluate_predictions.evaluate_dir(out, windows_run, BCFG, tmp_path / "re")
    stored = json.loads((out / "evaluation_results.json").read_text(encoding="utf-8"))
    for b, summ in s["summaries"].items():
        assert summ == stored["baselines"][b]["summary"]
        assert sha256_file(tmp_path / "re" / ("evaluation_%s.npz" % b)) == sha256_file(out / ("evaluation_%s.npz" % b))


def test_future_targets_cannot_influence_predictions(windows_run):
    bcfg, _ = load_baseline_config(BCFG)
    _, cfg, (ws, bank, extra), _ = verify_window_run(windows_run, bcfg)
    h1 = build_history_input(ws, bank, extra["history_target_bank_node"], cfg)
    ws2 = copy.deepcopy(ws)
    ws2.target_future_uv_px[:] = 1e9                         # attempt to inject future information
    ws2.future_timestamps[:] = -1.0
    h2 = build_history_input(ws2, bank, extra["history_target_bank_node"], cfg)
    for spec in bcfg["baselines"]:
        a, b = make_predictor(spec).predict(h1), make_predictor(spec).predict(h2)
        assert np.array_equal(a.positions, b.positions)
    assert np.allclose(h1.horizon_offsets_s, np.arange(1, 16) * 0.1)


def test_b1_uses_task2_causal_velocity_from_graph_bank(windows_run):
    bcfg, _ = load_baseline_config(BCFG)
    _, cfg, (ws, bank, extra), _ = verify_window_run(windows_run, bcfg)
    h = build_history_input(ws, bank, extra["history_target_bank_node"], cfg)
    tb = extra["history_target_bank_node"][:, -1]
    vel = bank.nodes["features"][tb][:, [NODE_FEATURES.index("vel_u_px_s"), NODE_FEATURES.index("vel_v_px_s")]]
    pr = make_predictor(bcfg["baselines"][1]).predict(h)
    use = pr.aux["velocity_source_index"] == 5
    assert use.any()
    assert np.allclose(pr.aux["velocity_px_s"][use], vel[use].astype(np.float64))
    assert h.velocity_source.startswith("task2_causal")


def test_input_verification_rejects_mismatches(windows_run, tmp_path):
    bad = tmp_path / "w"
    shutil.copytree(windows_run, bad)
    p = bad / "windows.npz"
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(WindowContractError, match="WINDOW_RUN_HASH"):
        run_baseline_experiment(bad, BCFG, tmp_path / "x")
    raw = json.loads(BCFG.read_text(encoding="utf-8"))
    raw["window_config_reference"]["sha256_resolved"] = "0" * 64
    c2 = tmp_path / "c2.json"
    c2.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(WindowContractError, match="WINDOW_CONFIG_MISMATCH"):
        run_baseline_experiment(windows_run, c2, tmp_path / "y")
    raw = json.loads(BCFG.read_text(encoding="utf-8"))
    raw["miss_rate_threshold_px"] = 0
    c3 = tmp_path / "c3.json"
    c3.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(WindowContractError, match="EVAL_CONFIG"):
        run_baseline_experiment(windows_run, c3, tmp_path / "z")


def test_output_never_overwritten_and_cli(run_a, windows_run, tmp_path, capsys):
    out, _ = run_a
    with pytest.raises(OutputDirectoryError):
        run_baseline_experiment(windows_run, BCFG, out)
    assert phase5_run_baselines.main(["--windows", str(windows_run), "--config", str(BCFG), "--out", str(tmp_path / "cli")]) == 0
    assert "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" in capsys.readouterr().out
    assert phase5_run_baselines.main(["--windows", str(windows_run), "--config", str(BCFG), "--out", str(tmp_path / "cli")]) == 2
    assert phase5_validate_evaluation.main(["--run", str(tmp_path / "cli")]) == 0
