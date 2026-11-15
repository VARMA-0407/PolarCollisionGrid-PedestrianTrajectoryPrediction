"""P5.7.5: neural evaluator - artifact verification, history-only inference, tensor contract, selected-mode vs oracle metrics, P5.4 integration,
baseline comparison, reproducibility and independent validation."""

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pedsafety.errors import ConfigError
from pedsafety.interaction.storage import load_npz
from pedsafety.output_dir import OutputDirectoryError
from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.evaluation import EvalConfig, evaluate_predictions
from pedsafety.phase5.evaluation.experiment import run_baseline_experiment
from pedsafety.phase5.models.neural_evaluation import (IN_SAMPLE, NeuralEvaluationConfig, load_trained_model, mode_diagnostics, predict_windows, run_neural_evaluation)
from p57_helpers import write_tiny_training_config

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import phase5_evaluate_neural  # noqa: E402
import phase5_validate_neural_evaluation  # noqa: E402

NCFG = REPO / "configs" / "phase5" / "neural_evaluation_default.json"
BCFG = REPO / "configs" / "phase5" / "baseline_evaluation_default.json"


@pytest.fixture(scope="module")
def eval_cfg(tmp_path_factory):
    d = json.loads(NCFG.read_text(encoding="utf-8"))
    d["batch_size"] = 8
    d["p54_reference_results_path"] = None
    p = tmp_path_factory.mktemp("p57evalcfg") / "neural_eval.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def eval_run(p57_train_run, p57_windows_run, eval_cfg, tmp_path_factory):
    train, _, _ = p57_train_run
    out = tmp_path_factory.mktemp("p57eval") / "run"
    m = run_neural_evaluation(train, p57_windows_run, eval_cfg, out, command="test")
    return out, m


def test_config_validation():
    c = NeuralEvaluationConfig.load(NCFG)
    assert c.sample_status == IN_SAMPLE and c.oracle_k == 5 and c.checkpoint == "final"
    d = json.loads(NCFG.read_text(encoding="utf-8"))
    for over, msg in (({"sample_status": "INDEPENDENT"}, "sample_status"), ({"label": "x"}, "label"), ({"batch_size": 0}, "batch_size"), ({"device": "tpu"}, "device"),
                      ({"checkpoint": "../evil.pt"}, "checkpoint"), ({"oracle_k": 0}, "oracle_k")):
        with pytest.raises(ConfigError, match=msg):
            NeuralEvaluationConfig(**{k: v for k, v in dict(d, **over).items() if not k.startswith("_")})


def test_verified_model_loading_and_provenance(p57_train_run, p57_windows_run):
    train, m, _ = p57_train_run
    lm = load_trained_model(train, p57_windows_run)
    v = lm.verification
    assert v["checkpoint_file_sha256"] == m["data_artifacts"][m["training"]["final_checkpoint"]]["sha256"] and v["scaler_buffers_equal_statistics"] and v["scaler_source_identity_verified"]
    assert not lm.model.training and v["checkpoint_epoch"] == 2
    assert v["scaler_statistics_sha256"] == m["inputs"]["scaler"]["statistics_sha256"] and lm.scaler_statistics.fit_partition == "train"


def _copy_train(train, tmp_path):
    d = tmp_path / "train_copy"
    shutil.copytree(train, d)
    return d


def test_tampered_checkpoint_scaler_manifest_and_windows_are_rejected(p57_train_run, p57_windows_run, tmp_path):
    train, m, _ = p57_train_run
    d = _copy_train(train, tmp_path)
    ck = d / m["training"]["final_checkpoint"]
    ck.write_bytes(ck.read_bytes() + b"\0")
    with pytest.raises(WindowContractError, match="TRAIN_ARTIFACT_HASH"):
        load_trained_model(d, p57_windows_run)
    # consistent forgery of the manifest hash for the checkpoint is still caught by the embedded canonical state hash chain
    d2 = tmp_path / "train_copy2"
    shutil.copytree(train, d2)
    sp = d2 / "scaler_statistics.json"
    body = json.loads(sp.read_text(encoding="utf-8"))
    body["edge"]["mean"][0] += 0.5
    sp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(WindowContractError, match="TRAIN_ARTIFACT_HASH"):
        load_trained_model(d2, p57_windows_run)
    mm = json.loads((d2 / "run_manifest.json").read_text(encoding="utf-8"))
    mm["data_artifacts"]["scaler_statistics.json"] = {"sha256": __import__("hashlib").sha256(sp.read_bytes()).hexdigest(), "bytes": sp.stat().st_size}
    (d2 / "run_manifest.json").write_text(json.dumps(mm), encoding="utf-8")
    with pytest.raises(WindowContractError, match="SCALER_HASH_MISMATCH"):
        load_trained_model(d2, p57_windows_run)
    w = tmp_path / "win"
    shutil.copytree(p57_windows_run, w)
    p = w / "windows.npz"
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(WindowContractError, match="MANIFEST_HASH_MISMATCH"):
        load_trained_model(train, w)
    with pytest.raises(WindowContractError, match="CHECKPOINT_NOT_IN_MANIFEST"):
        load_trained_model(train, p57_windows_run, checkpoint="checkpoints/nope.pt")


def test_predictions_follow_the_tensor_contract(p57_train_run, p57_windows_run):
    train, _, _ = p57_train_run
    lm = load_trained_model(train, p57_windows_run)
    ds = lm.dataset
    pred = predict_windows(lm.model, ds, 8)
    N, K, F = len(ds), 5, 15
    assert pred["positions_uv_px"].shape == (N, K, F, 2) and pred["valid"].shape == (N, K, F) and pred["mode_probabilities"].shape == (N, K)
    assert pred["positions_uv_px"].dtype == np.float32 and pred["valid"].all() and np.isfinite(pred["positions_uv_px"]).all()
    np.testing.assert_allclose(pred["mode_probabilities"].sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(pred["window_id"], ds.ws.window_id)
    assert np.array_equal(pred["anchor_uv_px"], ds.ws.history_target_uv_px[:, -1, :].astype(np.float32))        # raw pixel anchor
    # deterministic and independent of batch size
    again = predict_windows(lm.model, ds, 3)
    np.testing.assert_allclose(again["positions_uv_px"], pred["positions_uv_px"], atol=1e-3)
    assert np.array_equal(predict_windows(lm.model, ds, 8)["positions_uv_px"], pred["positions_uv_px"])


def test_inference_is_history_only_future_arrays_are_not_read(p57_train_run, p57_windows_run):
    train, _, _ = p57_train_run
    lm = load_trained_model(train, p57_windows_run)
    ds = lm.dataset
    base = predict_windows(lm.model, ds, 8)
    ds.ws.target_future_uv_px[:] = 1e9                              # destroy the ground truth in memory
    ds.ws.target_future_valid[:] = False
    ds.ws.future_frames[:] = -1
    mutated = predict_windows(lm.model, ds, 8)
    assert np.array_equal(base["positions_uv_px"], mutated["positions_uv_px"]) and np.array_equal(base["mode_probabilities"], mutated["mode_probabilities"])


def test_selected_mode_and_oracle_metrics_are_distinct_and_correct():
    N, K, F = 3, 2, 15
    gt = np.zeros((N, F, 2))
    gt[:, :, 0] = np.arange(F)[None] * 2.0
    good, bad = gt.copy(), gt.copy() + np.array([10.0, 0.0])
    pos = np.stack([np.stack([bad[i], good[i]]) for i in range(N)])                      # mode 0 bad (10 px off), mode 1 exact
    valid = np.ones((N, K, F), bool)
    probs = np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]])                                   # windows 0, 1 select the bad mode; window 2 the good one
    cfg = EvalConfig()
    r = evaluate_predictions(gt, np.ones((N, F), bool), pos, valid, np.arange(N), ["a", "a", "b"], coordinate_system="captured_1920x1080", unit="pixel", model_id="m",
                             horizon_offsets_s=np.arange(1, F + 1) / 10.0, cfg=cfg, mode_probabilities=probs)
    np.testing.assert_allclose(r.ade, [10.0, 10.0, 0.0])                                      # primary = most probable mode, NOT the closest
    np.testing.assert_allclose(r.fde, [10.0, 10.0, 0.0])
    np.testing.assert_allclose(r.min_ade, [0.0, 0.0, 0.0])                                    # oracle = closest mode
    s = r.summary
    assert s["window_weighted"]["ade_px"]["mean"] == pytest.approx(20 / 3) and s["oracle_best_of_k"]["min_ade_px"]["mean"] == 0.0
    assert s["track_weighted"]["ade_px_mean_of_track_means"] == pytest.approx((10.0 + 0.0) / 2)
    assert "ORACLE" in s["oracle_best_of_k"]["label"] and s["oracle_best_of_k"]["identical_to_ade_fde"] is False and s["miss_rate"]["status"] == "NOT_CONFIGURED"
    d = mode_diagnostics({"positions_uv_px": pos.astype(np.float32), "mode_probabilities": probs.astype(np.float32)}, gt, np.ones((N, F), bool))
    assert d["selected_equals_oracle_best_fraction"] == pytest.approx(1 / 3) and d["selected_mode_histogram"] == [2, 1] and d["oracle_best_mode_histogram"] == [0, 3]


def test_run_manifest_results_and_labels(eval_run):
    out, m = eval_run
    assert m["label"] == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" and m["sample_status"] == IN_SAMPLE and m["status"] == "completed"
    assert all(v is False for v in m["claims"].values() if isinstance(v, bool))
    assert set(m["data_artifacts"]) == {"predictions_neural.npz", "evaluation_neural.npz", "evaluation_results.json", "evaluation_report.md"}
    assert m["inputs"]["future_arrays_in_model_input"] is False and m["inputs"]["offline_kinematics_used"] is False and "git" in m["run_metadata"]
    res = json.loads((out / "evaluation_results.json").read_text(encoding="utf-8"))
    s = res["neural_model"]["summary"]
    assert res["sample_status"] == IN_SAMPLE and s["windows_evaluated"] == m["counts"]["windows"] == 20 and res["neural_model"]["num_modes"] == 5
    assert s["miss_rate"]["status"] == "NOT_CONFIGURED" and "ORACLE" in s["oracle_best_of_k"]["label"]
    assert s["oracle_best_of_k"]["min_ade_px"]["mean"] <= s["window_weighted"]["ade_px"]["mean"] + 1e-9
    assert {"window_weighted", "track_weighted", "per_horizon_mean_displacement_error_px"} <= set(s) and s["per_track"]
    rep = (out / "evaluation_report.md").read_text(encoding="utf-8")
    assert "IN-SAMPLE" in rep and "ORACLE" in rep and "not evidence of superiority" in rep and "most-probable" in rep.lower()
    assert "started_utc" not in json.dumps(res)


def test_baseline_comparison_uses_the_same_unchanged_evaluator(eval_run, p57_windows_run, tmp_path):
    out, m = eval_run
    res = json.loads((out / "evaluation_results.json").read_text(encoding="utf-8"))
    comp = res["baseline_comparison"]
    assert comp["n_windows"] == 20 and set(comp["per_model"]) == {"NEURAL_GATV2_TCN_K5", "B0_zero_velocity", "B1_last_valid_causal_velocity", "B2_history_linear_regression"}
    assert IN_SAMPLE in comp["status"]
    p54 = run_baseline_experiment(p57_windows_run, BCFG, tmp_path / "p54")                    # the real P5.4 pipeline
    p54r = json.loads((tmp_path / "p54" / "evaluation_results.json").read_text(encoding="utf-8"))
    for b in ("B0_zero_velocity", "B1_last_valid_causal_velocity", "B2_history_linear_regression"):
        assert comp["per_model"][b]["window_weighted"] == p54r["baselines"][b]["summary"]["window_weighted"]


def test_two_evaluations_are_bitwise_reproducible_and_validated(eval_run, p57_train_run, p57_windows_run, eval_cfg, tmp_path):
    out, m = eval_run
    train, _, _ = p57_train_run
    m2 = run_neural_evaluation(train, p57_windows_run, eval_cfg, tmp_path / "again")
    assert {k: v["sha256"] for k, v in m["data_artifacts"].items()} == {k: v["sha256"] for k, v in m2["data_artifacts"].items()}
    rep = phase5_validate_neural_evaluation.validate_run(out, compare_with=tmp_path / "again")
    assert rep["ok"], [k for k, v in rep["checks"].items() if not v["pass"]]
    assert "primary_most_probable_mode_ade_fde_recomputed" in rep["checks"] and "oracle_min_ade_fde_recomputed_separately" in rep["checks"]


def test_validator_detects_tampering(eval_run, tmp_path):
    out, m = eval_run
    bad = tmp_path / "bad"
    shutil.copytree(out, bad)
    p = bad / "predictions_neural.npz"
    p.write_bytes(p.read_bytes() + b"\0")
    assert not phase5_validate_neural_evaluation.validate_run(bad)["checks"]["data_artifact_hashes_match_manifest"]["pass"]
    # a consistent forgery of the saved predictions (hash updated) is caught by recomputation from the checkpoint
    forged = tmp_path / "forged"
    shutil.copytree(out, forged)
    arrs = load_npz(forged / "predictions_neural.npz")
    arrs["positions_uv_px"] = arrs["positions_uv_px"] + np.float32(1.0)
    from pedsafety.interaction.storage import write_npz
    sha = write_npz(forged / "predictions_neural.npz", arrs)
    mm = json.loads((forged / "run_manifest.json").read_text(encoding="utf-8"))
    mm["data_artifacts"]["predictions_neural.npz"] = {"sha256": sha, "bytes": (forged / "predictions_neural.npz").stat().st_size}
    (forged / "run_manifest.json").write_text(json.dumps(mm), encoding="utf-8")
    rep = phase5_validate_neural_evaluation.validate_run(forged)
    assert not rep["ok"] and not rep["checks"]["predictions_recomputed_from_checkpoint_bitwise"]["pass"]


def test_output_policy_and_cli(p57_train_run, p57_windows_run, eval_cfg, eval_run, tmp_path, capsys):
    train, _, _ = p57_train_run
    out, _ = eval_run
    with pytest.raises(OutputDirectoryError):
        run_neural_evaluation(train, p57_windows_run, eval_cfg, out)
    args = ["--train", str(train), "--windows", str(p57_windows_run), "--config", str(eval_cfg), "--out", str(tmp_path / "cli")]
    assert phase5_evaluate_neural.main(args) == 0 and "IN_SAMPLE" in capsys.readouterr().out
    assert phase5_evaluate_neural.main(args) == 2
    assert phase5_validate_neural_evaluation.main(["--run", str(tmp_path / "cli")]) == 0
    assert phase5_evaluate_neural.main(["--train", str(tmp_path / "nope"), "--windows", str(p57_windows_run), "--config", str(eval_cfg), "--out", str(tmp_path / "cli2")]) == 2
