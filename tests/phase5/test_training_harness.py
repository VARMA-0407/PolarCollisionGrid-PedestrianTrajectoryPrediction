"""P5.7.4: pilot training harness - configuration, seeds / determinism, checkpoints (round trip, resume, tamper), manifests, policies, CLI."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pedsafety.errors import ConfigError
from pedsafety.output_dir import OutputDirectoryError
from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.models.checkpoint import file_sha256, load_checkpoint, save_checkpoint, state_dict_sha256
from pedsafety.phase5.models.training_harness import TrainingConfig, build_optimizer, build_scheduler, epoch_order, resolve_device, run_training
from pedsafety.phase5.training_data import TrainingWindowDataset
from p57_helpers import write_tiny_training_config

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import phase5_train_pilot  # noqa: E402
import phase5_validate_training  # noqa: E402

DEFAULT = REPO / "configs" / "phase5" / "training_default.json"


def cfg_dict(**over):
    d = json.loads(DEFAULT.read_text(encoding="utf-8"))
    d.update(over)
    return d


# ---------------- configuration ----------------
def test_default_config_is_valid_and_development_only():
    c = TrainingConfig.load(DEFAULT)
    assert c.label == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" and c.split_policy == "PILOT_ALL_WINDOWS_AS_TRAIN_DEVELOPMENT_ONLY"
    assert c.checkpoint_selection == "FINAL_EPOCH_ONLY" and c.torch_threads == 1 and c.include_optional_context is False
    assert c.sha256() == TrainingConfig.load(DEFAULT).sha256() and c.sha256() != c.sha256_without_epochs()


@pytest.mark.parametrize("over,msg", [({"label": "INDEPENDENT"}, "label"), ({"epochs": 0}, "epochs"), ({"batch_size": -1}, "batch_size"),
                                      ({"learning_rate": 0}, "learning_rate"), ({"weight_decay": -1}, "weight_decay"), ({"grad_clip_norm": 0}, "grad_clip_norm"),
                                      ({"scheduler": "step"}, "scheduler"), ({"device": "tpu"}, "device"), ({"split_policy": "RANDOM_WINDOW_SPLIT"}, "split_policy"),
                                      ({"checkpoint_selection": "BEST_VALIDATION"}, "checkpoint_selection"), ({"include_optional_context": True}, "bicyclist"),
                                      ({"torch_threads": 0}, "torch_threads"), ({"seed": -1}, "seed"), ({"adam_beta1": 1.0}, "betas")])
def test_config_rejects_invalid_values(over, msg):
    with pytest.raises(ConfigError, match=msg):
        TrainingConfig.from_dict(cfg_dict(**over))


def test_config_rejects_unknown_and_missing_keys():
    with pytest.raises(ConfigError, match="unknown keys"):
        TrainingConfig.from_dict(cfg_dict(surprise=1))
    d = cfg_dict()
    del d["seed"]
    with pytest.raises(ConfigError, match="missing keys"):
        TrainingConfig.from_dict(d)


def test_epoch_order_depends_only_on_seed_and_epoch():
    a, b = epoch_order(7, 3, 50, True), epoch_order(7, 3, 50, True)
    assert np.array_equal(a, b) and sorted(a.tolist()) == list(range(50))
    assert not np.array_equal(a, epoch_order(7, 4, 50, True)) and not np.array_equal(a, epoch_order(8, 3, 50, True))
    assert np.array_equal(epoch_order(7, 3, 5, False), np.arange(5))


def test_optimizer_groups_and_scheduler():
    from pedsafety.phase5.models.forecaster import ForecastingModel
    from pedsafety.phase5.models import EncoderConfig
    from pedsafety.phase5.models.decoder import DecoderConfig
    from phase5_helpers import wcfg
    c = TrainingConfig.from_dict(cfg_dict())
    m = ForecastingModel(EncoderConfig.load(REPO / c.encoder_config_path), DecoderConfig.load(REPO / c.decoder_config_path), wcfg())
    opt = build_optimizer(m, c)
    assert type(opt).__name__ == "AdamW" and len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == c.weight_decay and opt.param_groups[1]["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in opt.param_groups[0]["params"]) and all(p.ndim < 2 for p in opt.param_groups[1]["params"])
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(m.parameters()))
    assert opt.param_groups[0]["lr"] == c.learning_rate and opt.param_groups[0]["betas"] == (c.adam_beta1, c.adam_beta2)
    s = build_scheduler(opt, c, 100)
    assert s.T_max == 100 and abs(s.eta_min - c.learning_rate * c.min_lr_ratio) < 1e-12
    assert build_scheduler(opt, TrainingConfig.from_dict(cfg_dict(scheduler="none")), 100) is None


def test_device_resolution():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        with pytest.raises(WindowContractError, match="CUDA_UNAVAILABLE"):
            resolve_device("cuda")


# ---------------- a recorded tiny run ----------------
def test_run_manifest_artifacts_and_policies(p57_train_run):
    out, m, cfgp = p57_train_run
    assert m["label"] == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" and m["status"] == "completed"
    t = m["training"]
    assert t["epochs"] == 2 and t["epochs_run_in_this_directory"] == 2 and t["optimizer"] == "AdamW" and t["checkpoint_selection"] == "FINAL_EPOCH_ONLY"
    assert "no best-checkpoint selection" in t["checkpoint_selection_limitation"] and t["epoch_summaries_in_sample"] is True
    assert m["claims"]["best_checkpoint_selected"] is False and m["claims"]["independent_validation"] is False and m["claims"]["independent_generalization"] is False
    assert all(v is False for v in m["claims"].values() if isinstance(v, bool))
    assert m["inputs"]["scaler"]["fit_partition"] == "train" and m["inputs"]["scaler"]["split_policy"] == "PILOT_ALL_WINDOWS_AS_TRAIN_DEVELOPMENT_ONLY"
    assert m["inputs"]["future_targets_in_model_input"] is False and m["inputs"]["offline_kinematics_used"] is False
    assert {"checkpoints/checkpoint_epoch_0001.pt", "checkpoints/checkpoint_epoch_0002.pt", "scaler_statistics.json", "train_log.jsonl", "config_resolved.json"} <= set(m["data_artifacts"])
    for k in ("git", "environment", "runtime_s", "peak_memory_bytes"):
        assert k in m["run_metadata"]
    assert m["run_metadata"]["environment"]["torch_threads"] == 1 and (out / "environment.json").is_file()
    assert any("one video" in s.lower() or "one pie video" in s.lower() for s in m["limitations"])


def test_independent_validator_passes_and_recomputes_scaler(p57_train_run):
    out, m, _ = p57_train_run
    rep = phase5_validate_training.validate_run(out, recompute_scaler=True)
    assert rep["ok"], [k for k, v in rep["checks"].items() if not v["pass"]]
    assert "scaler_statistics_independently_recomputed" in rep["checks"]


def test_log_is_deterministic_data_only_with_epoch_summaries(p57_train_run):
    out, m, _ = p57_train_run
    recs = [json.loads(l) for l in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    steps, epochs = [r for r in recs if r["type"] == "step"], [r for r in recs if r["type"] == "epoch"]
    assert len(epochs) == 2 and [e["epoch"] for e in epochs] == [1, 2] and len(steps) == m["training"]["final_global_step"] > 0
    assert epochs[-1] == m["training"]["final_epoch_summary"]
    assert all(np.isfinite(r["loss"]) and r["usable_windows"] > 0 and len(r["winner_counts"]) == 5 for r in steps)
    assert not any(k in r for r in recs for k in ("time", "seconds", "runtime", "timestamp"))


def test_two_identical_runs_have_identical_data_artifacts(p57_train_run, p57_windows_run, tmp_path):
    out, m, cfgp = p57_train_run
    m2 = run_training(p57_windows_run, cfgp, tmp_path / "again")
    assert {k: v["sha256"] for k, v in m["data_artifacts"].items()} == {k: v["sha256"] for k, v in m2["data_artifacts"].items()}
    assert phase5_validate_training.validate_run(out, compare_with=tmp_path / "again")["checks"]["deterministic_vs_second_run_all_data_artifacts_identical"]["pass"]


def test_different_seed_changes_the_run(p57_train_run, p57_windows_run, tmp_path):
    out, m, cfgp = p57_train_run
    c2 = write_tiny_training_config(tmp_path / "c2.json", seed=99)
    m2 = run_training(p57_windows_run, c2, tmp_path / "seed99")
    assert m2["data_artifacts"]["train_log.jsonl"]["sha256"] != m["data_artifacts"]["train_log.jsonl"]["sha256"]
    assert m2["data_artifacts"]["scaler_statistics.json"]["sha256"] == m["data_artifacts"]["scaler_statistics.json"]["sha256"]       # scaler is seed independent


def test_checkpoint_round_trip_restores_model_optimizer_scheduler(p57_train_run, p57_windows_run):
    out, m, _ = p57_train_run
    from pedsafety.phase5.models.neural_evaluation import load_trained_model
    loaded = load_trained_model(out, p57_windows_run)
    rec = m["data_artifacts"][m["training"]["final_checkpoint"]]
    p = load_checkpoint(out / m["training"]["final_checkpoint"], expected_file_sha256=rec["sha256"], expected_model_state_sha256=rec["model_state_sha256"])
    assert state_dict_sha256(loaded.model.state_dict()) == rec["model_state_sha256"] == p["model_state_sha256"]
    assert p["epoch"] == 2 and p["global_step"] == m["training"]["final_global_step"] and p["optimizer_state"] and p["scheduler_state"]["T_max"] == m["training"]["total_steps"]
    # optimizer / scheduler states restore into fresh objects
    from pedsafety.phase5.models.training_harness import TrainingConfig as TC
    tc = TC.from_dict(json.loads((out / "config_resolved.json").read_text(encoding="utf-8"))["training_config"])
    opt = build_optimizer(loaded.model, tc)
    sched = build_scheduler(opt, tc, m["training"]["total_steps"])
    load_checkpoint(out / m["training"]["final_checkpoint"], model=loaded.model, optimizer=opt, scheduler=sched)
    assert sched.last_epoch == m["training"]["final_global_step"] and len(opt.state_dict()["state"]) > 0
    assert opt.param_groups[0]["lr"] == p["scheduler_state"]["_last_lr"][0]


def test_resume_from_epoch_one_equals_the_continuous_run(p57_train_run, p57_windows_run, tmp_path):
    out, m, cfgp = p57_train_run
    ck1 = out / "checkpoints" / "checkpoint_epoch_0001.pt"
    m2 = run_training(p57_windows_run, cfgp, tmp_path / "resumed", resume_checkpoint=ck1)
    assert m2["training"]["epochs_run_in_this_directory"] == 1 and m2["training"]["resumed_from"] == str(ck1)
    a, b = m["data_artifacts"]["checkpoints/checkpoint_epoch_0002.pt"], m2["data_artifacts"]["checkpoints/checkpoint_epoch_0002.pt"]
    assert a["model_state_sha256"] == b["model_state_sha256"] and a["sha256"] == b["sha256"]      # byte-identical continuation


def test_resume_provenance_and_schedule_are_enforced(p57_train_run, p57_windows_run, tmp_path):
    out, m, cfgp = p57_train_run
    ck1 = out / "checkpoints" / "checkpoint_epoch_0001.pt"
    with pytest.raises(WindowContractError, match="RESUME_PROVENANCE_MISMATCH"):
        run_training(p57_windows_run, write_tiny_training_config(tmp_path / "s.json", seed=5), tmp_path / "r1", resume_checkpoint=ck1)
    with pytest.raises(WindowContractError, match="RESUME_SCHEDULE_MISMATCH"):
        run_training(p57_windows_run, write_tiny_training_config(tmp_path / "e.json", epochs=3), tmp_path / "r2", resume_checkpoint=ck1)
    with pytest.raises(WindowContractError, match="RESUME_NOTHING_TO_DO"):
        run_training(p57_windows_run, cfgp, tmp_path / "r3", resume_checkpoint=out / "checkpoints" / "checkpoint_epoch_0002.pt")
    with pytest.raises((WindowContractError, FileNotFoundError, OSError)):
        run_training(p57_windows_run, cfgp, tmp_path / "r4", resume_checkpoint=tmp_path / "missing.pt")


def test_checkpoint_tamper_and_overwrite_protection(p57_train_run, tmp_path):
    out, m, _ = p57_train_run
    src = out / "checkpoints" / "checkpoint_epoch_0002.pt"
    rec = m["data_artifacts"]["checkpoints/checkpoint_epoch_0002.pt"]
    bad = tmp_path / "bad.pt"
    raw = bytearray(src.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    bad.write_bytes(bytes(raw))
    with pytest.raises(WindowContractError, match="CHECKPOINT_HASH_MISMATCH"):
        load_checkpoint(bad, expected_file_sha256=rec["sha256"])
    with pytest.raises(WindowContractError, match="CHECKPOINT_HASH_MISMATCH|CHECKPOINT_UNREADABLE|CHECKPOINT_STATE_HASH_MISMATCH"):
        load_checkpoint(bad, expected_file_sha256=rec["sha256"], expected_model_state_sha256=rec["model_state_sha256"])
    with pytest.raises(WindowContractError, match="CHECKPOINT_STATE_HASH_MISMATCH"):
        load_checkpoint(src, expected_model_state_sha256="0" * 64)
    with pytest.raises(WindowContractError, match="CHECKPOINT_EXISTS"):
        save_checkpoint(src, model=torch.nn.Linear(1, 1), optimizer=None, scheduler=None, epoch=1, global_step=1, metadata={})
    with pytest.raises(WindowContractError, match="CHECKPOINT_METADATA"):
        save_checkpoint(tmp_path / "x.pt", model=torch.nn.Linear(1, 1), optimizer=None, scheduler=None, epoch=1, global_step=1, metadata={"t": torch.zeros(1)})
    assert file_sha256(src) == rec["sha256"]


def test_scaler_statistics_tamper_detected_by_validator(p57_train_run, tmp_path):
    out, m, _ = p57_train_run
    bad = tmp_path / "bad_run"
    shutil.copytree(out, bad)
    p = bad / "scaler_statistics.json"
    body = json.loads(p.read_text(encoding="utf-8"))
    body["node"]["mean"][0] += 1.0
    p.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(WindowContractError, match="SCALER_HASH_MISMATCH"):
        phase5_validate_training.validate_run(bad)                                       # hash / provenance verification fails loudly
    bad2 = tmp_path / "bad_run2"
    shutil.copytree(out, bad2)
    q = bad2 / "checkpoints" / "checkpoint_epoch_0001.pt"
    q.write_bytes(q.read_bytes() + b"\0")
    with pytest.raises(WindowContractError, match="CHECKPOINT_HASH_MISMATCH"):
        phase5_validate_training.validate_run(bad2)
    assert phase5_validate_training.main(["--run", str(bad2)]) == 2                      # surfaced as a contract failure by the CLI


def test_output_directory_is_never_overwritten(p57_train_run, p57_windows_run):
    out, m, cfgp = p57_train_run
    with pytest.raises(OutputDirectoryError):
        run_training(p57_windows_run, cfgp, out)


def test_tampered_window_run_is_rejected_before_training(p57_windows_run, p57_train_run, tmp_path):
    _, _, cfgp = p57_train_run
    bad = tmp_path / "w"
    shutil.copytree(p57_windows_run, bad)
    p = bad / "windows.npz"
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(WindowContractError, match="MANIFEST_HASH_MISMATCH"):
        run_training(bad, cfgp, tmp_path / "x")


def test_cli_exit_codes(p57_windows_run, tmp_path, capsys):
    cfgp = write_tiny_training_config(tmp_path / "c.json", epochs=1)
    args = ["--windows", str(p57_windows_run), "--config", str(cfgp), "--out", str(tmp_path / "cli")]
    assert phase5_train_pilot.main(args) == 0 and "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" in capsys.readouterr().out
    assert phase5_train_pilot.main(args) == 2
    assert phase5_validate_training.main(["--run", str(tmp_path / "cli")]) == 0


def test_validator_flags_a_modified_log(p57_train_run, tmp_path):
    out, m, _ = p57_train_run
    bad = tmp_path / "logbad"
    shutil.copytree(out, bad)
    (bad / "train_log.jsonl").write_text("{}\n" + (bad / "train_log.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    rep = phase5_validate_training.validate_run(bad)
    assert not rep["ok"] and not rep["checks"]["data_artifact_hashes_match_manifest"]["pass"]
