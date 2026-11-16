"""INTEGRATION tests of the multi-video orchestration on synthetic data: the REAL Phase 4 / 5 / 6 / 7 / 8 code runs on four synthetic videos
in four recording sets. These test software behaviour (resume, partition isolation, provenance, labels); none is a scientific result."""

import json
import shutil

import numpy as np
import pytest

from mv_helpers import make_pie_tree, write_phase3_dir
from pedsafety.multivideo.common import MARKER, StageError, check_marker, read_json
from pedsafety.multivideo.downstream import downstream_batch, downstream_config
from pedsafety.multivideo.evaluation import evaluate_all, render_evaluation_markdown
from pedsafety.multivideo.inventory import inventory_pie
from pedsafety.multivideo.preprocess import preprocess_batch, preprocess_config
from pedsafety.multivideo.report import build_report, final_status, marker_table, validate_schemas
from pedsafety.multivideo.split import build_split_manifest, freeze_split, resolve_official_split, video_partitions
from pedsafety.multivideo.training import load_multivideo_model, train_multivideo

REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
CFGS = {"image_kinematics": "configs/phase4/image_kinematics_default.json", "interaction": "configs/phase4/interaction_representation_default.json",
        "windows": "configs/phase5/window_generation_default.json"}
DCFGS = {"conflict": "configs/phase6/conflict_default.json", "risk": "configs/phase7/risk_default.json", "alert": "configs/phase8/alert_default.json",
         "windows": "configs/phase5/window_generation_default.json"}
LAYOUT = {"set01": 1, "set02": 1, "set03": 1, "set05": 1}
TINY = {"epochs": 2, "batch_size": 16, "device": "cpu", "torch_threads": 1, "checkpoint_every_epochs": 1, "log_every_steps": 1, "scaler_batch_size": 16}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mv")
    pie = make_pie_tree(tmp, LAYOUT)
    inv = inventory_pie(pie, hash_cache_path=tmp / "hash_cache.json", progress=lambda s: None)
    off = resolve_official_split([pie])
    split = freeze_split(build_split_manifest(inv, off), tmp / "splits")["manifest"]
    parts = video_partitions(split)
    art, work, q = tmp / "art", tmp / "work", tmp / "quarantine"
    p3 = {k: write_phase3_dir(tmp / "phase3" / v["set"] / k, k, seed=i) for i, (k, v) in enumerate(sorted(parts.items()))}
    pc = preprocess_config(REPO, CFGS, "synthetic")
    items = [{"key": k, "set": parts[k]["set"], "phase3_dir": p3[k]} for k in sorted(parts)]
    pre = preprocess_batch(REPO, items, artifacts_root=art, work_root=work, quarantine_root=q, pcfg=pc, progress=lambda s: None)
    wdirs = {k: art / "phase5_windows" / parts[k]["set"] / k for k in parts}
    return {"tmp": tmp, "pie": pie, "inv": inv, "split": split, "parts": parts, "art": art, "work": work, "q": q, "p3": p3, "pc": pc, "items": items, "pre": pre, "wdirs": wdirs}


@pytest.fixture(scope="module")
def trained(world, tmp_path_factory):
    w = world
    train = {k: d for k, d in w["wdirs"].items() if w["parts"][k]["partition"] == "train"}
    val = {k: d for k, d in w["wdirs"].items() if w["parts"][k]["partition"] == "validation"}
    out = w["tmp"] / "training"
    m = train_multivideo(train, w["split"], REPO / "configs/phase5/training_default.json", out, overrides=TINY, val_dirs=val, quarantine_root=w["q"],
                         progress=lambda s: None)
    return out, m, train, val


def test_inventory_split_and_partitions_are_recording_level(world):
    s = world["split"]
    assert world["inv"]["n_videos"] == 4 and world["inv"]["status"] == "MISSING_REQUIRED_DATA"          # set04 and set06 are absent: reported, not assumed
    assert s["frozen"] and s["official_split"]["status"] == "VERIFIED_FROM_SOURCE"
    assert s["partitions"] == {"train": ["set01", "set02"], "calibration": [], "validation": ["set05"], "test": ["set03"]}
    assert s["sets_in_official_split_without_videos"] == ["set04", "set06"]
    for v, p in world["parts"].items():
        assert v.startswith(p["set"] + "_")


def test_preprocess_completes_and_windows_carry_video_and_group_ids(world):
    assert world["pre"]["all_complete"], world["pre"]["failed"]
    from pedsafety.phase5.window_storage import read_dataset
    for k, d in world["wdirs"].items():
        _, _, extra = read_dataset(d)
        assert set(extra["video_id"].tolist()) == {k} and set(extra["group_id"].tolist()) == {world["parts"][k]["set"]}
        assert check_marker(d, stage="S5A_WINDOWS", config_sha256=read_json(d / MARKER)["config_sha256"], verify="sha256")[0] == "COMPLETE"


def test_preprocess_resume_skips_complete_and_recomputes_only_missing_substage(world):
    w = world
    again = preprocess_batch(REPO, w["items"], artifacts_root=w["art"], work_root=w["work"], quarantine_root=w["q"], pcfg=w["pc"], progress=lambda s: None)
    assert set(r["status"] for r in again["results"].values()) == {"SKIPPED_ALREADY_COMPLETE"}
    k = w["items"][0]["key"]
    wd = w["wdirs"][k]
    old = read_json(wd / MARKER)["files"]["windows.npz"]["sha256"]
    (wd / MARKER).unlink()                                                                  # simulate a disconnect before the marker was written
    r = preprocess_batch(REPO, [w["items"][0]], artifacts_root=w["art"], work_root=w["work"], quarantine_root=w["q"], pcfg=w["pc"], progress=lambda s: None)["results"][k]
    assert r["status"] == "COMPLETED" and r["computed_substages"] == ["S5A_WINDOWS"]       # S4A / S4B restored, not recomputed
    assert read_json(wd / MARKER)["files"]["windows.npz"]["sha256"] == old                 # deterministic
    assert any(p.name.startswith("phase5_windows__" + k) or k in p.name for p in w["q"].iterdir())   # incomplete folder moved aside, not deleted


def test_stale_configuration_is_refused_not_reused(world):
    w = world
    pc = dict(w["pc"], phase3_config_sha256="different")
    r = preprocess_batch(REPO, [w["items"][0]], artifacts_root=w["art"], work_root=w["work"], quarantine_root=w["q"], pcfg=pc, progress=lambda s: None)
    assert r["results"][w["items"][0]["key"]]["error_code"] == "STALE_ARTIFACT"


def test_training_uses_train_partition_only_and_is_resumable(world, trained):
    out, m, train, val = trained
    assert m["status"] == "completed" and {v["video"] for v in m["inputs"]["train_videos"]} == set(train)
    assert m["inputs"]["scaler"]["fit_partition"] == "train" and m["inputs"]["scaler"]["fit_windows"] == m["inputs"]["train_windows"]
    assert m["training"]["split_policy"] == "RECORDING_LEVEL_FROZEN_SPLIT_TRAIN_PARTITION_ONLY" and m["training"]["checkpoint_selection"] == "FINAL_EPOCH_ONLY"
    log = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in log if r["type"] == "validation_monitor"] == [1, 2] and all(r["used_for_selection"] is False for r in log if r["type"] == "validation_monitor")
    with pytest.raises(StageError) as e:                                                    # a validation video may never be trained on
        train_multivideo(dict(train, **val), world["split"], REPO / "configs/phase5/training_default.json", world["tmp"] / "leak", overrides=TINY)
    assert e.value.code == "TRAIN_PARTITION_LEAK"
    again = train_multivideo(train, world["split"], REPO / "configs/phase5/training_default.json", out, overrides=TINY, val_dirs=val, progress=lambda s: None)
    assert again.get("skipped_already_complete")
    # interrupted after epoch 1: the epoch-2 checkpoint and the manifest are missing -> resume continues from epoch 1 and reproduces epoch 2 exactly (CPU, 1 thread)
    res = world["tmp"] / "training_resume"
    shutil.copytree(out, res)
    final_sha = m["data_artifacts"]["checkpoints/checkpoint_epoch_0002.pt"]["model_state_sha256"]
    (res / "run_manifest.json").unlink()
    (res / "checkpoints" / "checkpoint_epoch_0002.pt").unlink()
    m2 = train_multivideo(train, world["split"], REPO / "configs/phase5/training_default.json", res, overrides=TINY, val_dirs=val, progress=lambda s: None)
    assert m2["data_artifacts"]["checkpoints/checkpoint_epoch_0002.pt"]["model_state_sha256"] == final_sha
    with pytest.raises(StageError) as e:
        train_multivideo(train, world["split"], REPO / "configs/phase5/training_default.json", out, overrides=dict(TINY, learning_rate=0.001))
    assert e.value.code == "STALE_TRAINING_RUN"


def test_model_loader_verifies_hashes(trained, tmp_path):
    out, m, _, _ = trained
    model, man, name, rec, stats, ver, wcfg = load_multivideo_model(out)
    assert ver["config_hashes_verified"] and name == "checkpoints/checkpoint_epoch_0002.pt"
    bad = tmp_path / "bad"
    shutil.copytree(out, bad)
    with open(bad / "scaler_statistics.json", "a") as fh:
        fh.write(" ")
    with pytest.raises(StageError) as e:
        load_multivideo_model(bad)
    assert e.value.code == "TRAIN_ARTIFACT_HASH"


def test_evaluation_downstream_and_report(world, trained):
    w = world
    out, m, _, _ = trained
    ev_root = w["art"] / "phase5_evaluation"
    summ = evaluate_all(out, w["wdirs"], w["split"], ev_root, w["work"], w["q"], neural_eval_config_path=REPO / "configs/phase5/neural_evaluation_default.json", repo=REPO,
                        device="cpu", phase9_gate={"status": "BLOCKED_NO_VALID_FROZEN_SPLIT_MANIFEST"}, progress=lambda s: None)
    pp = summ["per_partition"]
    assert set(pp) == {"train", "validation", "test"}
    assert pp["train"]["sample_status"].startswith("IN_SAMPLE") and pp["test"]["sample_status"].startswith("HELD_OUT_RECORDING_PSEUDO_TARGET")
    for p in pp.values():
        s = p["primary_most_probable_mode"]
        assert s["windows_evaluated"] > 0 and "oracle_best_of_k" in s and s["oracle_best_of_k"]["k"] == 5
        assert s["oracle_best_of_k"]["min_ade_px"]["mean"] <= s["window_weighted"]["ade_px"]["mean"] + 1e-9   # oracle is separate (and never worse)
    assert (ev_root / "test_access_log.jsonl").is_file()
    assert "HELD_OUT" in render_evaluation_markdown(summ)
    dcfg = downstream_config(REPO, DCFGS)
    items = [{"key": k, "set": w["parts"][k]["set"], "partition": w["parts"][k]["partition"], "windows_dir": w["wdirs"][k],
              "predictions_dir": ev_root / w["parts"][k]["set"] / k} for k in sorted(w["parts"])]
    kw = dict(artifacts_root=w["art"], work_root=w["work"], quarantine_root=w["q"], dcfg=dcfg, progress=lambda s: None)
    ds = downstream_batch(REPO, items, stages=("S6_CONFLICT",), **kw)
    assert ds["all_complete"], ds
    ds = downstream_batch(REPO, items, stages=("S7_RISK",), **kw)
    assert ds["all_complete"], ds
    ds = downstream_batch(REPO, items, stages=("S8_ALERTS",), **kw)
    assert ds["all_complete"], ds
    full = downstream_batch(REPO, items, **kw)
    assert all(v["status"] == "SKIPPED_ALREADY_COMPLETE" for r in full["results"].values() for v in r["stages"].values())
    k = items[0]["key"]
    risk = read_json(w["art"] / "phase7_risk" / items[0]["set"] / k / "risk_report.json")
    assert risk["calibration_status"] != "CALIBRATED" and risk["calibration"]["status"] == "BLOCKED_NO_VERIFIED_CONFLICT_LABELS"
    arr = np.load(w["art"] / "phase7_risk" / items[0]["set"] / k / "risk_arrays.npz")
    assert np.isnan(arr["calibrated_probability"]).all()
    sv = validate_schemas(w["p3"][k], w["art"] / "phase8_alerts" / items[0]["set"] / k, w["art"] / "phase7_risk" / items[0]["set"] / k)
    assert sv["ok"], sv
    vids = {kk: w["parts"][kk]["set"] for kk in w["parts"]}
    mt = marker_table({"S5A_WINDOWS": w["art"] / "phase5_windows", "S8_ALERTS": w["art"] / "phase8_alerts"}, vids, verify="sha256")
    assert set(v for t in mt.values() for v in t.values()) == {"COMPLETE"}
    assert final_status([{"name": "a", "result": "PASS"}, {"name": "b", "result": "NOT_RUN"}]) == "INCOMPLETE"
    assert final_status([{"name": "a", "result": "PASS"}, {"name": "b", "result": "FAIL"}]) == "FAILED"
    assert final_status([{"name": "a", "result": "PASS"}]) == "COMPLETED"
    rep = build_report(experiment_id="t", exp_dir=w["tmp"], lock={"locked_config": {}}, inventory=w["inv"], split=w["split"], tracker_data={"stages": {}},
                       environment={}, git={}, markers=mt, phase3={"results": {}}, training=m, evaluation=summ, downstream=full, tests=None,
                       validation={"overall_status": "INCOMPLETE", "checks": []}, execution={"statement": "synthetic"}, deviations=["x"])
    assert rep["claims"]["calibrated_probability"] is False and rep["realtime"]["claim_supported"] is False
    assert "Final status: INCOMPLETE" in (w["tmp"] / "EXPERIMENT_REPORT.md").read_text()
