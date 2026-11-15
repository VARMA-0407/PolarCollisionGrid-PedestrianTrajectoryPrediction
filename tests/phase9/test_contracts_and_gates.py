"""UNIT tests (synthetic fixtures): stage contracts, benchmark helpers and scientific-evaluation gates. No scientific result is produced here."""

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pedsafety.phase9.benchmark import StageClock, measure_repeated, realtime_assessment, summarize
from pedsafety.phase9.contracts import (validate_alert_outputs, validate_phase3_rows, validate_risk_records, validate_windows_history_only, verify_chain, verify_outputs)
from pedsafety.phase9.errors import EvaluationBlocked, MissingInputError, PipelineError, StageContractError
from pedsafety.phase9.evaluation import (BLOCKED_SCENES, BLOCKED_SPLIT, READY, alert_level_metrics, assess_independent_evaluation, blocked_evaluations,
                                         evaluate_conflict_independent, evaluate_trajectories_independent, require_independent, validate_split_manifest)
from pedsafety.phase9.orchestrator import PipelineConfig

REPO = Path(__file__).resolve().parents[2]
FPS = 30.0


def row(frame, track=1, cls="pedestrian", **over):
    r = {"video_id": "v", "frame_index": frame, "timestamp_s": frame / FPS, "track_id": track, "semantic_class": cls, "bbox_xyxy": [100.0, 100.0, 200.0, 400.0], "image_width": 1920,
         "image_height": 1080, "coordinate_system": "captured_1920x1080", "tracking_validity": True}
    r.update(over)
    return r


# ------------------------------------------------------------------ Stage 3 contract
def test_phase3_valid_rows_pass_and_are_counted():
    s = validate_phase3_rows([row(0), row(1), row(1, track=2, cls="car")], FPS)
    assert s == {"rows": 3, "frames": 2, "tracks": 2, "video_id": "v", "first_frame": 0, "last_frame": 1}


@pytest.mark.parametrize("bad", [
    row(1, coordinate_system="world_m"), row(1, bbox_xyxy=[1, 1, float("nan"), 5]), row(1, bbox_xyxy=[200, 100, 100, 400]), row(1, bbox_xyxy=[100, 100, 5000, 400]),
    row(1, timestamp_s=1.0), row(-1), row(True), row(1, semantic_class="scooter"), {k: v for k, v in row(1).items() if k != "bbox_xyxy"}, row(1, video_id="other"),
])
def test_phase3_violations_are_rejected_not_repaired(bad):
    with pytest.raises(StageContractError):
        validate_phase3_rows([row(0), bad], FPS)


def test_phase3_duplicates_backwards_and_variable_frame_rate():
    with pytest.raises(StageContractError, match="duplicate"):
        validate_phase3_rows([row(3), row(3)], FPS)
    with pytest.raises(StageContractError, match="backwards"):
        validate_phase3_rows([row(5), row(4)], FPS)
    with pytest.raises(StageContractError, match="timestamp"):                                  # a 25 fps clock fed to a 30 fps pipeline
        validate_phase3_rows([row(f, timestamp_s=f / 25.0) for f in range(10, 14)], FPS)
    s = validate_phase3_rows([row(0), row(90)], FPS)                                                  # gaps and absent frames are allowed and stay gaps
    assert s["frames"] == 2


def test_manifest_hash_and_chain_verification(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"abc")
    import hashlib
    good = {"outputs": {"a.bin": {"sha256": hashlib.sha256(b"abc").hexdigest()}}}
    assert verify_outputs(tmp_path, good, ["a.bin"])["a.bin"] == good["outputs"]["a.bin"]["sha256"]
    with pytest.raises(StageContractError) as e:
        verify_outputs(tmp_path, {"outputs": {"a.bin": {"sha256": "0" * 64}}}, ["a.bin"])
    assert e.value.code == "ARTIFACT_HASH"
    with pytest.raises(StageContractError) as e:
        verify_outputs(tmp_path, good, ["missing.bin"])
    assert e.value.code == "MANIFEST_ENTRY"
    (tmp_path / "a.bin").write_bytes(b"abd")
    with pytest.raises(StageContractError) as e:
        verify_outputs(tmp_path, good, ["a.bin"])
    assert e.value.code == "ARTIFACT_HASH"


def make_ws(n=3):
    anchor = np.arange(n, dtype=np.int64) * 3 + 60
    hist = anchor[:, None] + np.arange(-15, 1, 3)[None, :]
    fut = anchor[:, None] + np.arange(3, 48, 3)[None, :]
    return SimpleNamespace(n_windows=n, anchor_frame=anchor, history_frames=hist, future_frames=fut, history_timestamps=hist / FPS, future_timestamps=fut / FPS,
                           window_id=np.arange(n), target_track_id=np.array(["pie:v:%d" % i for i in range(n)]))


def test_windows_history_only_and_alignment_gates():
    ws = make_ws()
    assert validate_windows_history_only(ws, FPS)["history_only"]
    bad = copy.deepcopy(ws)
    bad.history_frames[1, -1] += 3                                                                      # a "history" frame after the anchor
    with pytest.raises(StageContractError):
        validate_windows_history_only(bad, FPS)
    bad = copy.deepcopy(ws)
    bad.future_frames[0, 0] = bad.anchor_frame[0]                                                       # future overlapping the anchor
    with pytest.raises(StageContractError) as e:
        validate_windows_history_only(bad, FPS)
    assert e.value.code == "FUTURE_ORDER"
    bad = copy.deepcopy(ws)
    bad.history_timestamps[2, 0] += 0.5
    with pytest.raises(StageContractError) as e:
        validate_windows_history_only(bad, FPS)
    assert e.value.code == "TIMESTAMP_ALIGNMENT"


def rec(**over):
    r = {"pedestrian_id": "pie:v:0", "vehicle_id": "7", "anchor_frame": 60, "window_id": 0, "calibrated_probability": None, "calibration_status": "NOT_CALIBRATED",
         "risk_category": "HIGH", "risk_score": 0.5, "masks": {"conflict_valid": True, "category_assigned": True, "out_of_distribution": False}}
    r.update(over)
    return r


def test_risk_record_identity_frame_and_probability_semantics():
    ws = make_ws()
    assert validate_risk_records([rec()], ws, FPS)["records"] == 1
    for bad, code in ((rec(window_id=99), "RECORD_WINDOW"), (rec(anchor_frame=63), "RECORD_ANCHOR"), (rec(pedestrian_id="pie:v:2"), "RECORD_IDENTITY"),
                      (rec(calibrated_probability=0.4), "RECORD_PROBABILITY"),
                      (rec(risk_category="INSUFFICIENT_EVIDENCE"), "RECORD_INSUFFICIENT")):
        with pytest.raises(StageContractError) as e:
            validate_risk_records([bad], ws, FPS)
        assert e.value.code == code
    ok = rec(risk_category="INSUFFICIENT_EVIDENCE", risk_score=None, masks={"conflict_valid": False, "category_assigned": False, "out_of_distribution": False})
    assert validate_risk_records([ok], ws, FPS)["insufficient_evidence"] == 1


def ev(frame=60, to="ACTIVE_WARNING", frm="WARNING_PENDING", valid=True, prob=None, status="NOT_CALIBRATED"):
    return {"event_id": "e", "event_type": "STATE_TRANSITION", "observation_time_s": frame / FPS, "frame": frame, "from_state": frm, "to_state": to, "evidence_valid": valid,
            "phase7": {"calibrated_probability": prob, "calibration_status": status}}


def alert(**over):
    a = {"alert_id": "a1", "pedestrian_id": "pie:v:0", "vehicle_id": "7", "clearance_frame": 90, "activated_frame": 60, "dedup_key": "k"}
    a.update(over)
    return a


def test_alert_outputs_never_arise_from_missing_evidence():
    recs = [rec()]
    assert validate_alert_outputs([ev()], [alert()], recs, FPS)["alerts"] == 1
    with pytest.raises(StageContractError) as e:
        validate_alert_outputs([ev(valid=False)], [alert()], recs, FPS)
    assert e.value.code == "ALERT_FROM_INVALID_EVIDENCE"
    with pytest.raises(StageContractError) as e:
        validate_alert_outputs([{**ev(), "observation_time_s": 9.9}], [alert()], recs, FPS)
    assert e.value.code == "EVENT_TIME"
    with pytest.raises(StageContractError) as e:
        validate_alert_outputs([ev()], [alert(pedestrian_id="pie:v:5")], recs, FPS)
    assert e.value.code == "ALERT_WITHOUT_HIGH_RISK"
    with pytest.raises(StageContractError) as e:
        validate_alert_outputs([ev()], [alert(clearance_frame=None)], recs, FPS)
    assert e.value.code == "ALERT_NOT_CLOSED"
    with pytest.raises(StageContractError) as e:
        validate_alert_outputs([ev(prob=0.9)], [alert()], recs, FPS)
    assert e.value.code == "EVENT_PROBABILITY"
    low = [rec(risk_category="MODERATE")]
    with pytest.raises(StageContractError):
        validate_alert_outputs([ev()], [alert()], low, FPS)                                             # an alert needs a HIGH/CRITICAL Phase 7 record


# ------------------------------------------------------------------ config
def test_pipeline_config_validation():
    d = json.loads((REPO / "configs/phase9/pipeline_pie_video_0004.json").read_text())
    assert PipelineConfig.from_dict(d).sha256() == PipelineConfig.from_dict(copy.deepcopy(d)).sha256()
    for k, v in (("label", "VALIDATED"), ("stages", ["S3_PERCEPTION", "S99"]), ("stages", []), ("inference_windows", "x"), ("benchmark", {"inference_warmup": -1, "inference_repeats": 1,
                 "preprocessing_extra_repeats": 0}), ("streaming_capable", "yes"), ("target_fps", 0)):
        with pytest.raises(PipelineError):
            PipelineConfig.from_dict({**d, k: v})
    with pytest.raises(PipelineError):
        PipelineConfig.from_dict({**d, "surprise": 1})


# ------------------------------------------------------------------ engineering measurement correctness
def test_statistics_and_warmup_separation():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == 2.5 and s["min"] == 1.0 and s["max"] == 4.0 and s["median"] == 2.5 and s["std_sample"] == pytest.approx(1.2909944, rel=1e-6)
    assert summarize([5.0])["std_sample"] is None and summarize([]) is None
    calls = []

    def fn():
        calls.append(1)
        time.sleep(0.02 if len(calls) > 2 else 0.2)                                                      # the first two (warm-up) calls are deliberately slow
    r = measure_repeated(fn, warmup=2, repeats=3)
    assert len(calls) == 5 and len(r["warmup_s"]) == 2 and len(r["steady_s"]) == 3
    assert min(r["warmup_s"]) > 0.15 and r["steady_stats"]["max"] < 0.15                                  # warm-up excluded from the steady-state statistics
    with pytest.raises(ValueError):
        measure_repeated(fn, 0, 0)


def test_stage_clock_records_wall_cpu_and_memory():
    c = StageClock()
    with c.stage("sleep"):
        time.sleep(0.05)
    with c.stage("spin"):
        t = time.process_time()
        while time.process_time() - t < 0.05:
            pass
    assert c.records["sleep"]["wall_s"] >= 0.05 and c.records["sleep"]["cpu_s"] < 0.04
    assert c.records["spin"]["cpu_s"] >= 0.05 and c.records["spin"]["mean_cores_busy"] > 0.5
    assert c.records["spin"]["process_peak_working_set_bytes_after"] is None or c.records["spin"]["process_peak_working_set_bytes_after"] > 1e6
    with pytest.raises(RuntimeError):
        with c.stage("boom"):
            raise RuntimeError("x")
    assert "boom" in c.records                                                                           # a failing stage is still timed


def test_realtime_claim_requires_every_condition():
    ok = realtime_assessment(fps_target=30, frames=3000, measured_seconds=60, perception_executed=True, streaming_capable=True)
    assert ok["claim_supported"] and ok["unmet_conditions"] == [] and ok["measured_frames_per_s_of_measured_stages"] == 50
    fast_module = realtime_assessment(fps_target=30, frames=9840, measured_seconds=10, perception_executed=False, streaming_capable=False)
    assert not fast_module["claim_supported"] and len(fast_module["unmet_conditions"]) == 2                # 984 fps of an isolated part is not a real-time claim
    slow = realtime_assessment(fps_target=30, frames=3000, measured_seconds=200, perception_executed=True, streaming_capable=True)
    assert not slow["claim_supported"] and "15.00" in slow["unmet_conditions"][0]
    assert not realtime_assessment(fps_target=30, frames=1, measured_seconds=0, perception_executed=True, streaming_capable=True)["claim_supported"]


# ------------------------------------------------------------------ independent evaluation gates
def group(g, vid, **over):
    r = {"videos": [vid], "annotation": {"name": "ann", "type": "official_dataset_annotation", "version": "1", "sha256": "a" * 64, "protocol_reference": "doc", "verified": True},
         "identity_mapping": {"verified": True}, "timestamp_alignment": {"verified": True}, "license": {"name": "L", "archive_sha256": "b" * 64, "acquisition_record": "rec"}}
    r.update(over)
    return r


def split(**over):
    m = {"manifest_version": "phase9-group-split-1.0.0", "frozen": True, "frozen_utc": "2026-12-01T00:00:00Z", "purge_gap_frames": 90,
         "partitions": {"train": ["g1", "g2"], "calibration": ["g3"], "validation": ["g4"], "test": ["g5", "g6"]},
         "groups": {g: group(g, "video_%s" % g) for g in ("g1", "g2", "g3", "g4", "g5", "g6")}}
    m.update(over)
    return m


def test_split_manifest_validation_problems():
    assert validate_split_manifest(split()) == []
    cases = {"overlap": split(partitions={"train": ["g1", "g2"], "calibration": ["g3"], "validation": ["g4"], "test": ["g2", "g5", "g6"]}),
             "unfrozen": split(frozen=False), "empty_test": split(partitions={"train": ["g1", "g2"], "calibration": ["g3"], "validation": ["g4"], "test": []}),
             "unassigned": {**split(), "groups": {**split()["groups"], "g7": group("g7", "video_g7")}},
             "video_overlap": {**split(), "groups": {**split()["groups"], "g6": group("g6", "video_g5")}},
             "unverified_annotation": {**split(), "groups": {**split()["groups"], "g5": group("g5", "video_g5", annotation={"verified": False})}},
             "unverified_mapping": {**split(), "groups": {**split()["groups"], "g5": group("g5", "video_g5", identity_mapping={"verified": False})}},
             "no_licence": {**split(), "groups": {**split()["groups"], "g5": group("g5", "video_g5", license={})}},
             "no_purge": split(purge_gap_frames=None), "bad_version": split(manifest_version="x")}
    for name, m in cases.items():
        assert validate_split_manifest(m), name


def fake_repo(tmp_path, n_groups):
    for i in range(n_groups):
        d = tmp_path / "experiments" / ("phase4_task3_pie_video_%04d" % i) / "r" / "primary"
        d.mkdir(parents=True)
        (d / "tracks.json").write_text(json.dumps({"tracks": [{"track_id": "pie:video_%04d:1" % i}]}))
    return tmp_path


def test_independent_evaluation_status_levels(tmp_path):
    assert assess_independent_evaluation(REPO)["status"] == BLOCKED_SCENES                              # the actual repository: one scene group
    one = assess_independent_evaluation(fake_repo(tmp_path / "one", 1))
    assert one["status"] == BLOCKED_SCENES and one["fabricated_results"] is False
    r3 = fake_repo(tmp_path / "three", 3)
    assert assess_independent_evaluation(r3)["status"] == BLOCKED_SPLIT                                  # scenes exist, no frozen split
    p = tmp_path / "split.json"
    p.write_text(json.dumps(split(frozen=False)))
    st = assess_independent_evaluation(r3, p)
    assert st["status"] == BLOCKED_SPLIT and st["problems"]
    p.write_text(json.dumps(split()))
    assert assess_independent_evaluation(r3, p)["status"] == READY


def ready():
    return {"status": READY, "problems": []}


def test_require_independent_partition_isolation():
    s = split()
    require_independent(ready(), s, evaluated_groups=["g5", "g6"], model_training_groups=["g1", "g2"])
    for ev_g, tr_g in ((["g5"], ["g1", "g5"]), (["g4"], ["g1"]), (["g5", "g3"], ["g1"]), ([], ["g1"]), (["g5"], ["g6"])):                    # leak, wrong partition, empty, non-train model
        with pytest.raises(EvaluationBlocked):
            require_independent(ready(), s, evaluated_groups=ev_g, model_training_groups=tr_g)
    with pytest.raises(EvaluationBlocked) as e:
        require_independent({"status": BLOCKED_SCENES, "problems": ["x"]}, s, evaluated_groups=["g5"], model_training_groups=["g1"])
    assert e.value.code == BLOCKED_SCENES


def test_real_repository_blocks_every_independent_metric():
    a = assess_independent_evaluation(REPO)
    assert a["status"] == BLOCKED_SCENES
    b = blocked_evaluations(a)
    assert len(b) == 5 and all(v["status"] == BLOCKED_SCENES for v in b.values())
    with pytest.raises(EvaluationBlocked):
        evaluate_trajectories_independent(a, split(), evaluated_groups=["g5"], model_training_groups=["g1"], gt_uv=None, gt_valid=None, pred_uv=None, pred_valid=None, window_id=None,
                                          track_id=None, mode_probabilities=None, horizon_offsets_s=None, eval_cfg=None, model_id="m")
    with pytest.raises(EvaluationBlocked):
        evaluate_conflict_independent(a, split(), evaluated_groups=["g5"], model_training_groups=["g1"], scores=None, labels=None, threshold=0.5)
    with pytest.raises(EvaluationBlocked):
        alert_level_metrics([], [], {}, fps=30, observed_duration_s=100, match_horizon_frames=30, assessment=a)


def test_trajectory_metric_arithmetic_with_synthetic_arrays_only():
    """Formula check on synthetic arrays under a synthetic READY gate; the output is NOT a project result."""
    from pedsafety.phase5.evaluation.experiment import eval_config_from, load_baseline_config
    ecfg = eval_config_from(load_baseline_config(REPO / "configs/phase5/baseline_evaluation_default.json")[0])
    N, K, F = 4, 3, 15
    gt = np.zeros((N, F, 2)) + np.arange(F)[None, :, None] * 5.0
    pred = np.repeat(gt[:, None], K, axis=1).copy()
    pred[:, 1] += 7.0                                                                                      # a worse second mode
    probs = np.tile([0.7, 0.2, 0.1], (N, 1))
    out = evaluate_trajectories_independent(ready(), split(), evaluated_groups=["g5"], model_training_groups=["g1"], gt_uv=gt, gt_valid=np.ones((N, F), bool), pred_uv=pred,
                                            pred_valid=np.ones((N, K, F), bool), window_id=np.arange(N), track_id=np.array(["t%d" % i for i in range(N)]), mode_probabilities=probs,
                                            horizon_offsets_s=np.arange(1, F + 1) / 10.0, eval_cfg=ecfg, model_id="synthetic")
    assert out["sample_status"] == "INDEPENDENT_HELD_OUT_GROUPS"
    ade = out["summary"]["window_weighted"]["ade_px"]["mean"]
    assert ade == pytest.approx(0.0, abs=1e-9)                                                             # primary metric uses the most-probable mode (exact here)


def test_alert_level_metric_arithmetic_with_synthetic_events_only():
    src = {"name": "n", "type": "independent_human_annotation", "version": "1", "sha256": "c" * 64}
    alerts = [{"alert_id": "a1", "dedup_key": "p|1", "activated_frame": 100, "clearance_frame": 160}, {"alert_id": "a2", "dedup_key": "p|1", "activated_frame": 120, "clearance_frame": 150},
              {"alert_id": "a3", "dedup_key": "p|2", "activated_frame": 500, "clearance_frame": 530}]
    events = [{"verified": True, "dedup_key": "p|1", "event_frame": 130}, {"verified": True, "dedup_key": "p|9", "event_frame": 700}]
    m = alert_level_metrics(alerts, events, src, fps=30.0, observed_duration_s=3600.0, match_horizon_frames=60, assessment=ready())
    assert m["alert_recall"] == 0.5 and m["missed_event_rate"] == 0.5
    assert m["alert_precision"] == pytest.approx(2 / 3) and m["false_alerts_per_hour"] == 1.0           # a3 matches nothing; a1 and a2 both match the first event
    assert m["duplicate_alert_rate"] == pytest.approx(1 / 3) and m["warning_coverage"] == 0.5
    assert m["lead_time_s"]["mean"] == pytest.approx(1.0) and m["time_to_clearance_s"]["n"] == 3
    with pytest.raises(EvaluationBlocked):
        alert_level_metrics(alerts, [{**events[0], "verified": False}], src, fps=30, observed_duration_s=1, match_horizon_frames=60, assessment=ready())
    with pytest.raises(EvaluationBlocked):
        alert_level_metrics(alerts, events, {**src, "type": "perceived_tracks"}, fps=30, observed_duration_s=1, match_horizon_frames=60, assessment=ready())
