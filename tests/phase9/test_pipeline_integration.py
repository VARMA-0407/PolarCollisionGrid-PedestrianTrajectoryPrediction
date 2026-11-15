"""INTEGRATION and ROBUSTNESS tests on the actual project artifacts (PIE pilot, in-sample). These test software behaviour; none is a scientific result."""

import copy
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pedsafety.phase6.errors import ConflictContractError
from pedsafety.phase8.config import AlertConfig
from pedsafety.phase8.manager import AlertManager
from pedsafety.phase8.runner import run_alerts
from pedsafety.phase9.__main__ import main
from pedsafety.phase9.contracts import validate_alert_outputs, validate_risk_records
from pedsafety.phase9.orchestrator import PipelineConfig, run_pipeline
import pedsafety.phase9.orchestrator as orch

REPO = Path(__file__).resolve().parents[2]
CFG_PATH = REPO / "configs/phase9/pipeline_pie_video_0004.json"
REF = json.loads(CFG_PATH.read_text())["reference_runs"]
have = all((REPO / v).exists() for v in REF.values()) and (REPO / "PIE_pilot_inputs/tracked_agents.jsonl").exists()
real = pytest.mark.skipif(not have, reason="real PIE pilot artifacts not present")
INFER = ["S3_PERCEPTION", "S5B_FORECAST", "S6_CONFLICT", "S7_RISK", "S8_ALERTS", "S9_EVALUATION_GATES"]


def make_cfg(tmp_path, **over):
    d = json.loads(CFG_PATH.read_text())
    d.update({"output_root": str(tmp_path / "runs"), "report_dir": None, "benchmark": {"inference_warmup": 0, "inference_repeats": 0, "preprocessing_extra_repeats": 0}})
    d.update(over)
    return PipelineConfig.from_dict(d)


@pytest.fixture(scope="module")
def inference_run(tmp_path_factory):
    """Precomputed-artifact execution: Phase 3 artifacts validated, Phases 5B-8 executed live with the verified checkpoint."""
    tmp = tmp_path_factory.mktemp("p9inf")
    cfg = make_cfg(tmp, stages=INFER)
    return run_pipeline(cfg, run_id="inf", benchmark=False), tmp


@real
def test_precomputed_artifact_execution_completes_and_every_stage_is_recorded(inference_run):
    m, tmp = inference_run
    assert m["status"] == "COMPLETED"
    st = {s: r["status"] for s, r in m["stages"].items()}
    assert st["S3_PERCEPTION"] == st["S5B_FORECAST"] == st["S6_CONFLICT"] == st["S7_RISK"] == st["S8_ALERTS"] == st["S9_EVALUATION_GATES"] == "COMPLETED"
    assert st["S4A_IMAGE_KINEMATICS"] == st["S4B_INTERACTION"] == st["S5A_WINDOWS"] == "NOT_SELECTED"
    for s in ("S5B_FORECAST", "S6_CONFLICT", "S7_RISK", "S8_ALERTS"):
        r = m["stages"][s]
        assert r["wall_s"] > 0 and r["cpu_s"] >= 0
        assert r["process_peak_working_set_bytes_after"] is None or r["process_peak_working_set_bytes_after"] > 0
    assert (tmp / "runs" / "inf" / "run_manifest.json").is_file() and (tmp / "runs" / "inf" / "S5B_forecast" / "predictions_neural.npz").is_file()


@real
def test_stage_outputs_reproduce_the_committed_reference_runs_deterministically(inference_run):
    m, _ = inference_run
    assert m["stages"]["S5B_FORECAST"]["detail"]["determinism"]["predictions_equal_reference"] is True        # bitwise equal to the Phase 5 artifact
    assert m["stages"]["S7_RISK"]["detail"]["determinism"] == {"risk_score_equal_reference": True, "risk_category_equal_reference": True}
    assert m["stages"]["S8_ALERTS"]["detail"]["determinism"] == {"n_alerts_equal_reference": True, "transitions_equal_reference": True}
    c = m["counts"]
    assert (c["windows_predicted"], c["pairs"], c["pairs_valid"], c["alerts"], c["alert_events"]) == (2327, 8233, 8090, 28, 1200)


@real
def test_no_future_information_reaches_inference_and_inputs_are_history_only(inference_run):
    m, _ = inference_run
    d = m["stages"]["S5B_FORECAST"]["detail"]
    assert d["future_isolation"]["passed"] is True and d["future_isolation"]["windows_perturbed"] >= 40         # scrambling all future targets changed no prediction
    assert d["history_only"]["history_only"] is True and d["history_only"]["min_future_gap_frames"] >= 1
    assert d["verification"]["config_hashes_verified"] and d["verification"]["scaler_buffers_equal_statistics"]
    p = m["stages"]["S6_CONFLICT"]["detail"]["exclusions"]
    assert m["stages"]["S6_CONFLICT"]["status"] == "COMPLETED" and p["pairs"] == 8233


@real
def test_provenance_hashes_propagate_to_every_alert_event(inference_run):
    m, tmp = inference_run
    res = m["stages"]["S8_ALERTS"]
    assert res["detail"]["summary"]["n_alerts"] == 28
    with gzip.open(tmp / "runs" / "inf" / "S8_alerts" / "alert_events.jsonl.gz", "rt", encoding="utf-8") as f:
        events = [json.loads(x) for x in f]
    ck = "40aff17f1a40eb2da24d0e0b63245eecefa5bba4e3f6e44c272dc07f3ab71b44"
    assert events and all(e["hashes"]["source_checkpoint_file_sha256"] == ck and e["hashes"]["calibration_artifact_sha256"] is None for e in events)
    assert all(e["phase7"]["calibrated_probability"] is None for e in events)
    assert all(e["video_id"] == "pie:video_0004" and abs(e["observation_time_s"] - e["frame"] / 30.0) < 1e-9 for e in events)
    man = json.loads((tmp / "runs" / "inf" / "S8_alerts" / "run_manifest.json").read_text())
    assert man["phase7_provenance"]["phase9_stage"] == "S8" and man["hashes"]["source_checkpoint_file_sha256"] == ck


@real
def test_track_identity_and_frame_alignment_are_preserved_end_to_end(inference_run):
    from pedsafety.phase5.window_storage import read_dataset
    m, tmp = inference_run
    ws, _, _ = read_dataset(REPO / REF["windows_dir"])
    with gzip.open(tmp / "runs" / "inf" / "S8_alerts" / "phase7_records_input.jsonl.gz", "rt", encoding="utf-8") as f:
        recs = [json.loads(x) for x in f]
    validate_risk_records(recs, ws, 30.0)                                                                      # window id, pedestrian id and anchor of every record match the windows
    ids = {str(t) for t in ws.target_track_id}
    assert {r["pedestrian_id"] for r in recs} <= ids
    assert {a["pedestrian_id"] for a in json.loads((tmp / "runs" / "inf" / "S8_alerts" / "alerts.json").read_text())["alerts"]} <= ids


@real
def test_invalid_and_insufficient_evidence_is_preserved_not_converted(inference_run):
    m, _ = inference_run
    c = m["counts"]
    assert c["pairs_invalid"] == 143 and c["insufficient_evidence_records"] == 1231
    assert c["alert_transitions_to_insufficient_evidence"] > 0                                                  # lost evidence is visible as a state, never as LOW
    assert c["risk_categories"]["INSUFFICIENT_EVIDENCE"] == 1231


@real
def test_scientific_gates_report_blocked_not_fabricated(inference_run):
    m, _ = inference_run
    ev = m["stages"]["S9_EVALUATION_GATES"]["detail"]
    assert ev["independent_evaluation"]["status"] == "BLOCKED_INSUFFICIENT_INDEPENDENT_SCENES" and ev["independent_evaluation"]["fabricated_results"] is False
    assert len(ev["blocked_evaluations"]) == 5 and all(v["status"] == "BLOCKED_INSUFFICIENT_INDEPENDENT_SCENES" for v in ev["blocked_evaluations"].values())
    assert m["sample_status"] == "IN_SAMPLE_PHASE5_MODEL_ONE_VIDEO" and not any(m["claims"].values())
    assert "independent_validation" not in json.dumps(m["claims"])


@real
def test_stage3_video_execution_is_reported_blocked_with_recorded_performance(inference_run):
    m, _ = inference_run
    v = m["stages"]["S3_PERCEPTION"]["detail"]["video_execution"]
    assert v["executed"] is False and v["status"] in ("BLOCKED_ENVIRONMENT", "NOT_REQUESTED_NO_RAW_VIDEO")
    rec = v["recorded_phase3_run"]
    assert rec["frames_processed"] == 9840 and rec["frames_skipped"] == 0 and rec["caveat"].startswith("RECORDED")
    assert rec["frames_per_s_wall_including_decode_and_io"] == pytest.approx(9840 / rec["wall_time_s"])
    assert m["performance"]["realtime_assessment"]["claim_supported"] is False


@real
def test_windows_stage_reexecution_reproduces_reference_data_and_chain(tmp_path):
    """Full-stage execution of S3 + S5A (window generation re-run from the reference Task 3 run): data files equal the reference, provenance chain verified."""
    cfg = make_cfg(tmp_path, stages=["S3_PERCEPTION", "S5A_WINDOWS"])
    m = run_pipeline(cfg, run_id="w", benchmark=False)
    r = m["stages"]["S5A_WINDOWS"]
    assert m["status"] == "COMPLETED" and r["status"] == "COMPLETED"
    assert r["detail"]["determinism"]["equal"] is True and set(r["detail"]["determinism"]["files"]) == {"windows.npz", "graph_bank.npz", "rejected_candidates.npz"}
    assert all(r["detail"]["chain"].values()) and r["detail"]["history_only"]["history_only"]
    assert m["counts"]["windows_accepted"] == 2327


@real
def test_committed_full_pipeline_manifest_is_complete_and_consistent():
    p = REPO / "experiments/phase9_end_to_end_pie_video_0004/run_manifest.json"
    if not p.is_file():
        pytest.skip("full pipeline run not yet recorded")
    m = json.loads(p.read_text())
    assert m["status"] == "COMPLETED" and all(r["status"] == "COMPLETED" for r in m["stages"].values())
    for s in ("S4A_IMAGE_KINEMATICS", "S4B_INTERACTION", "S5A_WINDOWS"):
        assert m["stages"][s]["detail"]["determinism"]["equal"] is True                                          # re-executed data equals the Phase 4/5 reference runs
    perf = m["performance"]
    ssum = sum(m["stages"][s]["wall_s"] for s in perf["downstream_stages_measured"])
    assert perf["downstream_wall_s_first_pass"] == pytest.approx(ssum)
    assert perf["downstream_frames_per_s_offline_batch"] == pytest.approx(9840 / ssum)
    assert m["total_wall_s"] >= ssum and perf["realtime_assessment"]["claim_supported"] is False
    assert perf["inference_chain"]["steady_state"]["inference_chain_total"]["n"] >= 5 and len(perf["inference_chain"]["warmup_total_s"]) >= 1
    assert perf["preprocessing"]["stats"]["S4A"]["n"] >= 3


# ------------------------------------------------------------------ robustness / failure propagation
def tiny_phase3(tmp, rows):
    d = tmp / "p3"
    d.mkdir()
    (d / "tracked_agents.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "agent_states.jsonl").write_text("{}\n")
    (d / "run_manifest.json").write_text("{}")
    return d


def good_row(f, track=1, cls="pedestrian", **o):
    r = {"video_id": "v", "frame_index": f, "timestamp_s": f / 30.0, "track_id": track, "semantic_class": cls, "bbox_xyxy": [10.0, 10.0, 50.0, 90.0], "image_width": 1920, "image_height": 1080,
         "coordinate_system": "captured_1920x1080", "tracking_validity": True}
    r.update(o)
    return r


def test_missing_phase3_input_fails_clearly_and_stops_dependants(tmp_path):
    cfg = make_cfg(tmp_path, stages=["S3_PERCEPTION", "S4A_IMAGE_KINEMATICS", "S4B_INTERACTION"], phase3={"dir": str(tmp_path / "nope")})
    m = run_pipeline(cfg, run_id="x", benchmark=False)
    assert m["status"] == "FAILED" and m["stages"]["S3_PERCEPTION"]["error_code"] == "S3_MISSING"
    assert m["stages"]["S4A_IMAGE_KINEMATICS"]["status"] == "NOT_RUN" and m["stages"]["S4B_INTERACTION"]["status"] == "NOT_RUN"
    assert not (Path(m["run_dir"]) / "S4A_image_kinematics").exists()                                           # nothing was fabricated
    assert m["failure_and_limitation_register"]["run_failures"][0]["code"] == "S3_MISSING"


@pytest.mark.parametrize("rows,code", [
    ([good_row(0), good_row(1, coordinate_system="world_m")], "S3_CONTRACT"),                                    # invalid coordinates
    ([good_row(0), good_row(3, bbox_xyxy=[10, 10, float("nan"), 90])], "S3_CONTRACT"),                                # invalid numeric value
    ([good_row(5), good_row(4)], "S3_CONTRACT"),                                                                  # out-of-order frames
    ([good_row(0), good_row(1, timestamp_s=9.0)], "S3_CONTRACT"),                                                 # inconsistent timestamps / variable frame rate
])
def test_corrupted_phase3_rows_stop_the_pipeline_without_downstream_output(tmp_path, rows, code):
    d = tiny_phase3(tmp_path, rows)
    cfg = make_cfg(tmp_path, stages=["S3_PERCEPTION", "S4A_IMAGE_KINEMATICS"], phase3={"dir": str(d)})
    m = run_pipeline(cfg, run_id="c", benchmark=False)
    assert m["status"] == "FAILED" and m["stages"]["S3_PERCEPTION"]["error_code"] == code and m["stages"]["S4A_IMAGE_KINEMATICS"]["status"] == "NOT_RUN"


def test_truncated_json_line_in_phase3_is_a_contract_failure(tmp_path):
    d = tiny_phase3(tmp_path, [good_row(0)])
    (d / "tracked_agents.jsonl").write_text(json.dumps(good_row(0)) + "\n" + json.dumps(good_row(1))[:40] + "\n")
    m = run_pipeline(make_cfg(tmp_path, stages=["S3_PERCEPTION"], phase3={"dir": str(d)}), run_id="t", benchmark=False)
    assert m["status"] == "FAILED" and "not valid JSON" in m["stages"]["S3_PERCEPTION"]["error"]


def test_starting_mid_pipeline_without_a_reference_run_fails_clearly(tmp_path):
    m = run_pipeline(make_cfg(tmp_path, stages=["S6_CONFLICT"], reference_runs={}), run_id="m", benchmark=False)
    assert m["status"] == "FAILED" and m["stages"]["S6_CONFLICT"]["error_code"] == "MISSING_INPUT"


@real
def test_model_and_window_provenance_mismatch_is_rejected_not_bypassed(tmp_path):
    wdir = tmp_path / "windows_copy"
    shutil.copytree(REPO / REF["windows_dir"], wdir)
    man = json.loads((wdir / "run_manifest.json").read_text())
    man["finished_utc"] = "2000-01-01T00:00:00Z"                                                                  # any change of the window-run manifest changes its hash
    (wdir / "run_manifest.json").write_text(json.dumps(man))
    refs = dict(REF, windows_dir=str(wdir))
    m = run_pipeline(make_cfg(tmp_path, stages=["S3_PERCEPTION", "S5B_FORECAST", "S6_CONFLICT"], reference_runs=refs), run_id="mm", benchmark=False)
    assert m["status"] == "FAILED" and m["stages"]["S5B_FORECAST"]["error_code"] == "DATASET_PROVENANCE_MISMATCH" and m["stages"]["S6_CONFLICT"]["status"] == "NOT_RUN"


@real
def test_corrupted_window_artifact_is_detected_by_hash(tmp_path):
    wdir = tmp_path / "windows_trunc"
    shutil.copytree(REPO / REF["windows_dir"], wdir)
    b = (wdir / "windows.npz").read_bytes()
    (wdir / "windows.npz").write_bytes(b[:len(b) // 2])
    refs = dict(REF, windows_dir=str(wdir))
    m = run_pipeline(make_cfg(tmp_path, stages=["S3_PERCEPTION", "S5B_FORECAST"], reference_runs=refs), run_id="cw", benchmark=False)
    assert m["status"] == "FAILED" and m["stages"]["S5B_FORECAST"]["status"] == "FAILED"


@real
def test_checkpoint_hash_mismatch_is_rejected(tmp_path):
    tdir = tmp_path / "train_copy"
    shutil.copytree(REPO / REF["training_dir"], tdir)
    ck = next((tdir / "checkpoints").glob("*.pt"))
    b = bytearray(ck.read_bytes())
    b[len(b) // 2] ^= 0xFF
    ck.write_bytes(bytes(b))
    refs = dict(REF, training_dir=str(tdir))
    m = run_pipeline(make_cfg(tmp_path, stages=["S3_PERCEPTION", "S5B_FORECAST"], reference_runs=refs), run_id="ck", benchmark=False)
    assert m["status"] == "FAILED" and m["stages"]["S5B_FORECAST"]["status"] == "FAILED"
    assert m["stages"]["S5B_FORECAST"]["error_code"] == "TRAIN_ARTIFACT_HASH"                                       # the manifest-recorded checkpoint hash no longer matches the file


@real
def test_no_vehicle_context_fails_safely_and_produces_no_warning(tmp_path, monkeypatch):
    def none(*a, **k):
        raise ConflictContractError("NO_PAIRS", "no context agent was observed at any selected anchor; nothing to build")
    monkeypatch.setattr(orch, "build_conflict_pairs", none)
    m = run_pipeline(make_cfg(tmp_path, stages=["S3_PERCEPTION", "S5B_FORECAST", "S6_CONFLICT", "S7_RISK", "S8_ALERTS"]), run_id="nv", benchmark=False)
    assert m["status"] == "FAILED" and m["stages"]["S6_CONFLICT"]["error_code"] == "NO_PAIRS"
    assert m["stages"]["S7_RISK"]["status"] == m["stages"]["S8_ALERTS"]["status"] == "NOT_RUN"
    assert not (Path(m["run_dir"]) / "S8_alerts").exists()


@pytest.fixture(scope="module")
def real_records(inference_run):
    _, tmp = inference_run
    with gzip.open(tmp / "runs" / "inf" / "S8_alerts" / "phase7_records_input.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


ACFG = AlertConfig.load(REPO / "configs/phase8/alert_default.json")


@real
def test_missing_frames_and_long_gaps_never_create_alerts_from_nothing(real_records):
    base = run_alerts(real_records, ACFG)
    for cut in ((3000, 4500), (0, 10 ** 6)):                                                                    # a long observation gap, and no observation at all
        kept = [r for r in real_records if not cut[0] <= r["anchor_frame"] < cut[1]]
        res = run_alerts(kept, ACFG)
        assert res["summary"]["n_alerts"] <= base["summary"]["n_alerts"]
        validate_alert_outputs(res["events"], res["alerts"], kept, 30.0)
    assert run_alerts([], ACFG)["summary"]["n_alerts"] == 0


@real
def test_all_evidence_insufficient_yields_no_warning(real_records):
    ins = []
    for r in real_records:
        q = dict(r, risk_category="INSUFFICIENT_EVIDENCE", risk_score=None, reason="TEST", masks={"conflict_valid": False, "category_assigned": False, "out_of_distribution": False})
        ins.append(q)
    res = run_alerts(ins, ACFG)
    assert res["summary"]["n_alerts"] == 0 and not any(e["to_state"] in ("ACTIVE_WARNING", "ESCALATED_WARNING") for e in res["events"])


@real
def test_track_identity_change_is_never_merged(real_records):
    base = run_alerts(real_records, ACFG)
    hot = base["alerts"][0]
    cut = hot["activated_frame"] + 6
    changed = [dict(r, pedestrian_id=r["pedestrian_id"] + "#2") if (r["pedestrian_id"] == hot["pedestrian_id"] and r["anchor_frame"] > cut) else r for r in real_records]
    res = run_alerts(changed, ACFG)
    old = [a for a in res["alerts"] if a["pedestrian_id"] == hot["pedestrian_id"]]
    new = [a for a in res["alerts"] if a["pedestrian_id"] == hot["pedestrian_id"] + "#2"]
    assert old and old[0]["activated_frame"] == hot["activated_frame"]                                          # the original identity alerts exactly as before the change
    old_ids = {a["alert_id"] for a in old}
    assert all(a["alert_id"] not in old_ids for a in new)                                                        # the new identity never inherits the old alert
    ev_new = [e for e in res["events"] if e["pedestrian_id"].endswith("#2") and e["alert_id"]]
    assert all(e["alert_id"] not in old_ids for e in ev_new)
    assert all(a["clearance_kind"] == "EVIDENCE_LOST" for a in old if a["clearance_frame"] > cut)                  # the old identity lost evidence; it was not closed by the new one's data


@real
def test_out_of_order_and_repeated_timestamps_are_rejected_at_the_alert_boundary(real_records):
    from pedsafety.phase8.errors import TimestampOrderError
    from pedsafety.phase8.phase7_input import parse_record
    obs = [parse_record(r, i) for i, r in enumerate(real_records[:50])]
    f0 = obs[0].frame
    m = AlertManager(ACFG)
    m.step(f0 + 30, [])
    with pytest.raises(TimestampOrderError):
        m.step(f0, [o for o in obs if o.frame == f0])
    with pytest.raises(TimestampOrderError):
        m.step(f0 + 33, [o for o in obs if o.frame == f0][:1] or [obs[0]])                                       # an observation that belongs to another frame


def test_cli_exit_codes(tmp_path, capsys):
    assert main(["evaluation-status", "--repo", str(REPO)]) == 0
    out = capsys.readouterr().out
    assert "BLOCKED_INSUFFICIENT_INDEPENDENT_SCENES" in out and "READY_BUT_NOT_RUN" not in out
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert main(["run", "--config", str(bad)]) == 2
    cfg = json.loads(CFG_PATH.read_text())
    cfg.update({"output_root": str(tmp_path / "r"), "report_dir": None, "phase3": {"dir": str(tmp_path / "missing")}, "stages": ["S3_PERCEPTION"]})
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    assert main(["run", "--config", str(p), "--run-id", "z", "--no-benchmark"]) == 1                                  # a failed stage gives exit code 1
    assert main(["run", "--config", str(p), "--run-id", "z", "--no-benchmark"]) == 2                                  # an existing run directory is never overwritten
