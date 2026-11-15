"""Tests of the honesty properties: execution mode, unavailable measurements, blocked claims, exact reference equality, end-to-end isolation, input immutability."""

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

import pedsafety.phase9.orchestrator as orch
from pedsafety.phase9.orchestrator import PipelineConfig, run_pipeline
from pedsafety.phase9.reporting import UNAVAILABLE, build_attribution, execution_mode
from pedsafety.phase9.verification import compare_run_with_references, hash_inputs
from pedsafety.phase6.runner import REPO

CFG_PATH = REPO / "configs/phase9/pipeline_pie_video_0004.json"
CFGD = json.loads(CFG_PATH.read_text())
have = all((REPO / v).exists() for v in CFGD["reference_runs"].values()) and (REPO / "PIE_pilot_inputs/tracked_agents.jsonl").exists()
real = pytest.mark.skipif(not have, reason="real PIE pilot artifacts not present")
INFER = ["S3_PERCEPTION", "S5B_FORECAST", "S6_CONFLICT", "S7_RISK", "S8_ALERTS", "S9_EVALUATION_GATES"]
COMMITTED = REPO / "experiments/phase9_end_to_end_pie_video_0004"


def cfg(tmp, **over):
    d = copy.deepcopy(CFGD)
    d.update({"output_root": str(tmp / "runs"), "report_dir": None, "benchmark": {"inference_warmup": 0, "inference_repeats": 0, "preprocessing_extra_repeats": 0}})
    d.update(over)
    return PipelineConfig.from_dict(d)


@pytest.fixture(scope="module")
def run_a(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lim_a")
    before = hash_inputs(CFGD, REPO)
    m = run_pipeline(cfg(tmp, stages=INFER), run_id="a", benchmark=False)
    return m, tmp, before


def events_of(tmp, rid):
    with gzip.open(tmp / "runs" / rid / "S8_alerts" / "alert_events.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


# ------------------------------------------------------------------ execution mode and unavailable measurements
@real
def test_manifest_states_the_execution_mode_and_never_claims_raw_video(run_a):
    m, _, _ = run_a
    assert m["execution_mode"] == "PRECOMPUTED_PHASE3_ARTIFACTS" and m["raw_video_executed"] is False
    assert m["raw_video_to_alert_pipeline_success"] is False
    assert "NOT a raw-video-to-alert execution" in m["statement"]
    v = m["stages"]["S3_PERCEPTION"]["detail"]["video_execution"]
    assert v["executed"] is False and v["status"].startswith(("BLOCKED", "NOT_REQUESTED"))
    assert m["stages"]["S3_PERCEPTION"]["detail"]["mode"] == "PRECOMPUTED_PHASE3_ARTIFACTS_VALIDATED"


@real
def test_incomplete_stage_selection_is_not_reported_as_integration_success(run_a):
    m, _, _ = run_a                                                                                        # S4A/S4B/S5A were not executed in this run
    assert m["status"] == "COMPLETED" and m["phase4_to_phase8_chain_complete"] is False
    assert m["precomputed_artifact_integration_success"] is False and m["raw_video_to_alert_pipeline_success"] is False


def test_failed_run_is_never_success(tmp_path):
    m = run_pipeline(cfg(tmp_path, stages=["S3_PERCEPTION", "S4A_IMAGE_KINEMATICS"], phase3={"dir": str(tmp_path / "none")}), run_id="f", benchmark=False)
    assert m["status"] == "FAILED" and m["precomputed_artifact_integration_success"] is False and m["raw_video_to_alert_pipeline_success"] is False


@real
def test_unavailable_stages_have_null_not_zero_timings(run_a):
    m, _, _ = run_a
    att = m["performance_attribution"]
    for k in ("video_decoding", "phase3_perception"):
        c = att["categories"][k]
        assert c["status"] == UNAVAILABLE and c["mean_s"] is None and c["median_s"] is None and c["n"] == 0
    assert att["categories"]["phase3_artifact_validation"]["scope"].endswith("NOT perception time")
    rec = att["categories"]["phase3_perception"]["recorded_not_measured_here"]
    assert rec["caveat"].startswith("RECORDED") and rec["frames_processed"] == 9840                             # recorded Colab values are labelled, never mixed with local ones
    assert "NOT raw-video end-to-end FPS" in att["phase4_to_phase8_offline_batch_throughput"]["WARNING"]
    assert att["memory"]["psutil"] == "unavailable" and att["memory"]["per_stage_isolated"] is False and att["memory"]["gpu_memory"] == "NOT_APPLICABLE_NO_CUDA_DEVICE"


@real
def test_missing_performance_metrics_are_explicit_in_attribution(run_a):
    m, _, _ = run_a
    assert m["performance"].get("inference_chain") is None                                                     # benchmark disabled in this run
    for k in ("phase5_inference_model_forward", "phase6_conflict_processing", "phase7_risk_processing", "phase8_alert_processing"):
        c = m["performance_attribution"]["categories"][k]
        assert c["status"] == UNAVAILABLE and c["mean_s"] is None and c["n"] == 0                                 # not measured => explicit, never an invented number
    assert m["performance_attribution"]["categories"]["file_io_and_serialization"]["status"] == "PARTIALLY_MEASURED"


def test_attribution_from_a_synthetic_manifest_without_any_timing():
    man = {"status": "COMPLETED", "stages": {"S3_PERCEPTION": {"status": "COMPLETED", "wall_s": 1.0, "detail": {"video_execution": {"executed": False}}}},
           "performance": {}, "counts": {}}
    att = build_attribution(man)
    assert att["raw_video_to_alert_pipeline_success"] is False and att["categories"]["phase5_inference_model_forward"]["status"] == UNAVAILABLE
    assert att["phase4_to_phase8_offline_batch_throughput"]["frames_per_s"] is None
    assert execution_mode({"status": "COMPLETED", "stages": {"S3_PERCEPTION": {"detail": {"video_execution": {"executed": True}}}}})["raw_video_executed"] is True


@real
def test_no_physical_quantities_or_generalization_claims_are_generated(run_a):
    m, tmp, _ = run_a
    assert not any(m["claims"].values()) and set(m["claims"]) >= {"independent_generalization", "physical_ttc_pet", "real_time_performance", "deployment_ready"}
    forbidden = ("ttc", "pet_s", "meter", "metre", "_m_", "distance_m", "speed_mps", "mph", "kmh")

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield str(k).lower()
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    bad = [k for k in keys(m) if any(f in k for f in forbidden) and k != "physical_ttc_pet"]
    assert bad == []
    ev = events_of(tmp, "a")
    assert all(e["severity"]["value"] is None and e["predicted_conflict_time"]["value"] is None and e["verified_event_time"]["value"] is None for e in ev)
    assert m["scientific_evaluation"]["independent"] == "BLOCKED_INSUFFICIENT_INDEPENDENT_SCENES" and m["sample_status"] == "IN_SAMPLE_PHASE5_MODEL_ONE_VIDEO"
    assert m["performance"]["realtime_assessment"]["claim_supported"] is False


@real
def test_phase7_calibration_and_phase8_insufficient_evidence_are_preserved(run_a):
    m, tmp, _ = run_a
    ev = events_of(tmp, "a")
    assert all(e["phase7"]["calibrated_probability"] is None and e["phase7"]["calibration_status"] in (None, "NOT_CALIBRATED") for e in ev)
    ins = [e for e in ev if e["to_state"] == "INSUFFICIENT_EVIDENCE"]
    assert ins and all(e["evidence_valid"] is False for e in ins)
    assert not any(e["to_state"] in ("ACTIVE_WARNING", "ESCALATED_WARNING") and e["evidence_valid"] is False and e["from_state"] not in ("ACTIVE_WARNING", "ESCALATED_WARNING") for e in ev)
    assert m["counts"]["insufficient_evidence_records"] == 1231


# ------------------------------------------------------------------ exact comparison with the committed reference artifacts
@real
def test_outputs_exactly_equal_the_committed_phase5_phase7_phase8_artifacts(run_a):
    m, tmp, _ = run_a
    chk = compare_run_with_references(tmp / "runs" / "a", CFGD, REPO)
    assert chk["all_compared_checks_pass"] is True
    assert chk["phase5_predictions_npz"]["sha256_equal"] and chk["phase5_predictions_npz"]["arrays"]["n_arrays"] == 11
    e = chk["phase8_events"]
    assert e["n"] == e["n_reference"] and e["event_ids_equal_in_order"] and e["alert_ids_equal_in_order"] and e["pairs_equal_in_order"] and e["transitions_equal_in_order"]
    assert e["all_fields_equal_except_provenance"] and set(e["provenance_keys_that_differ"]) <= {"phase9_stage", "phase9_execution_mode", "phase5_predictions_sha256"}
    assert chk["phase7_records"]["exactly_equal_lists"] and chk["phase8_alert_table"]["exactly_equal"]
    assert all(chk["phase7_arrays_vs_records"].values())
    # the expected values come from the committed reference run, not from this code: a reference-derived count must match the run
    ref = json.loads((REPO / CFGD["reference_runs"]["phase8_dir"] / "summary_report.json").read_text())
    assert (m["counts"]["alerts"], m["counts"]["alert_events"]) == (ref["n_alerts"], ref["n_events"])


@real
def test_comparison_detects_a_modified_output(run_a, tmp_path):
    import shutil
    m, tmp, _ = run_a
    dst = tmp_path / "copy"
    shutil.copytree(tmp / "runs" / "a", dst)
    p = dst / "S8_alerts" / "alerts.json"
    d = json.loads(p.read_text())
    d["alerts"][0]["peak_risk_score"] = 0.123456
    p.write_text(json.dumps(d))
    assert compare_run_with_references(dst, CFGD, REPO)["all_compared_checks_pass"] is False
    d2 = dst / "S5B_forecast" / "predictions_neural.npz"
    z = dict(np.load(d2))
    z["positions_uv_px"] = z["positions_uv_px"] + np.float32(1e-3)
    np.savez(d2, **z)
    c = compare_run_with_references(dst, CFGD, REPO)
    assert c["phase5_predictions_npz"]["sha256_equal"] is False and "positions_uv_px" in c["phase5_predictions_npz"]["arrays"]["unequal_arrays"]


@real
def test_source_artifacts_are_not_modified_by_repeated_inference(run_a, tmp_path):
    m, tmp, before = run_a
    assert m["inputs"]["unchanged_after_run"] is True and len(m["inputs"]["sha256_before"]) >= 25
    assert hash_inputs(CFGD, REPO) == before                                                                    # identical after the run
    m2 = run_pipeline(cfg(tmp_path, stages=INFER), run_id="b", benchmark=False)                                   # repeated inference
    assert hash_inputs(CFGD, REPO) == before and m2["inputs"]["unchanged_after_run"] is True
    assert events_of(tmp_path, "b") == events_of(tmp, "a")                                                        # identical inputs and configuration give identical events (provenance included)
    pa, pb = tmp / "runs" / "a" / "S5B_forecast" / "predictions_neural.npz", tmp_path / "runs" / "b" / "S5B_forecast" / "predictions_neural.npz"
    assert pa.read_bytes() == pb.read_bytes()


@real
def test_end_to_end_future_target_scrambling_changes_no_output(run_a, tmp_path, monkeypatch):
    """Scramble every future target the pipeline can see (inside the model's dataset AND in the window dataset used by Phase 6): all predictions, records and alert events stay identical."""
    m, tmp, _ = run_a
    real_load, real_read = orch.load_trained_model, orch.read_dataset

    def scramble(ws):
        ws.target_future_uv_px[...] = ws.target_future_uv_px[::-1] + 777.0
        ws.target_future_valid[...] = ~ws.target_future_valid
        ws.target_future_border[...] = ~ws.target_future_border

    def load(*a, **k):
        lm = real_load(*a, **k)
        scramble(lm.dataset.ws)
        return lm

    def read(*a, **k):
        ws, bank, extra = real_read(*a, **k)
        scramble(ws)
        return ws, bank, extra
    monkeypatch.setattr(orch, "load_trained_model", load)
    monkeypatch.setattr(orch, "read_dataset", read)
    m2 = run_pipeline(cfg(tmp_path, stages=INFER), run_id="s", benchmark=False)
    assert m2["status"] == "COMPLETED"
    assert events_of(tmp_path, "s") == events_of(tmp, "a")
    assert (tmp_path / "runs" / "s" / "S5B_forecast" / "predictions_neural.npz").read_bytes() == (tmp / "runs" / "a" / "S5B_forecast" / "predictions_neural.npz").read_bytes()


@real
def test_perturbing_history_inputs_does_change_the_output(run_a, tmp_path, monkeypatch):
    """Sensitivity control: the isolation test above is meaningful only if the pipeline does react to its real (history) inputs."""
    m, tmp, _ = run_a
    real_load = orch.load_trained_model

    def load(*a, **k):
        lm = real_load(*a, **k)
        for e in lm.dataset.bank.edges.values():                                                              # history interaction features (the dataset's own anchor guard rejects edits of target positions)
            e.edge_attr[...] = e.edge_attr * 3.0 + 1.0
        return lm
    monkeypatch.setattr(orch, "load_trained_model", load)
    m2 = run_pipeline(cfg(tmp_path, stages=["S3_PERCEPTION", "S5B_FORECAST"]), run_id="h", benchmark=False)
    a = np.load(tmp / "runs" / "a" / "S5B_forecast" / "predictions_neural.npz")["positions_uv_px"]
    b = np.load(tmp_path / "runs" / "h" / "S5B_forecast" / "predictions_neural.npz")["positions_uv_px"]
    assert m2["status"] == "COMPLETED" and not np.array_equal(a, b)


# ------------------------------------------------------------------ the committed full execution
@real
@pytest.mark.skipif(not COMMITTED.is_dir(), reason="full run record not present")
def test_committed_full_execution_record_is_consistent_and_honest():
    man = json.loads((COMMITTED / "run_manifest.json").read_text())
    att = json.loads((COMMITTED / "performance_attribution.json").read_text())
    ver = json.loads((COMMITTED / "verification_report.json").read_text())
    assert man["status"] == "COMPLETED" and att["execution_mode"] == "PRECOMPUTED_PHASE3_ARTIFACTS" and att["precomputed_artifact_integration_success"] is True
    assert att["raw_video_to_alert_pipeline_success"] is False and att["categories"]["phase3_perception"]["mean_s"] is None
    assert ver["checks"]["all_compared_checks_pass"] is True and "phase5_windows" in ver["checks"]                  # re-executed Phase 4/5 data compared as well
    assert all(v["equal"] for v in ver["checks"]["phase5_windows"].values())
    for s in ("S4A_IMAGE_KINEMATICS", "S4B_INTERACTION", "S5A_WINDOWS"):
        assert man["stages"][s]["detail"]["determinism"]["equal"] is True
    cur = hash_inputs(CFGD, REPO)
    assert all(cur[k] == v for k, v in ver["input_sha256_computed_after_the_run"].items() if k in cur)
    assert att["phase4_to_phase8_offline_batch_throughput"]["WARNING"].startswith("NOT raw-video end-to-end FPS")
    assert att["realtime_assessment"]["claim_supported"] is False and len(att["realtime_assessment"]["unmet_conditions"]) >= 2
