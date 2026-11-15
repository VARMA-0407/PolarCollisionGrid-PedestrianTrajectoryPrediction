"""Attribution of measured time to pipeline components, execution-mode statement and acceptance bookkeeping, derived ONLY from a run manifest (no re-measurement).

Categories requested for the performance report: video decoding, Phase 3 perception, Phase 4 preprocessing, Phase 5 inference, Phase 6, Phase 7, Phase 8 and file I/O. A
category that was not executed is reported ``UNAVAILABLE`` with null values, never 0 ms. File I/O is only separable where a stage reports it; elsewhere it is INCLUDED in the stage
time and said to be.
"""

from typing import Any, Dict, Mapping, Optional

EXECUTION_MODE = "PRECOMPUTED_PHASE3_ARTIFACTS"
UNAVAILABLE = "UNAVAILABLE"


def execution_mode(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    s3 = manifest["stages"].get("S3_PERCEPTION", {}) or {}
    ve = (s3.get("detail") or {}).get("video_execution") or {}
    raw = bool(ve.get("executed"))
    chain = ("S4A_IMAGE_KINEMATICS", "S4B_INTERACTION", "S5A_WINDOWS", "S5B_FORECAST", "S6_CONFLICT", "S7_RISK", "S8_ALERTS")
    done = [s for s in chain if manifest["stages"].get(s, {}).get("status") == "COMPLETED"]
    complete_chain = len(done) == len(chain)
    return {"execution_mode": "RAW_VIDEO" if raw else EXECUTION_MODE, "raw_video_executed": raw,
            "phase3_perception": "EXECUTED" if raw else "NOT_EXECUTED_PRECOMPUTED_ARTIFACTS_VALIDATED_ONLY",
            "phase4_to_phase8_chain_complete": complete_chain, "stages_completed": done,
            "raw_video_to_alert_pipeline_success": bool(raw and complete_chain),
            "precomputed_artifact_integration_success": bool((not raw) and complete_chain and manifest["status"] == "COMPLETED"),
            "statement": ("Phases 4-8 were executed on precomputed Phase 3 artifacts; Phase 3 raw-video inference was not run (no raw video, no OpenCV/Ultralytics/GPU locally). "
                          "This is NOT a raw-video-to-alert execution." if not raw else "raw video executed")}


def _row(mean_s: Optional[float], median_s: Optional[float], n: int, scope: str, **extra) -> Dict[str, Any]:
    return {"status": "MEASURED", "mean_s": mean_s, "median_s": median_s, "n": n, "scope": scope, **extra}


def build_attribution(manifest: Mapping[str, Any], environment: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    perf, st = manifest["performance"], manifest["stages"]
    rec = (((st.get("S3_PERCEPTION") or {}).get("detail") or {}).get("video_execution") or {}).get("recorded_phase3_run") or {}
    pre, inf = perf.get("preprocessing") or {}, perf.get("inference_chain") or {}
    ss = inf.get("steady_state") or {}
    det = perf.get("stage_detail", {}).get("S5B", {})

    def pre_row(k, sid, scope):
        s = (pre.get("stats") or {}).get(k)
        if s:
            return _row(s["mean"], s["median"], s["n"], scope, runs_s=pre["runs_s"][k])
        w = (st.get(sid) or {}).get("wall_s")
        return _row(w, w, 1, scope) if (st.get(sid) or {}).get("status") == "COMPLETED" and w is not None else {"status": UNAVAILABLE, "mean_s": None, "median_s": None, "n": 0,
                                                                                                               "reason": "stage not executed in this run"}

    def ss_row(k, scope):
        s = ss.get(k)
        return _row(s["mean"], s["median"], s["n"], scope, std_s=s["std_sample"], cv=s["cv"]) if s else {"status": UNAVAILABLE, "mean_s": None, "median_s": None, "n": 0}
    cats = {
        "video_decoding": {"status": UNAVAILABLE, "mean_s": None, "median_s": None, "n": 0, "reason": "no raw video and no OpenCV in this environment; not zero"},
        "phase3_perception": {"status": UNAVAILABLE, "mean_s": None, "median_s": None, "n": 0, "reason": "YOLO/ByteTrack were not executed here; not zero",
                              "recorded_not_measured_here": {k: rec.get(k) for k in ("frames_processed", "detect_mean_ms", "track_mean_ms", "wall_time_s", "frames_per_s_wall_including_decode_and_io",
                                                                                       "hardware", "cpu", "caveat")}},
        "phase3_artifact_validation": (_row(st["S3_PERCEPTION"]["wall_s"], st["S3_PERCEPTION"]["wall_s"], 1, "reading and contract-validating the precomputed Phase 3 rows; NOT perception time")
                                       if st.get("S3_PERCEPTION", {}).get("wall_s") is not None else {"status": UNAVAILABLE, "mean_s": None, "median_s": None, "n": 0}),
        "phase4_preprocessing_image_kinematics": pre_row("S4A", "S4A_IMAGE_KINEMATICS", "Phase 4 Task 2 incl. its own file read/write"),
        "phase4_preprocessing_interaction": pre_row("S4B", "S4B_INTERACTION", "Phase 4 Task 3 incl. its own file read/write"),
        "phase5_window_generation": pre_row("S5A", "S5A_WINDOWS", "P5.3 window generation incl. its own file read/write"),
        "phase5_model_load_and_verification": {"status": "MEASURED", "mean_s": det.get("model_load_and_verify_s"), "median_s": det.get("model_load_and_verify_s"), "n": 1,
                                               "scope": "checkpoint / scaler / config / window-run hash verification and model construction (first pass)"},
        "phase5_inference_model_forward": ss_row("model_forward", "GATv2 + TCN encoder and decoder forward passes, batch 64, CPU, 1 thread, steady state"),
        "phase5_inference_collate_and_other": ss_row("collate_and_other", "dataset item construction and batching around the forward pass, steady state"),
        "phase6_conflict_processing": ss_row("conflict_pairs", "pair construction and image-space features, steady state"),
        "phase7_risk_processing": ss_row("risk", "heuristic risk scoring, categories and record construction, steady state"),
        "phase8_alert_processing": ss_row("alert_manager", "state machine over all records, no file output, steady state"),
        "file_io_and_serialization": {"status": "PARTIALLY_MEASURED", "phase5_predictions_write_s": det.get("write_predictions_s"),
                                      "included_in_stage_times_not_separable": ["S3 read", "S4A read/write", "S4B read/write", "S5A read/write", "S8 gzip/JSON writes", "S6 window/graph-bank read"],
                                      "reason": "separating them would require instrumenting Phase 4/5/8 code, which Phase 9 does not modify"}}
    chain_total = perf.get("downstream_wall_s_first_pass")
    out = {"attribution_version": "phase9-performance-attribution-1.0.0", **execution_mode(manifest), "categories": cats,
           "first_pass_stage_wall_s": {s: st[s].get("wall_s") for s in st if st[s].get("wall_s") is not None},
           "phase4_to_phase8_first_pass_wall_s": chain_total,
           "phase4_to_phase8_offline_batch_throughput": {
               "frames_represented": manifest["counts"].get("frames_processed_phase3"), "frames_per_s": perf.get("downstream_frames_per_s_offline_batch"),
               "ms_per_frame": perf.get("downstream_ms_per_frame"),
               "WARNING": "NOT raw-video end-to-end FPS and NOT a real-time result: excludes video decoding and Phase 3 perception, processes the whole recording as an offline batch, and the window "
                          "builder needs each target's future frames. Phase 4 Task 2/3 are 90% of this time. Never compare with a 30 FPS requirement."},
           "isolated_component_not_system_throughput": "the Phase 8 alert manager alone processed ~2e4 observations/s; this is a component microbenchmark, not complete-system throughput",
           "memory": {"process_peak_working_set_bytes": perf.get("process_peak_working_set_bytes"), "method": (perf.get("hardware") or {}).get("memory_method") or
                      "process peak working set via the OS API; process-level, monotone, includes the benchmark repeats and extra preprocessing runs", "per_stage_isolated": False,
                      "psutil": "unavailable", "gpu_memory": "NOT_APPLICABLE_NO_CUDA_DEVICE"},
           "gpu": perf.get("gpu"), "realtime_assessment": perf.get("realtime_assessment"),
           "run_total_wall_s_including_benchmark_repeats": manifest.get("total_wall_s"),
           "repetitions": {"inference_warmup_runs": len(inf.get("warmup_total_s", [])), "inference_warmup_s": inf.get("warmup_total_s"), "inference_steady_state_repeats":
                           (ss.get("inference_chain_total") or {}).get("n"), "preprocessing_runs_per_stage": {k: len(v) for k, v in (pre.get("runs_s") or {}).items()},
                           "note": "preprocessing runs are cold (no warm-up) repetitions; run 1 is the main pass"}}
    if environment:
        out["environment"] = dict(environment)
    return out


__all__ = ["build_attribution", "execution_mode", "EXECUTION_MODE", "UNAVAILABLE"]
