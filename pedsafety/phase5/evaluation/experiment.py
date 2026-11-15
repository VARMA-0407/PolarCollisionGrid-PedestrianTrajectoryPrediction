"""P5.4 experiment: verify a P5.3 window run -> run baselines on history only -> save predictions -> evaluate -> report.

Output directory (new or empty; never overwritten):
    predictions_<baseline>.npz     predictions only (no targets): window_id, track index, anchor, positions [N,F,2], masks, reasons, aux
    evaluation_<baseline>.npz      per-window ADE/FDE/step errors/oracle metrics/reasons (traceable by window_id)
    evaluation_results.json        configuration echo, per-baseline summaries, common-window comparison (no run metadata)
    evaluation_report.md           human-readable report (no run metadata)
    run_manifest.json              provenance, run-specific metadata, and the data-artifact hashes above
Data artifacts are deterministic; run-specific metadata (timestamps, runtime, memory, git) lives only in run_manifest.json.
"""

import datetime
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...image_kinematics_run import sha256_file
from ...interaction.runner import peak_memory_bytes
from ...interaction.storage import load_npz, write_json, write_npz, write_text
from ...output_dir import check_output_dir
from ..baselines import Prediction, build_history_input, make_predictor
from ..errors import WindowContractError
from ..window_config import WindowConfig
from ..window_manifest import PILOT_LABEL, git_provenance
from ..window_storage import read_dataset
from .metrics import WINDOW_INVALID_REASONS, EvalConfig, EvaluationResult, common_window_mask, evaluate_predictions, summarize

REPO = Path(__file__).resolve().parents[3]
MANIFEST_VERSION = "phase5-baseline-evaluation-run-manifest-1.0.0"
REQUIRED_KEYS = ("config_version", "pilot_label", "window_config_reference", "dataset_identifier", "target_source_mode", "prediction_horizon_steps",
                 "prediction_horizon_s", "sampling_hz", "coordinate_system", "units", "baselines", "metrics", "aggregation",
                 "partial_target_policy", "invalid_prediction_policy", "miss_rate_threshold_px")
CLAIMS = {"coordinate_space": "image (captured raster pixels)", "metric_world_coordinates": False, "physical_distance": False,
          "ttc_pet_or_collision_risk": False, "independent_ground_truth_targets": False, "independent_generalization": False,
          "research_grade_performance": False, "statistically_significant_ranking": False, "learned_model": False, "real_time_claim": False,
          "note": "Pseudo-targets are perceived tracks of one video (PILOT_SMOKE); results are pipeline diagnostics only."}
LIMITATIONS = [
    "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT: one PIE video, one scene group, 2,327 heavily overlapping windows from 84 tracks; no split; "
    "no generalization, cross-scene or research-grade claim; no ranking of baselines is statistically supported.",
    "Targets are perceived YOLO11n + ByteTrack footpoints (pseudo-targets) with unmeasured detection/tracking error.",
    "Errors are image-plane pixel distances: depth-dependent and affected by camera ego-motion; not metric.",
    "B1 uses Task 2 one-frame (1/30 s) causal velocities with ~48 px/s per-axis jitter, which is amplified over a 1.5 s horizon.",
    "Strict windows (all history and future observed) favour long, continuously tracked pedestrians.",
    "No confidence intervals: windows are correlated and only one video exists; track-weighted summaries are reported alongside window-weighted ones.",
]


def scalar_str(a) -> str:
    """Read a string stored by ``write_npz`` (0-d values are saved as 1-element arrays)."""
    arr = np.asarray(a).reshape(-1)
    if arr.shape != (1,):
        raise WindowContractError("PREDICTION_FILE", "expected a single string value, got shape %s" % (np.asarray(a).shape,))
    return str(arr[0])


def _canon_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def load_baseline_config(path) -> Tuple[Dict[str, Any], str]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WindowContractError("BASELINE_CONFIG", "cannot read %s: %s" % (path, exc))
    data = {k: v for k, v in raw.items() if not k.startswith("_")}
    missing = [k for k in REQUIRED_KEYS if k not in data]
    unknown = [k for k in data if k not in REQUIRED_KEYS]
    if missing or unknown:
        raise WindowContractError("BASELINE_CONFIG", "missing keys %s, unknown keys %s" % (missing, unknown))
    if data["pilot_label"] != PILOT_LABEL:
        raise WindowContractError("BASELINE_CONFIG", "pilot_label must be %s" % PILOT_LABEL)
    if data["coordinate_system"] != "captured_1920x1080" or data["units"].get("position") != "pixel":
        raise WindowContractError("COORDINATE_SYSTEM_MISMATCH", "baseline config must use captured_1920x1080 pixels")
    ids = [b.get("baseline_id") for b in data["baselines"]]
    if len(set(ids)) != len(ids) or not ids:
        raise WindowContractError("BASELINE_CONFIG", "baseline ids must be unique and non-empty")
    for b in data["baselines"]:
        make_predictor(b)                                              # validates parameters
    EvalConfig(data["partial_target_policy"], data["invalid_prediction_policy"], data["miss_rate_threshold_px"],
               data["aggregation"]["std_ddof"], tuple(data["aggregation"]["report_horizons_s"]))
    return data, _canon_sha(data)


def eval_config_from(data: Dict[str, Any]) -> EvalConfig:
    return EvalConfig(data["partial_target_policy"], data["invalid_prediction_policy"], data["miss_rate_threshold_px"],
                      data["aggregation"]["std_ddof"], tuple(data["aggregation"]["report_horizons_s"]))


def verify_window_run(run_dir, bcfg: Dict[str, Any]):
    """Verify the P5.3 run manifest, output hashes and configuration reference; return (manifest, WindowConfig, dataset, file hashes)."""
    d = Path(run_dir)
    mp = d / "run_manifest.json"
    if not mp.is_file():
        raise WindowContractError("WINDOW_RUN_MISSING", "missing %s" % mp)
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m.get("status") != "completed" or m.get("label") != PILOT_LABEL or m.get("validation", {}).get("ok") is not True:
        raise WindowContractError("WINDOW_RUN_STATUS", "P5.3 run must be completed, validated and labelled %s" % PILOT_LABEL)
    hashes = {}
    for name, rec in m["outputs"].items():
        p = d / name
        if not p.is_file() or p.stat().st_size != rec["bytes"] or sha256_file(p) != rec["sha256"]:
            raise WindowContractError("WINDOW_RUN_HASH", "P5.3 output %s missing or does not match its manifest hash" % name)
        hashes[name] = rec["sha256"]
    ref = bcfg["window_config_reference"]
    wcfg = WindowConfig.from_dict(m["config"]["content"])
    if wcfg.sha256() != m["config"]["sha256_resolved"] or wcfg.sha256() != ref["sha256_resolved"] or wcfg.config_version != ref["config_version"]:
        raise WindowContractError("WINDOW_CONFIG_MISMATCH", "P5.3 window configuration does not match the baseline configuration reference")
    if wcfg.target_source_mode != bcfg["target_source_mode"] or wcfg.coordinate_system != bcfg["coordinate_system"]:
        raise WindowContractError("WINDOW_CONFIG_MISMATCH", "target source or coordinate system differs from the baseline configuration")
    if wcfg.prediction_length != bcfg["prediction_horizon_steps"] or abs(wcfg.future_horizon_s - bcfg["prediction_horizon_s"]) > 1e-9 \
            or abs(wcfg.target_sampling_hz - bcfg["sampling_hz"]) > 1e-9:
        raise WindowContractError("WINDOW_CONFIG_MISMATCH", "horizon or sampling differs from the baseline configuration")
    ws, bank, extra = read_dataset(d)
    return m, wcfg, (ws, bank, extra), dict(hashes, **{"run_manifest.json": sha256_file(mp)})


def write_prediction(path: Path, pred: Prediction, ws) -> str:
    arrays = {"window_id": pred.window_id, "target_track_index": ws.target_track_index, "anchor_frame": ws.anchor_frame,
              "positions_uv_px": pred.positions, "valid": pred.valid, "window_valid": pred.window_valid, "invalid_reason": pred.invalid_reason,
              "invalid_reason_names": np.array(pred.invalid_reason_names if pred.invalid_reason_names else ("",)),
              "horizon_offsets_s": pred.horizon_offsets_s, "baseline_id": np.array(pred.baseline_id),
              "coordinate_system": np.array(pred.coordinate_system), "position_unit": np.array(pred.position_unit)}
    for k, v in sorted(pred.aux.items()):
        arrays["aux__" + k] = v
    return write_npz(path, arrays)


def write_evaluation(path: Path, res: EvaluationResult) -> str:
    return write_npz(path, {"window_id": res.window_id, "track_id": res.track_id, "evaluated": res.evaluated, "invalid_reason": res.invalid_reason,
                            "invalid_reason_names": np.array(WINDOW_INVALID_REASONS), "ade_px": res.ade, "fde_px": res.fde,
                            "step_error_px": res.step_error, "min_ade_px": res.min_ade, "min_fde_px": res.min_fde,
                            "oracle_evaluated": res.oracle_evaluated, "k": np.array(res.k)})


def evaluate_prediction_arrays(preds: Dict[str, Dict[str, np.ndarray]], ws, ecfg: EvalConfig) -> Tuple[Dict[str, EvaluationResult], Dict[str, Any]]:
    """Evaluate saved/in-memory prediction arrays against the P5.3 pseudo-targets (targets read only here)."""
    results: Dict[str, EvaluationResult] = {}
    for bid, a in preds.items():
        if not np.array_equal(a["window_id"], ws.window_id):
            raise WindowContractError("WINDOW_ID_MISMATCH", "%s predictions are not aligned with the window dataset" % bid)
        results[bid] = evaluate_predictions(ws.target_future_uv_px.astype(np.float64), ws.target_future_valid, a["positions_uv_px"], a["valid"],
                                            ws.window_id, ws.target_track_id, coordinate_system=scalar_str(a["coordinate_system"]),
                                            unit=scalar_str(a["position_unit"]), model_id=bid, horizon_offsets_s=a["horizon_offsets_s"], cfg=ecfg)
    common = common_window_mask(list(results.values()))
    hz = next(iter(preds.values()))["horizon_offsets_s"]
    comparison = {"definition": "windows evaluated by every baseline (same window set for all)", "n_windows": int(common.sum()),
                  "n_tracks": int(np.unique(ws.target_track_id[common]).shape[0]),
                  "per_baseline": {bid: {k: v for k, v in summarize(r, hz, ecfg, common).items() if k != "per_track"} for bid, r in results.items()}}
    return results, comparison


def render_report(results_json: Dict[str, Any]) -> str:
    L = ["# Phase 5 P5.4 baseline evaluation — %s" % PILOT_LABEL, "",
         "**All numbers below are PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT diagnostics**: one PIE video (set01/video_0004), perceived-track "
         "pseudo-targets, heavily overlapping windows, image-space pixels (`captured_1920x1080`). They do not establish forecasting accuracy, "
         "generalization, or that any baseline is better than another.", "",
         "Windows in dataset: %d. Policies: %s targets, %s predictions. Miss rate: %s." % (
             results_json["dataset"]["windows"], results_json["policies"]["partial_target_policy"], results_json["policies"]["invalid_prediction_policy"],
             results_json["policies"]["miss_rate"]), "",
         "## Window-weighted (each evaluated window weight 1; windows are not independent)", "",
         "| Baseline | evaluated | not evaluated | ADE mean | ADE median | ADE std | FDE mean | FDE median | FDE std | tracks |",
         "|---|---|---|---|---|---|---|---|---|---|"]

    def f(x):
        return "—" if x is None else "%.2f" % x
    for bid, s in results_json["baselines"].items():
        e = s["summary"]
        a, d = e["window_weighted"]["ade_px"], e["window_weighted"]["fde_px"]
        L.append("| %s | %d | %d | %s | %s | %s | %s | %s | %s | %d |" % (bid, e["windows_evaluated"], e["windows_not_evaluated"], f(a["mean"]),
                 f(a["median"]), f(a["std"]), f(d["mean"]), f(d["median"]), f(d["std"]), e["unique_target_tracks_evaluated"]))
    L += ["", "## Track-weighted (mean of per-track means; each track weight 1)", "", "| Baseline | tracks | ADE mean | ADE median | FDE mean | FDE median |",
          "|---|---|---|---|---|---|"]
    for bid, s in results_json["baselines"].items():
        t = s["summary"]["track_weighted"]
        L.append("| %s | %d | %s | %s | %s | %s |" % (bid, t["n_tracks"], f(t["ade_px_mean_of_track_means"]), f(t["ade_px_median_of_track_means"]),
                 f(t["fde_px_mean_of_track_means"]), f(t["fde_px_median_of_track_means"])))
    c = results_json["common_window_comparison"]
    L += ["", "## Common evaluated windows (%d windows, %d tracks; same set for every baseline)" % (c["n_windows"], c["n_tracks"]), "",
          "| Baseline | ADE mean | FDE mean | error @0.5 s | @1.0 s | @1.5 s |", "|---|---|---|---|---|---|"]
    for bid, e in c["per_baseline"].items():
        at = e["per_horizon_mean_displacement_error_px"]["at_report_horizons"]
        L.append("| %s | %s | %s | %s | %s | %s |" % (bid, f(e["window_weighted"]["ade_px"]["mean"]), f(e["window_weighted"]["fde_px"]["mean"]),
                 f(at.get("0.5s")), f(at.get("1.0s")), f(at.get("1.5s"))))
    L += ["", "## Invalid predictions", ""]
    for bid, s in results_json["baselines"].items():
        L.append("- %s: %s; evaluator exclusions %s" % (bid, s["prediction_invalid_reasons"], s["summary"]["not_evaluated_reasons"]))
    L += ["", "minADE@1 / minFDE@1 equal ADE / FDE (K = 1 for every baseline; best-of-K oracle metrics are not reported as primary).",
          "Units: pixels. No metric distance, TTC, PET or collision risk is computed.", ""]
    return "\n".join(L)


def run_baseline_experiment(windows_run, baseline_config_path, out_dir, *, command: Optional[str] = None) -> Dict[str, Any]:
    out = check_output_dir(out_dir)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    git = git_provenance()
    bcfg, bsha = load_baseline_config(baseline_config_path)
    ecfg = eval_config_from(bcfg)
    wman, wcfg, (ws, bank, extra), in_hashes = verify_window_run(windows_run, bcfg)
    t1 = time.perf_counter()
    hist = build_history_input(ws, bank, extra["history_target_bank_node"], wcfg)
    preds: Dict[str, Prediction] = {}
    pred_time: Dict[str, float] = {}
    for spec in bcfg["baselines"]:
        s = time.perf_counter()
        pr = make_predictor(spec)
        preds[spec["baseline_id"]] = pr.predict(hist)
        pred_time[spec["baseline_id"]] = time.perf_counter() - s
    t2 = time.perf_counter()
    arrays = {bid: {"window_id": p.window_id, "positions_uv_px": p.positions, "valid": p.valid, "horizon_offsets_s": p.horizon_offsets_s,
                    "coordinate_system": np.array(p.coordinate_system), "position_unit": np.array(p.position_unit)} for bid, p in preds.items()}
    results, comparison = evaluate_prediction_arrays(arrays, ws, ecfg)
    t3 = time.perf_counter()

    out.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Dict[str, Any]] = {}

    def record(name, sha):
        files[name] = {"sha256": sha, "bytes": (out / name).stat().st_size}
    for bid, p in preds.items():
        record("predictions_%s.npz" % bid, write_prediction(out / ("predictions_%s.npz" % bid), p, ws))
        record("evaluation_%s.npz" % bid, write_evaluation(out / ("evaluation_%s.npz" % bid), results[bid]))
    results_json = {
        "label": PILOT_LABEL, "results_version": "phase5-baseline-evaluation-results-1.0.0",
        "dataset": {"identifier": bcfg["dataset_identifier"], "windows_run": _rel(windows_run), "windows_run_manifest_sha256": in_hashes["run_manifest.json"],
                    "windows": ws.n_windows, "unique_target_tracks": int(np.unique(ws.target_track_id).shape[0]), "videos": 1, "scene_groups": 1,
                    "target_source_mode": wcfg.target_source_mode},
        "baseline_config": {"path": _rel(baseline_config_path), "sha256_canonical": bsha, "content": bcfg},
        "coordinates": {"coordinate_system": wcfg.coordinate_system, "units": bcfg["units"], "origin": "top-left", "u": "+right", "v": "+down",
                        "world_coordinates": "unavailable"},
        "horizon_offsets_s": [round(float(x), 9) for x in hist.horizon_offsets_s],
        "policies": {"partial_target_policy": ecfg.partial_target_policy, "invalid_prediction_policy": ecfg.invalid_prediction_policy,
                     "miss_rate": "NOT_CONFIGURED" if ecfg.miss_rate_threshold_px is None else "%.3f px" % ecfg.miss_rate_threshold_px,
                     "aggregation": bcfg["aggregation"]},
        "baselines": {bid: {"config": p.config, "predictions_valid": int(p.window_valid.sum()), "predictions_invalid": int((~p.window_valid).sum()),
                            "prediction_invalid_reasons": p.invalid_counts(), "summary": results[bid].summary,
                            "diagnostics": _diagnostics(bid, p)} for bid, p in preds.items()},
        "common_window_comparison": comparison,
        "claims": CLAIMS, "limitations": LIMITATIONS}
    record("evaluation_results.json", write_json(out / "evaluation_results.json", results_json))
    record("evaluation_report.md", write_text(out / "evaluation_report.md", render_report(results_json)))
    finished = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    peak = peak_memory_bytes()                                         # measured before any version lookup
    try:
        from importlib.metadata import version as _v                   # metadata only: torch is never imported
        torch_v = _v("torch") + " (installed; not imported or used)"
    except Exception:                                                  # noqa: BLE001
        torch_v = None
    manifest = {
        "manifest_version": MANIFEST_VERSION, "task": "phase5_p5.4_baselines_and_evaluation", "status": "completed", "label": PILOT_LABEL,
        "data_artifacts": dict(sorted(files.items())),
        "data_artifacts_note": "deterministic outputs: identical inputs and configuration give identical hashes",
        "inputs": {"windows_run": _rel(windows_run), "windows_run_files": in_hashes, "windows_run_git": wman["git"].get("commit"),
                   "window_config_sha256": wcfg.sha256(), "baseline_config": {"path": _rel(baseline_config_path), "sha256_canonical": bsha,
                                                                             "sha256_file": sha256_file(Path(baseline_config_path))},
                   "future_targets_read_by": "evaluator only (after all predictions were produced)",
                   "velocity_source": hist.velocity_source, "offline_kinematics_used": False},
        "counts": {"windows": ws.n_windows, "unique_target_tracks": int(np.unique(ws.target_track_id).shape[0]),
                   "per_baseline": {bid: {"valid_predictions": int(p.window_valid.sum()), "invalid_predictions": int((~p.window_valid).sum()),
                                          "evaluated_windows": int(results[bid].evaluated.sum())} for bid, p in preds.items()},
                   "common_evaluated_windows": comparison["n_windows"]},
        "coordinates": results_json["coordinates"], "target_source": wcfg.target_source_mode, "baseline_ids": list(preds),
        "run_metadata": {"git": git, "started_utc": started, "finished_utc": finished, "command": command,
                         "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch_v, "platform": platform.platform()},
                         "runtime_s": {"verify_and_load_inputs": t1 - t0, "predict_total": t2 - t1, "predict_per_baseline": pred_time,
                                       "evaluate": t3 - t2, "total_before_writing": t3 - t0,
                                       "note": "offline CPU timing on one machine; not a real-time claim"},
                         "peak_memory_bytes": peak,
                         "peak_memory_note": "process peak working set at the end of evaluation (includes Python, numpy and the loaded P5.3 dataset)"},
        "claims": CLAIMS, "limitations": LIMITATIONS}
    write_json(out / "run_manifest.json", manifest)
    return manifest


def _diagnostics(bid: str, p: Prediction) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    if "velocity_source_index" in p.aux:
        src = p.aux["velocity_source_index"]
        d["velocity_source_history_index_counts"] = {str(k): int((src == k).sum()) for k in sorted(set(src.tolist()))}
        d["note"] = "history index 5 = anchor sample; lower = older sample found by backward search; -1 = none (invalid)"
    if "valid_history_samples" in p.aux:
        c = p.aux["valid_history_samples"]
        d["valid_history_sample_counts"] = {str(k): int((c == k).sum()) for k in sorted(set(c.tolist()))}
    return d


def load_predictions_dir(pred_dir, baseline_ids: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for bid in baseline_ids:
        a = load_npz(Path(pred_dir) / ("predictions_%s.npz" % bid))
        if scalar_str(a["baseline_id"]) != bid:
            raise WindowContractError("PREDICTION_FILE", "file for %s declares %s" % (bid, a["baseline_id"]))
        out[bid] = a
    return out


__all__ = ["run_baseline_experiment", "load_baseline_config", "verify_window_run", "evaluate_prediction_arrays", "load_predictions_dir",
           "eval_config_from", "render_report", "scalar_str", "CLAIMS", "LIMITATIONS"]
