"""Phase 5 P5.7.5 CLI: independently validate a neural evaluation run (read-only unless --report names a new file).

Checks: data-artifact hashes; labels, in-sample status and claims; the training run and checkpoint / scaler verification (re-done);
the saved predictions are RECOMPUTED from the verified checkpoint (bitwise); an independent plain-numpy implementation recomputes the
primary (most-probable mode) ADE / FDE and the oracle minADE@K / minFDE@K from the saved predictions and the window targets and compares
them with the saved per-window values and the summary; shapes / units; probabilities sum to one; primary metrics differ from the
oracle metrics in definition; optional bitwise comparison of two evaluation runs.
Exit codes: 0 all checks pass, 1 a check failed, 2 contract failure, 3 unexpected failure.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.image_kinematics_run import sha256_file  # noqa: E402
from pedsafety.interaction.storage import load_npz  # noqa: E402
from pedsafety.phase5.models.neural_evaluation import NeuralEvaluationConfig, load_trained_model, predict_windows  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def validate_run(run_dir, compare_with=None) -> dict:
    d = Path(run_dir)
    m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    res = json.loads((d / "evaluation_results.json").read_text(encoding="utf-8"))
    out = {"run": str(d), "checks": {}}

    def chk(name, ok, detail=None):
        out["checks"][name] = {"pass": bool(ok), "detail": detail}
    bad = [n for n, r in m["data_artifacts"].items() if not (d / n).is_file() or sha256_file(d / n) != r["sha256"] or (d / n).stat().st_size != r["bytes"]]
    chk("data_artifact_hashes_match_manifest", not bad, bad or sorted(m["data_artifacts"]))
    chk("labels_in_sample_and_claims_false", m["label"] == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" and m["sample_status"] == "IN_SAMPLE_MODEL_TRAINED_ON_THESE_WINDOWS"
        and res["sample_status"] == m["sample_status"] and all(v is False for v in m["claims"].values() if isinstance(v, bool)))
    inp = m["inputs"]
    ncfg = NeuralEvaluationConfig.load(REPO / inp["neural_evaluation_config"]["path"])
    if ncfg.torch_threads is not None:                                  # bitwise reproducibility holds for the recorded thread count (multi-thread matmuls differ in the last bits)
        torch.set_num_threads(ncfg.torch_threads)
    chk("recorded_thread_count_used_for_recomputation", m["run_metadata"]["versions"]["torch_threads"] == (ncfg.torch_threads or m["run_metadata"]["versions"]["torch_threads"]))
    chk("neural_config_hash_matches", ncfg.sha256() == inp["neural_evaluation_config"]["sha256_canonical"])
    train_dir, wrun = REPO / inp["training_run"], REPO / inp["windows_run"]
    chk("training_manifest_hash_matches", sha256_file(train_dir / "run_manifest.json") == inp["training_run_manifest_sha256"])
    loaded = load_trained_model(train_dir, wrun, checkpoint=inp["checkpoint"])
    v = loaded.verification
    chk("checkpoint_and_scaler_reverified", v["checkpoint_file_sha256"] == inp["checkpoint_file_sha256"] and v["checkpoint_model_state_sha256"] == inp["checkpoint_model_state_sha256"]
        and v["scaler_statistics_sha256"] == inp["scaler_statistics_sha256"] and v["scaler_buffers_equal_statistics"] and v["scaler_source_identity_verified"])
    ds = loaded.dataset
    saved = load_npz(d / "predictions_neural.npz")
    fresh = predict_windows(loaded.model, ds, ncfg.batch_size)
    chk("predictions_recomputed_from_checkpoint_bitwise", np.array_equal(saved["positions_uv_px"], fresh["positions_uv_px"]) and np.array_equal(saved["mode_probabilities"], fresh["mode_probabilities"])
        and np.array_equal(saved["valid"], fresh["valid"]) and np.array_equal(saved["window_id"], fresh["window_id"]))
    N, K, F = len(ds), ncfg.oracle_k, ds.config.prediction_length
    chk("shapes_units_and_probabilities", saved["positions_uv_px"].shape == (N, K, F, 2) and saved["valid"].shape == (N, K, F) and saved["mode_probabilities"].shape == (N, K)
        and np.allclose(saved["mode_probabilities"].sum(axis=1), 1.0, atol=1e-5) and np.all(saved["mode_probabilities"] >= 0)
        and str(saved["coordinate_system"].reshape(-1)[0]) == "captured_1920x1080" and str(saved["position_unit"].reshape(-1)[0]) == "pixel")
    chk("anchor_is_raw_last_history_position", np.array_equal(saved["anchor_uv_px"], ds.ws.history_target_uv_px[:, -1, :].astype(np.float32)))
    gt, gv = ds.ws.target_future_uv_px.astype(np.float64), ds.ws.target_future_valid
    ev = load_npz(d / "evaluation_neural.npz")
    pos, pr = saved["positions_uv_px"].astype(np.float64), saved["mode_probabilities"].astype(np.float64)
    ade_k = np.full((N, K), np.nan)
    fde_k = np.full((N, K), np.nan)
    for i in range(N):                                                   # independent plain loops (no evaluator code)
        if not gv[i].all():
            continue
        for k in range(K):
            dist = [math.hypot(pos[i, k, t, 0] - gt[i, t, 0], pos[i, k, t, 1] - gt[i, t, 1]) for t in range(F)]
            ade_k[i, k], fde_k[i, k] = sum(dist) / F, dist[-1]
    sel = pr.argmax(axis=1)
    rows = np.arange(N)
    p_ade, p_fde = ade_k[rows, sel], fde_k[rows, sel]
    o_ade, o_fde = np.nanmin(ade_k, axis=1), np.nanmin(fde_k, axis=1)
    same = lambda a, b: bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-9, rtol=0))   # noqa: E731
    chk("primary_most_probable_mode_ade_fde_recomputed", same(p_ade, ev["ade_px"]) and same(p_fde, ev["fde_px"]))
    chk("oracle_min_ade_fde_recomputed_separately", same(o_ade, ev["min_ade_px"]) and same(o_fde, ev["min_fde_px"]))
    chk("oracle_never_larger_than_primary", bool(np.all(o_ade <= p_ade + 1e-9)) and bool(np.all(o_fde <= p_fde + 1e-9)))
    s = res["neural_model"]["summary"]
    chk("summary_means_recomputed", abs(s["window_weighted"]["ade_px"]["mean"] - float(np.nanmean(p_ade))) < 1e-9 and abs(s["window_weighted"]["fde_px"]["mean"] - float(np.nanmean(p_fde))) < 1e-9
        and abs(s["oracle_best_of_k"]["min_ade_px"]["mean"] - float(np.nanmean(o_ade))) < 1e-9 and abs(s["oracle_best_of_k"]["min_fde_px"]["mean"] - float(np.nanmean(o_fde))) < 1e-9)
    chk("oracle_labelled_in_report", "ORACLE" in s["oracle_best_of_k"]["label"] and s["oracle_best_of_k"]["identical_to_ade_fde"] is False and s["miss_rate"]["status"] == "NOT_CONFIGURED")
    comp = res.get("baseline_comparison")
    if comp:
        chk("baseline_comparison_present_and_in_sample", comp["n_windows"] == N and IN_SAMPLE_WORD in comp["status"] and set(comp["per_model"]) >= {ncfg.model_id, "B0_zero_velocity", "B1_last_valid_causal_velocity", "B2_history_linear_regression"})
    ref = res.get("p54_reference_check")
    if ref:
        chk("recomputed_baselines_equal_committed_p54_results", all(r["equal"] for r in ref.values()), ref)
    if compare_with:
        o = json.loads((Path(compare_with) / "run_manifest.json").read_text(encoding="utf-8"))
        chk("deterministic_vs_second_run_all_data_artifacts_identical", {k: v["sha256"] for k, v in o["data_artifacts"].items()} == {k: v["sha256"] for k, v in m["data_artifacts"].items()})
    out["ok"] = all(v["pass"] for v in out["checks"].values())
    out["n_checks"] = len(out["checks"])
    return out


IN_SAMPLE_WORD = "IN_SAMPLE"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare-with", help="a second evaluation run directory that must be bitwise identical")
    ap.add_argument("--report", help="write the JSON report to this NEW file (refuses to overwrite)")
    a = ap.parse_args(argv)
    try:
        rep = validate_run(a.run, a.compare_with)
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    if a.report:
        p = Path(a.report)
        if p.exists():
            print("error: report path exists; refusing to overwrite: %s" % p, file=sys.stderr)
            return 2
        p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    failed = [k for k, v in rep["checks"].items() if not v["pass"]]
    print("%s: %d/%d checks passed%s" % ("PASS" if rep["ok"] else "FAIL", rep["n_checks"] - len(failed), rep["n_checks"], "" if not failed else "; failed: %s" % failed))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
