"""Phase 5 P5.4 CLI: independently validate a baseline-evaluation run (read-only unless --report names a new file).

Checks: data-artifact hashes equal the manifest; P5.3 input re-verifies and its manifest hash equals the recorded one;
predictions re-generated from history alone are bit-identical to the saved ones; predictions are invariant to any change
of the future target arrays (they are never passed to predictors); every metric is recomputed with an independent
per-window loop implementation and matches the saved per-window values and summaries; shapes, units, coordinate
system, labels and claims are as required.

Example:
    python scripts/phase5_validate_evaluation.py --run experiments/phase5_task4_baselines_pie_video_0004/<run>/primary
Exit codes: 0 all checks pass, 1 a check failed, 2 contract failure, 3 unexpected failure.
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.image_kinematics_run import sha256_file  # noqa: E402
from pedsafety.interaction.storage import load_npz  # noqa: E402
from pedsafety.phase5.baselines import build_history_input, make_predictor  # noqa: E402
from pedsafety.phase5.evaluation.experiment import load_baseline_config, scalar_str, verify_window_run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOL = 1e-9


def _loop_metrics(gt, gv, pred, pv):
    """Independent reference: plain Python loops, STRICT_FULL_HORIZON, K = 1."""
    n, f = gt.shape[:2]
    ade = np.full(n, np.nan)
    fde = np.full(n, np.nan)
    for i in range(n):
        if not (all(gv[i]) and all(pv[i])):
            continue
        d = [math.hypot(pred[i, t, 0] - gt[i, t, 0], pred[i, t, 1] - gt[i, t, 1]) for t in range(f)]
        ade[i] = sum(d) / f
        fde[i] = d[-1]
    return ade, fde


def _close(a, b):
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=TOL, rtol=0))


def validate_run(run_dir) -> dict:
    d = Path(run_dir)
    m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    out = {"run": str(d), "checks": {}}

    def chk(name, ok, detail=None):
        out["checks"][name] = {"pass": bool(ok), "detail": detail}
    bad = [n for n, r in m["data_artifacts"].items() if not (d / n).is_file() or sha256_file(d / n) != r["sha256"] or (d / n).stat().st_size != r["bytes"]]
    chk("data_artifact_hashes_match_manifest", not bad, bad or sorted(m["data_artifacts"]))
    chk("label_pilot_only", m["label"] == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT")
    chk("claims_all_false", all(v is False for v in m["claims"].values() if isinstance(v, bool)))
    cfg_path = REPO / m["inputs"]["baseline_config"]["path"]
    bcfg, bsha = load_baseline_config(cfg_path)
    chk("baseline_config_hash_matches", bsha == m["inputs"]["baseline_config"]["sha256_canonical"])
    wrun = REPO / m["inputs"]["windows_run"]
    _, wcfg, (ws, bank, extra), in_hashes = verify_window_run(wrun, bcfg)
    chk("window_run_manifest_hash_matches", in_hashes["run_manifest.json"] == m["inputs"]["windows_run_files"]["run_manifest.json"])
    hist = build_history_input(ws, bank, extra["history_target_bank_node"], wcfg)
    res = json.loads((d / "evaluation_results.json").read_text(encoding="utf-8"))
    gt = ws.target_future_uv_px.astype(np.float64)
    gv = ws.target_future_valid
    ws_perturbed = copy.deepcopy(ws)
    ws_perturbed.target_future_uv_px[:] = -1e6                          # any future change must not affect predictions
    hist_p = build_history_input(ws_perturbed, bank, extra["history_target_bank_node"], wcfg)
    for spec in bcfg["baselines"]:
        bid = spec["baseline_id"]
        sp = load_npz(d / ("predictions_%s.npz" % bid))
        se = load_npz(d / ("evaluation_%s.npz" % bid))
        fresh = make_predictor(spec).predict(hist)
        chk("%s.predictions_regenerate_bit_identical" % bid, np.array_equal(np.nan_to_num(fresh.positions, nan=7.7e7), np.nan_to_num(sp["positions_uv_px"], nan=7.7e7))
            and np.array_equal(fresh.valid, sp["valid"]) and np.array_equal(fresh.invalid_reason, sp["invalid_reason"]))
        pert = make_predictor(spec).predict(hist_p)
        chk("%s.predictions_invariant_to_future_targets" % bid, np.array_equal(np.nan_to_num(pert.positions, nan=7.7e7), np.nan_to_num(fresh.positions, nan=7.7e7)))
        chk("%s.shape_units_coordinates" % bid, sp["positions_uv_px"].shape == (ws.n_windows, wcfg.prediction_length, 2)
            and scalar_str(sp["coordinate_system"]) == "captured_1920x1080" and scalar_str(sp["position_unit"]) == "pixel"
            and scalar_str(sp["baseline_id"]) == bid
            and np.array_equal(sp["window_id"], ws.window_id) and np.array_equal(sp["anchor_frame"], ws.anchor_frame))
        chk("%s.invalid_predictions_are_nan_not_zero" % bid, bool(np.isnan(sp["positions_uv_px"][~sp["valid"]]).all()))
        ade, fde = _loop_metrics(gt, gv, sp["positions_uv_px"], sp["valid"])
        chk("%s.per_window_metrics_recomputed" % bid, _close(ade, se["ade_px"]) and _close(fde, se["fde_px"]) and np.array_equal(se["window_id"], ws.window_id))
        s = res["baselines"][bid]["summary"]["window_weighted"]
        ev = ~np.isnan(ade)
        chk("%s.summary_recomputed" % bid, s["ade_px"]["n"] == int(ev.sum())
            and (int(ev.sum()) == 0 or (abs(s["ade_px"]["mean"] - float(np.mean(ade[ev]))) < 1e-9 and abs(s["fde_px"]["mean"] - float(np.mean(fde[ev]))) < 1e-9
                                        and abs(s["ade_px"]["median"] - float(np.median(ade[ev]))) < 1e-9)))
        chk("%s.min_metrics_equal_ade_fde_for_k1" % bid, _close(se["min_ade_px"], se["ade_px"]) and _close(se["min_fde_px"], se["fde_px"]))
    out["ok"] = all(v["pass"] for v in out["checks"].values())
    out["n_checks"] = len(out["checks"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--report", help="write the JSON report to this NEW file (refuses to overwrite)")
    a = ap.parse_args(argv)
    try:
        rep = validate_run(a.run)
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
    print("%s: %d/%d checks passed%s" % ("PASS" if rep["ok"] else "FAIL", rep["n_checks"] - len(failed), rep["n_checks"],
                                         "" if not failed else "; failed: %s" % failed))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
