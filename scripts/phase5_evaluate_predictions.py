"""Phase 5 P5.4 CLI: (re-)evaluate SAVED prediction files against a verified P5.3 window run.

Model-agnostic: reads ``predictions_<id>.npz`` (window_id, positions_uv_px [N,F,2], valid [N,F], horizon_offsets_s,
coordinate_system, position_unit) produced by any predictor and writes per-window metrics and summaries to a NEW
directory. Used to show that results can be recomputed from saved predictions alone.

Example:
    python scripts/phase5_evaluate_predictions.py --predictions <baseline run>/primary --windows <P5.3 run>/primary \
        --config configs/phase5/baseline_evaluation_default.json --out <new dir>
Exit codes: 0 success, 2 contract failure, 3 unexpected failure.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.interaction.storage import write_json  # noqa: E402
from pedsafety.output_dir import check_output_dir  # noqa: E402
from pedsafety.phase5.evaluation.experiment import (eval_config_from, evaluate_prediction_arrays, load_baseline_config,  # noqa: E402
                                                    load_predictions_dir, verify_window_run, write_evaluation)
from pedsafety.phase5.window_manifest import PILOT_LABEL  # noqa: E402


def evaluate_dir(pred_dir, windows, config, out=None):
    bcfg, bsha = load_baseline_config(config)
    _, _, (ws, _, _), _ = verify_window_run(windows, bcfg)
    ids = [b["baseline_id"] for b in bcfg["baselines"]]
    results, comparison = evaluate_prediction_arrays(load_predictions_dir(pred_dir, ids), ws, eval_config_from(bcfg))
    summary = {"label": PILOT_LABEL, "baseline_config_sha256_canonical": bsha, "predictions_dir": str(pred_dir),
               "summaries": {b: r.summary for b, r in results.items()}, "common_window_comparison": comparison}
    if out is not None:
        o = check_output_dir(out)
        o.mkdir(parents=True, exist_ok=True)
        for b, r in results.items():
            write_evaluation(o / ("evaluation_%s.npz" % b), r)
        write_json(o / "recomputed_summary.json", summary)
    return results, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="new or empty output directory")
    a = ap.parse_args(argv)
    try:
        _, s = evaluate_dir(a.predictions, a.windows, a.config, a.out)
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    print(json.dumps({b: {"evaluated": v["windows_evaluated"], "ade_mean_px": v["window_weighted"]["ade_px"]["mean"],
                          "fde_mean_px": v["window_weighted"]["fde_px"]["mean"]} for b, v in s["summaries"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
