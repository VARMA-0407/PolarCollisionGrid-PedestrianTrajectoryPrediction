"""Phase 5 P5.4 CLI: run baselines B0-B2 on a verified P5.3 window run, save predictions, evaluate, report (CPU, no model training).

Example:
    python scripts/phase5_run_baselines.py \
        --windows experiments/phase5_task3_windows_pie_video_0004/<run>/primary \
        --config configs/phase5/baseline_evaluation_default.json \
        --out experiments/phase5_task4_baselines_pie_video_0004/<UTC>_pie_video_0004/primary

Exit codes: 0 success, 2 contract/configuration/output-directory failure, 3 unexpected failure.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.phase5.evaluation.experiment import run_baseline_experiment  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", required=True, help="P5.3 window run directory (run_manifest.json, windows.npz, graph_bank.npz, ...)")
    ap.add_argument("--config", required=True, help="baseline evaluation configuration")
    ap.add_argument("--out", required=True, help="new or empty output directory")
    a = ap.parse_args(argv)
    args = list(argv if argv is not None else sys.argv[1:])
    try:
        m = run_baseline_experiment(a.windows, a.config, a.out, command=" ".join(["phase5_run_baselines.py"] + args))
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    c = m["counts"]
    print("completed [%s]: %d windows, %d tracks; %s; common evaluated windows %d -> %s" % (
        m["label"], c["windows"], c["unique_target_tracks"],
        "; ".join("%s valid %d invalid %d" % (b, v["valid_predictions"], v["invalid_predictions"]) for b, v in c["per_baseline"].items()),
        c["common_evaluated_windows"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
