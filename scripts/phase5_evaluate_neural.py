"""Phase 5 P5.7.5 CLI: evaluate a trained pilot model (verified checkpoint + scaler) with the unchanged P5.4 evaluator, history-only inference.

Example:
    python scripts/phase5_evaluate_neural.py --train experiments/phase5_task7_training/<run> \
        --windows experiments/phase5_task3_windows_pie_video_0004/<run>/primary \
        --config configs/phase5/neural_evaluation_default.json --out experiments/phase5_task7_neural_evaluation/<UTC>_pie_video_0004
All results are PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT and IN-SAMPLE. Exit codes: 0 success, 2 contract/verification failure, 3 unexpected failure.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.phase5.models.neural_evaluation import run_neural_evaluation  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", required=True, help="training run directory (run_manifest.json, checkpoints/, scaler_statistics.json)")
    ap.add_argument("--windows", required=True, help="P5.3 window run directory")
    ap.add_argument("--config", required=True, help="neural evaluation configuration")
    ap.add_argument("--out", required=True, help="new or empty output directory")
    a = ap.parse_args(argv)
    args = list(argv if argv is not None else sys.argv[1:])
    try:
        m = run_neural_evaluation(a.train, a.windows, a.config, a.out, command=" ".join(["phase5_evaluate_neural.py"] + args))
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    c = m["counts"]
    print("completed [%s, %s]: %d windows, K=%d, common windows with baselines %s -> %s" % (m["label"], m["sample_status"], c["windows_evaluated"], c["num_modes"],
                                                                                           c["common_windows_with_baselines"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
