"""Phase 5 P5.5 CLI: forward-pass smoke test of the UNTRAINED spatial-temporal encoder on a verified P5.3 window run.

No training, no accuracy, no comparison with baselines. Example:
    python scripts/phase5_run_encoder_smoke.py \
        --windows experiments/phase5_task3_windows_pie_video_0004/<run>/primary \
        --config configs/phase5/encoder_default.json \
        --out experiments/phase5_task5_encoder_smoke_pie_video_0004/<UTC>_pie_video_0004/primary
Exit codes: 0 success, 2 contract/configuration/output-directory failure, 3 unexpected failure.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.phase5.models.smoke import run_smoke  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", required=True, help="P5.3 window run directory")
    ap.add_argument("--config", required=True, help="encoder configuration")
    ap.add_argument("--out", required=True, help="new or empty output directory")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--optional-bicycle-windows", type=int, default=128, help="windows for the optional bicycle-context pass (0 disables nothing; must be >= 1)")
    a = ap.parse_args(argv)
    args = list(argv if argv is not None else sys.argv[1:])
    try:
        m = run_smoke(a.windows, a.config, a.out, batch_size=a.batch_size, optional_bicycle_windows=a.optional_bicycle_windows,
                      command=" ".join(["phase5_run_encoder_smoke.py"] + args))
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    c = m["counts"]
    print("completed [%s]: %d windows, %d parameters, numerically valid: %s -> %s" % (m["label"], c["windows"], c["parameter_count"], c["valid"], a.out))
    return 0 if c["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
