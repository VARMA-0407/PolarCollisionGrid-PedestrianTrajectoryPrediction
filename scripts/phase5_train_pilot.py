"""Phase 5 P5.7.4 CLI: pilot training of encoder + decoder on a verified P5.3 window run (development only; all losses are in-sample).

Example:
    python scripts/phase5_train_pilot.py \
        --windows experiments/phase5_task3_windows_pie_video_0004/<run>/primary \
        --config configs/phase5/training_default.json \
        --out experiments/phase5_task7_training/<UTC>_pie_video_0004
Resume (new directory, original untouched):  --resume <run>/checkpoints/checkpoint_epoch_0010.pt
Exit codes: 0 success, 2 contract/configuration/output-directory failure, 3 unexpected failure.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.phase5.models.training_harness import run_training  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", required=True, help="P5.3 window run directory (hash-verified)")
    ap.add_argument("--config", required=True, help="training configuration")
    ap.add_argument("--out", required=True, help="new or empty output directory (never overwritten)")
    ap.add_argument("--resume", help="checkpoint to resume from (provenance must match)")
    a = ap.parse_args(argv)
    args = list(argv if argv is not None else sys.argv[1:])
    try:
        m = run_training(a.windows, a.config, a.out, resume_checkpoint=a.resume, command=" ".join(["phase5_train_pilot.py"] + args))
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    t = m["training"]
    f = t["final_epoch_summary"]
    print("completed [%s]: %d epochs (%d in this run), %d steps, in-sample mean loss first %.4f -> final %.4f, final checkpoint %s -> %s"
          % (m["label"], t["epochs"], t["epochs_run_in_this_directory"], t["final_global_step"], t["first_epoch_summary"]["mean_loss"], f["mean_loss"], t["final_checkpoint"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
