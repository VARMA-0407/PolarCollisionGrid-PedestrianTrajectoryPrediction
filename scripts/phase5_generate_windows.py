"""Phase 5 P5.3 CLI: pilot trajectory-window generation from an existing Phase 4 Task 3 run (no inference, no model).

Example:
    python scripts/phase5_generate_windows.py \
        --task3 experiments/phase4_task3_pie_video_0004/<run>/primary \
        --phase3-manifest PIE_pilot_inputs/run_manifest.json --tracked-agents PIE_pilot_inputs/tracked_agents.jsonl \
        --config configs/phase5/window_generation_default.json \
        --out experiments/phase5_task3_windows_pie_video_0004/<UTC>_pie_video_0004/primary

Exit codes: 0 success, 2 contract/configuration/output-directory failure, 3 unexpected failure.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.phase5.runner import generate_windows  # noqa: E402
from pedsafety.phase5.window_config import WindowConfig  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task3", required=True, help="Task 3 run directory (run_manifest.json, nodes.npz, edges_*.npz, ...)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="new or empty output directory")
    ap.add_argument("--phase3-manifest", help="Phase 3 run_manifest.json (fps, frame range, label map)")
    ap.add_argument("--tracked-agents", help="Phase 3 tracked_agents.jsonl (hashed for provenance only)")
    ap.add_argument("--video-id")
    ap.add_argument("--group-id")
    ap.add_argument("--skip-upstream", action="store_true", help="skip Phase 3/Task 2 hash and source-line verification (tests only)")
    a = ap.parse_args(argv)
    args = list(argv if argv is not None else sys.argv[1:])
    try:
        cfg = WindowConfig.load(a.config)
        m = generate_windows(a.task3, cfg, a.out, config_path=a.config, command=" ".join(["phase5_generate_windows.py"] + args),
                             phase3_manifest=Path(a.phase3_manifest) if a.phase3_manifest else None,
                             tracked_agents=Path(a.tracked_agents) if a.tracked_agents else None,
                             video_id=a.video_id, group_id=a.group_id, check_upstream=not a.skip_upstream)
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    c = m["counts"]
    print("completed [%s]: %d candidates, %d accepted, %d rejected (%s); %d unique pedestrian targets; validation %d/%d checks passed -> %s"
          % (m["label"], c["candidate_windows"], c["accepted_windows"], c["rejected_windows"],
             ", ".join("%s %d" % kv for kv in c["rejection_reasons"].items()), c["unique_pedestrian_target_tracks_accepted"],
             sum(v["pass"] for v in m["validation"]["checks"].values()), m["validation"]["n_checks"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
