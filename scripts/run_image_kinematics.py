"""Phase 4 Task 2 CLI: image-space kinematics from an existing Phase 3 agent_states.jsonl (no inference is run).

Example:
    python scripts/run_image_kinematics.py --input <dir>/agent_states.jsonl \
        --config configs/phase4/image_kinematics_default.json \
        --out experiments/phase4_task2_pie_video_0004/<UTC>_run \
        --run-manifest <dir>/run_manifest.json --tracked-agents <dir>/tracked_agents.jsonl

Exit codes: 0 success, 2 invalid input/configuration/output directory, 3 unexpected failure.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.image_kinematics import ImageKinematicsConfig  # noqa: E402
from pedsafety.image_kinematics_run import run_image_kinematics  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="Phase 3 agent_states.jsonl (read only)")
    ap.add_argument("--config", required=True, help="Phase 4 Task 2 configuration JSON")
    ap.add_argument("--out", required=True, help="new or empty output directory")
    ap.add_argument("--run-manifest", default=None, help="optional Phase 3 run_manifest.json (hashed and excerpted for provenance)")
    ap.add_argument("--tracked-agents", default=None, help="optional Phase 3 tracked_agents.jsonl (hashed for provenance only)")
    a = ap.parse_args(argv)
    try:
        cfg = ImageKinematicsConfig.load(a.config)
        m = run_image_kinematics(a.input, cfg, a.out, config_path=a.config, run_manifest_path=a.run_manifest,
                                 tracked_agents_path=a.tracked_agents, command=" ".join(["run_image_kinematics.py"] + list(argv if argv is not None else sys.argv[1:])))
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    c = m["counts"]
    print("completed: %d rows, %d tracks, valid velocity %d, valid acceleration %d, valid heading %d, quarantined %d -> %s"
          % (c["causal_rows"], c["tracks"], c["valid_velocity"], c["valid_acceleration"], c["valid_heading"], c["quarantined"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
