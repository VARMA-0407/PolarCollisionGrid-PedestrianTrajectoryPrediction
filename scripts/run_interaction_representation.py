"""Phase 4 Task 3 CLI: image-space interaction representation from an existing Task 2 causal sidecar (no inference).

Example:
    python scripts/run_interaction_representation.py \
        --causal <task2_run>/causal_kinematics.jsonl --task2-manifest <task2_run>/run_manifest.json \
        --phase3 PIE_pilot_inputs/agent_states.jsonl --config configs/phase4/interaction_representation_default.json \
        --out experiments/phase4_task3_pie_video_0004/<UTC>_run

Exit codes: 0 success, 2 contract/configuration/output-directory failure, 3 unexpected failure.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.interaction.config import InteractionConfig  # noqa: E402
from pedsafety.interaction.runner import run_interaction_representation  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--causal", required=True, help="Task 2 causal_kinematics.jsonl (read only; the offline file is refused)")
    ap.add_argument("--task2-manifest", required=True, help="the Task 2 run_manifest.json of that run")
    ap.add_argument("--phase3", required=True, help="the Phase 3 agent_states.jsonl the Task 2 run was made from (read only)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="new or empty output directory")
    a = ap.parse_args(argv)
    args = list(argv if argv is not None else sys.argv[1:])
    try:
        cfg = InteractionConfig.load(a.config)
        m = run_interaction_representation(a.causal, a.task2_manifest, a.phase3, cfg, a.out, config_path=a.config,
                                           command=" ".join(["run_interaction_representation.py"] + args))
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    c = m["counts"]
    print("completed: %d nodes in %d frames, %d edges (%s) -> %s" % (c["nodes"], c["frames_with_nodes"], c["edges_total"],
                                                                     ", ".join("%s %d" % kv for kv in c["edges_by_relation"].items()), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
