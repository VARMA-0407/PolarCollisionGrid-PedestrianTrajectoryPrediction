"""CLI: ``python -m pedsafety.phase8 <command>``. Research prototype; JSON in / JSON out; no network access."""

import argparse
import json
import sys

from .config import AlertConfig
from .runner import read_records, run_alerts, run_on_phase7_artifacts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pedsafety.phase8", description="Pedestrian safety early-warning event generation (RESEARCH PROTOTYPE, not for operational use)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("from-records", help="run on a JSONL(.gz) file of Phase 7 risk records")
    a.add_argument("--records", required=True)
    a.add_argument("--alert-config", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("from-phase7", help="regenerate and cross-check the Phase 7 records from the committed artifacts, then run")
    for k in ("windows-run", "neural-eval-dir", "phase7-dir", "conflict-config", "risk-config", "window-config", "alert-config", "out"):
        b.add_argument("--" + k, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "from-records":
        res = run_alerts(read_records(args.records), AlertConfig.load(args.alert_config), None, None, args.out)
    else:
        res = run_on_phase7_artifacts(args.windows_run, args.neural_eval_dir, args.phase7_dir, args.conflict_config, args.risk_config, args.window_config,
                                      args.alert_config, args.out)
    print(json.dumps(res["summary"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
