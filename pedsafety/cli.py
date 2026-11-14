"""Stage 6 command line: ``python -m pedsafety <command> ...``.

Exit codes: 0 ok | 1 smoke/consistency check failed | 2 configuration error |
3 input/data error | 4 split refused | 5 calibration error | 6 other pedsafety error |
7 output directory refused (exists and is not empty, or is a file) | 70 unexpected internal error.

Output policy: every command writes only into a new or EMPTY directory; nothing is overwritten or deleted. Use a fresh
directory per run. A run directory is complete only if its completion file exists (run_manifest.json, plot_manifest.json
or hbs_smoke_report.json) and error.json is absent. Failures print one line to stderr and, when ``--out`` is
given, write ``<out>/error.json`` (stage, code, message, details). Use ``--debug`` for a traceback.
"""

import argparse
import json
import sys
import traceback
from typing import List, Optional

from . import pipeline
from .errors import ConfigError
from .io.synthetic_corpus import write_synthetic_corpus
from .pipeline_config import load_config


def _common(p: argparse.ArgumentParser, out_required: bool = True) -> None:
    p.add_argument("config", help="pipeline config JSON")
    p.add_argument("--out", required=out_required, help="output directory (created if absent)")
    p.add_argument("--input", default=None, help="override input.csv (read-only)")
    p.add_argument("--strict", action="store_true", help="fail on the first parse/validation issue")
    p.add_argument("--fail-on-issues", action="store_true", help="exit 3 if any row was rejected or any warning exists")
    p.add_argument("--debug", action="store_true", help="show tracebacks")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m pedsafety", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate-config", help="load and check a config without reading data")
    p.add_argument("config")
    p.add_argument("--debug", action="store_true")
    for name, helptext in (("ingest", "CSV -> schema-valid records (+ optional calibration projection); reports only"),
                           ("features", "descriptive causal features over all records (no split)"),
                           ("build", "full pipeline: split, windows, train-only normalisation, run manifest")):
        _common(sub.add_parser(name, help=helptext))
    p = sub.add_parser("plot", help="figures: trajectories, kinematics, projection, split, windows")
    _common(p)
    p.add_argument("--which", default=None, help="comma-separated subset of figure names")
    p = sub.add_parser("hbs-smoke", help="read-only HBS check; asserts nothing metric is produced and the split is refused")
    _common(p)
    p.add_argument("--max-rows", type=int, default=None, help="use a copy of the first N data rows (source untouched)")
    p = sub.add_parser("synthetic", help="write a synthetic multi-scene CSV and a matching config")
    p.add_argument("--out", required=True)
    p.add_argument("--n-scenes", type=int, default=6)
    p.add_argument("--n-frames", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-px", type=float, default=0.0)
    p.add_argument("--debug", action="store_true")
    return ap


def _run(args) -> int:
    if args.command == "synthetic":
        info = write_synthetic_corpus(args.out, n_scenes=args.n_scenes, n_frames=args.n_frames, seed=args.seed, noise_px=args.noise_px)
        print(json.dumps(info, indent=2))
        return pipeline.EXIT_OK
    cfg = load_config(args.config, input_csv_override=getattr(args, "input", None))
    if args.command == "validate-config":
        print("config OK  sha256=%s  grouping=%s  windows=%s" % (cfg.sha256(), "set" if cfg.grouping else "absent",
                                                                  "set" if cfg.windows else "absent"))
        return pipeline.EXIT_OK
    kw = {"strict": args.strict}
    if args.command == "ingest":
        m = pipeline.run_ingest(cfg, args.out, fail_on_issues=args.fail_on_issues, **kw)
    elif args.command == "features":
        m = pipeline.run_features(cfg, args.out, fail_on_issues=args.fail_on_issues, **kw)
    elif args.command == "build":
        m = pipeline.run_build(cfg, args.out, fail_on_issues=args.fail_on_issues, **kw).manifest
    elif args.command == "plot":
        m = pipeline.run_plot(cfg, args.out, which=args.which.split(",") if args.which else None, **kw)
    elif args.command == "hbs-smoke":
        rep = pipeline.run_hbs_smoke(cfg, args.out, max_rows=args.max_rows)
        print("HBS smoke: %s" % rep["status"])
        for k, v in rep["checks"].items():
            print("  [%s] %s" % ("ok" if v else "FAIL", k))
        return pipeline.EXIT_OK if rep["status"] == "pass" else pipeline.EXIT_CHECK_FAILED
    else:  # pragma: no cover - argparse guarantees a command
        raise ConfigError("unknown command %r" % args.command)
    print("%s: %s" % (args.command, m.get("status", "completed")))
    for w in m.get("warnings", []):
        print("  warning: %s" % w)
    return pipeline.EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except SystemExit:
        raise
    except BaseException as exc:                                     # noqa: BLE001 - mapped to a stable exit code
        if isinstance(exc, KeyboardInterrupt):
            raise
        info = pipeline.classify_error(exc)
        print("error [%s/%s]: %s" % (info["stage"], info["code"], info["message"]), file=sys.stderr)
        for d in info["details"][:5]:
            print("  - %s" % d, file=sys.stderr)
        if getattr(args, "out", None):
            written = pipeline.write_error(args.out, info, owned=getattr(exc, "pedsafety_output_owned", None) is not None)
            if written is None:
                print("  (error.json not written: the output directory was refused or is not owned by this run; "
                      "existing files were left untouched)", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc()
        return info["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
