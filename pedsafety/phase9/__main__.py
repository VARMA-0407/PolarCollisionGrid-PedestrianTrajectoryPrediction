"""CLI: ``python -m pedsafety.phase9 run|evaluation-status``. Research prototype; exit code 1 if any stage failed, 2 for a configuration error."""

import argparse
import dataclasses
import json
import sys

from .errors import PipelineError
from .evaluation import assess_independent_evaluation, blocked_evaluations
from .orchestrator import PipelineConfig, run_pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pedsafety.phase9", description="End-to-end pedestrian safety pipeline orchestration (RESEARCH PROTOTYPE, not for operational use)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the selected stages and write a consolidated manifest")
    r.add_argument("--config", required=True)
    r.add_argument("--stages", nargs="*", help="override the configured stage selection")
    r.add_argument("--run-id")
    r.add_argument("--output-root")
    r.add_argument("--report-dir")
    r.add_argument("--no-benchmark", action="store_true")
    e = sub.add_parser("evaluation-status", help="report which independent evaluations are possible")
    e.add_argument("--repo", default=".")
    e.add_argument("--split-manifest")
    v = sub.add_parser("verify-run", help="compare a finished run directory with the committed reference artifacts (exact equality) and hash the inputs")
    v.add_argument("--run-dir", required=True)
    v.add_argument("--config", required=True)
    v.add_argument("--out")
    t = sub.add_parser("attribute", help="derive the performance attribution and execution-mode statement from an existing run manifest (no re-measurement)")
    t.add_argument("--manifest", required=True)
    t.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "evaluation-status":
            st = assess_independent_evaluation(a.repo, a.split_manifest)
            print(json.dumps({"assessment": st, "evaluations": blocked_evaluations(st)}, indent=1))
            return 0
        if a.cmd == "verify-run":
            from ..phase6.runner import REPO
            from .verification import compare_run_with_references, hash_inputs, inventory_run
            cfgd = PipelineConfig.load(a.config).to_dict()
            rep = {"run_dir": a.run_dir, "checks": compare_run_with_references(a.run_dir, cfgd, REPO), "input_sha256_computed_after_the_run": hash_inputs(cfgd, REPO), **inventory_run(a.run_dir),
                   "note": "input hashes are computed now and are tied to the run by the upstream manifests' recorded hashes (Phase 4 input sha, Phase 5 window and model hashes)"}
            txt = json.dumps(rep, indent=1, sort_keys=True)
            if a.out:
                open(a.out, "w", encoding="utf-8").write(txt)
            print(json.dumps({"all_compared_checks_pass": rep["checks"]["all_compared_checks_pass"], "compared": [k for k in rep["checks"] if k != "all_compared_checks_pass"]}, indent=1))
            return 0 if rep["checks"]["all_compared_checks_pass"] else 1
        if a.cmd == "attribute":
            from .benchmark import hardware_info
            from .reporting import build_attribution
            m = json.load(open(a.manifest, encoding="utf-8"))
            att = build_attribution(m, {**hardware_info(), "measured_after_run_on_the_same_machine": True})
            txt = json.dumps(att, indent=1, sort_keys=True, default=str)
            if a.out:
                open(a.out, "w", encoding="utf-8").write(txt)
            print(txt)
            return 0
        cfg = PipelineConfig.load(a.config)
        over = {}
        if a.stages:
            over["stages"] = tuple(a.stages)
        if a.output_root:
            over["output_root"] = a.output_root
        if a.report_dir is not None:
            over["report_dir"] = a.report_dir
        if over:
            cfg = PipelineConfig.from_dict({**{k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(cfg).items()}, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in over.items()}})
        m = run_pipeline(cfg, run_id=a.run_id, benchmark=not a.no_benchmark)
    except PipelineError as exc:
        print(json.dumps({"status": "CONFIG_OR_OUTPUT_ERROR", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": m["status"], "run_dir": m["run_dir"], "stages": {s: r["status"] for s, r in m["stages"].items()}, "counts": m["counts"]}, indent=1, default=str))
    return 1 if m["status"] == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
