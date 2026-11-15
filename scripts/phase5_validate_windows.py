"""Phase 5 P5.3 CLI: independently re-validate a stored window dataset against its manifest and its Task 3 source.

Checks: every output file's size and SHA-256 equal the manifest; the configuration in the manifest re-validates and its
hash matches; the Task 3 source re-verifies (hashes, schema, masks) and its manifest hash equals the recorded one;
all window/bank invariants (``pedsafety.phase5.window_validation``) hold on the data re-read from disk; a dense batch
of windows can be collated with explicit masks. Read-only unless ``--report`` names a new file.

Example:
    python scripts/phase5_validate_windows.py --run experiments/phase5_task3_windows_pie_video_0004/<run>/primary
Exit codes: 0 all checks pass, 1 a check failed, 2 contract failure, 3 unexpected failure.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.image_kinematics_run import sha256_file  # noqa: E402
from pedsafety.phase5.graph_window_adapter import collate_windows  # noqa: E402
from pedsafety.phase5.task3_source import load_task3_run  # noqa: E402
from pedsafety.phase5.window_config import WindowConfig  # noqa: E402
from pedsafety.phase5.window_storage import read_dataset  # noqa: E402
from pedsafety.phase5.window_validation import validate_window_dataset  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def validate_run(run_dir) -> dict:
    d = Path(run_dir)
    m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    out = {"run": str(d), "checks": {}}

    def chk(name, ok, detail=None):
        out["checks"][name] = {"pass": bool(ok), "detail": detail}
    bad = [n for n, r in m["outputs"].items() if not (d / n).is_file() or (d / n).stat().st_size != r["bytes"] or sha256_file(d / n) != r["sha256"]]
    chk("output_hashes_match_manifest", not bad, bad or sorted(m["outputs"]))
    cfg = WindowConfig.from_dict(m["config"]["content"])
    chk("config_hash_matches_manifest", cfg.sha256() == m["config"]["sha256_resolved"])
    chk("label_pilot_only", m.get("label") == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT")
    chk("claims_all_false", all(v is False for v in m["claims"].values() if isinstance(v, bool)))
    t3dir = REPO / m["sources"]["task3_run_dir"]
    run = load_task3_run(t3dir, cfg)
    chk("task3_manifest_hash_matches", run.manifest_sha256 == m["sources"]["task3_run_manifest"]["sha256"])
    ws, bank, extra = read_dataset(d)
    rep = validate_window_dataset(ws, bank, extra["history_target_bank_node"], cfg, run)
    for k, v in rep["checks"].items():
        out["checks"]["dataset." + k] = v
    c = m["counts"]
    chk("counts_match_manifest", c["accepted_windows"] == ws.n_windows and c["rejected_windows"] == ws.n_rejected
        and c["rejection_reasons"] == ws.rejection_counts())
    n = min(16, ws.n_windows)
    if n:
        b = collate_windows(bank, ws, range(n), cfg)
        ok = bool(b["present"][:, :, 0].all()) and b["node_features"].shape[:2] == (n, cfg.history_length) \
            and bool((b["future_frames"] > b["history_frames"][:, -1:]).all())
        b2 = collate_windows(bank, ws, range(n), cfg, include_optional=True)
        chk("collate_batch_default_and_optional_context", ok and b2["node_features"].shape[2] >= b["node_features"].shape[2],
            {"windows": n, "agents_default": int(b["node_features"].shape[2]), "agents_with_optional": int(b2["node_features"].shape[2]),
             "relations_default": list(b["active_relations"]), "relations_with_optional": list(b2["active_relations"])})
    out["ok"] = all(v["pass"] for v in out["checks"].values())
    out["n_checks"] = len(out["checks"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--report", help="write the JSON report to this NEW file (refuses to overwrite)")
    a = ap.parse_args(argv)
    try:
        rep = validate_run(a.run)
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    if a.report:
        p = Path(a.report)
        if p.exists():
            print("error: report path exists; refusing to overwrite: %s" % p, file=sys.stderr)
            return 2
        p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    failed = [k for k, v in rep["checks"].items() if not v["pass"]]
    print("%s: %d/%d checks passed%s" % ("PASS" if rep["ok"] else "FAIL", rep["n_checks"] - len(failed), rep["n_checks"],
                                         "" if not failed else "; failed: %s" % failed))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
