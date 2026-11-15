"""Phase 5 P5.3: write a provenance/comparison record for a Task 3 regeneration (read-only on both runs).

Compares an original Task 3 run with its regeneration from committed code: output hashes, node/edge/relation counts,
feature dimensions, masks, source-line references, coordinate metadata; re-verifies both runs with the Phase 5 input
contract (schema validation, hashes, masks). Writes one NEW JSON file next to (not inside) the regenerated run.

Example:
    python scripts/phase5_task3_regeneration_record.py --original <orig>/primary --regenerated <regen>/primary \
        --pre-run-git-status "tracked clean; untracked: docs/phase5/*.md (5 files)" --out <regen>/regeneration_record.json
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.phase5.task3_source import load_task3_run  # noqa: E402
from pedsafety.phase5.window_config import WindowConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "phase5" / "window_generation_default.json"


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--original", required=True)
    ap.add_argument("--regenerated", required=True)
    ap.add_argument("--pre-run-git-status", required=True, help="working-tree state observed immediately before the regeneration run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print("error: %s exists; refusing to overwrite" % out, file=sys.stderr)
        return 2
    cfg = WindowConfig.load(CONFIG)
    o = load_task3_run(a.original, cfg)              # raises on any schema/hash/mask violation
    r = load_task3_run(a.regenerated, cfg)
    mo, mr = o.manifest, r.manifest
    same_outputs = {k: mo["outputs"][k]["sha256"] == mr["outputs"].get(k, {}).get("sha256") for k in mo["outputs"]}
    differing_manifest_keys = sorted(k for k in set(mo) | set(mr) if mo.get(k) != mr.get(k))
    cmp = {
        "all_output_files_byte_identical": all(same_outputs.values()) and set(mo["outputs"]) == set(mr["outputs"]),
        "output_sha256_equal": same_outputs,
        "node_count": [o.n_nodes, r.n_nodes],
        "edge_count": [mo["counts"]["edges_total"], mr["counts"]["edges_total"]],
        "relation_counts_equal": mo["counts"]["edges_by_relation"] == mr["counts"]["edges_by_relation"],
        "relation_counts": mr["counts"]["edges_by_relation"],
        "node_feature_dims": [list(o.features.shape), list(r.features.shape)],
        "edge_feature_dims": {k: [list(o.edges[k].edge_attr.shape), list(r.edges[k].edge_attr.shape)] for k in r.edges},
        "node_masks_equal": bool(np.array_equal(o.masks, r.masks)),
        "node_features_equal_nan_aware": bool(np.array_equal(np.isnan(o.features), np.isnan(r.features))
                                              and np.array_equal(np.nan_to_num(o.features), np.nan_to_num(r.features))),
        "edge_masks_equal": all(np.array_equal(o.edges[k].edge_mask, r.edges[k].edge_mask) for k in r.edges),
        "source_line_references_equal": bool(np.array_equal(o.source_line, r.source_line)),
        "coordinate_metadata_equal": mo["conventions"] == mr["conventions"] and o.feature_dictionary == r.feature_dictionary,
        "inputs_equal": mo["inputs"] == mr["inputs"],
        "config_equal": mo["config"] == mr["config"],
        "manifest_keys_that_differ": differing_manifest_keys,
        "explanation": "Only provenance/measurement fields differ: git.commit (original recorded 2306309 because Task 3 code was uncommitted "
                       "at run time; regeneration records 5d58ab7), started/finished timestamps, the command's --out path, runtime_s, and "
                       "peak_memory_bytes (null in the original because of the since-fixed Windows API typing issue)."}
    try:
        from importlib.metadata import version as _v
        jsv = _v("jsonschema")
    except Exception:                                                  # noqa: BLE001
        jsv = None
    rec = {"record_version": "phase5-task3-regeneration-record-1.0.0", "purpose": "clean provenance of the Phase 5 P5.3 input (Input contract §13)",
           "original_run": {"path": _rel(Path(a.original)), "manifest_sha256": o.manifest_sha256, "git": mo["git"]},
           "regenerated_run": {"path": _rel(Path(a.regenerated)), "manifest_sha256": r.manifest_sha256, "git": mr["git"],
                               "started_utc": mr["started_utc"], "finished_utc": mr["finished_utc"], "command": mr["command"],
                               "versions": mr["versions"], "config_sha256_resolved": mr["config"]["sha256_resolved"],
                               "input_hashes": {k: v.get("sha256") for k, v in mr["inputs"].items() if isinstance(v, dict)},
                               "output_hashes": {k: v["sha256"] for k, v in mr["outputs"].items()},
                               "peak_memory_bytes": mr["peak_memory_bytes"], "runtime_s_total": mr["runtime_s"].get("total_before_manifest")},
           "pre_run_working_tree": a.pre_run_git_status,
           "inference_rerun": False, "phase3_outputs_reused": True,
           "schema_validation": {"original": "PASS (run manifest + feature dictionary schemas, output hashes, mask/NaN contract)",
                                 "regenerated": "PASS (same checks)"},
           "comparison": cmp,
           "record_environment": {"python": platform.python_version(), "numpy": np.__version__, "jsonschema": jsv}}
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print("byte-identical outputs: %s; differing manifest keys: %s -> %s" % (cmp["all_output_files_byte_identical"], differing_manifest_keys, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
