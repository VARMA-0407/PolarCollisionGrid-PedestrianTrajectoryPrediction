"""Phase 4 Task 3: independent read-only check of a completed run directory (loads the files from disk; recomputes nothing
upstream). Writes one JSON report (refuses to overwrite).

Checks: manifest output hashes and sizes; dtypes and shapes; node order; edge endpoints in bounds, in the same frame as
the edge, of the relation's types, never self-edges; per-relation edge counts equal the complete-graph bound from node
counts; NaN exactly where masks are False; reversal identities on real data (antisymmetric features negate, symmetric
features match) for the three relation pairs; flags consistent with sep_over_h and node border flags; every feature
physical: false; no world-coordinate key anywhere in the outputs.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedsafety.interaction.storage import load_npz  # noqa: E402

ANTI = ("du_px", "dv_px", "dv_row_px", "log_h_ratio", "dvel_u_px_s", "dvel_v_px_s", "bearing_sin", "bearing_cos")
SYM = ("sep_px", "sep_over_h", "closing_rate_px_s", "overlap_ios")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run, out = Path(a.run), Path(a.out)
    if out.exists():
        print("refusing to overwrite %s" % out, file=sys.stderr)
        return 2
    m = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    rep = {"run": str(run), "checks": {}}
    ck = rep["checks"]
    ck["manifest_hashes_and_sizes"] = all(sha(run / k) == v["sha256"] and (run / k).stat().st_size == v["bytes"] for k, v in m["outputs"].items())
    n = load_npz(run / "nodes.npz")
    N = n["frame"].shape[0]
    types = list(n["node_type_names"])
    ck["node_dtypes"] = (n["features"].dtype == np.float32 and n["masks"].dtype == bool and n["frame"].dtype == np.int64 and n["track_index"].dtype == np.int32)
    order = np.lexsort((n["track_index"], n["frame"]))
    ck["node_order_canonical"] = bool((order == np.arange(N)).all())
    fn = list(n["feature_names"])
    mk = list(n["mask_names"])
    fm = {"vel_u_px_s": "velocity_valid", "vel_v_px_s": "velocity_valid", "speed_px_s": "velocity_valid", "heading_sin": "heading_valid",
          "heading_cos": "heading_valid", "frames_since_prev_obs": "frames_since_prev_obs_valid"}
    nan_ok = True
    for f, mname in fm.items():
        col, msk = n["features"][:, fn.index(f)], n["masks"][:, mk.index(mname)]
        nan_ok &= bool((np.isnan(col) == ~msk).all())
    for f in ("u_px", "v_px", "box_h_px", "track_age_frames"):
        nan_ok &= bool(np.isfinite(n["features"][:, fn.index(f)]).all())
    ck["node_nan_exactly_where_masked"] = nan_ok
    counts = {}
    for f in sorted(run.glob("edges_*.npz")):
        e = load_npz(f)
        s, d = (str(x) for x in e["relation"])
        E = e["frame"].shape[0]
        src, dst = e["edge_index"]
        ok = {
            "dtypes": e["edge_attr"].dtype == np.float32 and e["edge_index"].dtype == np.int64 and e["edge_mask"].dtype == bool,
            "in_bounds": bool(((src >= 0) & (src < N) & (dst >= 0) & (dst < N)).all()),
            "no_self_edges": bool((src != dst).all()),
            "same_frame": bool(((n["frame"][src] == e["frame"]) & (n["frame"][dst] == e["frame"])).all()),
            "types": bool(((n["node_type"][src] == types.index(s)) & (n["node_type"][dst] == types.index(d))).all()),
            "nan_exactly_where_masked": bool((np.isnan(e["edge_attr"]) == ~e["edge_mask"]).all()),
            "truncation_flag": bool((e["flags"][:, 0] == (n["border"][src] | n["border"][dst])).all()),
        }
        names = list(e["feature_names"])
        flags = list(e["flag_names"])
        soh = e["edge_attr"][:, names.index("sep_over_h")]
        ok["radius_flags"] = all(bool((e["flags"][:, flags.index(fl)] == (soh <= float(fl.split("_le_")[1]))).all()) for fl in flags if "_le_" in fl)
        counts["%s__%s" % (s, d)] = {"edges": int(E), **ok, "_e": e}
    for k, v in counts.items():
        ck["edges_" + k] = {kk: vv for kk, vv in v.items() if kk != "_e"}
    # complete-graph bound
    theo = {}
    for fr in np.unique(n["frame"]):
        c = np.bincount(n["node_type"][n["frame"] == fr], minlength=3)
        for si, s in enumerate(types):
            for di, d in enumerate(types):
                theo["%s__%s" % (s, d)] = theo.get("%s__%s" % (s, d), 0) + int(c[si]) * (int(c[di]) - (si == di))
    ck["edges_equal_complete_graph_bound"] = {k: v["edges"] == theo[k] for k, v in counts.items()}
    ck["pairs_not_represented"] = {k: theo[k] for k in theo if k not in counts}
    # reversal identities on real edges
    rev = {}
    for a_, b_ in (("pedestrian__pedestrian", "pedestrian__pedestrian"), ("vehicle__pedestrian", "pedestrian__vehicle"),
                   ("bicyclist__pedestrian", "pedestrian__bicyclist"), ("bicyclist__vehicle", "vehicle__bicyclist")):
        ea, eb = counts[a_]["_e"], counts[b_]["_e"]
        names = list(ea["feature_names"])
        key_b = {(int(s), int(d)): i for i, (s, d) in enumerate(eb["edge_index"].T)}
        idx = np.array([key_b[(int(d), int(s))] for s, d in ea["edge_index"].T])
        res = {}
        for f in ANTI + SYM:
            x, y = ea["edge_attr"][:, names.index(f)].astype(np.float64), eb["edge_attr"][idx, names.index(f)].astype(np.float64)
            both = np.isfinite(x) & np.isfinite(y)
            target = -y if f in ANTI else y
            res[f] = float(np.max(np.abs(x[both] - target[both]))) if both.any() else None
        rev["%s vs %s" % (a_, b_)] = {"pairs": int(idx.size), "max_abs_deviation": res}
    ck["reversal_identities_max_abs_deviation"] = rev
    d = json.loads((run / "feature_dictionary.json").read_text(encoding="utf-8"))
    ck["all_features_physical_false"] = all(f["physical"] is False for f in d["node_features"] + d["edge_features"] + d["edge_flags"])
    txt = "".join((run / k).read_text(encoding="utf-8") for k in ("tracks.json", "interaction_diagnostics.json", "feature_dictionary.json"))
    ck["no_world_coordinate_values_in_outputs"] = '"world_x":' not in txt and '"world_y":' not in txt
    rep["nodes"] = int(N)
    rep["edges_total"] = int(sum(v["edges"] for v in counts.values()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    flat_ok = all(v if isinstance(v, bool) else all(x for x in v.values() if isinstance(x, bool)) for k, v in ck.items()
                  if k not in ("pairs_not_represented", "reversal_identities_max_abs_deviation"))
    print(json.dumps({"all_boolean_checks_passed": flat_ok, "nodes": N, "edges": rep["edges_total"], "reversal": rev}, indent=1))
    return 0 if flat_ok else 1


if __name__ == "__main__":
    sys.exit(main())
