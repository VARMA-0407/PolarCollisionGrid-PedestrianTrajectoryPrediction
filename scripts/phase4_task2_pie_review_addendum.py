"""Phase 4 Task 2 final-review addendum for a completed real-data run (read-only; nothing is re-processed or overwritten).

Adds three things the original analysis (scripts/phase4_task2_pie_analysis.py v1.0.0) did not report precisely:

1. Exact gap boundaries. Gap convention implemented in pedsafety/track_continuity.py:
       frame_difference = before_frame - after_frame          (consecutive accepted observations of one track)
       missing_frames   = frame_difference // frame_step - 1  (frames with no row for that track)
       elapsed_s        = timestamp(before) - timestamp(after)
   A track's first/last frame are NOT gap boundaries.
2. A stride test that separates frame-to-frame jitter from smooth motion better than a single-stride estimate:
       d2_k = p(f+k) - 2 p(f) + p(f-k)   for rows at f-k, f, f+k of one track (k frames apart, any rows in between ignored)
       sigma_k = 1.4826 * MAD(d2_k) / sqrt(6)   per axis
   Independent white position noise gives sigma_k constant in k; a smooth (locally constant) acceleration a adds
   a (k dt)^2, i.e. grows as k^2.
3. Final consistency checks between the primary run and the determinism re-run, the input hashes, the run
   configuration, causal/offline labelling, world fields and the sidecar schema.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedsafety.image_kinematics_run import load_sidecar_validator  # noqa: E402

STRIDES = (1, 2, 3, 4, 6, 8)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def jl(p):
    with open(p, "r", encoding="utf-8") as fh:
        for l in fh:
            if l.strip():
                yield json.loads(l)


def robust_sigma(values):
    a = np.asarray(values, dtype=float)
    return None if a.size == 0 else float(1.4826 * np.median(np.abs(a - np.median(a))) / math.sqrt(6.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="the timestamped real-data run directory")
    ap.add_argument("--input-dir", required=True, help="directory with the original Phase 3 files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    run, inp, out = Path(a.run_dir), Path(a.input_dir), Path(a.out)
    if out.exists():
        print("refusing to overwrite %s" % out, file=sys.stderr)
        return 2
    prim, rer = run / "primary_causal", run / "determinism_rerun"
    integ = json.loads((run / "input_integrity.json").read_text(encoding="utf-8"))

    # ---- 1. exact gap boundaries from the continuity report, cross-checked against the original input ----
    cont = json.loads((prim / "track_continuity_report.json").read_text(encoding="utf-8"))
    frames_in = defaultdict(list)
    for r in jl(inp / "agent_states.jsonl"):
        frames_in[r["track_id"]].append((r["frame_index"], r["timestamp"]))
    gaps = []
    for t in cont["per_track"]:
        for g in t["gaps"]:
            gaps.append({"track_id": t["track_id"], "agent_class": t["agent_class"], "last_observed_frame_before_gap": g["after_frame"],
                         "first_observed_frame_after_gap": g["before_frame"], "frame_difference": g["before_frame"] - g["after_frame"],
                         "missing_frames": g["missing_frames"], "elapsed_s": g["elapsed_s"],
                         "track_first_frame": t["first_frame"], "track_last_frame": t["last_frame"]})
    mismatch = 0
    for tid, rows in frames_in.items():
        expected = [(b[0] - a_[0] - 1, a_[0], b[0]) for a_, b in zip(rows, rows[1:]) if b[0] - a_[0] > 1]
        reported = [(g["missing_frames"], g["last_observed_frame_before_gap"], g["first_observed_frame_after_gap"]) for g in gaps if g["track_id"] == tid]
        mismatch += expected != reported
    gaps.sort(key=lambda g: (-g["missing_frames"], g["track_id"]))
    gap_section = {"convention": "missing_frames = (first_observed_frame_after_gap - last_observed_frame_before_gap) // frame_step - 1; "
                                 "elapsed_s from record timestamps; track first/last frames are not gap boundaries",
                   "n_gaps": len(gaps), "missing_frames_total": sum(g["missing_frames"] for g in gaps),
                   "tracks_where_input_and_report_disagree": mismatch, "longest_gaps": gaps[:a.top]}

    # ---- 2. stride test ----
    by = defaultdict(dict)
    for r in jl(prim / "causal_kinematics.jsonl"):
        if r["position_state"] == "ok":
            by[r["track_id"]][r["frame_index"]] = (r["position_x"], r["position_y"], r["agent_class"])
    stride = {}
    for k in STRIDES:
        d = defaultdict(lambda: ([], []))
        for fm in by.values():
            for f, (x, y, c) in fm.items():
                if f - k in fm and f + k in fm:
                    (xm, ym, _), (xp, yp, _) = fm[f - k], fm[f + k]
                    for cls in (c, "all"):
                        d[cls][0].append(xp - 2 * x + xm)
                        d[cls][1].append(yp - 2 * y + ym)
        stride[str(k)] = {cls: {"n": len(v[0]), "sigma_x_px": robust_sigma(v[0]), "sigma_y_px": robust_sigma(v[1])} for cls, v in sorted(d.items())}
    s1, s8 = stride["1"]["all"], stride["8"]["all"]
    smooth_bound = {ax: s8[ax] / 64.0 for ax in ("sigma_x_px", "sigma_y_px")}
    stride_section = {
        "formula": "sigma_k = 1.4826 * MAD(p(f+k) - 2p(f) + p(f-k)) / sqrt(6), per axis, rows of one track at f-k, f, f+k",
        "by_stride": stride,
        "smooth_motion_upper_bound_at_k1_px": smooth_bound,
        "interpretation": ("sigma_k grows with k (about 1.1 px at k=1 to about 2.5-3.2 px at k=8), so the second differences are not pure white "
                           "noise. If all of the k=8 spread were locally constant acceleration over 16 frames, its share at k=1 would be at most "
                           "sigma_8/64 (about 0.04-0.05 px), far below the observed k=1 spread (about 1.1 px). Under that assumption most of the "
                           "frame-to-frame second difference is not smooth motion: detector box jitter and/or high-frequency image motion such "
                           "as camera vibration (which is real image motion, not detector error). The assumption fails for abrupt motion, so "
                           "this is evidence, not an isolation of noise from movement."),
    }

    # ---- 3. consistency ----
    mp = json.loads((prim / "run_manifest.json").read_text(encoding="utf-8"))
    mr = json.loads((rer / "run_manifest.json").read_text(encoding="utf-8"))
    file_cmp = {}
    for name in sorted(mp["outputs"]):
        hp, hr = sha256(prim / name), sha256(rer / name)
        file_cmp[name] = {"primary": hp, "rerun": hr, "identical": hp == hr, "matches_manifest": hp == mp["outputs"][name]["sha256"]}
    differing_manifest_keys = sorted(k for k in set(mp) | set(mr) if mp.get(k) != mr.get(k))
    out_diff = sorted(k for k in mp["outputs"] if mp["outputs"][k] != mr["outputs"].get(k))
    inputs_now = {n: sha256(inp / n) for n in integ["files"]}
    cfg = json.loads((prim / "config_resolved.json").read_text(encoding="utf-8"))
    v = load_sidecar_validator()
    counts = {"causal_rows": 0, "offline_rows": 0, "schema_violations": 0, "world_populated": 0, "causal_not_backward": 0, "offline_not_central": 0}
    for name, key, ok_view, ok_method in (("causal_kinematics.jsonl", "causal_rows", "CAUSAL", "backward_difference_actual_dt"),
                                          ("offline_kinematics.jsonl", "offline_rows", "OFFLINE_NON_CAUSAL", "central_difference_actual_dt")):
        for r in jl(prim / name):
            counts[key] += 1
            counts["schema_violations"] += next(v.iter_errors(r), None) is not None
            counts["world_populated"] += r["world_x"] is not None or r["world_y"] is not None or r["world_position_valid"] is not False
            if r["view"] != ok_view or r["velocity_method"] != ok_method:
                counts["causal_not_backward" if ok_view == "CAUSAL" else "offline_not_central"] += 1
    consistency = {
        "output_files": file_cmp,
        "all_outputs_identical_except": sorted(n for n, c in file_cmp.items() if not c["identical"]),
        "manifest_keys_that_differ": differing_manifest_keys, "manifest_output_entries_that_differ": out_diff,
        "input_hashes_now": inputs_now, "input_hashes_at_integrity_check": {n: d["sha256"] for n, d in integ["files"].items()},
        "inputs_unchanged": all(inputs_now[n] == integ["files"][n]["sha256"] for n in inputs_now),
        "config": {"max_derivative_gap_frames": cfg["max_derivative_gap_frames"], "smoothing": cfg["smoothing"],
                   "min_heading_speed_px_per_s": cfg["min_heading_speed_px_per_s"], "expected_image_coordinate_system": cfg["expected_image_coordinate_system"]},
        "row_checks": counts,
    }
    report = {"addendum_version": "phase4-task2-review-addendum-1.0.0", "run_dir": str(run), "data": "REAL PIE set01/video_0004",
              "original_analysis_script_sha256_v1_0_0": "ea989bbabbb9766e48a26cc17a12bd3952a789e12f43158cb5140919a72755dd",
              "gaps": gap_section, "stride_noise_test": stride_section, "consistency": consistency}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gap_mismatch_tracks": mismatch, "longest": gaps[0], "identical_except": consistency["all_outputs_identical_except"],
                      "manifest_keys_differ": differing_manifest_keys, "inputs_unchanged": consistency["inputs_unchanged"],
                      "config": consistency["config"], "rows": counts, "smooth_bound": smooth_bound}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
