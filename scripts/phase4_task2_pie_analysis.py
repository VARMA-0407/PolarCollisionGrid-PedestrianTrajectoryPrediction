"""Phase 4 Task 2: read-only analysis of a completed real-data run (no re-processing of kinematics, no inference).

Reads the run's causal/offline sidecars and reports, plus the original agent_states.jsonl (for box heights), and writes
one JSON summary (refuses to overwrite). Independent checks performed from the files on disk:

* every causal and offline row re-validated against the sidecar schema;
* world fields: none populated;
* causal velocities recomputed here as backward differences from the causal file's own positions/timestamps and
  compared with the stored values (proves the causal file holds no central-difference / offline values);
* determinism: output hashes of two independent runs compared (optional second run directory).

Derived diagnostics: gap distribution with percentages and cumulative shares, derivative losses caused by gaps, and a
robust footpoint-noise estimate from second differences on consecutive frames (sigma = 1.4826 * MAD(d2) / sqrt(6) per
axis, valid when true motion changes little over 2 frames). The noise estimate is EVIDENCE for a later smoothing
decision; it does not select a time constant.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedsafety.image_kinematics_run import load_sidecar_validator  # noqa: E402

DT = 1.0 / 30.0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def jl(p):
    with open(p, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def robust_sigma(d2):
    a = np.asarray(d2, dtype=float)
    if a.size == 0:
        return None
    return float(1.4826 * np.median(np.abs(a - np.median(a))) / math.sqrt(6.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runner output directory (primary)")
    ap.add_argument("--rerun", default=None, help="second runner output directory for the determinism check")
    ap.add_argument("--input", required=True, help="the original agent_states.jsonl (read only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run, out = Path(a.run), Path(a.out)
    if out.exists():
        print("refusing to overwrite %s" % out, file=sys.stderr)
        return 2
    in_sha_before = sha256(a.input)
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    causal, offline = jl(run / "causal_kinematics.jsonl"), jl(run / "offline_kinematics.jsonl")
    cont = json.loads((run / "track_continuity_report.json").read_text(encoding="utf-8"))
    qual = json.loads((run / "kinematic_quality_report.json").read_text(encoding="utf-8"))
    src = {n: r for n, r in enumerate(jl(a.input), start=1)}          # source_line -> record (blank lines: none in this file)

    # ---- independent checks ----
    v = load_sidecar_validator()
    schema_viol = {"causal": sum(1 for r in causal if next(v.iter_errors(r), None) is not None),
                   "offline": sum(1 for r in offline if next(v.iter_errors(r), None) is not None)}
    world = sum(1 for r in causal + offline if r["world_x"] is not None or r["world_y"] is not None or r["world_position_valid"] is not False)
    views = {"causal": dict(Counter("%s|%s" % (r["view"], r["velocity_method"]) for r in causal).most_common()),
             "offline": dict(Counter("%s|%s" % (r["view"], r["velocity_method"]) for r in offline).most_common())}
    by_track = defaultdict(list)
    for r in causal:
        by_track[r["track_id"]].append(r)
    max_dev, n_cmp = 0.0, 0
    for rows in by_track.values():
        for p, c in zip(rows, rows[1:]):
            if c["velocity_state"] == "valid":
                dt = c["timestamp"] - p["timestamp"]
                for k, ax in (("velocity_x", "position_x"), ("velocity_y", "position_y")):
                    max_dev = max(max_dev, abs((c[ax] - p[ax]) / dt - c[k]))
                n_cmp += 1
    causal_recompute = {"rows_compared": n_cmp, "max_abs_difference_px_per_s": max_dev,
                        "all_valid_rows_follow_previous_row_consecutively": all(
                            c["frame_index"] - p["frame_index"] == 1 for rows in by_track.values() for p, c in zip(rows, rows[1:]) if c["velocity_state"] == "valid")}

    # ---- gaps ----
    hist = {int(k): n for k, n in cont["summary"]["gap_missing_frames_histogram"].items()}
    n_gaps = sum(hist.values())
    cum, run_total = [], 0
    for k in sorted(hist):
        run_total += hist[k]
        cum.append({"missing_frames": k, "n_gaps": hist[k], "pct_of_gaps": 100.0 * hist[k] / n_gaps, "cumulative_pct": 100.0 * run_total / n_gaps})
    buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 30), (31, 10 ** 9)]
    bucket_tab = [{"missing_frames": "%d-%s" % (lo, "inf" if hi > 10 ** 8 else hi) if lo != hi else str(lo),
                   "n_gaps": sum(n for k, n in hist.items() if lo <= k <= hi),
                   "pct_of_gaps": 100.0 * sum(n for k, n in hist.items() if lo <= k <= hi) / n_gaps} for lo, hi in buckets]
    missing_total = sum(k * n for k, n in hist.items())
    rows_n = len(causal)
    vs, acs = Counter(r["velocity_state"] for r in causal), Counter(r["acceleration_state"] for r in causal)
    prev_unavail_after_gap = sum(1 for rows in by_track.values() for p, c in zip(rows, rows[1:])
                                 if c["acceleration_state"] == "previous_velocity_unavailable" and p["velocity_state"] == "gap_before")
    losses = {"velocity_rows_lost_to_gaps": vs["gap_before"], "velocity_rows_lost_to_gaps_pct_of_rows": 100.0 * vs["gap_before"] / rows_n,
              "velocity_rows_lost_to_track_start": vs["insufficient_samples"],
              "acceleration_rows_lost_to_gaps": acs["gap_before"] + prev_unavail_after_gap,
              "acceleration_rows_lost_to_gaps_pct_of_rows": 100.0 * (acs["gap_before"] + prev_unavail_after_gap) / rows_n,
              "acceleration_rows_lost_to_track_start": acs["insufficient_samples"] + acs["previous_velocity_unavailable"] - prev_unavail_after_gap,
              "heading_rows_lost_to_gaps": Counter(r["heading_state"] for r in causal)["gap_before"],
              "what_if_bridging_counts_only": [{"max_bridge_frames": m, "gaps_within": sum(n for k, n in hist.items() if k <= m),
                                                "pct_of_gaps": 100.0 * sum(n for k, n in hist.items() if k <= m) / n_gaps}
                                               for m in (1, 2, 3, 5, 10, 30)],
              "note": "what-if rows are counts of gaps only; bridging is NOT enabled and no bridged value was computed"}

    # ---- footpoint noise from second differences on consecutive triples ----
    d2 = defaultdict(lambda: {"x": [], "y": [], "x_rel": [], "y_rel": []})
    for rows in by_track.values():
        for r0, r1, r2 in zip(rows, rows[1:], rows[2:]):
            if r1["frame_index"] - r0["frame_index"] == 1 and r2["frame_index"] - r1["frame_index"] == 1:
                b = src[r1["source_line"]]["bbox_xyxy"]
                h = b[3] - b[1]
                for ax, key in (("x", "position_x"), ("y", "position_y")):
                    val = r2[key] - 2 * r1[key] + r0[key]
                    for cls in (r1["agent_class"], "all"):
                        d2[cls][ax].append(val)
                        d2[cls][ax + "_rel"].append(val / h)
    noise = {}
    for cls, dd in sorted(d2.items()):
        sx, sy = robust_sigma(dd["x"]), robust_sigma(dd["y"])
        s = math.sqrt((sx ** 2 + sy ** 2) / 2.0)
        acc_axis = math.sqrt(6.0) * s / DT ** 2                        # backward-difference acceleration noise per axis
        noise[cls] = {"triples": len(dd["x"]), "sigma_x_px": sx, "sigma_y_px": sy,
                      "sigma_x_over_box_height": robust_sigma(dd["x_rel"]), "sigma_y_over_box_height": robust_sigma(dd["y_rel"]),
                      "implied_raw_velocity_noise_per_axis_px_per_s": math.sqrt(2.0) * s / DT,
                      "implied_raw_acceleration_noise_median_magnitude_px_per_s2": 1.1774 * acc_axis}
    measured_acc_median = qual["causal_raw"]["acceleration_magnitude_px_per_s2"]["p50"]
    noise_note = ("Robust estimate assuming independent per-frame footpoint noise and negligible true change of velocity over two frames. "
                  "Real detector jitter is correlated and class-dependent; motion of the ego-camera also enters. Evidence only; no tau selected.")

    # ---- examples ----
    pt = cont["per_track"]
    pick = lambda key, n=5: [{k: t[k] for k in ("track_id", "agent_class", "first_frame", "last_frame", "n_valid_positions", "n_gaps",  # noqa: E731
                                              "max_gap_frames", "missing_frames_total", "n_border_touching_rows", "n_suspicious_jumps_at_lowest_threshold")}
                             for t in sorted(pt, key=key)[:n]]
    examples = {"most_gaps": pick(lambda t: (-t["n_gaps"], t["track_id"])),
                "longest_single_gap": pick(lambda t: (-t["max_gap_frames"], t["track_id"])),
                "most_suspicious_jumps_at_0.25_box_heights": pick(lambda t: (-t["n_suspicious_jumps_at_lowest_threshold"], t["track_id"])),
                "most_border_touching_rows": pick(lambda t: (-t["n_border_touching_rows"], t["track_id"])),
                "largest_consecutive_jumps": cont["anomaly_indicators"]["suspicious_jumps_consecutive_frames"]["largest_items"][:5],
                "closest_possible_fragment_pairs": cont["anomaly_indicators"]["possible_fragment_pairs"]["items"][:5]}
    gaps_over_buffer = sum(n for k, n in hist.items() if k > 30)

    # ---- determinism ----
    det = None
    if a.rerun:
        m2 = json.loads((Path(a.rerun) / "run_manifest.json").read_text(encoding="utf-8"))
        same = {k: man["outputs"][k]["sha256"] == m2["outputs"][k]["sha256"] for k in man["outputs"] if k != "processing_log.txt"}
        det = {"rerun": a.rerun, "outputs_identical_excluding_log_and_manifest": same, "all_identical": all(same.values()),
               "manifest_differs_only_in": sorted(k for k in man if man[k] != m2.get(k)),
               "note": "processing_log.txt and run_manifest.json contain wall-clock timestamps and the output directory name, so they differ by design"}
    in_sha_after = sha256(a.input)
    summary = {
        "analysis_version": "phase4-task2-pie-analysis-1.0.0", "data": "REAL PIE set01/video_0004 Phase 3 outputs (not synthetic)",
        "run": str(run), "input": {"path": a.input, "sha256_before": in_sha_before, "sha256_after": in_sha_after, "unchanged": in_sha_before == in_sha_after},
        "runtime_and_size": {"run_wall_time_note": "measured by the caller (see report)", "outputs": man["outputs"]},
        "independent_checks": {"sidecar_schema_violations": schema_viol, "world_fields_populated_rows": world, "view_and_method": views,
                               "causal_backward_difference_recomputation": causal_recompute, "determinism": det},
        "gaps": {"n_gaps": n_gaps, "missing_frames_total": missing_total, "gaps_longer_than_bytetrack_buffer_30_frames": gaps_over_buffer,
                 "buckets": bucket_tab, "per_value": cum},
        "derivative_losses": losses,
        "footpoint_noise_estimate": {"by_class": noise, "measured_raw_acceleration_median_px_per_s2": measured_acc_median, "note": noise_note},
        "examples": examples,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"schema_violations": schema_viol, "world": world, "recompute": causal_recompute, "input_unchanged": in_sha_before == in_sha_after,
                      "determinism": None if det is None else det["all_identical"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
