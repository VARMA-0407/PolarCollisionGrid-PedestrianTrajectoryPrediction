"""Phase 4 Task 2: track-continuity diagnostics for sequence-scoped tracks (image space).

Descriptive only. Nothing here changes, merges, splits or re-identifies a track: original track ids are preserved.
Two kinds of finding are kept apart:

* measured facts: observation counts, frame/time spans, missing-frame gaps and their durations (actual timestamps),
  duplicate / out-of-order / non-monotonic rows (from the Stage 2 grouping rules), border-touching boxes;
* ANOMALY INDICATORS: ``suspicious_jump`` (footpoint displacement large relative to the box height and/or an abrupt box
  height change between consecutive observations of one track) and ``possible_fragment_pair`` (one track ends and
  another of the same class starts shortly after and nearby). These are labelled "possible ID switch" / "possible
  fragment" indicators. They are NOT confirmed ID switches: confirmation needs independent evidence (for example
  ground-truth identities), which this module does not have. ``confirmed_id_switches`` is therefore always null.

Every anomaly threshold is reported as a sensitivity table over the configured values; no single threshold is
declared correct.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .image_kinematics import CausalBatchResult, ImageKinematicsConfig

CONTINUITY_VERSION = "track-continuity-0.1.0-phase4-task2"


def _pct(values: Sequence[float]) -> Optional[Dict[str, float]]:
    v = [float(x) for x in values if x is not None and math.isfinite(x)]
    if not v:
        return None
    a = np.asarray(v)
    return {"n": int(a.size), "min": float(a.min()), "p05": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "mean": float(a.mean())}


def continuity_report(records: Sequence[Mapping[str, Any]], causal: CausalBatchResult, cfg: ImageKinematicsConfig) -> Dict[str, Any]:
    cc = cfg.continuity
    issue_counts: Dict[str, Counter] = {}
    for iss in causal.issues:
        issue_counts.setdefault(iss.track_id, Counter())[iss.code] += 1
    rows_per_track: Counter = Counter()
    seq_of: Dict[str, str] = {}
    class_of: Dict[str, Counter] = {}
    for i, c in enumerate(causal.checks):
        if c is not None and c.placeable:
            tid = records[i]["track_id"]
            rows_per_track[tid] += 1
            seq_of.setdefault(tid, records[i]["sequence_id"])
            class_of.setdefault(tid, Counter())[records[i].get("agent_class")] += 1

    per_track: List[Dict[str, Any]] = []
    gap_hist: Counter = Counter()
    gap_durations: List[float] = []
    jumps: List[Dict[str, Any]] = []
    ends: Dict[str, Dict[str, Any]] = {}
    for tid in sorted(rows_per_track, key=lambda t: (seq_of[t], t)):
        idxs = [i for i in causal.accepted.get(tid, []) if i in causal.steps]
        frames = [records[i]["frame_index"] for i in idxs]
        times = [float(records[i]["timestamp"]) for i in idxs]
        valid = [causal.checks[i].position is not None for i in idxs]
        gaps = []
        for a in range(1, len(idxs)):
            missing = (frames[a] - frames[a - 1]) // cfg.frame_step - 1
            if missing > 0:
                gaps.append({"after_frame": frames[a - 1], "before_frame": frames[a], "missing_frames": missing,
                             "elapsed_s": times[a] - times[a - 1]})
                gap_hist[missing] += 1
                gap_durations.append(times[a] - times[a - 1])
            ia, ib = idxs[a - 1], idxs[a]
            ca, cb = causal.checks[ia], causal.checks[ib]
            if ca.position is not None and cb.position is not None and ca.box_height and cb.box_height:
                d = math.hypot(cb.position[0] - ca.position[0], cb.position[1] - ca.position[1])
                href = 0.5 * (ca.box_height + cb.box_height)
                jumps.append({"track_id": tid, "from_frame": frames[a - 1], "to_frame": frames[a], "missing_frames": max(missing, 0),
                              "displacement_px": d, "displacement_over_box_height": d / href,
                              "box_height_ratio": max(ca.box_height, cb.box_height) / min(ca.box_height, cb.box_height)})
        cls = class_of[tid].most_common(1)[0][0] if class_of.get(tid) else None
        ic = issue_counts.get(tid, Counter())
        n_border = sum(1 for i in idxs if causal.checks[i].touches_border)
        entry = {"track_id": tid, "sequence_id": seq_of[tid], "agent_class": cls, "classes_seen": dict(class_of[tid]),
                 "n_input_rows": rows_per_track[tid], "n_accepted": len(idxs), "n_valid_positions": int(sum(valid)),
                 "first_frame": frames[0] if frames else None, "last_frame": frames[-1] if frames else None,
                 "first_time_s": times[0] if times else None, "last_time_s": times[-1] if times else None,
                 "duration_s": (times[-1] - times[0]) if times else None,
                 "span_frames": (frames[-1] - frames[0] + 1) if frames else None,
                 "missing_frames_total": int(sum(g["missing_frames"] for g in gaps)), "n_gaps": len(gaps),
                 "max_gap_frames": max((g["missing_frames"] for g in gaps), default=0), "gaps": gaps,
                 "n_duplicate_rows": ic.get("duplicate_observation", 0) + ic.get("duplicate_or_out_of_order", 0),
                 "n_out_of_order_rows": ic.get("out_of_order_frame", 0), "n_timestamp_not_monotonic": ic.get("timestamp_not_monotonic", 0),
                 "n_invalid_time": ic.get("invalid_time", 0), "n_off_grid": ic.get("off_grid_frame", 0),
                 "n_track_sequence_conflict": ic.get("track_sequence_conflict", 0), "n_border_touching_rows": n_border,
                 "n_suspicious_jumps_at_lowest_threshold": 0}
        per_track.append(entry)
        vi = [i for i in idxs if causal.checks[i].position is not None]
        if vi:
            first, last = vi[0], vi[-1]
            ends[tid] = {"seq": seq_of[tid], "cls": cls, "first_frame": records[first]["frame_index"], "last_frame": records[last]["frame_index"],
                         "first_xy": causal.checks[first].position, "last_xy": causal.checks[last].position,
                         "last_h": causal.checks[last].box_height}

    # ---- suspicious jumps (consecutive observations only; gap pairs reported separately) ----
    consec = [j for j in jumps if j["missing_frames"] == 0]
    lowest = cc.jump_ratio_thresholds[0]
    per_track_jumps = Counter(j["track_id"] for j in consec if j["displacement_over_box_height"] > lowest)
    for e in per_track:
        e["n_suspicious_jumps_at_lowest_threshold"] = per_track_jumps.get(e["track_id"], 0)
    jump_table = [{"displacement_over_box_height_gt": t, "n_pairs": sum(1 for j in consec if j["displacement_over_box_height"] > t),
                   "n_tracks": len({j["track_id"] for j in consec if j["displacement_over_box_height"] > t})} for t in cc.jump_ratio_thresholds]
    box_table = [{"box_height_ratio_gt": r, "n_pairs": sum(1 for j in consec if j["box_height_ratio"] > r),
                  "n_tracks": len({j["track_id"] for j in consec if j["box_height_ratio"] > r})} for r in cc.box_height_ratio_thresholds]
    both = [j for j in consec if j["displacement_over_box_height"] > cc.jump_ratio_thresholds[-1] and j["box_height_ratio"] > cc.box_height_ratio_thresholds[0]]
    top = sorted(consec, key=lambda j: -j["displacement_over_box_height"])[:cc.max_listed_items]

    # ---- possible fragment pairs (never merged) ----
    by_start: Dict[tuple, List[str]] = {}
    for tid, e in ends.items():
        by_start.setdefault((e["seq"], e["first_frame"]), []).append(tid)
    pairs = []
    wmax = cc.fragment_max_gap_frames[-1]
    for a, ea in ends.items():
        if not ea["last_h"]:
            continue
        for gapf in range(1, wmax + 1):
            for b in by_start.get((ea["seq"], ea["last_frame"] + gapf), []):
                eb = ends[b]
                if b == a or eb["cls"] != ea["cls"]:
                    continue
                d = math.hypot(eb["first_xy"][0] - ea["last_xy"][0], eb["first_xy"][1] - ea["last_xy"][1]) / ea["last_h"]
                if d <= cc.fragment_max_distance_box_heights:
                    pairs.append({"earlier_track_id": a, "later_track_id": b, "agent_class": ea["cls"], "end_frame": ea["last_frame"],
                                  "start_frame": eb["first_frame"], "frame_gap": gapf, "distance_over_box_height": d,
                                  "label": "possible_fragment_pair (not confirmed; not merged)"})
    frag_table = [{"max_gap_frames": w, "n_pairs": sum(1 for p in pairs if p["frame_gap"] <= w)} for w in cc.fragment_max_gap_frames]

    lengths = [e["n_valid_positions"] for e in per_track]
    durations = [e["duration_s"] for e in per_track if e["duration_s"] is not None]
    per_class: Dict[str, Dict[str, Any]] = {}
    for e in per_track:
        pc = per_class.setdefault(str(e["agent_class"]), {"n_tracks": 0, "n_valid_positions": 0, "n_gaps": 0, "missing_frames_total": 0, "lengths": []})
        pc["n_tracks"] += 1
        pc["n_valid_positions"] += e["n_valid_positions"]
        pc["n_gaps"] += e["n_gaps"]
        pc["missing_frames_total"] += e["missing_frames_total"]
        pc["lengths"].append(e["n_valid_positions"])
    for pc in per_class.values():
        pc["track_length_valid_positions"] = _pct(pc.pop("lengths"))
    issue_totals: Counter = Counter()
    for c in issue_counts.values():
        issue_totals.update(c)
    return {
        "continuity_version": CONTINUITY_VERSION,
        "frame_step": cfg.frame_step,
        "max_derivative_gap_frames": cfg.max_derivative_gap_frames,
        "summary": {
            "n_tracks": len(per_track), "n_sequences": len(set(seq_of.values())),
            "track_length_valid_positions": _pct(lengths), "track_duration_s": _pct(durations),
            "short_tracks": [{"valid_positions_lt": t, "n_tracks": sum(1 for n in lengths if n < t)} for t in cc.short_track_thresholds],
            "n_gaps": int(sum(gap_hist.values())), "n_tracks_with_gaps": sum(1 for e in per_track if e["n_gaps"]),
            "gap_missing_frames_histogram": {str(k): v for k, v in sorted(gap_hist.items())},
            "gap_elapsed_s": _pct(gap_durations),
            "grouping_and_stream_issues": dict(sorted(issue_totals.items())),
            "n_border_touching_rows": int(sum(e["n_border_touching_rows"] for e in per_track)),
            "per_class": dict(sorted(per_class.items())),
        },
        "anomaly_indicators": {
            "note": "Indicators only. A suspicious jump or a fragment pair is a POSSIBLE ID switch / fragmentation, never a confirmed one; "
                    "no track was merged, split or renamed.",
            "suspicious_jumps_consecutive_frames": {"displacement_table": jump_table, "box_height_change_table": box_table,
                                                    "n_pairs_evaluated": len(consec),
                                                    "possible_id_switch_indicators": {"rule": "displacement/box_height > %r AND box_height_ratio > %r"
                                                                                      % (cc.jump_ratio_thresholds[-1], cc.box_height_ratio_thresholds[0]),
                                                                                      "n_pairs": len(both), "items": both[:cc.max_listed_items]},
                                                    "largest_items": top},
            "displacement_across_gaps": _pct([j["displacement_over_box_height"] for j in jumps if j["missing_frames"] > 0]),
            "possible_fragment_pairs": {"max_distance_box_heights": cc.fragment_max_distance_box_heights, "table": frag_table,
                                        "items": sorted(pairs, key=lambda p: (p["frame_gap"], p["distance_over_box_height"]))[:cc.max_listed_items]},
            "confirmed_id_switches": None,
            "confirmed_id_switches_note": "not determinable: no independent identity evidence was supplied",
        },
        "per_track": per_track,
    }


__all__ = ["CONTINUITY_VERSION", "continuity_report"]
