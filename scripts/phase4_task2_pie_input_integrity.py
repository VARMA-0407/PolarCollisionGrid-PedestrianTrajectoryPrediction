"""Phase 4 Task 2: read-only integrity check of the Phase 3 PIE pilot inputs before kinematic processing.

The Phase 3 run_manifest.json records no per-file hashes, so integrity is checked by (a) recording SHA-256 of each
input and (b) cross-checking every count the manifest does record against the files themselves, plus a row-by-row
correspondence between tracked_agents.jsonl and agent_states.jsonl and the Phase 4 coordinate contract.
Writes one JSON report (refuses to overwrite). Opens inputs read-only.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rows(p):
    with open(p, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                yield n, json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", required=True, help="path of the JSON report to create (must not exist)")
    ap.add_argument("--expected-video", default="video_0004.mp4")
    ap.add_argument("--expected-raster", default="captured_1920x1080")
    a = ap.parse_args(argv)
    d = Path(a.input_dir)
    out = Path(a.out)
    if out.exists():
        print("refusing to overwrite %s" % out, file=sys.stderr)
        return 2
    files = {n: d / n for n in ("agent_states.jsonl", "tracked_agents.jsonl", "run_manifest.json")}
    missing = [n for n, p in files.items() if not p.is_file()]
    if missing:
        print("missing input files: %s" % missing, file=sys.stderr)
        return 2
    hashes = {n: {"sha256": sha256(p), "bytes": p.stat().st_size} for n, p in files.items()}
    man = json.loads(files["run_manifest.json"].read_text(encoding="utf-8"))
    res, fs = man.get("results", {}), man.get("components", {}).get("frame_source", {}).get("parameters", {})
    checks = []

    def check(name, expected, actual):
        checks.append({"check": name, "expected": expected, "actual": actual, "ok": expected == actual})

    check("manifest.video", a.expected_video, man.get("video"))
    check("manifest.is_mock", False, man.get("is_mock"))
    check("manifest.backend", "yolo", man.get("backend"))
    check("manifest.frame_source.width/height/fps", ["1920", "1080", "30.0"], [fs.get("width"), fs.get("height"), fs.get("fps")])
    check("manifest.frames_processed", 9840, res.get("frames_processed"))
    check("manifest.world_coordinates", "none: outputs are image-space pixels only", man.get("world_coordinates"))

    # agent_states
    a_cls, a_rast, a_tracks, a_seq, a_scene, a_src = Counter(), Counter(), set(), Counter(), Counter(), Counter()
    a_frames, a_world_bad, a_kin_bad, a_fp_bad, a_unit_bad, a_obs_bad, a_interp_bad, a_nonfinite_t = [], 0, 0, 0, 0, 0, 0, 0
    a_keys = Counter()
    akey = []
    for n, r in rows(files["agent_states.jsonl"]):
        a_cls[r.get("agent_class")] += 1
        a_rast[r.get("image_coordinate_system")] += 1
        a_tracks.add(r.get("track_id"))
        a_seq[r.get("sequence_id")] += 1
        a_scene[r.get("scene_id")] += 1
        a_src[r.get("source_dataset")] += 1
        a_frames.append(r.get("frame_index"))
        a_keys[tuple(sorted(r))] += 1
        if r.get("world_x") is not None or r.get("world_y") is not None or r.get("world_position_valid") is not False:
            a_world_bad += 1
        if any(r.get(k) is not None for k in ("velocity_x", "velocity_y", "speed", "acceleration_x", "acceleration_y", "heading")):
            a_kin_bad += 1
        b, fp = r.get("bbox_xyxy"), r.get("footpoint")
        if not (b and fp and abs(fp[0] - (b[0] + b[2]) / 2) <= 1e-6 and abs(fp[1] - b[3]) <= 1e-6 and r.get("image_x") == fp[0] and r.get("image_y") == fp[1]):
            a_fp_bad += 1
        a_unit_bad += r.get("image_coordinate_unit") != "pixel"
        a_obs_bad += r.get("observation_valid") is not True
        a_interp_bad += r.get("interpolation_flag") is not False
        t = r.get("timestamp")
        a_nonfinite_t += not (isinstance(t, (int, float)) and math.isfinite(t))
        akey.append((r.get("frame_index"), str(r.get("track_id")).rsplit(":", 1)[-1], [round(v, 6) for v in b] if b else None, r.get("timestamp")))
    # tracked_agents
    t_cls, t_rast, t_tracks, t_frames, tkey = Counter(), Counter(), set(), [], []
    for n, r in rows(files["tracked_agents.jsonl"]):
        t_cls[r.get("semantic_class")] += 1
        t_rast[r.get("coordinate_system")] += 1
        t_tracks.add(r.get("track_id"))
        t_frames.append(r.get("frame_index"))
        tkey.append((r.get("frame_index"), str(r.get("track_id")), [round(v, 6) for v in r["bbox_xyxy"]], r.get("timestamp_s")))
    n_a, n_t = len(akey), len(tkey)
    check("agent_states rows == manifest.agent_states.records", man.get("agent_states", {}).get("records"), n_a)
    check("tracked_agents rows == manifest.results.track_rows", res.get("track_rows"), n_t)
    check("tracked_agents unique track ids == manifest.results.unique_track_ids", res.get("unique_track_ids"), len(t_tracks))
    check("agent_states unique track ids == tracked_agents unique track ids", len(t_tracks), len(a_tracks))
    check("tracked_agents rows by class == manifest", res.get("track_rows_by_semantic_class"), dict(sorted(t_cls.items())))
    # manifest first/last_frame_index are the first/last frames PROCESSED (pedsafety/perception/runner.py), not the first/last
    # frames that contain a track row; track rows must lie inside that range.
    lo, hi = res.get("first_frame_index"), res.get("last_frame_index")
    check("track-row frames lie inside the processed frame range", True, bool(lo is not None and hi is not None and min(t_frames) >= lo and max(t_frames) <= hi))
    check("agent_states frames == tracked_agents frames (min, max)", [min(t_frames), max(t_frames)], [min(a_frames), max(a_frames)])
    check("agent_states source_dataset == manifest", {man.get("agent_states", {}).get("source_dataset"): n_a}, dict(a_src))
    mism = sum(1 for x, y in zip(akey, tkey) if x != y) if n_a == n_t else None
    check("row-by-row correspondence (frame, track number, bbox, timestamp)", 0, mism)
    check("agent_states raster", {a.expected_raster: n_a}, dict(a_rast))
    check("tracked_agents raster", {a.expected_raster: n_t}, dict(t_rast))
    check("agent_states world fields null / world_position_valid false (violations)", 0, a_world_bad)
    check("agent_states kinematic fields null (violations)", 0, a_kin_bad)
    check("footpoint == bbox bottom-centre == image_x/y (violations)", 0, a_fp_bad)
    check("image_coordinate_unit == pixel (violations)", 0, a_unit_bad)
    check("observation_valid true (violations)", 0, a_obs_bad)
    check("interpolation_flag false (violations)", 0, a_interp_bad)
    check("non-finite timestamps", 0, a_nonfinite_t)
    report = {
        "check_version": "phase4-task2-pie-input-integrity-1.0.0",
        "input_dir": str(d), "files": hashes,
        "manifest_has_per_file_hashes": False,
        "manifest_has_per_file_hashes_note": "the Phase 3 run_manifest.json records no file hashes; the SHA-256 values above become the reference for this and later runs",
        "agent_states": {"rows": n_a, "unique_tracks": len(a_tracks), "classes": dict(sorted(a_cls.items(), key=lambda kv: str(kv[0]))),
                         "sequence_ids": dict(a_seq), "scene_ids": dict(a_scene), "key_sets": len(a_keys),
                         "track_id_example": sorted(a_tracks, key=str)[0]},
        "tracked_agents": {"rows": n_t, "unique_tracks": len(t_tracks), "classes": dict(sorted(t_cls.items())),
                           "first_frame_with_track": min(t_frames), "last_frame_with_track": max(t_frames),
                           "frames_with_at_least_one_track": len(set(t_frames))},
        "manifest_excerpt": {"video": man.get("video"), "created_utc": man.get("created_utc"), "is_mock": man.get("is_mock"),
                             "detector_weights_sha256": man.get("components", {}).get("detector", {}).get("parameters", {}).get("weights_sha256"),
                             "tracker_class_flips": man.get("tracker_class_flips"), "issues_by_code": res.get("issues_by_code"),
                             "frame_source": fs},
        "checks": checks, "all_ok": all(c["ok"] for c in checks),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_ok": report["all_ok"], "failed": [c for c in checks if not c["ok"]]}, indent=1))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
