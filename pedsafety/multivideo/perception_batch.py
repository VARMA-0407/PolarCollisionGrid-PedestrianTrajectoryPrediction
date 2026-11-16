"""Phase 3 perception for every PIE video with video-level resume.

Each video is processed by the EXISTING CLI ``scripts/run_perception.py`` (which calls ``pedsafety.perception.runner.execute_run``)
in its own subprocess, so all per-video memory (the agent-state adapter keeps every frame result until the end of a video) is released
after each video. Per video:

1. copy the MP4 from Drive to local scratch and check its size (and SHA-256 when the inventory hashed it);
2. write a per-video perception configuration = the repository configuration with ``video.sequence_id = <set>_<video>`` (PIE reuses
   ``video_0001`` in several sets, so the file stem alone is not a unique identity) and the verified weights path;
3. run the CLI (YOLO + ByteTrack, GPU when available), then validate ``tracked_agents.jsonl`` with the Phase 9 Phase-3 contract;
4. optionally evaluate detection / tracking against the PIE ``*_annt.xml`` with the existing Phase 3 Task 4 evaluator;
5. gzip ``detections.jsonl`` (no later stage reads it; the uncompressed SHA-256 is recorded) and publish to Drive with verification.

A video whose Drive folder holds a valid completion marker for the same configuration is skipped (resume).
"""

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .common import StageError, canonical_sha256, check_marker, move_aside, publish_dir, sha256_file, utc_now, write_json, write_marker

STAGE = "S3_PERCEPTION"
EVAL_STAGE = "S3_PERCEPTION_PIE_EVALUATION"


def perception_stage_config(base_config_path, *, weights_sha256: str, device: str, max_frames: Optional[int], save_video: bool, write_detections: bool,
                            compress_detections: bool) -> Dict[str, Any]:
    base = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    return {"stage": STAGE, "base_config_sha256": canonical_sha256(base), "weights_sha256": weights_sha256, "max_frames": max_frames, "save_video": save_video,
            "write_detections": write_detections, "compress_detections": compress_detections, "sequence_id_rule": "<set>_<video>", "scene_id_rule": "pie_<set>_<video>",
            "source_dataset": "PIE", "device_requested": device}


def _config_sha(stage_cfg: Dict[str, Any]) -> str:
    return canonical_sha256({k: v for k, v in stage_cfg.items() if k != "device_requested"})          # the device does not change the definition of the outputs


def _gzip(src: Path, dst: Path) -> Dict[str, Any]:
    h = hashlib.sha256()
    with open(src, "rb") as fin, open(dst, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fout:
        for block in iter(lambda: fin.read(1 << 20), b""):
            h.update(block)
            fout.write(block)
    return {"uncompressed_sha256": h.hexdigest(), "uncompressed_bytes": src.stat().st_size, "compressed_file": dst.name, "compressed_sha256": sha256_file(dst)}


def run_perception_video(repo, video: Dict[str, Any], *, base_config_path, weights_path, weights_sha256: str, device: str, phase3_root, eval_root, work_root,
                         quarantine_root, stage_cfg: Dict[str, Any], run_pie_evaluation: bool = True, source_fps: float = 30.0, git_commit: Optional[str] = None,
                         timeout_s: Optional[float] = None, verify_only: bool = False) -> Dict[str, Any]:
    repo = Path(repo)
    key, set_id = video["key"], video["set"]
    final = Path(phase3_root) / set_id / key
    cfg_sha = _config_sha(stage_cfg)
    state, marker = check_marker(final, stage=STAGE, config_sha256=cfg_sha)
    if state == "COMPLETE":
        return {"key": key, "status": "SKIPPED_ALREADY_COMPLETE", "dir": str(final), "detail": marker.get("detail", {})}
    if state == "STALE":
        raise StageError("STALE_ARTIFACT", "%s holds Phase 3 outputs of a different configuration; use a new experiment" % final)
    if verify_only:
        return {"key": key, "status": "MISSING_VERIFY_ONLY", "marker_state": state}
    work = Path(work_root) / "phase3" / key
    if work.exists():
        shutil.rmtree(work)
    (work / "input").mkdir(parents=True)
    t0 = time.perf_counter()
    local_video = work / "input" / (key + ".mp4")
    shutil.copyfile(video["path"], local_video)
    if local_video.stat().st_size != video["bytes"]:
        raise StageError("VIDEO_COPY_SIZE", "local copy of %s has %d bytes, expected %d" % (video["relpath"], local_video.stat().st_size, video["bytes"]))
    if video.get("sha256") and sha256_file(local_video) != video["sha256"]:
        raise StageError("VIDEO_COPY_HASH", "local copy of %s differs from the inventoried SHA-256" % video["relpath"])
    t_copy = time.perf_counter() - t0
    cfg = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    cfg["video"]["sequence_id"] = key
    if stage_cfg.get("max_frames"):
        cfg["video"]["max_frames"] = int(stage_cfg["max_frames"])
    cfg_path = work / "perception_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    out_root = work / "run"
    cmd = [sys.executable, "-W", "ignore", str(repo / "scripts" / "run_perception.py"), "--config", str(cfg_path), "--video", str(local_video),
           "--output-root", str(out_root), "--run-name", key, "--model", str(weights_path), "--device", device, "--scene-id", "pie_%s" % key, "--source-dataset", "PIE"]
    if stage_cfg.get("save_video"):
        cmd.append("--save-video")
    if not stage_cfg.get("write_detections", True):
        cmd.append("--no-detections-jsonl")
    t1 = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, timeout=timeout_s)
    t_run = time.perf_counter() - t1
    log = "COMMAND: %s\nEXIT: %d\n--- stdout ---\n%s\n--- stderr ---\n%s\n" % (" ".join(cmd), p.returncode, p.stdout, p.stderr[-20000:])
    runs = sorted(d for d in out_root.iterdir() if d.is_dir()) if out_root.is_dir() else []
    if p.returncode != 0 or len(runs) != 1:
        (work / "perception_failure.log").write_text(log, encoding="utf-8")
        raise StageError("PERCEPTION_EXIT", "run_perception.py exit %d for %s (%d run dirs): %s" % (p.returncode, key, len(runs), (p.stderr or p.stdout)[-1500:]))
    run = runs[0]
    (run / "perception_stdout.log").write_text(log, encoding="utf-8")
    shutil.copyfile(cfg_path, run / "perception_config.json")
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    for f in ("tracked_agents.jsonl", "agent_states.jsonl", "run_manifest.json"):
        if not (run / f).is_file():
            raise StageError("PERCEPTION_OUTPUT_MISSING", "%s missing for %s" % (f, key))
    if man.get("is_mock") or man.get("backend") != "yolo":
        raise StageError("PERCEPTION_MOCK", "perception run for %s is not a real YOLO run (backend %s, is_mock %s)" % (key, man.get("backend"), man.get("is_mock")))
    from ..phase9.contracts import validate_phase3_file
    contract = validate_phase3_file(run / "tracked_agents.jsonl", source_fps)
    res = man["results"]
    detail = {"frames_seen": res["frames_seen"], "frames_processed": res["frames_processed"], "frames_skipped": res["frames_skipped"], "detections": res["detections"],
              "track_rows": res["track_rows"], "unique_track_ids": res["unique_track_ids"], "issues_by_code": res["issues_by_code"], "wall_time_s_perception_run": man["wall_time_s"],
              "frames_per_s_including_decode_and_io": res["frames_processed"] / man["wall_time_s"] if man["wall_time_s"] else None,
              "detect_mean_ms": man.get("stage_latency_ms_in_pipeline_run", {}).get("detect", {}).get("mean_ms"),
              "track_mean_ms": man.get("stage_latency_ms_in_pipeline_run", {}).get("track", {}).get("mean_ms"),
              "gpu": (man.get("environment", {}).get("cuda") or {}).get("gpus"), "video_copy_s": t_copy, "subprocess_wall_s": t_run, "phase3_contract": contract,
              "sequence_id": key, "set": set_id, "video": video["video"], "video_sha256": video.get("sha256"), "weights_sha256": weights_sha256,
              "max_frames": stage_cfg.get("max_frames"), "partial_video": bool(stage_cfg.get("max_frames")),
              "timing_note": "offline batch processing of a recorded video on Colab; not an end-to-end or real-time latency measurement"}
    # ---- optional PIE annotation evaluation (existing Phase 3 Task 4 evaluator), needs the uncompressed detections
    ann = (video.get("annotations") or {}).get("annt")
    eval_rec = {"status": "NOT_REQUESTED"}
    if run_pie_evaluation:
        if not ann:
            eval_rec = {"status": "BLOCKED_NO_ANNOTATION_XML"}
        elif not (run / "detections.jsonl").is_file():
            eval_rec = {"status": "BLOCKED_NO_DETECTIONS_JSONL"}
        else:
            from ..perception.pie_perception_evaluator import run_evaluation
            ev_out = work / "pie_eval"
            try:
                r = run_evaluation(ann, run / "detections.jsonl", run / "tracked_agents.jsonl", ev_out, video_id=key, run_manifest=run / "run_manifest.json")
                eval_rec = {"status": "COMPLETED", "annotation_xml": ann, "annotation_sha256": sha256_file(ann),
                            "summary_keys": sorted(r) if isinstance(r, dict) else None}
                ev_final = Path(eval_root) / set_id / key
                files = publish_dir(ev_out, ev_final, quarantine_root)
                write_marker(ev_final, stage=EVAL_STAGE, key=key, config_sha256=cfg_sha, files=files, detail=eval_rec, git_commit=git_commit)
                eval_rec["dir"] = str(ev_final)
            except Exception as exc:                                                                  # noqa: BLE001 - recorded per video
                eval_rec = {"status": "FAILED", "error": "%s: %s" % (type(exc).__name__, str(exc)[:800])}
    detail["pie_annotation_evaluation"] = eval_rec
    if stage_cfg.get("compress_detections", True) and (run / "detections.jsonl").is_file():
        detail["detections_compression"] = _gzip(run / "detections.jsonl", run / "detections.jsonl.gz")
        (run / "detections.jsonl").unlink()
    write_json(run / "phase3_publication.json", {"created_utc": utc_now(), "key": key, "set": set_id, "source_video": video["relpath"], "detail": detail,
                                                 "note": "detections.jsonl is stored gzip-compressed when detections_compression is present; no later stage reads it"})
    files = publish_dir(run, final, quarantine_root)
    write_marker(final, stage=STAGE, key=key, config_sha256=cfg_sha, files=files, detail=detail, upstream={"video_sha256": video.get("sha256"), "video_bytes": video["bytes"]},
                 git_commit=git_commit)
    shutil.rmtree(work, ignore_errors=True)
    return {"key": key, "status": "COMPLETED", "dir": str(final), "detail": detail, "total_wall_s": time.perf_counter() - t0}


def find_reusable_phase3(video: Dict[str, Any], reuse_roots, stage_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A COMPLETE Phase 3 folder of the same configuration in another experiment (explicitly configured by the user), verified by SHA-256."""
    for root in reuse_roots or ():
        d = Path(root) / video["set"] / video["key"]
        state, marker = check_marker(d, stage=STAGE, config_sha256=_config_sha(stage_cfg), verify="sha256")
        if state == "COMPLETE" and marker.get("upstream", {}).get("video_sha256") in (None, video.get("sha256")):
            return {"key": video["key"], "status": "REUSED_VERIFIED", "dir": str(d), "detail": dict(marker.get("detail", {}), reused_from=str(d))}
    return None


def run_perception_batch(repo, videos: List[Dict[str, Any]], *, progress: Callable[[str], None] = print, stop_on_error: bool = False, reuse_roots=(),
                         **kw) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for i, v in enumerate(videos, 1):
        progress("[%s] Phase 3 %d/%d %s (%.2f GB)" % (utc_now(), i, len(videos), v["key"], v["bytes"] / 1e9))
        try:
            r = find_reusable_phase3(v, reuse_roots, kw["stage_cfg"]) or run_perception_video(repo, v, **kw)
        except Exception as exc:                                                                      # noqa: BLE001 - recorded per video
            r = {"key": v["key"], "status": "FAILED", "error_code": getattr(exc, "code", type(exc).__name__), "error": str(exc)[:2000]}
        d = r.get("detail", {})
        if r["status"] == "FAILED":
            progress("    -> FAILED %s" % r["error"][:500])
        else:
            fps = d.get("frames_per_s_including_decode_and_io")
            progress("    -> %s frames=%s track_rows=%s offline_fps=%s" % (r["status"], d.get("frames_processed"), d.get("track_rows"), "%.1f" % fps if fps else "n/a"))
        results[v["key"]] = r
        if r["status"] == "FAILED" and stop_on_error:
            break
    return summarize_batch(results, len(videos))


OK_VIDEO_STATES = ("COMPLETED", "SKIPPED_ALREADY_COMPLETE", "REUSED_VERIFIED", "PRECOMPUTED_VALIDATED")


def summarize_batch(results: Dict[str, Any], n_expected: int) -> Dict[str, Any]:
    by: Dict[str, int] = {}
    for r in results.values():
        by[r["status"]] = by.get(r["status"], 0) + 1
    failed = sorted(k for k, r in results.items() if r["status"] == "FAILED")
    missing = sorted(k for k, r in results.items() if r["status"] not in OK_VIDEO_STATES and r["status"] != "FAILED")
    return {"n_videos": n_expected, "n_attempted": len(results), "by_status": by, "failed": failed, "missing": missing, "results": results,
            "all_complete": not failed and not missing and len(results) == n_expected}


def register_precomputed_phase3(videos: List[Dict[str, Any]], precomputed_root, *, source_fps: float = 30.0, progress: Callable[[str], None] = print) -> Dict[str, Any]:
    """PRECOMPUTED-ARTIFACT mode: validate existing Phase 3 folders (``<root>/<set>/<set>_<video>/`` or ``<root>/<set>_<video>/``) without running
    perception. Nothing is copied; the folder and the SHA-256 of its three inputs are recorded, and the execution mode of these videos is
    PRECOMPUTED_PHASE3_ARTIFACTS (never raw-video end-to-end)."""
    from ..phase9.contracts import validate_phase3_file
    results: Dict[str, Any] = {}
    for v in videos:
        cands = [Path(precomputed_root) / v["set"] / v["key"], Path(precomputed_root) / v["key"]]
        d = next((c for c in cands if (c / "tracked_agents.jsonl").is_file()), None)
        if d is None:
            results[v["key"]] = {"key": v["key"], "status": "MISSING_PRECOMPUTED", "searched": [str(c) for c in cands]}
            progress("  %s: no precomputed Phase 3 folder" % v["key"])
            continue
        try:
            for f in ("agent_states.jsonl", "run_manifest.json"):
                if not (d / f).is_file():
                    raise StageError("S3_MISSING", "%s missing in %s" % (f, d))
            man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
            if man.get("is_mock") is not False:
                raise StageError("PERCEPTION_MOCK", "%s is a mock perception run" % d)
            contract = validate_phase3_file(d / "tracked_agents.jsonl", source_fps)
            results[v["key"]] = {"key": v["key"], "status": "PRECOMPUTED_VALIDATED", "dir": str(d), "detail": {
                "execution_mode": "PRECOMPUTED_PHASE3_ARTIFACTS", "phase3_contract": contract, "frames_processed": man.get("results", {}).get("frames_processed"),
                "track_rows": man.get("results", {}).get("track_rows"), "wall_time_s_perception_run": None, "recorded_wall_time_s": man.get("wall_time_s"),
                "video_id_in_rows": contract.get("video_id"), "sha256": {f: sha256_file(d / f) for f in ("agent_states.jsonl", "tracked_agents.jsonl", "run_manifest.json")}}}
            progress("  %s: precomputed Phase 3 validated (%s rows)" % (v["key"], contract["rows"]))
        except Exception as exc:                                                                      # noqa: BLE001 - recorded per video
            results[v["key"]] = {"key": v["key"], "status": "FAILED", "error_code": getattr(exc, "code", type(exc).__name__), "error": str(exc)[:2000]}
    return summarize_batch(results, len(videos))


__all__ = ["run_perception_video", "run_perception_batch", "perception_stage_config", "find_reusable_phase3", "register_precomputed_phase3", "summarize_batch",
           "STAGE", "OK_VIDEO_STATES"]
