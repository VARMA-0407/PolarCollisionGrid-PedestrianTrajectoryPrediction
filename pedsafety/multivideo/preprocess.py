"""Phase 4 Task 2 (image kinematics), Phase 4 Task 3 (interaction representation) and Phase 5 P5.3 (window generation) per video.

Every sub-stage calls the existing public API with the repository configuration files, exactly as the Phase 9 orchestrator does
(``run_image_kinematics``, ``run_interaction_representation``, ``generate_windows``), and applies the same Phase 9 contract checks
(Phase 3 row contract, Task 2 row-count / quarantine check, Task 3 node count, history-only windows, sha256 provenance chain).

The Task 3 and P5.3 runners re-verify upstream files at the paths recorded in the upstream manifests, so the three sub-stages of a video
run inside one deterministic local scratch folder ``<work_root>/preprocess/<video>/{phase3,S4A,S4B,S5A}``. Each finished sub-stage is
published to Drive with SHA-256 verification and its own completion marker; on resume, finished sub-stages are restored into the same
scratch paths (keep WORK_ROOT unchanged between sessions) and only the missing ones are computed. Windows are labelled with
``video_id = <set>_<video>`` and ``group_id = <set>`` (the recording group of the frozen split). No gap filling, interpolation or
forward-fill is added (the repository configurations keep these disabled).
"""

import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .common import StageError, canonical_sha256, check_marker, copy_dir_verified, publish_dir, read_json, sha256_file, utc_now, write_marker

SUBSTAGES = ("S4A_IMAGE_KINEMATICS", "S4B_INTERACTION", "S5A_WINDOWS")
DIRNAMES = {"S4A_IMAGE_KINEMATICS": "phase4_task2_kinematics", "S4B_INTERACTION": "phase4_task3_interaction", "S5A_WINDOWS": "phase5_windows"}


def preprocess_config(repo, configs: Dict[str, str], phase3_config_sha256: str) -> Dict[str, Any]:
    repo = Path(repo)
    h = {k: sha256_file(repo / configs[k]) for k in ("image_kinematics", "interaction", "windows")}
    return {"configs": dict(configs), "config_file_sha256": h, "phase3_config_sha256": phase3_config_sha256,
            "labels": {"video_id": "<set>_<video>", "group_id": "<set>"}}


def _sha(cfg: Dict[str, Any], sub: str) -> str:
    return canonical_sha256({"substage": sub, **cfg})


def preprocess_video(repo, key: str, set_id: str, phase3_dir, artifacts_root, work_root, quarantine_root, pcfg: Dict[str, Any], *,
                     git_commit: Optional[str] = None, keep_scratch: bool = False, until: str = "S5A_WINDOWS", verify_only: bool = False) -> Dict[str, Any]:
    """Run the sub-stages up to and including ``until``. ``verify_only``: never compute; report which sub-stages are COMPLETE."""
    if until not in SUBSTAGES:
        raise StageError("UNTIL", "until must be one of %s" % (SUBSTAGES,))
    wanted = SUBSTAGES[:SUBSTAGES.index(until) + 1]
    from ..image_kinematics import ImageKinematicsConfig
    from ..image_kinematics_run import run_image_kinematics
    from ..interaction.config import InteractionConfig
    from ..interaction.runner import run_interaction_representation
    from ..phase5.runner import generate_windows
    from ..phase5.window_config import WindowConfig
    from ..phase5.window_storage import read_dataset
    from ..phase9.contracts import validate_phase3_file, validate_windows_history_only, verify_chain, verify_outputs

    repo, phase3_dir = Path(repo), Path(phase3_dir)
    finals = {s: Path(artifacts_root) / DIRNAMES[s] / set_id / key for s in wanted}
    states = {s: check_marker(finals[s], stage=s, config_sha256=_sha(pcfg, s)) for s in wanted}
    for s, (st, _) in states.items():
        if st == "STALE":
            raise StageError("STALE_ARTIFACT", "%s holds %s outputs of a different configuration; use a new experiment" % (finals[s], s))
    if all(st == "COMPLETE" for st, _ in states.values()):
        return {"key": key, "status": "SKIPPED_ALREADY_COMPLETE", "dirs": {s: str(d) for s, d in finals.items()},
                "detail": {s: m.get("detail", {}) for s, (_, m) in states.items()}}
    if verify_only:
        return {"key": key, "status": "MISSING_VERIFY_ONLY", "marker_states": {s: st for s, (st, _) in states.items()}}
    work = Path(work_root) / "preprocess" / key
    loc = {"phase3": work / "phase3", **{s: work / s for s in SUBSTAGES}}
    work.mkdir(parents=True, exist_ok=True)
    if loc["phase3"].exists():
        shutil.rmtree(loc["phase3"])
    loc["phase3"].mkdir()
    for f in ("agent_states.jsonl", "tracked_agents.jsonl", "run_manifest.json"):
        if not (phase3_dir / f).is_file():
            raise StageError("S3_MISSING", "%s missing in %s" % (f, phase3_dir))
        shutil.copyfile(phase3_dir / f, loc["phase3"] / f)
    states3, tracked, p3man = loc["phase3"] / "agent_states.jsonl", loc["phase3"] / "tracked_agents.jsonl", loc["phase3"] / "run_manifest.json"
    wcfg = WindowConfig.load(repo / pcfg["configs"]["windows"])
    fps = wcfg.source_fps
    detail: Dict[str, Any] = {}
    timing: Dict[str, float] = {}
    p3stats = validate_phase3_file(tracked, fps)
    n_rows = sum(1 for line in open(states3, encoding="utf-8") if line.strip())
    up = {"phase3_agent_states_sha256": sha256_file(states3), "phase3_tracked_agents_sha256": sha256_file(tracked), "phase3_manifest_sha256": sha256_file(p3man)}

    def done_or_restore(s) -> bool:
        st, m = states[s]
        if st == "COMPLETE":
            local_ok = loc[s].is_dir() and all((loc[s] / rel).is_file() and (loc[s] / rel).stat().st_size == r["bytes"] and sha256_file(loc[s] / rel) == r["sha256"]
                                               for rel, r in m["files"].items())
            if not local_ok:
                copy_dir_verified(finals[s], loc[s])
            detail[s] = dict(m.get("detail", {}), restored_from_drive=not local_ok, reused_local_scratch=local_ok)
            return True
        if loc[s].exists():
            shutil.rmtree(loc[s])
        return False

    def publish(s, det):
        files = publish_dir(loc[s], finals[s], quarantine_root)
        write_marker(finals[s], stage=s, key=key, config_sha256=_sha(pcfg, s), upstream=up, detail=det, files=files, git_commit=git_commit)
        detail[s] = det

    # ---- Phase 4 Task 2
    s = "S4A_IMAGE_KINEMATICS"
    if not done_or_restore(s):
        t = time.perf_counter()
        c = repo / pcfg["configs"]["image_kinematics"]
        m = run_image_kinematics(states3, ImageKinematicsConfig.load(c), loc[s], config_path=str(c), run_manifest_path=p3man, tracked_agents_path=tracked,
                                 command="multivideo S4A %s" % key)
        verify_outputs(loc[s], m, ["causal_kinematics.jsonl"])
        if m["counts"]["quarantined"] or m["counts"]["causal_rows"] != p3stats["rows"]:
            raise StageError("S4A_ROWS", "Task 2 quarantined rows or changed the row count (%s vs %d phase 3 rows)" % (m["counts"], p3stats["rows"]))
        timing[s] = time.perf_counter() - t
        publish(s, {"counts": m["counts"], "wall_s": timing[s], "phase3_contract": p3stats, "agent_state_rows": n_rows})
    # ---- Phase 4 Task 3
    s = "S4B_INTERACTION"
    if s in wanted and not done_or_restore(s):
        t = time.perf_counter()
        c = repo / pcfg["configs"]["interaction"]
        t2 = loc["S4A_IMAGE_KINEMATICS"]
        m = run_interaction_representation(t2 / "causal_kinematics.jsonl", t2 / "run_manifest.json", states3, InteractionConfig.load(c), loc[s], config_path=str(c),
                                           command="multivideo S4B %s" % key)
        t2m = read_json(t2 / "run_manifest.json")
        if m["counts"]["nodes"] != t2m["counts"]["causal_rows"]:
            raise StageError("S4B_NODES", "node count %s differs from the causal row count %s" % (m["counts"]["nodes"], t2m["counts"]["causal_rows"]))
        timing[s] = time.perf_counter() - t
        publish(s, {"counts": {k: m["counts"][k] for k in ("nodes", "tracks") if k in m["counts"]}, "wall_s": timing[s]})
    # ---- Phase 5 P5.3 windows
    s = "S5A_WINDOWS"
    if s in wanted and not done_or_restore(s):
        t = time.perf_counter()
        c = repo / pcfg["configs"]["windows"]
        t2, t3 = loc["S4A_IMAGE_KINEMATICS"], loc["S4B_INTERACTION"]
        m = generate_windows(t3, wcfg, loc[s], config_path=str(c), command="multivideo S5A %s" % key, phase3_manifest=p3man, tracked_agents=tracked,
                             video_id=key, group_id=set_id)
        verify_outputs(loc[s], m, ["windows.npz", "graph_bank.npz"])
        ws, _, extra = read_dataset(loc[s])
        hist = validate_windows_history_only(ws, fps) if ws.n_windows else {"history_only": True, "note": "no accepted windows"}
        chain = verify_chain(phase3_agent_states=states3, task2_dir=t2, task3_dir=t3, windows_dir=loc[s], t2=read_json(t2 / "run_manifest.json"),
                             t3=read_json(t3 / "run_manifest.json"), w=m)
        ids = sorted(set(map(str, extra["video_id"].tolist()))), sorted(set(map(str, extra["group_id"].tolist())))
        if ws.n_windows and (ids[0] != [key] or ids[1] != [set_id]):
            raise StageError("S5A_IDENTITY", "windows carry video/group ids %s, expected %s / %s" % (ids, key, set_id))
        timing[s] = time.perf_counter() - t
        publish(s, {"accepted_windows": m["counts"]["accepted_windows"], "rejected_windows": m["counts"]["rejected_windows"],
                    "rejection_reasons": m["counts"]["rejection_reasons"], "history_only": hist, "chain": chain, "wall_s": timing[s],
                    "window_label_note": "the P5.3 generator labels every run %s; the experiment partition comes from the frozen split" % m.get("label")})
    if not keep_scratch:
        shutil.rmtree(work, ignore_errors=True)
    return {"key": key, "status": "COMPLETED", "dirs": {s: str(d) for s, d in finals.items()}, "detail": detail, "computed_substages": sorted(timing),
            "wall_s": timing}


def preprocess_batch(repo, items: List[Dict[str, Any]], *, progress: Callable[[str], None] = print, **kw) -> Dict[str, Any]:
    """``items``: [{"key", "set", "phase3_dir"}]. Failures are recorded per video; other videos continue."""
    results: Dict[str, Any] = {}
    for i, it in enumerate(items, 1):
        progress("[%s] Phase 4/5A %d/%d %s" % (utc_now(), i, len(items), it["key"]))
        try:
            r = preprocess_video(repo, it["key"], it["set"], it["phase3_dir"], **kw)
            w = r["detail"].get("S5A_WINDOWS", {})
            progress("    -> %s computed=%s windows=%s" % (r["status"], r.get("computed_substages", []), w.get("accepted_windows")))
        except Exception as exc:                                                                      # noqa: BLE001 - recorded per video
            r = {"key": it["key"], "status": "FAILED", "error_code": getattr(exc, "code", type(exc).__name__), "error": str(exc)[:2000]}
            progress("    -> FAILED %s" % r["error"][:500])
        results[it["key"]] = r
    failed = sorted(k for k, r in results.items() if r["status"] == "FAILED")
    missing = sorted(k for k, r in results.items() if r["status"] == "MISSING_VERIFY_ONLY")
    by: Dict[str, int] = {}
    for r in results.values():
        by[r["status"]] = by.get(r["status"], 0) + 1
    return {"n_videos": len(items), "by_status": by, "failed": failed, "missing": missing, "results": results, "all_complete": not failed and not missing}


__all__ = ["preprocess_video", "preprocess_batch", "preprocess_config", "SUBSTAGES", "DIRNAMES"]
