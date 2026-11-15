"""Orchestrate P5.3: verify inputs -> build windows -> build graph bank -> validate -> write dataset and manifest."""

import datetime
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..interaction.storage import write_json
from ..output_dir import check_output_dir
from .errors import WindowContractError
from .graph_window_adapter import build_graph_bank, window_target_bank_nodes
from .task3_source import NODE_TYPE_NAMES, Task3Run, load_task3_run
from .trajectory_targets import ANNOTATED_TARGET_REQUIREMENTS, PILOT_SMOKE_TARGET_SOURCE
from .upstream import verify_upstream
from .window_builder import REJECTION_REASONS, WindowSet, build_windows
from .window_config import WindowConfig
from .window_manifest import CLAIMS, LIMITATIONS, MANIFEST_VERSION, PILOT_LABEL, git_provenance, versions
from .window_storage import write_bank, write_rejected, write_windows
from .window_validation import validate_window_dataset

REPO = Path(__file__).resolve().parents[2]


def _rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def context_statistics(ws: WindowSet, bank, cfg: WindowConfig) -> Dict[str, Any]:
    """Distinct context tracks per window (target excluded) observed in any history frame, by node type."""
    bn = bank.nodes
    W = ws.n_windows
    per_type = np.zeros((W, len(NODE_TYPE_NAMES)), np.int64)
    max_nodes_default = max_nodes_all = 0
    default_codes = [NODE_TYPE_NAMES.index(t) for t in cfg.default_context_node_types]
    for w in range(W):
        seen = {c: set() for c in range(len(NODE_TYPE_NAMES))}
        for f in ws.history_frames[w].tolist():
            k = bank.frame_pos(int(f))
            a, b = int(bank.node_ptr[k]), int(bank.node_ptr[k + 1])
            ty = bn["node_type"][a:b]
            max_nodes_all = max(max_nodes_all, b - a)
            max_nodes_default = max(max_nodes_default, int(np.isin(ty, default_codes).sum()))
            for t, c in zip(bn["track_index"][a:b].tolist(), ty.tolist()):
                if t != int(ws.target_track_index[w]):
                    seen[c].add(t)
        per_type[w] = [len(seen[c]) for c in range(len(NODE_TYPE_NAMES))]

    def summ(x):
        return {"mean": float(x.mean()) if W else 0.0, "median": float(np.median(x)) if W else 0.0, "max": int(x.max()) if W else 0,
                "windows_with_at_least_one": int((x > 0).sum())}
    return {"distinct_context_tracks_per_window": {t: summ(per_type[:, i]) for i, t in enumerate(NODE_TYPE_NAMES)},
            "windows_without_any_context_agent_default_types": int((per_type[:, [NODE_TYPE_NAMES.index(t) for t in cfg.default_context_node_types]].sum(axis=1) == 0).sum()) if W else 0,
            "max_nodes_in_one_history_frame_default_types": int(max_nodes_default),
            "max_nodes_in_one_history_frame_all_types": int(max_nodes_all),
            "note": "context agents are counted when they have a Task 3 node in at least one of the window's history frames; absence at other "
                    "history frames is an explicit presence mask"}


def generate_windows(task3_dir, cfg: WindowConfig, out_dir, *, config_path: Optional[str] = None, command: Optional[str] = None,
                     phase3_manifest: Optional[Path] = None, tracked_agents: Optional[Path] = None, video_id: Optional[str] = None,
                     group_id: Optional[str] = None, check_upstream: bool = True) -> Dict[str, Any]:
    out = check_output_dir(out_dir)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    git = git_provenance()
    run: Task3Run = load_task3_run(task3_dir, cfg)
    t1 = time.perf_counter()
    upstream = verify_upstream(run, cfg, phase3_manifest, tracked_agents) if check_upstream else {"checked": False}
    t2 = time.perf_counter()
    seqs = {t["track_id"].rsplit(":", 1)[0] for t in run.tracks}
    if len(seqs) != 1:
        raise WindowContractError("MULTIPLE_SEQUENCES", "a Task 3 run must hold exactly one video sequence, found %s" % sorted(seqs))
    vid = video_id or seqs.pop()
    gid = group_id or vid
    last = None
    if check_upstream and upstream.get("phase3_manifest", {}).get("provided"):
        last = upstream["phase3_manifest"]["last_frame_index"]
    first = upstream["phase3_manifest"]["first_frame_index"] if last is not None else 0
    ws = build_windows(run, cfg, video_first_frame=first, video_last_frame=last)
    t3 = time.perf_counter()
    bank = build_graph_bank(run, ws, cfg)
    tbn = window_target_bank_nodes(bank, ws)
    t4 = time.perf_counter()
    report = validate_window_dataset(ws, bank, tbn, cfg, run)
    if not report["ok"]:
        failed = [k for k, v in report["checks"].items() if not v["pass"]]
        raise WindowContractError("WINDOW_VALIDATION_FAILED", "validation failed: %s" % failed)
    t5 = time.perf_counter()

    out.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Dict[str, Any]] = {}

    def record(name, sha):
        files[name] = {"sha256": sha, "bytes": (out / name).stat().st_size}
    record("windows.npz", write_windows(out / "windows.npz", ws, tbn, vid, gid))
    record("graph_bank.npz", write_bank(out / "graph_bank.npz", bank))
    record("rejected_candidates.npz", write_rejected(out / "rejected_candidates.npz", ws, REJECTION_REASONS))
    lo, hi = cfg.dependency_offsets
    smoke = {"label": PILOT_LABEL, "split_created": False, "independent_groups": 1, "video_ids": [vid], "group_ids": [gid],
             "partitions": {"pilot_development": {"n_windows": ws.n_windows, "window_ids": "all windows in windows.npz",
                                                  "windows_sha256": files["windows.npz"]["sha256"]}},
             "permitted_uses": ["loading and batching tests", "data-pipeline smoke tests", "preliminary engineering diagnostics"],
             "forbidden_uses": ["train/validation/test evaluation", "model selection reported as research evidence",
                                "forecasting-accuracy claims", "generalization claims"],
             "formal_split_requirements": {
                 "split_unit": "independent group (proposal: PIE recording set, after verifying set semantics); never windows, frames or tracks",
                 "dependency_interval_offsets_frames": [lo, hi], "dependency_interval_length_frames": cfg.dependency_length_frames,
                 "minimum_purge_gap_frames_for_chronological_blocks": cfg.minimum_purge_gap_frames,
                 "note": "recompute if H, F, stride or any derived-feature lookback changes"}}
    record("smoke_manifest.json", write_json(out / "smoke_manifest.json", smoke))

    rc = ws.rejection_counts()
    ped_tracks = [t for t in run.tracks if t["node_type"] == "pedestrian"]
    cand_tracks = len(set(ws.target_track_index.tolist()) | set(ws.rejected_track_index.tolist()))
    bank_by_type = {t: int((bank.nodes["node_type"] == i).sum()) for i, t in enumerate(NODE_TYPE_NAMES)}
    rel_default = list(bank.active_relations(cfg, False))
    rel_all = list(bank.active_relations(cfg, True))
    counts = {
        "candidate_windows": ws.n_windows + ws.n_rejected, "accepted_windows": ws.n_windows, "rejected_windows": ws.n_rejected,
        "rejection_reasons": rc, "rejection_reason_precedence": list(REJECTION_REASONS),
        "pedestrian_tracks_in_task3": len(ped_tracks), "pedestrian_tracks_with_candidates": cand_tracks,
        "unique_pedestrian_target_tracks_accepted": int(np.unique(ws.target_track_index).shape[0]),
        "windows_target_border_at_anchor": int(ws.target_border_at_anchor.sum()),
        "windows_any_target_future_border": int(ws.target_future_border.any(axis=1).sum()) if ws.n_windows else 0,
        "windows_target_missing_unsampled_frames": int(sum(
            any(run.node_at(int(ws.target_track_index[w]), f) is None
                for f in range(int(ws.history_frames[w, 0]), int(ws.future_frames[w, -1]) + 1))
            for w in range(ws.n_windows))),
        "windows_target_missing_unsampled_frames_note": "accepted windows whose target has no node at one or more NON-sampled frames between "
                                                        "h_0 and f_{F-1}; allowed by the contract (sampled positions are all observed), "
                                                        "never filled; affected Task 2 velocities are already masked",
        "video_count": 1, "scene_group_count": 1,
        "graph_bank_frames": int(bank.frames.shape[0]), "graph_bank_nodes": bank.n_nodes, "graph_bank_nodes_by_type": bank_by_type,
        "graph_bank_edges_by_relation": {k: int(e.frame.shape[0]) for k, e in bank.edges.items()},
        "active_relations_default_context": rel_default, "active_relations_with_optional_context": rel_all,
        "context": context_statistics(ws, bank, cfg)}
    finished = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "manifest_version": MANIFEST_VERSION, "task": "phase5_p5.3_pilot_window_generation", "status": "completed", "label": PILOT_LABEL,
        "versions": versions(), "git": git, "started_utc": started, "finished_utc": finished, "command": command,
        "config": {"path": config_path, "sha256_resolved": cfg.sha256(), "content": cfg.to_dict(), "derived": cfg.derived()},
        "sources": {"task3_run_dir": _rel(run.run_dir), "task3_run_manifest": {"path": _rel(run.run_dir / "run_manifest.json"),
                                                                               "sha256": run.manifest_sha256},
                    "task3_git": run.manifest["git"], "task3_counts": run.manifest["counts"],
                    "task3_files_consumed": {k: v for k, v in run.file_hashes.items() if k.endswith(".npz") or k in ("tracks.json", "feature_dictionary.json", "config_resolved.json")},
                    "feature_dictionary_version": run.feature_dictionary["dictionary_version"],
                    "upstream_verification": upstream},
        "coordinates": {"coordinate_system": cfg.coordinate_system, "unit": cfg.coordinate_unit, "origin": "top-left", "u_axis": "+right",
                        "v_axis": "+down", "position": "bbox footpoint (bottom-centre)", "velocity_unit": "px/s (image plane, not physical)",
                        "time_unit": "s", "world_position_valid": False},
        "target_source": {"mode": cfg.target_source_mode, "description": PILOT_SMOKE_TARGET_SOURCE,
                          "annotated_target": {"status": "NOT_IMPLEMENTED_FUTURE_EXTENSION", "requirements": list(ANNOTATED_TARGET_REQUIREMENTS)}},
        "input_source": {"name": cfg.input_source, "causal_only": True,
                         "history_inputs": "Task 3 nodes and edges at the 6 history frames only (graph_bank.npz)",
                         "future_use": "target node positions only (windows.npz target_future_*); no future nodes of other agents, no future edges",
                         "offline_kinematics_used": False},
        "sampling": {"source_fps": cfg.source_fps, "target_sampling_hz": cfg.target_sampling_hz, "stride_frames": cfg.sampling_stride_frames,
                     "history_length": cfg.history_length, "prediction_length": cfg.prediction_length, "history_span_s": cfg.history_span_s,
                     "future_horizon_s": cfg.future_horizon_s, "anchor_grid": "frame %% %d == %d" % (cfg.anchor_stride_frames, cfg.anchor_phase_frames)},
        "policies": {"missing_observation": cfg.missing_observation_policy, "track_continuity": cfg.track_continuity_policy,
                     "context_agents": "present at a history frame only if Task 3 has a node there; absence = presence mask False; never a reason to reject",
                     "target_node_types": list(cfg.target_node_types), "default_context_node_types": list(cfg.default_context_node_types),
                     "optional_context_node_types": list(cfg.optional_context_node_types),
                     "semantically_uncertain_node_types": list(cfg.semantically_uncertain_node_types)},
        "counts": counts,
        "dependency_and_overlap": {"dependency_interval_offsets_frames": [lo, hi], "dependency_interval_length_frames": cfg.dependency_length_frames,
                                   "minimum_purge_gap_frames": cfg.minimum_purge_gap_frames,
                                   "overlap_note": "anchors every 3 frames with a %d-frame dependency interval: consecutive windows of one track share "
                                                   "most frames; any future partition must assign whole independent groups and purge at least the "
                                                   "dependency length between chronological blocks" % cfg.dependency_length_frames},
        "validation": report,
        "runtime_s": {"load_and_verify_task3": t1 - t0, "upstream_verification": t2 - t1, "build_windows": t3 - t2, "build_graph_bank": t4 - t3,
                      "validation": t5 - t4, "note": "offline preprocessing on one machine; not a real-time claim"},
        "outputs": dict(sorted(files.items())), "output_total_bytes": int(sum(f["bytes"] for f in files.values())),
        "claims": CLAIMS, "limitations": LIMITATIONS}
    write_json(out / "run_manifest.json", manifest)
    return manifest


__all__ = ["generate_windows", "context_statistics"]
