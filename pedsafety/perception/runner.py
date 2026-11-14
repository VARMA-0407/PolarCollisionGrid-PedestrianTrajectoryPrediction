"""Execution layer shared by the local CLI, the Colab notebook and the tests (Phase 3, Task 1).

``execute_run`` is the single implementation of "video -> YOLO -> ByteTrack -> JSONL (+ annotated video, + agent states)".
``scripts/run_perception.py`` is a thin argument parser around it and the Colab notebook calls that script, so there is no
second implementation. Frames are processed one at a time and every output is written incrementally.
"""

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..errors import ConfigError, PedsafetyError
from .adapter import AdapterConfig, tracks_to_agent_states
from .bytetrack_tracker import ByteTrackTracker
from .config import PerceptionConfig
from .detector import Detector
from .export import JsonlWriter, detection_rows, track_rows
from .performance import PerformanceRecorder
from .pipeline import FrameResult, PerceptionPipeline, PerceptionResult
from .runtime_info import collect_environment, git_commit, hardware_string
from .sources import FrameSource
from .types import ComponentInfo, FrameMeta

RUN_MANIFEST_VERSION = "perception-run-manifest-0.1.0-task1"


class PayloadTap(FrameSource):
    """Delegating source that remembers the most recent payload, so the visualiser can draw on the frame the pipeline just
    processed without the pipeline ever exposing pixels."""

    def __init__(self, inner: FrameSource):
        self.inner = inner
        self.last_payload: Any = None

    @property
    def info(self) -> ComponentInfo:
        return self.inner.info

    def frames(self) -> Iterator[Tuple[FrameMeta, Any]]:
        for meta, payload in self.inner.frames():
            self.last_payload = payload
            yield meta, payload


def make_run_dir(root: Path, name: str) -> Path:
    """Create a NEW, unique experiment directory ``<root>/<UTC timestamp>_<name>``; never reuses or overwrites one."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name) or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / ("%s_%s" % (stamp, safe))
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_detector(cfg: PerceptionConfig, backend: str = "yolo", replay_gt: Optional[Dict[str, Any]] = None) -> Detector:
    if backend == "yolo":
        from .yolo_detector import UltralyticsDetector
        return UltralyticsDetector(cfg.detector)
    if backend == "replay":
        from .synthetic_video import ReplayDetector
        if replay_gt is None:
            raise ConfigError("backend 'replay' needs a synthetic ground-truth file")
        return ReplayDetector(replay_gt)
    raise ConfigError("backend must be 'yolo' or 'replay', got %r" % (backend,))


class _Stats:
    def __init__(self):
        self.frames = self.processed = self.detections = self.track_rows = 0
        self.by_class: Counter = Counter()
        self.det_by_class: Counter = Counter()
        self.ids = set()
        self.issues: Counter = Counter()
        self.max_tracks_in_frame = 0
        self.first_frame: Optional[int] = None
        self.last_frame: Optional[int] = None

    def add(self, fr: FrameResult) -> None:
        self.frames += 1
        self.processed += 1 if fr.processed else 0
        self.detections += len(fr.detections)
        self.track_rows += len(fr.tracks)
        self.max_tracks_in_frame = max(self.max_tracks_in_frame, len(fr.tracks))
        for d in fr.detections:
            self.det_by_class[d.category.value] += 1
        for t in fr.tracks:
            self.by_class[t.category.value] += 1
            self.ids.add(t.track_id)
        for i in fr.issues:
            self.issues[i.code.value] += 1
        if self.first_frame is None:
            self.first_frame = fr.frame.frame_index
        self.last_frame = fr.frame.frame_index

    def as_dict(self) -> Dict[str, Any]:
        return {"frames_seen": self.frames, "frames_processed": self.processed, "frames_skipped": self.frames - self.processed,
                "detections": self.detections, "detections_by_semantic_class": dict(sorted(self.det_by_class.items())),
                "track_rows": self.track_rows, "track_rows_by_semantic_class": dict(sorted(self.by_class.items())),
                "unique_track_ids": len(self.ids), "max_tracks_in_one_frame": self.max_tracks_in_frame,
                "issues_by_code": dict(sorted(self.issues.items())), "first_frame_index": self.first_frame, "last_frame_index": self.last_frame}


def execute_run(cfg: PerceptionConfig, video_path: str, out_dir: Path, *, backend: str = "yolo", replay_gt: Optional[Dict[str, Any]] = None,
                save_video: bool = False, write_detections: bool = True, scene_id: Optional[str] = None,
                source_dataset: str = "video_perception", extra_manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the perception pipeline on one video and write all outputs into the (already created, empty) ``out_dir``.

    Outputs: ``tracked_agents.jsonl``, ``detections.jsonl`` (optional), ``annotated.mp4`` (optional), ``agent_states.jsonl``
    (only if ``scene_id`` is given), ``run_manifest.json``. Returns the manifest dict.
    """
    from .video_source import OpenCVVideoSource
    out_dir = Path(out_dir)
    t_start = time.perf_counter()
    env = collect_environment()
    notes: List[str] = list(cfg.warnings())
    source = OpenCVVideoSource(video_path, sequence_id=cfg.video.sequence_id, fps_override=cfg.video.fps_override, max_frames=cfg.video.max_frames)
    tap = PayloadTap(source)
    try:
        detector = build_detector(cfg, backend, replay_gt)
        tracker = ByteTrackTracker(cfg.tracker)
        recorder = PerformanceRecorder()
        pipeline = PerceptionPipeline(tap, detector, tracker, cfg.pipeline_config(), recorder)
        stats = _Stats()
        kept: List[FrameResult] = []
        writer = None
        track_w = JsonlWriter(str(out_dir / "tracked_agents.jsonl"))
        det_w = JsonlWriter(str(out_dir / "detections.jsonl")) if write_detections else None
        try:
            for fr in pipeline.iter_results():
                stats.add(fr)
                track_w.write_rows(track_rows(fr))
                if det_w is not None:
                    det_w.write_rows(detection_rows(fr))
                if scene_id is not None:
                    kept.append(fr)
                if save_video and tap.last_payload is not None:
                    from .visualize import AnnotatedVideoWriter, draw_frame_result
                    if writer is None:
                        writer = AnnotatedVideoWriter(str(out_dir / "annotated.mp4"), source.fps, source.width, source.height)
                    writer.write(draw_frame_result(tap.last_payload, fr))
        finally:
            track_w.close()
            if det_w is not None:
                det_w.close()
            if writer is not None:
                writer.close()
        issues = [i for f in kept for i in f.issues]
        perf = recorder.summary((source.info, detector.info, tracker.info), hardware_string(env))
        agent_info: Optional[Dict[str, Any]] = None
        if scene_id is not None:
            result = PerceptionResult(kept, issues, source.info, detector.info, tracker.info)
            ds = source_dataset if not (result.is_mock and "mock" not in source_dataset.lower()) else source_dataset + "_mock"
            ar = tracks_to_agent_states(result, AdapterConfig(ds, scene_id))
            with open(out_dir / "agent_states.jsonl", "w", encoding="utf-8", newline="\n") as fh:
                for rec in ar.records:
                    fh.write(json.dumps(rec, separators=(",", ":"), allow_nan=False) + "\n")
            agent_info = {"records": len(ar.records), "dropped_predicted": ar.dropped_predicted, "source_dataset": ds, "scene_id": scene_id,
                          "is_mock": ar.is_mock}
        is_mock = source.info.is_mock or detector.info.is_mock or tracker.info.is_mock
        stage_ms = {k: {"count": v.count, "mean_ms": v.mean_s * 1e3, "p50_ms": v.p50_s * 1e3, "p95_ms": v.p95_s * 1e3, "max_ms": v.max_s * 1e3}
                    for k, v in perf.stages.items()}
        manifest: Dict[str, Any] = {
            "manifest_version": RUN_MANIFEST_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "video": os.path.basename(video_path),
            "backend": backend, "is_mock": is_mock, "config": cfg.to_dict(), "config_notes": notes, "environment": env, "git_commit": git_commit(),
            "components": {c.kind: {"name": c.name, "version": c.version, "is_mock": c.is_mock, "parameters": dict(c.parameters)}
                           for c in (source.info, detector.info, tracker.info)},
            "video_stats": source.stats, "results": stats.as_dict(), "tracker_class_flips": tracker.class_flips,
            "agent_states": agent_info, "outputs": sorted([p.name for p in out_dir.iterdir()] + ["run_manifest.json"]),
            "stage_latency_ms_in_pipeline_run": stage_ms, "wall_time_s": time.perf_counter() - t_start,
            "performance_note": "latencies in this manifest come from a normal run (no warm-up, no isolation of stages); "
                                "use scripts/benchmark_perception.py for a performance report",
            "world_coordinates": "none: outputs are image-space pixels only"}
        manifest.update(extra_manifest or {})
        with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        return manifest
    finally:
        source.close()
