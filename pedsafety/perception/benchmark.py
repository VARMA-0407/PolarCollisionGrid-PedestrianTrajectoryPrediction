"""Performance instrumentation for the perception frontend (Phase 3, Task 1G).

Four SEPARATE measurements, never mixed:

* ``decode_only``   - video decoding latency (the source's own per-frame timer).
* ``detection_only``- decode + YOLO + detection validation, no tracker (a no-op tracker is plugged in).
* ``tracking_only`` - ByteTrack updates replayed on the detections captured in the detection-only pass; no decoding, no detector.
* ``combined``      - the full pipeline; per-frame wall time between consecutive results (includes decoding), plus the stage split.

Definitions: latency = wall time of one frame for that stage (``time.perf_counter``); P50/P95 = nearest-rank percentiles;
processing FPS = frames / total elapsed wall time of the pass (NOT 1 / mean latency). Warm-up inferences run before timing and
are excluded. Peak GPU memory is ``torch.cuda.max_memory_allocated/reserved`` over the detection-only and combined passes when
CUDA is present, else null. Python-heap and native/pinned memory are not measured.

This is perception-frontend performance only. It says nothing about the PCG predictor or an end-to-end safety system, and a
report is marked as supporting performance claims only if no component is a mock, the clock is real and hardware is declared.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .bytetrack_tracker import ByteTrackTracker
from .config import PerceptionConfig
from .detector import Detector
from .performance import PerformanceRecorder, percentile
from .pipeline import FrameResult, PerceptionPipeline
from .runtime_info import collect_environment, hardware_string
from .sources import FrameSource
from .tracker import Tracker
from .types import ComponentInfo, DetectionRecord, FrameMeta, TrackObservation

BENCHMARK_VERSION = "perception-benchmark-0.1.0-task1"


class NoOpTracker(Tracker):
    """Tracker that tracks nothing; used so the detection-only pass measures no tracking work. It is not a mock result source:
    it produces no tracks at all."""

    @property
    def info(self) -> ComponentInfo:
        return ComponentInfo("tracker", "no-op-tracker", "1", False, (("purpose", "detection-only benchmark pass"),))

    def update(self, frame: FrameMeta, detections: Sequence[DetectionRecord]) -> Sequence[TrackObservation]:
        return []

    def reset(self) -> None:
        return None


def latency_stats(latencies_s: Sequence[float], elapsed_s: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Millisecond statistics of a list of per-frame latencies; ``None`` for an empty list (nothing is invented)."""
    v = list(latencies_s)
    if not v:
        return None
    total = elapsed_s if elapsed_s is not None else sum(v)
    return {"frames": len(v), "mean_ms": 1e3 * sum(v) / len(v), "p50_ms": 1e3 * percentile(v, 50), "p95_ms": 1e3 * percentile(v, 95),
            "max_ms": 1e3 * max(v), "processing_fps": (len(v) / total) if total > 0 else None, "elapsed_s": total}


def _cuda_reset() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return True
    except ImportError:
        pass
    return False


def _cuda_peak() -> Optional[Dict[str, float]]:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20}
    except ImportError:
        pass
    return None


def run_benchmark(make_source: Callable[[], FrameSource], detector: Detector, cfg: PerceptionConfig, *, warmup: int = 3,
                  environment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the four passes. ``make_source`` must return a FRESH source each call (sources are one-shot)."""
    env = environment or collect_environment()
    hardware = hardware_string(env)
    plc = cfg.pipeline_config()

    # ---- pass 1: decode only
    src = make_source()
    t0 = time.perf_counter()
    n = 0
    first_meta = None
    for meta, _ in src.frames():
        first_meta = first_meta or meta
        n += 1
    decode_elapsed = time.perf_counter() - t0
    decode_lat = list(getattr(src, "decode_latencies_s", []))
    if first_meta is None:
        raise ValueError("the video produced no frames; nothing to benchmark")

    # ---- warm-up (excluded)
    warm = getattr(detector, "warmup", None)
    if callable(warm) and warmup > 0:
        warm(first_meta.width, first_meta.height, warmup)

    # ---- pass 2: detection only (also captures the detections for the tracking-only replay)
    gpu_on = _cuda_reset()
    src = make_source()
    rec_d = PerformanceRecorder()
    pipe_d = PerceptionPipeline(src, detector, NoOpTracker(), plc, rec_d)
    captured: List[FrameResult] = []
    t0 = time.perf_counter()
    for fr in pipe_d.iter_results():
        captured.append(fr)
    det_elapsed = time.perf_counter() - t0
    det_summary = rec_d.summary((src.info, detector.info), hardware)
    gpu_det = _cuda_peak() if gpu_on else None
    detect_stage = det_summary.stages.get("detect")

    # ---- pass 3: tracking only (replay of the captured detections; no source, no detector)
    tracker = ByteTrackTracker(cfg.tracker)
    track_lat: List[float] = []
    tracker.reset()
    t0 = time.perf_counter()
    for fr in captured:
        if not fr.processed:
            continue
        a = time.perf_counter()
        tracker.update(fr.frame, fr.detections)
        track_lat.append(time.perf_counter() - a)
    track_elapsed = time.perf_counter() - t0

    # ---- pass 4: combined
    gpu_on = _cuda_reset()
    tracker2 = ByteTrackTracker(cfg.tracker)
    src = make_source()
    rec_c = PerformanceRecorder()
    pipe_c = PerceptionPipeline(src, detector, tracker2, plc, rec_c)
    comb_lat: List[float] = []
    t_run = time.perf_counter()
    prev = t_run
    for _ in pipe_c.iter_results():
        now = time.perf_counter()
        comb_lat.append(now - prev)
        prev = now
    comb_elapsed = prev - t_run
    comb_summary = rec_c.summary((src.info, detector.info, tracker2.info), hardware)
    gpu_comb = _cuda_peak() if gpu_on else None

    def stage_ms(name: str) -> Optional[Dict[str, Any]]:
        s = comb_summary.stages.get(name)
        return None if s is None else {"frames": s.count, "mean_ms": s.mean_s * 1e3, "p50_ms": s.p50_s * 1e3, "p95_ms": s.p95_s * 1e3, "max_ms": s.max_s * 1e3}

    return {
        "benchmark_version": BENCHMARK_VERSION, "environment": env, "hardware_declared": hardware,
        "frames": n, "frame_size": [first_meta.width, first_meta.height], "source_fps": first_meta.nominal_fps, "warmup_inferences": warmup,
        "components": {c.kind: {"name": c.name, "version": c.version, "is_mock": c.is_mock, "parameters": dict(c.parameters)}
                       for c in comb_summary.components},
        "decode_only": latency_stats(decode_lat, decode_elapsed),
        "detection_only": {"detect_stage": None if detect_stage is None else {"frames": detect_stage.count, "mean_ms": detect_stage.mean_s * 1e3,
                                                                               "p50_ms": detect_stage.p50_s * 1e3, "p95_ms": detect_stage.p95_s * 1e3,
                                                                               "max_ms": detect_stage.max_s * 1e3},
                           "pass_elapsed_s": det_elapsed, "processing_fps_incl_decode": (n / det_elapsed) if det_elapsed > 0 else None,
                           "gpu_memory": gpu_det},
        "tracking_only": latency_stats(track_lat, track_elapsed),
        "combined": {"per_frame_wall_incl_decode": latency_stats(comb_lat, comb_elapsed), "detect_stage": stage_ms("detect"),
                     "track_stage": stage_ms("track"), "gpu_memory": gpu_comb, "pipeline_counters": {
                         "detections": sum(len(f.detections) for f in captured), "note": "detections counted in the detection-only pass"}},
        "supports_performance_claims": comb_summary.supports_performance_claims,
        "claim_rule": "true only when no component is a mock, the clock is the real monotonic clock and hardware is declared",
        "scope": "perception frontend only (decode, YOLO, ByteTrack, validation); not the PCG predictor and not end-to-end system FPS",
        "gpu_memory_measured": bool(gpu_det or gpu_comb)}
