"""Phase 3 Task 1G: perception-frontend benchmark (decode-only, detection-only, tracking-only, combined).

Local CPU example (do NOT use this locally for GPU claims):
  python scripts/benchmark_perception.py --video clip.mp4 --model weights/yolo11n.pt --device cpu --max-frames 100 --output-root outputs/perception_bench

Colab T4 example: see notebooks/phase3_task1_perception_colab.ipynb (it calls this script).
Writes benchmark.json into a NEW timestamped directory and prints a summary table.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pedsafety.errors import ConfigError, PedsafetyError  # noqa: E402
from pedsafety.perception.benchmark import run_benchmark  # noqa: E402
from pedsafety.perception.config import load_config  # noqa: E402
from pedsafety.perception.runner import build_detector, make_run_dir  # noqa: E402
from pedsafety.perception.runtime_errors import DetectorLoadError, TrackerInitError, VideoSourceError  # noqa: E402
from pedsafety.perception.video_source import OpenCVVideoSource  # noqa: E402


def _row(name, s):
    if not s:
        return "%-26s  (no data)" % name
    return "%-26s  n=%-5d mean=%8.2f ms  p50=%8.2f  p95=%8.2f  max=%8.2f" % (name, s["frames"], s["mean_ms"], s["p50_ms"], s["p95_ms"], s["max_ms"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "phase3" / "perception_default.json"))
    ap.add_argument("--video", required=True)
    ap.add_argument("--output-root", default="outputs/perception_bench")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--backend", choices=("yolo", "replay"), default="yolo")
    ap.add_argument("--replay-gt", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=3, help="dummy inferences before timing (excluded)")
    ap.add_argument("--plumbing-only", action="store_true",
                    help="mark the report as NOT supporting performance claims (random weights, synthetic video, or any other non-representative setup)")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(Path(a.config))
        det = cfg.detector
        if a.model is not None or a.device is not None:
            det = dataclasses.replace(det, model_path=a.model or det.model_path, device=a.device or det.device)
        vid = dataclasses.replace(cfg.video, max_frames=a.max_frames if a.max_frames is not None else cfg.video.max_frames)
        cfg = dataclasses.replace(cfg, detector=det, video=vid)
        gt = json.loads(Path(a.replay_gt).read_text(encoding="utf-8")) if a.replay_gt else None
        detector = build_detector(cfg, a.backend, gt)
        out_dir = make_run_dir(Path(a.output_root), a.run_name or Path(a.video).stem)

        def make_source():
            return OpenCVVideoSource(a.video, sequence_id=cfg.video.sequence_id, fps_override=cfg.video.fps_override, max_frames=cfg.video.max_frames)

        rep = run_benchmark(make_source, detector, cfg, warmup=a.warmup)
        if a.plumbing_only:
            rep["supports_performance_claims"] = False
            rep["caveats"] = ["marked --plumbing-only by the operator: setup is not representative (e.g. random weights and/or synthetic video)"]
        rep["config"] = cfg.to_dict()
        rep["video"] = Path(a.video).name
        (out_dir / "benchmark.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    except (ConfigError, VideoSourceError, DetectorLoadError, TrackerInitError, OSError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    except PedsafetyError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 3
    print("hardware: %s" % rep["hardware_declared"])
    print("frames: %d  size: %s  source fps: %s  warm-up: %d" % (rep["frames"], rep["frame_size"], rep["source_fps"], rep["warmup_inferences"]))
    print(_row("decode only", rep["decode_only"]))
    print(_row("detection only (detect)", rep["detection_only"]["detect_stage"]))
    print(_row("tracking only", rep["tracking_only"]))
    c = rep["combined"]
    print(_row("combined (incl. decode)", c["per_frame_wall_incl_decode"]))
    print("combined processing FPS: %.2f" % (c["per_frame_wall_incl_decode"]["processing_fps"] or 0))
    print("GPU peak memory: %s" % (c["gpu_memory"] or "not measured (no CUDA)"))
    print("supports performance claims: %s%s" % (rep["supports_performance_claims"], "" if rep["supports_performance_claims"] else "  (%s)" % ("; ".join(rep.get("caveats", [])) or "mock component or undeclared hardware")))
    print("written: %s" % (out_dir / "benchmark.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
