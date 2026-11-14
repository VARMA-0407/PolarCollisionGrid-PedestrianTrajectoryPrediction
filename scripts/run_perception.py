"""Phase 3 Task 1: run the perception frontend (video -> YOLO -> ByteTrack -> JSONL) on one video.

Same script locally (CPU) and in Colab (GPU). Examples (repository root, `pedsafety-perception` environment):

  python scripts/run_perception.py --video clip.mp4 --model weights/yolo11n.pt --device cpu --save-video
  python scripts/run_perception.py --video synthetic.mp4 --backend replay --replay-gt synthetic.gt.json --save-video --scene-id synthetic_scene

Exit codes: 0 ok, 2 bad configuration/input (missing video or weights, invalid config), 3 other pedsafety error.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pedsafety.errors import ConfigError, PedsafetyError  # noqa: E402
from pedsafety.perception.config import load_config  # noqa: E402
from pedsafety.perception.runner import execute_run, make_run_dir  # noqa: E402
from pedsafety.perception.runtime_errors import DetectorLoadError, TrackerInitError, VideoSourceError  # noqa: E402


def _drop_if_empty(path) -> None:
    """A failed run must not leave an empty experiment directory behind (a directory with any content is never touched)."""
    try:
        if path is not None and path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "phase3" / "perception_default.json"))
    ap.add_argument("--video", required=True, help="input video file")
    ap.add_argument("--output-root", default="outputs/perception", help="a NEW timestamped run directory is created inside it")
    ap.add_argument("--run-name", default=None, help="suffix of the run directory (default: video file stem)")
    ap.add_argument("--backend", choices=("yolo", "replay"), default="yolo", help="'replay' = MOCK ground-truth detector for the synthetic video")
    ap.add_argument("--replay-gt", default=None, help="ground-truth JSON of the synthetic video (backend replay only)")
    ap.add_argument("--model", default=None, help="override detector.model_path (the file must already exist; nothing is downloaded)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="override detector.device")
    ap.add_argument("--max-frames", type=int, default=None, help="override video.max_frames")
    ap.add_argument("--save-video", action="store_true", help="write annotated.mp4 incrementally")
    ap.add_argument("--no-detections-jsonl", action="store_true")
    ap.add_argument("--scene-id", default=None, help="if given, also write agent_states.jsonl through the existing adapter")
    ap.add_argument("--source-dataset", default="video_perception")
    a = ap.parse_args(argv)
    out_dir = None
    try:
        cfg = load_config(Path(a.config))
        det = cfg.detector
        if a.model is not None or a.device is not None:
            det = dataclasses.replace(det, model_path=a.model or det.model_path, device=a.device or det.device)
        vid = dataclasses.replace(cfg.video, max_frames=a.max_frames if a.max_frames is not None else cfg.video.max_frames)
        cfg = dataclasses.replace(cfg, detector=det, video=vid)
        gt = json.loads(Path(a.replay_gt).read_text(encoding="utf-8")) if a.replay_gt else None
        out_dir = make_run_dir(Path(a.output_root), a.run_name or Path(a.video).stem)
        m = execute_run(cfg, a.video, out_dir, backend=a.backend, replay_gt=gt, save_video=a.save_video, write_detections=not a.no_detections_jsonl,
                        scene_id=a.scene_id, source_dataset=a.source_dataset)
    except (ConfigError, VideoSourceError, DetectorLoadError, TrackerInitError, OSError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        _drop_if_empty(out_dir)
        return 2
    except PedsafetyError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        _drop_if_empty(out_dir)
        return 3
    r = m["results"]
    print("run directory : %s" % out_dir)
    print("backend       : %s%s" % (m["backend"], "  (MOCK detector: plumbing only, not a detection result)" if m["is_mock"] else ""))
    print("frames        : %d seen, %d processed, %d skipped" % (r["frames_seen"], r["frames_processed"], r["frames_skipped"]))
    print("detections    : %d   track rows: %d   unique track ids: %d" % (r["detections"], r["track_rows"], r["unique_track_ids"]))
    print("issues        : %s" % (r["issues_by_code"] or "none"))
    print("outputs       : %s" % ", ".join(m["outputs"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
