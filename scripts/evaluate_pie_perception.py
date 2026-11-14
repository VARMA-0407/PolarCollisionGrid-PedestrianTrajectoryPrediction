"""Phase 3 Task 4: evaluate existing YOLO + ByteTrack outputs against the official PIE annotations of one video.

Reads existing JSONL files only (no model, no video). Example:

  python scripts/evaluate_pie_perception.py \
      --annotations /content/PIE_Pilot/annotations/annotations/set01/video_0004_annt.xml \
      --detections  <run>/detections.jsonl --tracked-agents <run>/tracked_agents.jsonl \
      --run-manifest <run>/run_manifest.json --output-dir <new empty dir> --video-id video_0004

Exit codes: 0 ok, 2 bad input/configuration (missing file, wrong video id, output directory not empty), 3 other pedsafety error.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pedsafety.errors import ConfigError, PedsafetyError  # noqa: E402
from pedsafety.perception.pie_perception_evaluator import FRAME_POLICIES, PieEvaluationError, run_evaluation  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True, help="PIE CVAT XML (video_XXXX_annt.xml)")
    ap.add_argument("--detections", required=True, help="detections.jsonl of the perception run")
    ap.add_argument("--tracked-agents", required=True, help="tracked_agents.jsonl of the perception run")
    ap.add_argument("--output-dir", required=True, help="a NEW (or empty) directory; existing content is never overwritten")
    ap.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.50, 0.75])
    ap.add_argument("--video-id", default="video_0004", help="video_id value to select in the prediction rows")
    ap.add_argument("--run-manifest", default=None, help="optional run_manifest.json (gives the processed frame range and weights hash)")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="optional detection confidence floor (default: use every detection)")
    ap.add_argument("--frame-offset", type=int, default=0, help="prediction frame = annotation frame + offset (default 0; sanity-checked in the report)")
    ap.add_argument("--frame-policy", choices=FRAME_POLICIES, default="annotated_any", help="primary evaluated-frame set (the other is reported as sensitivity)")
    ap.add_argument("--exclude-fully-occluded", action="store_true", help="drop fully occluded GT boxes (default: retained and reported by occlusion level)")
    a = ap.parse_args(argv)
    try:
        r = run_evaluation(a.annotations, a.detections, a.tracked_agents, a.output_dir, video_id=a.video_id, iou_thresholds=a.iou_thresholds,
                           min_confidence=a.min_confidence, frame_offset=a.frame_offset, frame_policy=a.frame_policy,
                           exclude_fully_occluded=a.exclude_fully_occluded, run_manifest=a.run_manifest)
    except (ConfigError, PieEvaluationError, OSError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    except PedsafetyError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 3
    prim = r["detection"]["results"]["primary"]
    print("output directory: %s" % r["output_dir"])
    print("evaluated frames: %d (policy %s)" % (prim["frames"]["count"], prim["frame_policy"]))
    for t, d in prim["by_iou_threshold"].items():
        o = d["overall_supported_classes"]
        print("detection IoU %s: P=%s R=%s F1=%s (TP %d FP %d FN %d)" % (t, o["precision"], o["recall"], o["f1"], o["tp"], o["fp"], o["fn"]))
    for t, m in r["tracking"]["results"]["primary"]["variants"]["primary_tracking_validity_true"].items():
        print("tracking  IoU %s: IDF1=%s MOTA=%s IDSW=%d" % (t, m["idf1"], m["mota"], m["id_switches"]))
    print("files: pie_detection_metrics.json, pie_tracking_metrics.json, pie_evaluation_summary.md, pie_evaluation_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
