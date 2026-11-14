"""Generate the local synthetic test video and its ground-truth JSON (plumbing validation only; no download).

  python scripts/make_synthetic_video.py --out outputs/synthetic/synthetic_scene.mp4
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pedsafety.perception.synthetic_video import generate_synthetic_video, save_ground_truth  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output .mp4 path (parent directories are created)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--frames", type=int, default=120)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print("ERROR: %s already exists; refusing to overwrite" % out, file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    gt = generate_synthetic_video(str(out), width=a.width, height=a.height, fps=a.fps, n_frames=a.frames)
    gt_path = out.with_suffix(".gt.json")
    save_ground_truth(gt, str(gt_path))
    print("video: %s\nground truth: %s\nframes: %d, %dx%d @ %s fps (synthetic rectangles: plumbing validation only)"
          % (out, gt_path, gt["n_frames"], gt["width"], gt["height"], gt["fps"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
