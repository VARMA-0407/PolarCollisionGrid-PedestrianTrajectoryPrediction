"""Stage 6 fixtures: CSV corpora written to tmp dirs and the configs that drive the CLI.
All data are synthetic image-pixel tracks; nothing here describes a real dataset."""

import csv
import json
from pathlib import Path

from pedsafety.io.synthetic import SYNTHETIC_COLUMNS, _CSV_ORDER
from pedsafety.io.synthetic_corpus import write_synthetic_corpus
from stage3_fixtures import RASTER, make_manifest
from stage5_fixtures import corpus_observations


def synthetic(tmp: Path, **kw):
    info = write_synthetic_corpus(tmp, **kw)
    return Path(info["csv"]), Path(info["config"])


def load_cfg(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_cfg(path: Path, cfg: dict) -> Path:
    Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return Path(path)


def write_obs_csv(obs, path: Path) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_ORDER)
        for o in obs:
            b = o.bbox_xyxy
            row = {"sequence": o.sequence_id, "scene": o.scene_id, "track": o.track_id, "frame": o.frame_index, "t_s": repr(o.timestamp),
                   "cls": o.source_label, "foot_x": repr(o.pos_x), "foot_y": repr(o.pos_y), "x0": repr(b[0]), "y0": repr(b[1]),
                   "x1": repr(b[2]), "y1": repr(b[3]), "vx": "", "vy": "", "conf": ""}
            w.writerow([row[c] for c in _CSV_ORDER])
    return path


def calibrated_setup(tmp: Path, *, mode: str, promote: bool, manifest=None):
    """Synthetic multi-scene CSV inside the calibration raster + a config that projects it."""
    csv_path = write_obs_csv(corpus_observations(n_scenes=4, seqs_per_scene=1, tracks_per_seq=2, n_frames=14), tmp / "cal.csv")
    man = manifest or make_manifest()
    (tmp / "manifest.json").write_text(json.dumps(man.to_dict()), encoding="utf-8")
    cols = {k: v for k, v in SYNTHETIC_COLUMNS.items() if k not in ("vel_x", "vel_y", "confidence")}
    cfg = {
        "config_version": 1,
        "input": {"csv": csv_path.name, "source_dataset": "SYN", "columns": cols, "position_semantics": "image_pixel",
                  "position_unit": "pixel", "position_coordinate_system": RASTER.coordinate_system},
        "calibration": {"manifest": "manifest.json", "observed_raster": RASTER.to_dict(), "promote_eligible_to_canonical": promote},
        "trajectory": {"coordinate_mode": mode},
        "grouping": {"unit": "scene", "independence_basis": "synthetic scenes are independent by construction"},
        "split": {"seed": 3},
        "windows": {"obs_len": 4, "pred_len": 3},
    }
    return csv_path, save_cfg(tmp / "cal_config.json", cfg)
