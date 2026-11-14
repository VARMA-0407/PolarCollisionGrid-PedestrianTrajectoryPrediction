"""Multi-scene SYNTHETIC corpus + matching pipeline config (Stage 6).

Everything is invented image-pixel data (see ``synthetic.py``). Each scene has its own
``scene_id`` and ``sequence_id`` and is independent BY CONSTRUCTION, which is the only
reason the generated config may state a scene-level independence basis. It says nothing
about any real dataset.
"""

import csv
import json
from pathlib import Path
from typing import Union

from ..output_dir import check_output_dir
from .synthetic import SYNTHETIC_COLUMNS, SYNTHETIC_COORDINATE_SYSTEM, _CSV_ORDER, AgentSpec, default_specs, generate_scene


def _specs_for_scene(s: int):
    shift = 37.0 * s
    return tuple(AgentSpec(a.track_id, a.label, (a.start_xy[0] + shift, a.start_xy[1] + 11.0 * s), a.velocity, a.acceleration, a.box_size,
                           a.first_frame, a.last_frame, a.missing_frames) for a in default_specs())


def write_synthetic_corpus(out_dir: Union[str, Path], *, n_scenes: int = 6, n_frames: int = 40, fps: float = 10.0, seed: int = 0,
                           noise_px: float = 0.0) -> dict:
    if isinstance(n_scenes, bool) or not isinstance(n_scenes, int) or n_scenes < 1:
        raise ValueError("n_scenes must be an integer >= 1, got %r" % (n_scenes,))
    out = check_output_dir(out_dir)          # new or empty directory only; never overwrites
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "synthetic_corpus.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_ORDER)
        for s in range(n_scenes):
            sc = generate_scene(seed=seed + s, n_frames=n_frames, fps=fps, sequence_id="syn_seq_%03d" % s, scene_id="syn_scene_%03d" % s,
                                specs=_specs_for_scene(s), position_noise_px=noise_px)
            for o in sc.observations:
                b = o.bbox_xyxy
                w.writerow([o.sequence_id, o.scene_id, o.track_id, o.frame_index, repr(o.timestamp), o.source_label, repr(o.pos_x), repr(o.pos_y),
                            repr(b[0]), repr(b[1]), repr(b[2]), repr(b[3]), repr(o.vel_x), repr(o.vel_y), repr(o.detection_confidence)])
    config = {
        "config_version": 1,
        "input": {"csv": csv_path.name, "source_dataset": "SYNTHETIC", "columns": dict(SYNTHETIC_COLUMNS),
                  "position_semantics": "image_pixel", "position_unit": "pixel", "position_coordinate_system": SYNTHETIC_COORDINATE_SYSTEM},
        "agent_state": {"velocity_contract": {"space": "image", "position_unit": "pixel", "time_unit": "s"}},
        "trajectory": {"coordinate_mode": "image_pixel"},
        "grouping": {"unit": "scene", "independence_basis": "synthetic scenes are generated independently by construction; "
                                                           "this says nothing about real data"},
        "split": {"seed": 7},
        "windows": {"obs_len": 5, "pred_len": 5},
        "view": "causal",
    }
    cfg_path = out / "synthetic_pipeline_config.json"
    cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "config": str(cfg_path), "n_scenes": n_scenes}
