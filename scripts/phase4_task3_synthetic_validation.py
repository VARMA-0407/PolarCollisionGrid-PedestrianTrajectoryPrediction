"""Phase 4 Task 3 synthetic end-to-end validation run (synthetic data only; deterministic).

Builds a small synthetic Phase 3-style agent_states file (2 pedestrians, 2 vehicles, 2 bicyclists, 30 frames, one
border-touching vehicle, one 3-frame gap), runs the REAL Task 2 runner on it, then the Task 3 runner on Task 2's causal
output. Writes everything under a new/empty output directory:
    inputs/agent_states.jsonl, task2/ (Task 2 outputs), task3/ (Task 3 outputs)
Optionally copies a mutation-check result file into the directory for provenance.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.image_kinematics import ImageKinematicsConfig  # noqa: E402
from pedsafety.image_kinematics_run import run_image_kinematics  # noqa: E402
from pedsafety.interaction.config import InteractionConfig  # noqa: E402
from pedsafety.interaction.runner import run_interaction_representation  # noqa: E402
from pedsafety.output_dir import check_output_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def record(track, frame, fx, fy, h, w, cls, label):
    b = [fx - w / 2.0, fy - h, fx + w / 2.0, fy]
    return {"track_id": "synthetic:scene:%s" % track, "agent_id": None, "agent_class": cls, "timestamp": frame / 30.0, "frame_index": frame,
            "sequence_id": "synthetic:scene", "scene_id": "synthetic:scene", "bbox_xyxy": b, "bbox_center": [fx, fy - h / 2.0],
            "footpoint": [fx, fy], "detection_confidence": 0.9, "image_x": fx, "image_y": fy, "image_coordinate_system": "captured_1920x1080",
            "image_coordinate_unit": "pixel", "world_x": None, "world_y": None, "world_coordinate_system": None, "world_coordinate_unit": None,
            "world_position_valid": False, "velocity_x": None, "velocity_y": None, "speed": None, "acceleration_x": None, "acceleration_y": None,
            "kinematic_coordinate_space": None, "kinematic_position_unit": None, "kinematic_time_unit": None, "heading": None, "visibility": None,
            "occlusion_status": None, "tracking_confidence": None, "observation_valid": True, "interpolation_flag": False,
            "source_dataset": "synthetic", "source_label": label}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mutation-results", default=None)
    a = ap.parse_args(argv)
    out = check_output_dir(a.out)
    (out / "inputs").mkdir(parents=True)
    recs = []
    for f in range(30):
        recs.append(record("1", f, 300 + 3 * f, 600, 70, 25, "pedestrian", "pedestrian"))
        if f not in (12, 13, 14):                                          # 3-frame gap for track 2
            recs.append(record("2", f, 520 - 2 * f, 610, 72, 25, "pedestrian", "pedestrian"))
        recs.append(record("3", f, 900 - 8 * f, 1080, 220, 320, "vehicle", "car"))   # touches the bottom border
        recs.append(record("4", f, 1400, 700 + f, 140, 220, "vehicle", "truck"))
        recs.append(record("5", f, 200 + 6 * f, 750, 95, 40, "bicyclist", "bike"))
        recs.append(record("6", f, 1700 - 5 * f, 760, 95, 40, "bicyclist", "bike"))
    p3 = out / "inputs" / "agent_states.jsonl"
    p3.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    run_image_kinematics(p3, ImageKinematicsConfig.load(REPO / "configs" / "phase4" / "image_kinematics_default.json"), out / "task2")
    m = run_interaction_representation(out / "task2" / "causal_kinematics.jsonl", out / "task2" / "run_manifest.json", p3,
                                       InteractionConfig.load(REPO / "configs" / "phase4" / "interaction_representation_default.json"), out / "task3",
                                       config_path="configs/phase4/interaction_representation_default.json", command="phase4_task3_synthetic_validation.py")
    if a.mutation_results:
        shutil.copyfile(a.mutation_results, out / "mutation_check_results.txt")
    print(json.dumps(m["counts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
