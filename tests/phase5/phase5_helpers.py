"""Synthetic scene for the Phase 5 P5.3 tests (no PIE data, no model, no video).

Frames 0..89 at 30 fps (timestamps frame/30). Tracks (sequence pie:video_0004):
    1  pedestrian, frames 0..89, continuous, linear motion           -> reference target
    2  pedestrian, frames 0..89 except frame 30 (a sampled frame)     -> windows sampling frame 30 are rejected
    3  pedestrian, frames 0..89 except frame 31 (an unsampled frame)  -> windows kept; velocity at 32 masked by Task 2
    4  vehicle,    frames 0..89                                        -> context only, never a target
    5  bicyclist (source label 'bike'), frames 10..60                  -> semantically uncertain optional context
    6  pedestrian, frames 0..40;  7 pedestrian, frames 41..89 continuing 6's path -> fragments, never merged
    8  pedestrian, frames 0..10                                        -> too short
    9  vehicle,    frames 50..89                                       -> appears mid-sequence (absent = masked, never filled)
"""

import json
from pathlib import Path

from pedsafety.interaction.runner import run_interaction_representation
from pedsafety.phase5.window_config import WindowConfig
from task2_helpers import rec
from task3_helpers import cfg3, make_task2_run

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "phase5" / "window_generation_default.json"
N_FRAMES = 90


def wcfg(**over) -> WindowConfig:
    d = json.loads(CONFIG.read_text(encoding="utf-8"))
    d.update(over)
    return WindowConfig.from_dict(d)


def scene_records():
    out = []
    for f in range(N_FRAMES):
        out.append(rec(track="1", frame=f, fx=100.0 + 2.0 * f, fy=600.0))
        if f != 30:
            out.append(rec(track="2", frame=f, fx=300.0, fy=500.0 + 1.0 * f, h=70.0))
        if f != 31:
            out.append(rec(track="3", frame=f, fx=500.0 - 1.5 * f, fy=650.0, h=80.0))
        out.append(rec(track="4", frame=f, fx=900.0 + 4.0 * f, fy=800.0, h=150.0, w=240.0, cls="vehicle", source_label="car"))
        if 10 <= f <= 60:
            out.append(rec(track="5", frame=f, fx=1200.0, fy=700.0 + f, h=90.0, w=40.0, cls="bicyclist", source_label="bike"))
        if f <= 40:
            out.append(rec(track="6", frame=f, fx=1400.0 + f, fy=620.0, h=75.0))
        else:
            out.append(rec(track="7", frame=f, fx=1400.0 + f, fy=620.0, h=75.0))
        if f <= 10:
            out.append(rec(track="8", frame=f, fx=1600.0, fy=640.0, h=60.0))
        if f >= 50:
            out.append(rec(track="9", frame=f, fx=200.0 + 3.0 * f, fy=900.0, h=160.0, w=260.0, cls="vehicle", source_label="truck"))
    out.sort(key=lambda r: (r["frame_index"], r["track_id"]))
    return out


def build_scene_task3_run(tmp: Path) -> Path:
    causal, t2m, p3 = make_task2_run(tmp, scene_records(), name="scene")
    out = tmp / "task3_run"
    run_interaction_representation(causal, t2m, p3, cfg3(), out, config_path="test", command="test")
    return out


def track_index(run, label: str) -> int:
    tid = "pie:video_0004:%s" % label
    return next(t["track_index"] for t in run.tracks if t["track_id"] == tid)
