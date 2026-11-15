"""Synthetic fixtures for the Phase 4 Task 3 tests (no PIE data, no model, no video)."""

import hashlib
import json
import math
from pathlib import Path
from typing import List, Optional

from pedsafety.image_kinematics import ImageKinematicsConfig
from pedsafety.image_kinematics_run import run_image_kinematics
from pedsafety.interaction.config import InteractionConfig
from pedsafety.interaction.input_adapter import JoinedRow
from task2_helpers import DEFAULT_CONFIG as T2_CONFIG
from task2_helpers import rec, write_jsonl

REPO = Path(__file__).resolve().parents[3]
T3_CONFIG = REPO / "configs" / "phase4" / "interaction_representation_default.json"
RASTER = "captured_1920x1080"


def cfg3(**over) -> InteractionConfig:
    d = json.loads(T3_CONFIG.read_text(encoding="utf-8"))
    d.update(over)
    return InteractionConfig.from_dict(d)


def jr(track: str, frame: int, u: float, v: float, h: float = 60.0, w: float = 20.0, vel=None, cls: str = "pedestrian",
       border: bool = False, gap: Optional[int] = 0, label: Optional[str] = None, line: int = 1, heading: Optional[float] = None,
       obs: str = "observed", pos: str = "ok") -> JoinedRow:
    """A JoinedRow with footpoint (u, v); vel=(vx, vy) makes velocity/heading valid (heading defaults to atan2)."""
    vx = vy = sp = hd = None
    vs = hs = "insufficient_samples"
    if vel is not None:
        vx, vy = float(vel[0]), float(vel[1])
        sp = math.hypot(vx, vy)
        vs = "valid"
        hd = math.atan2(vy, vx) if heading is None else heading
        hs = "valid" if sp > 0 else "below_min_speed"
        if hs != "valid":
            hd = None
    return JoinedRow(line, "pie:video_0004:%s" % track, "pie:video_0004", frame, frame / 30.0, cls, obs, pos, vs, hs,
                     u if pos == "ok" else None, v if pos == "ok" else None, vx, vy, sp, hd, border, gap, RASTER,
                     (u - w / 2.0, v - h, u + w / 2.0, v), label or {"pedestrian": "pedestrian", "vehicle": "car", "bicyclist": "bike"}[cls])


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_task2_run(tmp: Path, records: List[dict], name: str = "t2", raw_lines: Optional[List[str]] = None):
    """Write a Phase 3 agent_states file (records, or raw lines verbatim) and run the REAL Task 2 runner on it."""
    p3 = tmp / ("%s_agent_states.jsonl" % name)
    if raw_lines is not None:
        p3.write_text("".join(l + "\n" for l in raw_lines), encoding="utf-8")
    else:
        write_jsonl(p3, records)
    out = tmp / ("%s_run" % name)
    run_image_kinematics(p3, ImageKinematicsConfig.load(T2_CONFIG), out)
    return out / "causal_kinematics.jsonl", out / "run_manifest.json", p3


def read_rows(p) -> List[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_rows(p, rows) -> None:
    Path(p).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def resync(manifest: Path, causal: Path, phase3: Path) -> None:
    """Simulate an internally consistent upstream after a deliberate corruption: refresh the Task 2 manifest hashes."""
    m = json.loads(manifest.read_text(encoding="utf-8"))
    m["outputs"]["causal_kinematics.jsonl"]["sha256"] = sha(causal)
    m["outputs"]["causal_kinematics.jsonl"]["rows"] = len(read_rows(causal))
    h = sha(phase3)
    m["input"]["sha256_before"] = m["input"]["sha256_after"] = h
    manifest.write_text(json.dumps(m, indent=2), encoding="utf-8")


def scene_records():
    """Two pedestrians, one car, one bicyclist over 4 frames; the car touches the bottom border."""
    out = []
    for f in range(4):
        out.append(rec(track="1", frame=f, fx=100.0 + 3 * f, fy=500.0))
        out.append(rec(track="2", frame=f, fx=160.0, fy=420.0, h=80.0))
        out.append(rec(track="3", frame=f, fx=600.0 - 6 * f, fy=1080.0, h=200.0, w=300.0, cls="vehicle", source_label="car"))
        out.append(rec(track="4", frame=f, fx=900.0, fy=700.0 + 2 * f, h=90.0, w=40.0, cls="bicyclist", source_label="bike"))
    return out
