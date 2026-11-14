"""Deterministic synthetic scene generator for tests.

All positions are in IMAGE PIXELS (``position_semantics='image_pixel'``,
unit ``pixel``); velocity is pixel/s and time is seconds (frame/fps). The
synthetic data says nothing about any real dataset or any metric scale.

Geometry: ``pos_x, pos_y`` is the bottom-centre footpoint of a box of fixed
size, ``bbox = (x - w/2, y - h, x + w/2, y)``. Motion is analytic:
``p(t) = p0 + v t + a t^2 / 2`` with ``t = frame / fps``. Optional Gaussian
noise on the footpoint uses ``numpy.random.default_rng(seed)``; with
``position_noise_px == 0`` the output is exactly the analytic truth.
"""

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np

from .base import RawObservation
from .csv_adapter import CsvAdapterConfig

SYNTHETIC_COLUMNS = {
    "sequence_id": "sequence", "scene_id": "scene", "track_id": "track",
    "frame_index": "frame", "timestamp": "t_s", "label": "cls",
    "x": "foot_x", "y": "foot_y",
    "bbox_x_min": "x0", "bbox_y_min": "y0", "bbox_x_max": "x1", "bbox_y_max": "y1",
    "vel_x": "vx", "vel_y": "vy", "confidence": "conf",
}
_CSV_ORDER = ["sequence", "scene", "track", "frame", "t_s", "cls", "foot_x", "foot_y",
              "x0", "y0", "x1", "y1", "vx", "vy", "conf"]


@dataclass(frozen=True)
class AgentSpec:
    track_id: str
    label: str
    start_xy: Tuple[float, float]
    velocity: Tuple[float, float]               # pixel/s
    acceleration: Tuple[float, float] = (0.0, 0.0)  # pixel/s^2
    box_size: Tuple[float, float] = (20.0, 50.0)    # (w, h) pixels
    first_frame: int = 0
    last_frame: Optional[int] = None            # inclusive; None = to the end
    missing_frames: FrozenSet[int] = frozenset()


@dataclass
class SyntheticScene:
    observations: List[RawObservation]
    truth: Dict[Tuple[str, int], Dict[str, float]]  # (track_id, frame) -> x,y,vx,vy
    fps: float
    seed: int
    sequence_id: str
    scene_id: str
    specs: Tuple[AgentSpec, ...]
    position_noise_px: float


SYNTHETIC_COORDINATE_SYSTEM = "synthetic_image"


def _finite(name: str, value) -> float:
    """Return ``value`` as float, rejecting non-numeric, bool, NaN and inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a finite number, got %r" % (name, value))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite, got %r" % (name, value))
    return value


def default_specs() -> Tuple[AgentSpec, ...]:
    return (
        AgentSpec("ped_const", "pedestrian", (100.0, 300.0), (30.0, 0.0)),
        AgentSpec("ped_accel", "pedestrian", (400.0, 350.0), (0.0, -10.0), (5.0, 2.0)),
        AgentSpec("veh_const", "car", (50.0, 500.0), (120.0, 0.0), box_size=(90.0, 60.0)),
        AgentSpec("ped_gap", "pedestrian", (200.0, 250.0), (-20.0, 5.0), missing_frames=frozenset(range(10, 14))),
        AgentSpec("ped_late_early", "pedestrian", (300.0, 200.0), (10.0, 10.0), first_frame=5, last_frame=24),
    )


def generate_scene(
    *,
    seed: int = 0,
    n_frames: int = 40,
    fps: float = 10.0,
    sequence_id: str = "syn_seq_000",
    scene_id: str = "syn_scene_000",
    specs: Optional[Tuple[AgentSpec, ...]] = None,
    position_noise_px: float = 0.0,
    source_dataset: str = "SYNTHETIC",
    confidence: float = 0.9,
) -> SyntheticScene:
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 1:
        raise ValueError("n_frames must be an integer >= 1, got %r" % (n_frames,))
    fps = _finite("fps", fps)
    position_noise_px = _finite("position_noise_px", position_noise_px)
    confidence = _finite("confidence", confidence)
    if fps <= 0:
        raise ValueError("fps must be > 0, got %r" % (fps,))
    if position_noise_px < 0:
        raise ValueError("position_noise_px must be >= 0, got %r" % (position_noise_px,))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1], got %r" % (confidence,))
    specs = tuple(specs) if specs is not None else default_specs()
    rng = np.random.default_rng(seed)
    observations: List[RawObservation] = []
    truth: Dict[Tuple[str, int], Dict[str, float]] = {}
    for spec in specs:
        last = n_frames - 1 if spec.last_frame is None else min(spec.last_frame, n_frames - 1)
        for frame in range(spec.first_frame, last + 1):
            t = frame / fps
            x = spec.start_xy[0] + spec.velocity[0] * t + 0.5 * spec.acceleration[0] * t * t
            y = spec.start_xy[1] + spec.velocity[1] * t + 0.5 * spec.acceleration[1] * t * t
            vx = spec.velocity[0] + spec.acceleration[0] * t
            vy = spec.velocity[1] + spec.acceleration[1] * t
            # Draw noise for every frame (even missing ones) so that the
            # random stream does not depend on which frames are dropped.
            nx, ny = (rng.normal(0.0, position_noise_px, 2) if position_noise_px > 0 else (0.0, 0.0))
            if frame in spec.missing_frames:
                continue
            truth[(spec.track_id, frame)] = {"x": x, "y": y, "vx": vx, "vy": vy}
            ox, oy = float(x + nx), float(y + ny)
            w, h = spec.box_size
            observations.append(
                RawObservation(
                    source_dataset=source_dataset,
                    sequence_id=sequence_id,
                    scene_id=scene_id,
                    track_id=spec.track_id,
                    frame_index=frame,
                    timestamp=t,
                    source_label=spec.label,
                    bbox_xyxy=(ox - w / 2, oy - h, ox + w / 2, oy),
                    pos_x=ox,
                    pos_y=oy,
                    position_semantics="image_pixel",
                    position_unit="pixel",
                    position_coordinate_system=SYNTHETIC_COORDINATE_SYSTEM,
                    vel_x=float(vx),
                    vel_y=float(vy),
                    detection_confidence=confidence,
                    source_line=0,
                    extras=(),
                )
            )
    observations.sort(key=lambda o: (o.frame_index, o.track_id))
    return SyntheticScene(observations, truth, fps, seed, sequence_id, scene_id, specs, position_noise_px)


def write_scene_csv(scene: SyntheticScene, path: Union[str, Path]) -> Path:
    """Write the scene using ``repr`` floats so a parse round-trips exactly."""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_ORDER)
        for o in scene.observations:
            b = o.bbox_xyxy
            w.writerow([o.sequence_id, o.scene_id, o.track_id, o.frame_index, repr(o.timestamp),
                        o.source_label, repr(o.pos_x), repr(o.pos_y),
                        repr(b[0]), repr(b[1]), repr(b[2]), repr(b[3]),
                        repr(o.vel_x), repr(o.vel_y), repr(o.detection_confidence)])
    return path


def synthetic_csv_config(source_dataset: str = "SYNTHETIC", strict: bool = False) -> CsvAdapterConfig:
    return CsvAdapterConfig(
        source_dataset=source_dataset,
        columns=dict(SYNTHETIC_COLUMNS),
        position_semantics="image_pixel",
        position_unit="pixel",
        position_coordinate_system=SYNTHETIC_COORDINATE_SYSTEM,
        strict=strict,
    )
