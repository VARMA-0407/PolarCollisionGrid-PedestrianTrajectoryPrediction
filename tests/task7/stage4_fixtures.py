"""Synthetic Stage 4 fixtures: analytic tracks pushed through the real Stage 2
(and optionally Stage 3) code. All numbers are invented."""

import copy

import numpy as np

from pedsafety.agent_state import build_agent_states
from pedsafety.io.base import RawObservation
from pedsafety.projection import TransformConfig, transform_agent_states
from stage3_fixtures import RASTER, make_manifest

FPS = 10.0


def make_track(track="t1", frames=range(10), x0=300.0, y0=400.0, vx=30.0, vy=-10.0, ax=0.0, ay=0.0, fps=FPS, label="pedestrian",
               cs="synthetic_camera_image", seq="s1", scene="sc1", dataset="SYN", times=None, drop=()):
    """Analytic image-space footpoint track: p(t) = p0 + v t + a t^2 / 2, t = frame / fps (or ``times``)."""
    out = []
    for k, f in enumerate(frames):
        if f in drop:
            continue
        t = times[k] if times is not None else f / fps
        x = x0 + vx * t + 0.5 * ax * t * t
        y = y0 + vy * t + 0.5 * ay * t * t
        out.append(RawObservation(
            source_dataset=dataset, sequence_id=seq, scene_id=scene, track_id=track, frame_index=f, timestamp=t, source_label=label,
            bbox_xyxy=(x - 10.0, y - 50.0, x + 10.0, y), pos_x=x, pos_y=y, position_semantics="image_pixel", position_unit="pixel",
            position_coordinate_system=cs, source_line=f + 2))
    return out


def stage2(observations):
    res = build_agent_states(observations)
    assert res.report.ok, res.report.summary()
    return res.records, res.provenance


def stage3(records, provenance, manifest=None, promote=False, raster=RASTER):
    return transform_agent_states(records, provenance, manifest if manifest is not None else make_manifest(), raster,
                                  TransformConfig(promote_eligible_to_canonical=promote))


def snapshot(records, provenance, projections=None):
    return copy.deepcopy(records), list(provenance), None if projections is None else list(projections)


def col(traj, name):
    return np.asarray(traj.arrays[name])


def same(a, b):
    return np.array_equal(a, b, equal_nan=np.asarray(a).dtype.kind == "f")
