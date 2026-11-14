"""Synthetic Stage 3 fixtures. Every number here is invented for testing and
describes no real camera, road or dataset. The 'plane' is a made-up frame; the
unit 'm' appears only so the eligibility logic has something to evaluate."""

import numpy as np

from pedsafety.agent_state import build_agent_states
from pedsafety.calibration import (
    CalibrationManifest, CalibrationProvenance, ControlPoint, RasterSpec, VerificationClaim,
)
from pedsafety.geometry import apply_homography
from pedsafety.io.base import RawObservation

H_TRUE = np.array([[0.02, 0.001, -3.0], [0.0005, 0.025, -2.0], [1e-5, 2e-5, 1.0]])
RASTER = RasterSpec("synthetic_raster_A", 1920, 1080, "synthetic_camera_image", "pixel_center_half", "as_captured")
REGION = ((100.0, 100.0), (1800.0, 100.0), (1800.0, 1000.0), (100.0, 1000.0))
FIT_IMG = [(300.0, 200.0), (1600.0, 220.0), (1500.0, 900.0), (350.0, 880.0), (960.0, 540.0), (700.0, 400.0)]
HOLD_IMG = [(500.0, 300.0), (1200.0, 700.0), (900.0, 850.0)]


def plane_of(pts, H=H_TRUE):
    return apply_homography(H, pts).xy


def control_points(fit=FIT_IMG, hold=HOLD_IMG, source="survey", H=H_TRUE):
    cps = []
    for role, pts in (("fit", fit), ("holdout", hold)):
        for k, (img, pl) in enumerate(zip(pts, plane_of(pts, H))):
            cps.append(ControlPoint("%s%d" % (role, k), img, (float(pl[0]), float(pl[1])), role, source))
    return tuple(cps)


def make_manifest(**over):
    kw = dict(
        provenance=CalibrationProvenance("synthetic-cal", "v1", "synthetic test calibration", "test-author", "2026-11-12"),
        raster=RASTER, homography=H_TRUE, plane_coordinate_system="synthetic_plane", plane_unit="m",
        plane_origin="synthetic origin", plane_axes="X right, Y forward", handedness="right",
        scale_basis="synthetic known points", valid_region=REGION, control_points=control_points(),
        holdout_error_threshold=1e-6,
        verification=VerificationClaim("independently_verified", "synthetic-verifier", "2026-11-12", "synthetic-ref-001"),
    )
    kw.update(over)
    return CalibrationManifest(**kw)


def raw(track="t1", frame=0, x=960.0, y=540.0, sem="image_pixel", unit="pixel", cs="synthetic_camera_image",
        bbox="auto", label="pedestrian", **kw):
    if bbox == "auto":
        bbox = (x - 10.0, y - 50.0, x + 10.0, y) if sem == "image_pixel" else None
    base = dict(source_dataset="SYN", sequence_id="s1", scene_id="sc1", track_id=track, frame_index=frame,
                timestamp=frame / 10.0, source_label=label, bbox_xyxy=bbox, pos_x=x, pos_y=y,
                position_semantics=sem, position_unit=unit, position_coordinate_system=cs, source_line=frame + 2)
    base.update(kw)
    return RawObservation(**base)


def stage2(observations):
    res = build_agent_states(observations)
    assert res.report.ok, res.report.summary()
    return res.records, res.provenance


def default_points():
    return [(300.0, 200.0), (960.0, 540.0), (1500.0, 900.0), (500.0, 300.0)]


def default_stage2():
    return stage2([raw("t%d" % i, 0, x, y) for i, (x, y) in enumerate(default_points())])
