import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pedsafety.agent_state import build_agent_states
from pedsafety.calibration import RasterSpec, RasterTransform, VerificationClaim
from pedsafety.geometry import apply_homography
from pedsafety.io.csv_adapter import CsvAdapterConfig, CsvAnnotationAdapter
from pedsafety.projection import (
    KIND_ELIGIBLE, KIND_PROJECTIVE, ProjectionError, ProjectionStatus as S, TransformConfig, transform_agent_states,
)
from pedsafety.schema import collect_violations
from stage3_fixtures import (
    H_TRUE, RASTER, default_points, default_stage2, make_manifest, raw, stage2,
)

PROMOTE = TransformConfig(promote_eligible_to_canonical=True)
WORLD_FIELDS = ("world_x", "world_y", "world_coordinate_system", "world_coordinate_unit")
IMAGE_FIELDS = ("bbox_xyxy", "bbox_center", "footpoint", "image_x", "image_y", "image_coordinate_system",
                "image_coordinate_unit", "detection_confidence")


def statuses(res):
    return [p.status for p in res.projections]


def expected_plane(pts):
    return apply_homography(H_TRUE, pts).xy


# ----------------------------------------------------- Stage 2 baseline
def test_stage2_output_has_null_world_fields_and_is_unchanged_by_stage3_import():
    recs, _ = default_stage2()
    assert all(r["world_position_valid"] is False and all(r[f] is None for f in WORLD_FIELDS) for r in recs)
    assert all(collect_violations(r) == [] for r in recs)


# ------------------------------------------------ 14 projective stays canonical-null
def test_unverified_projective_transform_keeps_canonical_world_fields_null():
    recs, prov = default_stage2()
    manifest = make_manifest(plane_unit=None)                      # relative/projective, no declared scale
    res = transform_agent_states(recs, prov, manifest, RASTER, PROMOTE)       # promotion requested but not eligible
    assert statuses(res) == [S.PROJECTED] * 4 and res.counters["promoted"] == 0
    for out, p, exp in zip(res.records, res.projections, expected_plane(default_points())):
        assert out["world_position_valid"] is False and all(out[f] is None for f in WORLD_FIELDS)
        assert p.coordinate_kind == KIND_PROJECTIVE and not p.promoted_to_canonical and p.manifest_eligible is False
        assert p.plane_xy == pytest.approx(tuple(exp), rel=1e-12) and p.plane_unit is None
        assert p.plane_coordinate_system == "synthetic_plane"
    assert res.preflight.eligibility.manifest_eligible is False


def test_eligible_manifest_without_explicit_promotion_still_keeps_canonical_null():
    recs, prov = default_stage2()
    res = transform_agent_states(recs, prov, make_manifest(), RASTER)               # default config: promote False
    assert res.preflight.eligibility.manifest_eligible
    assert all(r["world_position_valid"] is False and r["world_x"] is None for r in res.records)
    assert all(p.coordinate_kind == KIND_ELIGIBLE and not p.promoted_to_canonical for p in res.projections)
    assert res.counters["promoted"] == 0


# ------------------------------------------------ 15 metric needs evidence
def test_metric_promotion_only_with_full_evidence_and_schema_valid():
    recs, prov = default_stage2()
    res = transform_agent_states(recs, prov, make_manifest(), RASTER, PROMOTE)
    assert res.counters["promoted"] == 4
    for out, p, exp in zip(res.records, res.projections, expected_plane(default_points())):
        assert out["world_position_valid"] is True
        assert (out["world_x"], out["world_y"]) == pytest.approx(tuple(exp), rel=1e-12)
        assert (out["world_coordinate_system"], out["world_coordinate_unit"]) == ("synthetic_plane", "m")
        assert collect_violations(out) == []
        assert p.promoted_to_canonical and p.physical_verification_established_by_software is False
        assert p.eligibility_basis == "software_manifest_checks_only"


@pytest.mark.parametrize("over", [dict(plane_coordinate_system=None), dict(plane_unit=None), dict(control_points=()),
                                  dict(valid_region=None), dict(verification=VerificationClaim()),
                                  dict(holdout_error_threshold=None)])
def test_any_missing_evidence_means_no_canonical_world_values(over):
    recs, prov = default_stage2()
    res = transform_agent_states(recs, prov, make_manifest(**over), RASTER, PROMOTE)
    assert res.counters["promoted"] == 0
    assert all(r["world_position_valid"] is False and all(r[f] is None for f in WORLD_FIELDS) for r in res.records)


# ------------------------------------------- 20-21 originals preserved, schema valid
def test_image_space_data_and_inputs_are_preserved():
    recs, prov = default_stage2()
    snapshot = copy.deepcopy(recs)
    res = transform_agent_states(recs, prov, make_manifest(), RASTER, PROMOTE)
    assert recs == snapshot                                          # inputs untouched
    assert res.records is not recs and res.records[0] is not recs[0]
    for before, after in zip(snapshot, res.records):
        for f in IMAGE_FIELDS + ("track_id", "sequence_id", "scene_id", "source_dataset", "source_label", "timestamp", "frame_index",
                                 "agent_class", "interpolation_flag", "observation_valid"):
            assert before[f] == after[f], f
        changed = {k for k in before if before[k] != after[k]}
        assert changed == set(WORLD_FIELDS) | {"world_position_valid"}
    assert all(collect_violations(r) == [] for r in res.records)


def test_footpoint_source_and_mismatch_tolerance():
    recs, prov = default_stage2()
    res = transform_agent_states(recs, prov, make_manifest(), RASTER, TransformConfig(position_source="footpoint"))
    assert all(p.source_point == "footpoint" for p in res.projections) and statuses(res) == [S.PROJECTED] * 4
    shifted = copy.deepcopy(recs)
    shifted[0]["image_x"] += 5.0
    ok = transform_agent_states(shifted, prov, make_manifest(), RASTER, TransformConfig(footpoint_tolerance_px=10.0))
    bad = transform_agent_states(shifted, prov, make_manifest(), RASTER, TransformConfig(footpoint_tolerance_px=1.0))
    assert statuses(ok)[0] == S.PROJECTED and statuses(bad)[0] == S.FOOTPOINT_MISMATCH
    assert bad.records[0]["world_x"] is None


# -------------------------------------------- 10-11 outside image/region
def test_point_outside_image_rejected_not_clipped():
    recs, prov = stage2([raw("a", 0, 2500.0, 500.0, bbox=None), raw("b", 0, -5.0, 500.0, bbox=None), raw("c", 0, 960.0, 1080.5, bbox=None)])
    res = transform_agent_states(recs, prov, make_manifest(), RASTER, PROMOTE)
    assert statuses(res) == [S.OUTSIDE_IMAGE] * 3
    assert all(p.plane_xy is None and r["world_x"] is None for p, r in zip(res.projections, res.records))
    assert [r["image_x"] for r in res.records] == [2500.0, -5.0, 960.0]            # unchanged


def test_point_outside_valid_region_rejected_inside_still_projected():
    recs, prov = stage2([raw("in", 0, 960.0, 540.0, bbox=None), raw("edge", 0, 100.0, 540.0, bbox=None),
                         raw("out1", 0, 50.0, 540.0, bbox=None), raw("out2", 0, 960.0, 1050.0, bbox=None)])
    res = transform_agent_states(recs, prov, make_manifest(), RASTER, PROMOTE)
    assert statuses(res) == [S.PROJECTED, S.PROJECTED, S.OUTSIDE_CALIBRATION_REGION, S.OUTSIDE_CALIBRATION_REGION]
    assert res.records[2]["world_position_valid"] is False and res.counters["promoted"] == 2


def test_without_a_region_only_the_raster_limits_the_domain():
    recs, prov = stage2([raw("a", 0, 50.0, 50.0, bbox=None)])
    res = transform_agent_states(recs, prov, make_manifest(valid_region=None), RASTER, PROMOTE)
    assert statuses(res) == [S.PROJECTED] and res.counters["promoted"] == 0     # region missing => not eligible


def test_behind_horizon_and_unstable_denominator_flagged():
    H = np.array([[0.02, 0.0, -3.0], [0.0, 0.025, -2.0], [-1e-3, 0.0, 1.0]])        # w = 1 - 1e-3 x (zero at x = 1000)
    m = make_manifest(homography=H, control_points=(), valid_region=((100, 100), (900, 100), (900, 1000), (100, 1000)))
    recs, prov = stage2([raw("a", 0, 500.0, 500.0, bbox=None), raw("b", 0, 1200.0, 500.0, bbox=None), raw("c", 0, 1000.0, 500.0, bbox=None)])
    res = transform_agent_states(recs, prov, dataclasses.replace(m, valid_region=None), RASTER)
    assert statuses(res) == [S.PROJECTED, S.BEHIND_HORIZON, S.DENOMINATOR_UNSTABLE]


# ------------------------------------------------- 12-13 raster handling
def test_different_observed_raster_without_documented_transform_is_refused():
    recs, prov = default_stage2()
    other = dataclasses.replace(RASTER, width=960, height=540)
    res = transform_agent_states(recs, prov, make_manifest(), other, PROMOTE)
    assert statuses(res) == [S.RASTER_MISMATCH] * 4 and "no documented RasterTransform" in res.projections[0].reason
    assert all(p.plane_xy is None for p in res.projections) and res.preflight.error
    assert all(r["world_x"] is None for r in res.records)


def test_resized_raster_with_documented_transform_projects_equivalently():
    obs_pts = [(150.0, 100.0), (480.0, 270.0), (750.0, 450.0)]
    recs, prov = stage2([raw("t%d" % i, 0, x, y, bbox=None, cs="small_image") for i, (x, y) in enumerate(obs_pts)])
    small = RasterSpec("small", 960, 540, "small_image", "pixel_center_half", "as_captured")
    res = transform_agent_states(recs, prov, make_manifest(), small, PROMOTE, RasterTransform(0.5, 0.5))
    assert statuses(res) == [S.PROJECTED] * 3 and all(p.raster_adapted for p in res.projections)
    exp = expected_plane([(2 * x, 2 * y) for x, y in obs_pts])
    assert np.allclose([p.plane_xy for p in res.projections], exp, rtol=1e-12)
    assert res.preflight.effective_homography_sha256 != transform_agent_states(*default_stage2(), make_manifest(), RASTER).preflight.effective_homography_sha256


def test_wrong_resize_metadata_refused():
    recs, prov = stage2([raw("a", 0, 480.0, 270.0, bbox=None, cs="small_image")])
    small = RasterSpec("small", 960, 540, "small_image", "pixel_center_half", "as_captured")
    res = transform_agent_states(recs, prov, make_manifest(), small, PROMOTE, RasterTransform(0.25, 0.25))
    assert statuses(res) == [S.RASTER_MISMATCH] and "predicts" in res.projections[0].reason


def test_record_coordinate_system_must_match_observed_raster():
    recs, prov = stage2([raw("a", 0, 500.0, 500.0, bbox=None, cs="some_other_image")])
    res = transform_agent_states(recs, prov, make_manifest(), RASTER, PROMOTE)
    assert statuses(res) == [S.COORDINATE_SYSTEM_MISMATCH]


def test_padding_area_is_outside_the_calibrated_region():
    t = RasterTransform(0.5, 0.5, pad_left=20, pad_top=10, pad_right=20, pad_bottom=10)
    padded = RasterSpec("padded", 1000, 560, "pad_image", "pixel_center_half", "as_captured")
    recs, prov = stage2([raw("pad", 0, 5.0, 5.0, bbox=None, cs="pad_image"), raw("img", 0, 520.0, 290.0, bbox=None, cs="pad_image")])
    res = transform_agent_states(recs, prov, make_manifest(), padded, PROMOTE, t)
    assert statuses(res) == [S.OUTSIDE_CALIBRATION_REGION, S.PROJECTED]


# ---------------------------------------------- skipped / not applicable
def test_non_image_semantics_unverified_and_world_declared_are_skipped_and_untouched():
    obs = [raw("u", 0, 49.9, 41.1, sem="unverified_planar", unit=None, cs="hbs_local"),
           raw("w", 0, 24.7, 11.2, sem="world_coordinate", unit="m", cs="site_map")]
    recs, prov = stage2(obs)
    res = transform_agent_states(recs, prov, make_manifest(), RASTER, PROMOTE)
    assert statuses(res) == [S.SKIPPED_NOT_IMAGE_SPACE] * 2
    assert res.records == recs and all(r["world_position_valid"] is False for r in res.records)
    assert prov[0].withheld_position_xy == (49.9, 41.1)             # Stage 2 sidecar untouched and not used as calibration


def test_record_without_image_position():
    recs, prov = stage2([raw("n", 0, None, None, bbox=None, pos_x=None, pos_y=None)])
    res = transform_agent_states(recs, prov, make_manifest(), RASTER, PROMOTE)
    assert statuses(res) == [S.NO_IMAGE_POSITION]


def test_no_manifest_leaves_everything_unprojected():
    recs, prov = default_stage2()
    res = transform_agent_states(recs, prov, None, RASTER, PROMOTE)
    assert statuses(res) == [S.NO_CALIBRATION] * 4 and res.preflight is None
    assert res.records == recs and all(p.calibration_id is None for p in res.projections)


# ----------------------------------------------------------- modes / contracts
def test_strict_mode_raises_on_failed_projection_but_not_on_skips():
    recs, prov = stage2([raw("a", 0, 2500.0, 500.0, bbox=None)])
    with pytest.raises(ProjectionError) as ei:
        transform_agent_states(recs, prov, make_manifest(), RASTER, TransformConfig(strict=True))
    assert "outside_image" in str(ei.value) and ei.value.result.projections[0].status == S.OUTSIDE_IMAGE
    rs, ps = stage2([raw("u", 0, 1.0, 1.0, sem="unverified_planar", unit=None, cs="x")])
    transform_agent_states(rs, ps, make_manifest(), RASTER, TransformConfig(strict=True))
    transform_agent_states(rs, ps, None, RASTER, TransformConfig(strict=True))


def test_parallel_input_contract():
    recs, prov = stage2([raw("a", 0, 500.0, 500.0, bbox=None), raw("a", 1, 510.0, 500.0, bbox=None)])
    with pytest.raises(ValueError, match="parallel"):
        transform_agent_states(recs, prov[:-1], make_manifest(), RASTER)
    with pytest.raises(ValueError, match="does not describe"):
        transform_agent_states(recs, list(reversed(prov)), make_manifest(), RASTER)


def test_transform_config_validation():
    from pedsafety.errors import ConfigError
    with pytest.raises(ConfigError):
        TransformConfig(position_source="center")
    with pytest.raises(ConfigError):
        TransformConfig(footpoint_tolerance_px=-1.0)


def test_empty_input():
    res = transform_agent_states([], [], make_manifest(), RASTER, PROMOTE)
    assert res.records == [] and res.projections == [] and res.counters == {"promoted": 0}


# ------------------------------------------------- 22 determinism + provenance
def test_results_are_deterministic_across_runs():
    recs, prov = default_stage2()
    a = transform_agent_states(recs, prov, make_manifest(), RASTER, PROMOTE)
    b = transform_agent_states(recs, prov, make_manifest(), RASTER, PROMOTE)
    assert a.records == b.records and a.projections == b.projections and a.counters == b.counters and a.preflight == b.preflight


def test_calibration_provenance_is_attached_to_every_projection():
    recs, prov = default_stage2()
    m = make_manifest()
    res = transform_agent_states(recs, prov, m, RASTER, PROMOTE)
    for p in res.projections:
        assert (p.calibration_id, p.calibration_version, p.manifest_sha256) == ("synthetic-cal", "v1", m.sha256())
        assert (p.calibration_raster_id, p.observed_raster_id, p.projection_version.startswith("projection-")) == \
            ("synthetic_raster_A", "synthetic_raster_A", True)
        assert p.effective_homography_sha256 == res.preflight.effective_homography_sha256
        assert p.image_xy_used is not None and p.source_point == "image_xy"
    other = transform_agent_states(recs, prov, make_manifest(homography=H_TRUE * 1.0000001, control_points=()), RASTER)
    assert other.projections[0].manifest_sha256 != res.projections[0].manifest_sha256


def test_numerical_stability_near_raster_corners_and_batch_size():
    pts = [(0.0, 0.0), (1920.0, 1080.0), (1920.0, 0.0), (0.0, 1080.0), (960.0, 540.0)]
    recs, prov = stage2([raw("t%d" % i, 0, x, y, bbox=None) for i, (x, y) in enumerate(pts)])
    res = transform_agent_states(recs, prov, make_manifest(valid_region=None), RASTER)
    assert statuses(res) == [S.PROJECTED] * 5
    assert np.allclose([p.plane_xy for p in res.projections], expected_plane(pts), rtol=1e-13, atol=1e-13)


# --------------------------------------------------- HBS read-only smoke test
HBS = Path(__file__).resolve().parents[2] / "Data" / "HBS" / "hbs.csv"


@pytest.mark.skipif(not HBS.is_file(), reason="local HBS file not present")
def test_hbs_declarations_stay_unverified_and_read_only(tmp_path):
    before = hashlib.sha256(HBS.read_bytes()).hexdigest()
    lines = HBS.read_text(encoding="utf-8").splitlines()[:601]           # header + first 600 rows, copied; source untouched
    sample = tmp_path / "hbs_sample.csv"
    sample.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = CsvAdapterConfig("HBS", {"track_id": "agent_id", "frame_index": "frame_id", "timestamp": "timestamp", "label": "label",
                                   "x": "pos_x", "y": "pos_y", "vel_x": "vel_x", "vel_y": "vel_y", "scene_id": "scene_id"},
                           sequence_id="hbs_seq", position_semantics="unverified_planar", position_unit=None,
                           position_coordinate_system="hbs_unverified_local")
    parsed = CsvAnnotationAdapter(sample, cfg).parse()
    assert parsed.issues == [] and len(parsed.observations) == 600
    built = build_agent_states(parsed.observations)
    assert built.report.n_records > 0 and built.report.ok, built.report.summary()
    res = transform_agent_states(built.records, built.provenance, make_manifest(), RASTER, PROMOTE)
    assert set(statuses(res)) == {S.SKIPPED_NOT_IMAGE_SPACE} and res.counters["promoted"] == 0
    assert all(r["world_position_valid"] is False and r["world_x"] is None and r["image_x"] is None for r in res.records)
    assert all(p.position_unit is None and p.position_semantics == "unverified_planar" for p in built.provenance)
    assert built.report.counters["unverified_planar_position_withheld"] == len(built.records)
    assert hashlib.sha256(HBS.read_bytes()).hexdigest() == before
