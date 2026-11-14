import dataclasses
import json

import numpy as np
import pytest

from pedsafety.calibration import (
    CalibrationManifest, CalibrationProvenance, ControlPoint, EligibilityPolicy, RasterSpec, RasterTransform,
    VerificationClaim, evaluate_metric_eligibility, reference_sign, resolve_effective_calibration,
)
from pedsafety.geometry import CalibrationError, NumericTolerances, apply_homography
from stage3_fixtures import FIT_IMG, H_TRUE, HOLD_IMG, RASTER, REGION, control_points, make_manifest, plane_of


def code_of(fn, *a, **k):
    with pytest.raises(CalibrationError) as ei:
        fn(*a, **k)
    return ei.value.code


def failed(res):
    return {n for n, ok in res.checks if not ok}


# ------------------------------------------------------------- RasterSpec
def test_raster_domain_depends_on_pixel_convention():
    assert RASTER.domain() == (0.0, 1920.0, 0.0, 1080.0)
    r = dataclasses.replace(RASTER, pixel_convention="pixel_center_integer")
    assert r.domain() == (-0.5, 1919.5, -0.5, 1079.5)
    assert RASTER.contains([[0, 0], [1920, 1080], [1920.1, 5], [-0.1, 5], [np.nan, 1]]).tolist() == [True, True, False, False, False]


@pytest.mark.parametrize("kw", [dict(width=0), dict(height=-1), dict(width=1920.0), dict(width=True), dict(raster_id=""),
                                dict(coordinate_system=" "), dict(pixel_convention="corner"), dict(pixel_domain="rectified"),
                                dict(origin="bottom_left")])
def test_invalid_raster_spec(kw):
    with pytest.raises(CalibrationError):
        dataclasses.replace(RASTER, **kw)


# ------------------------------------------------ RasterTransform / adaptation
def test_transform_matrix_known_values_both_conventions():
    t = RasterTransform(0.5, 0.5)
    assert np.allclose(t.matrix("pixel_center_half") @ [100, 50, 1], [50, 25, 1])
    # integer-centre: pixel 0 centre (0,0) -> edge 0.5 -> 0.25 -> centre -0.25
    assert np.allclose(t.matrix("pixel_center_integer") @ [0, 0, 1], [-0.25, -0.25, 1])
    tc = RasterTransform(1.0, 1.0, crop_x=100, crop_y=50, pad_left=3, pad_top=4)
    assert np.allclose(tc.matrix("pixel_center_half") @ [100, 50, 1], [3, 4, 1])


def test_expected_size_with_crop_scale_pad():
    t = RasterTransform(0.5, 0.5, crop_x=0, crop_y=0, crop_width=1920, crop_height=1080, pad_left=10, pad_right=10)
    assert t.expected_size(RASTER) == (980.0, 540.0)
    assert code_of(RasterTransform(1, 1, crop_x=1000, crop_width=1000).expected_size, RASTER) == "BAD_CROP"


@pytest.mark.parametrize("kw", [dict(scale_x=0, scale_y=1), dict(scale_x=1, scale_y=float("nan")), dict(scale_x=1, scale_y=1, crop_x=-1),
                                dict(scale_x=1, scale_y=1, pad_left=float("inf")), dict(scale_x=1, scale_y=1, crop_width=0)])
def test_invalid_raster_transform(kw):
    with pytest.raises(CalibrationError):
        RasterTransform(**kw)


def test_adapted_homography_equals_original_on_mapped_points():
    m = make_manifest()
    t = RasterTransform(0.5, 0.5, crop_x=100, crop_y=50, crop_width=1700, crop_height=900, pad_left=4, pad_top=6)
    obs = RasterSpec("obs", 854, 906, "obs_image", "pixel_center_half", "as_captured")   # 1700*.5+4 , 900*.5+6 (+pad)
    obs = dataclasses.replace(obs, width=854, height=456)
    eff = resolve_effective_calibration(m, obs, t)
    assert eff.adapted and eff.to_calibration_raster is not None
    pts = np.array([[300.0, 200.0], [960.0, 540.0], [1500.0, 900.0]])
    A = t.matrix("pixel_center_half")
    mapped = (A @ np.vstack([pts.T, np.ones(3)]))[:2].T
    assert np.allclose(apply_homography(eff.homography, mapped).xy, apply_homography(H_TRUE, pts).xy, rtol=1e-12, atol=1e-12)


def test_integer_convention_adaptation_roundtrip():
    cal = dataclasses.replace(RASTER, pixel_convention="pixel_center_integer")
    reg = tuple((x, y) for x, y in REGION)
    m = make_manifest(raster=cal, valid_region=reg, control_points=control_points())
    t = RasterTransform(0.5, 0.5)
    obs = RasterSpec("o", 960, 540, "o_img", "pixel_center_integer", "as_captured")
    eff = resolve_effective_calibration(m, obs, t)
    p = np.array([[301.0, 201.0]])
    q = (t.matrix("pixel_center_integer") @ [301.0, 201.0, 1.0])[:2]
    assert np.allclose(apply_homography(eff.homography, [q]).xy, apply_homography(H_TRUE, p).xy, rtol=1e-12)


def test_identical_raster_needs_no_transform_and_not_adapted():
    eff = resolve_effective_calibration(make_manifest(), RASTER)
    assert not eff.adapted and eff.to_calibration_raster is None and np.array_equal(eff.homography, H_TRUE)


# ---------------------------- 12-13 raster mismatches
@pytest.mark.parametrize("kw", [dict(width=1280), dict(height=720), dict(raster_id="other"), dict(coordinate_system="other_cs"),
                                dict(pixel_domain="undistorted"), dict(pixel_convention="pixel_center_integer")])
def test_raster_mismatch_without_transform_rejected(kw):
    obs = dataclasses.replace(RASTER, **kw)
    assert code_of(resolve_effective_calibration, make_manifest(), obs) == "RASTER_MISMATCH"


def test_resize_metadata_must_match_observed_size():
    m = make_manifest()
    obs = RasterSpec("o", 960, 540, "o_img", "pixel_center_half", "as_captured")
    resolve_effective_calibration(m, obs, RasterTransform(0.5, 0.5))                       # consistent: ok
    assert code_of(resolve_effective_calibration, m, obs, RasterTransform(0.25, 0.25)) == "RASTER_MISMATCH"
    assert code_of(resolve_effective_calibration, m, dataclasses.replace(obs, width=962), RasterTransform(0.5, 0.5)) == "RASTER_MISMATCH"
    resolve_effective_calibration(m, dataclasses.replace(obs, width=961), RasterTransform(0.5, 0.5))   # within rounding
    assert code_of(resolve_effective_calibration, m, dataclasses.replace(obs, width=961), RasterTransform(0.5, 0.5),
                   NumericTolerances(size_rounding_px=0.0)) == "RASTER_MISMATCH"


def test_crop_metadata_mismatch_and_domain_change_not_fixable_by_resize():
    m = make_manifest()
    obs = RasterSpec("o", 960, 540, "o_img", "pixel_center_half", "as_captured")
    assert code_of(resolve_effective_calibration, m, obs, RasterTransform(0.5, 0.5, crop_x=100)) == "RASTER_MISMATCH"
    assert code_of(resolve_effective_calibration, m, dataclasses.replace(obs, pixel_domain="undistorted"),
                   RasterTransform(0.5, 0.5)) == "RASTER_MISMATCH"
    assert code_of(resolve_effective_calibration, m, dataclasses.replace(obs, pixel_convention="pixel_center_integer"),
                   RasterTransform(0.5, 0.5)) == "RASTER_MISMATCH"


# ------------------------------------------------------ manifest construction
def test_valid_manifest_builds_and_homography_is_immutable_tuple():
    m = make_manifest()
    assert isinstance(m.homography, tuple) and np.array_equal(m.H, H_TRUE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.plane_unit = "cm"


@pytest.mark.parametrize("kw,code", [
    (dict(homography=np.zeros((3, 3))), "SINGULAR"),
    (dict(homography=np.eye(2)), "BAD_SHAPE"),
    (dict(homography=np.full((3, 3), np.nan)), "NON_FINITE"),
    (dict(direction="plane_to_image"), "BAD_DIRECTION"),
    (dict(direction=None), "BAD_DIRECTION"),
    (dict(plane_unit="furlong"), "BAD_UNIT"),
    (dict(handedness="up"), "BAD_HANDEDNESS"),
    (dict(holdout_error_threshold=0.0), "BAD_THRESHOLD"),
    (dict(holdout_error_threshold=float("nan")), "BAD_THRESHOLD"),
    (dict(valid_region=((0, 0), (10, 10))), "BAD_REGION"),
    (dict(valid_region=((0, 0), (2000, 0), (2000, 500), (0, 500))), "BAD_REGION"),                     # beyond raster
    (dict(valid_region=((100, 100), (800, 800), (800, 100), (100, 800))), "BAD_REGION"),               # bow-tie
    (dict(valid_region=((100, 100), (200, 100), (300, 100))), "BAD_REGION"),                           # zero area
])
def test_invalid_manifest_rejected(kw, code):
    assert code_of(make_manifest, **kw) == code


def test_region_crossing_the_vanishing_line_rejected():
    H = np.array([[0.02, 0.0, -3.0], [0.0, 0.025, -2.0], [-1e-3, 0.0, 1.0]])    # w = 1 - 1e-3 x : zero at x = 1000
    assert code_of(make_manifest, homography=H, control_points=(), valid_region=((100, 100), (1800, 100), (1800, 1000), (100, 1000))) \
        == "REGION_CROSSES_HORIZON"


def test_agent_trajectory_cannot_be_calibration_ground_truth():
    for src in ("agent_trajectory", "tracker", "detector", ""):
        assert code_of(ControlPoint, "p", (1, 1), (2, 2), "fit", src) == "BAD_CONTROL_POINT_SOURCE"
    assert "trajectories" in str(pytest.raises(CalibrationError, ControlPoint, "p", (1, 1), (2, 2), "fit", "agent_trajectory").value)


@pytest.mark.parametrize("args", [("", (1, 1), (2, 2), "fit", "survey"), ("p", (1, np.nan), (2, 2), "fit", "survey"),
                                  ("p", (1, 1), (2,), "fit", "survey"), ("p", (1, 1), (2, 2), "test", "survey")])
def test_invalid_control_points(args):
    with pytest.raises(CalibrationError):
        ControlPoint(*args)


def test_duplicate_ids_and_out_of_raster_control_points_rejected():
    cps = control_points()
    assert code_of(make_manifest, control_points=cps + (cps[0],)) == "BAD_CONTROL_POINT"
    assert code_of(make_manifest, control_points=cps + (ControlPoint("far", (5000, 5), (1, 1), "fit", "survey"),)) == "BAD_CONTROL_POINT"


def test_manifest_dict_roundtrip_hash_and_strictness():
    m = make_manifest()
    d = json.loads(json.dumps(m.to_dict()))
    m2 = CalibrationManifest.from_dict(d)
    assert m2 == m and m2.sha256() == m.sha256() and len(m.sha256()) == 64
    assert make_manifest(homography=H_TRUE * 1.0000001).sha256() != m.sha256()
    assert code_of(CalibrationManifest.from_dict, {**d, "surprise": 1}) == "BAD_MANIFEST"
    assert code_of(CalibrationManifest.from_dict, {k: v for k, v in d.items() if k != "raster"}) == "BAD_MANIFEST"
    assert code_of(CalibrationManifest.from_dict, {**d, "provenance": {"calibration_id": "x"}}) == "BAD_MANIFEST"
    assert code_of(CalibrationManifest.from_dict, [1]) == "BAD_MANIFEST"


def test_loading_a_manifest_does_not_make_it_eligible():
    bare = CalibrationManifest.from_dict({"provenance": {"calibration_id": "c", "version": "1"},
                                          "raster": RASTER.to_dict(), "homography": H_TRUE.tolist()})
    res = evaluate_metric_eligibility(bare)
    assert not res.manifest_eligible and len(res.reasons) >= 8


def test_reference_sign_positive_for_fixture():
    assert reference_sign(make_manifest()) == 1


# ------------------------------------------------------------- eligibility
def test_fully_evidenced_manifest_is_eligible_but_not_physically_verified_by_software():
    res = evaluate_metric_eligibility(make_manifest())
    assert res.manifest_eligible and res.reasons == () and all(ok for _, ok in res.checks)
    assert res.basis == "software_manifest_checks_only" and res.physical_verification_established_by_software is False
    assert (res.n_fit, res.n_holdout, res.error_unit) == (6, 3, "m")
    assert res.holdout_max_error < 1e-6 and res.fit_max_error < 1e-6
    assert json.dumps(res.to_dict())


def test_homography_alone_is_not_metric_eligible_and_all_reasons_reported():
    bare = make_manifest(plane_coordinate_system=None, plane_unit=None, plane_origin=None, plane_axes=None, handedness=None,
                         scale_basis=None, valid_region=None, control_points=(), holdout_error_threshold=None,
                         verification=VerificationClaim())
    res = evaluate_metric_eligibility(bare)
    assert not res.manifest_eligible
    assert {"named_destination_frame", "physical_unit_declared", "frame_definition_documented", "scale_basis_documented",
            "valid_region_documented", "fit_correspondences_sufficient_and_nondegenerate", "holdout_present",
            "error_threshold_declared", "independent_verification_claimed"} <= failed(res)
    assert res.holdout_max_error is None


def test_missing_destination_frame_blocks_eligibility():
    assert "named_destination_frame" in failed(evaluate_metric_eligibility(make_manifest(plane_coordinate_system=None)))
    assert "named_destination_frame" in failed(evaluate_metric_eligibility(make_manifest(plane_coordinate_system="  ")))


def test_missing_physical_unit_blocks_eligibility():
    res = evaluate_metric_eligibility(make_manifest(plane_unit=None))
    assert not res.manifest_eligible and "physical_unit_declared" in failed(res)


@pytest.mark.parametrize("over,check", [
    (dict(plane_origin=None), "frame_definition_documented"),
    (dict(plane_axes=""), "frame_definition_documented"),
    (dict(handedness=None), "frame_definition_documented"),
    (dict(scale_basis=None), "scale_basis_documented"),
    (dict(valid_region=None), "valid_region_documented"),
    (dict(raster=dataclasses.replace(RASTER, pixel_domain="unknown")), "raster_pixel_domain_known"),
    (dict(holdout_error_threshold=None), "error_threshold_declared"),
    (dict(verification=VerificationClaim()), "independent_verification_claimed"),
    (dict(verification=VerificationClaim("self_reported", "me", "2026", "ref")), "independent_verification_claimed"),
    (dict(verification=VerificationClaim("independently_verified", None, "2026", "ref")), "independent_verification_claimed"),
    (dict(verification=VerificationClaim("independently_verified", "me", "2026", None)), "independent_verification_claimed"),
])
def test_each_missing_piece_of_evidence_blocks_eligibility(over, check):
    res = evaluate_metric_eligibility(make_manifest(**over))
    assert not res.manifest_eligible and check in failed(res)


# 18: correspondences
def test_insufficient_fit_correspondences_ineligible():
    res = evaluate_metric_eligibility(make_manifest(control_points=control_points(fit=FIT_IMG[:3])))
    assert not res.manifest_eligible and "fit_correspondences_sufficient_and_nondegenerate" in failed(res)


def test_collinear_fit_correspondences_ineligible():
    line = [(300.0, 200.0), (500.0, 300.0), (700.0, 400.0), (900.0, 500.0), (1100.0, 600.0)]
    res = evaluate_metric_eligibility(make_manifest(control_points=control_points(fit=line, hold=[(1200.0, 700.0)])))
    assert not res.manifest_eligible and "fit_correspondences_sufficient_and_nondegenerate" in failed(res)


def test_policy_minimums_are_configurable_but_never_below_four():
    four = control_points(fit=FIT_IMG[:4])
    assert evaluate_metric_eligibility(make_manifest(control_points=four)).manifest_eligible
    assert not evaluate_metric_eligibility(make_manifest(control_points=four), EligibilityPolicy(min_fit_points=5)).manifest_eligible
    assert not evaluate_metric_eligibility(make_manifest(), EligibilityPolicy(min_holdout_points=4)).manifest_eligible
    with pytest.raises(CalibrationError):
        EligibilityPolicy(min_fit_points=3)
    with pytest.raises(CalibrationError):
        EligibilityPolicy(min_holdout_points=0)


# 19: held-out residuals
def test_no_holdout_points_ineligible():
    res = evaluate_metric_eligibility(make_manifest(control_points=control_points(hold=[])))
    assert not res.manifest_eligible and {"holdout_present", "holdout_error_within_threshold"} <= failed(res)


def test_holdout_residual_above_threshold_ineligible_and_recomputed_not_trusted():
    cps = list(control_points())
    k = next(i for i, c in enumerate(cps) if c.role == "holdout")
    bad = cps[k]
    cps[k] = ControlPoint(bad.point_id, bad.image_xy, (bad.plane_xy[0] + 0.5, bad.plane_xy[1]), "holdout", "survey")
    res = evaluate_metric_eligibility(make_manifest(control_points=tuple(cps)))
    assert not res.manifest_eligible and "holdout_error_within_threshold" in failed(res)
    assert res.holdout_max_error == pytest.approx(0.5, abs=1e-6)
    assert evaluate_metric_eligibility(make_manifest(control_points=tuple(cps), holdout_error_threshold=0.6)).manifest_eligible


def test_homography_that_does_not_reproduce_its_fit_points_ineligible():
    H = H_TRUE.copy()
    H[0, 2] += 0.5
    res = evaluate_metric_eligibility(make_manifest(homography=H))
    assert not res.manifest_eligible and {"fit_reproduced_by_homography", "holdout_error_within_threshold"} <= failed(res)


def test_holdout_must_be_distinct_from_fit_points():
    dup = control_points(fit=FIT_IMG, hold=[FIT_IMG[0]])
    res = evaluate_metric_eligibility(make_manifest(control_points=dup))
    assert "holdout_disjoint_from_fit" in failed(res) and not res.manifest_eligible


def test_control_points_must_lie_inside_the_valid_region():
    small = ((400.0, 400.0), (800.0, 400.0), (800.0, 800.0), (400.0, 800.0))
    res = evaluate_metric_eligibility(make_manifest(valid_region=small))
    assert "control_points_inside_region" in failed(res) and not res.manifest_eligible


def test_larger_error_in_physical_units_is_reported_with_unit():
    res = evaluate_metric_eligibility(make_manifest(plane_unit="cm"))
    assert res.error_unit == "cm"


def test_eligibility_is_deterministic():
    a, b = evaluate_metric_eligibility(make_manifest()), evaluate_metric_eligibility(make_manifest())
    assert a == b
