import numpy as np
import pytest

from pedsafety.errors import ConfigError
from pedsafety.geometry import (
    EPS, CalibrationError, NumericTolerances, apply_homography, check_correspondences, fit_homography,
    points_are_collinear, points_in_polygon, polygon_is_simple, polygon_signed_area, transform_point,
    validate_homography,
)

I3 = np.eye(3)


def code_of(fn, *a, **k):
    with pytest.raises(CalibrationError) as ei:
        fn(*a, **k)
    return ei.value.code


# ------------------------------------------------------------ 1-4 application
def test_identity_homography_returns_input():
    pts = np.array([[0.0, 0.0], [10.5, -3.0], [1920.0, 1080.0]])
    out = apply_homography(I3, pts)
    assert out.valid.all() and np.array_equal(out.xy, pts) and out.reasons == ["ok"] * 3


def test_non_identity_affine_known_values():
    H = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [0.0, 0.0, 1.0]])
    assert transform_point(H, 5.0, 4.0) == ((11.0, 11.0), "ok")


def test_perspective_known_value():
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.001, 0.0, 1.0]])
    xy, why = transform_point(H, 100.0, 50.0)
    assert why == "ok" and xy == pytest.approx((100.0 / 1.1, 50.0 / 1.1), rel=1e-15)


def test_batch_matches_pointwise_and_accepts_lists():
    H = np.array([[0.02, 0.001, -3.0], [0.0005, 0.025, -2.0], [1e-5, 2e-5, 1.0]])
    pts = [[300.0, 200.0], [1600.0, 220.0], [10.0, 10.0], [0.0, 1080.0], [960.0, 540.0]]
    batch = apply_homography(H, pts)
    for i, (x, y) in enumerate(pts):
        assert transform_point(H, x, y)[0] == pytest.approx(tuple(batch.xy[i]), rel=1e-15)


def test_empty_batch_returns_empty_result():
    for empty in ([], np.empty((0, 2))):
        out = apply_homography(I3, empty)
        assert out.xy.shape == (0, 2) and out.valid.shape == (0,) and out.reasons == []


def test_single_1d_point_accepted():
    assert apply_homography(I3, [3.0, 4.0]).xy.tolist() == [[3.0, 4.0]]


# ------------------------------------------------------------ 5-7 matrix checks
@pytest.mark.parametrize("H,code", [
    ([[1, 2, 3], [2, 4, 6], [0, 0, 1]], "SINGULAR"),
    (np.zeros((3, 3)), "SINGULAR"),
    (np.diag([1.0, 1e-12, 1.0]), "ILL_CONDITIONED"),
    (np.eye(2), "BAD_SHAPE"),
    (np.ones((3, 4)), "BAD_SHAPE"),
    (np.arange(9.0), "BAD_SHAPE"),
    (np.eye(3).reshape(3, 3, 1), "BAD_SHAPE"),
    ([["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]], "NON_NUMERIC"),
    ([[None, 0, 0], [0, 1, 0], [0, 0, 1]], "NON_NUMERIC"),
    (np.array([[True, False, False], [False, True, False], [False, False, True]]), "NON_NUMERIC"),
])
def test_bad_homography_rejected(H, code):
    assert code_of(validate_homography, H) == code
    assert code_of(apply_homography, H, [[1.0, 1.0]]) == code


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_matrix_rejected(bad):
    H = np.eye(3)
    H[0, 2] = bad
    assert code_of(validate_homography, H) == "NON_FINITE"


def test_condition_limit_is_configurable():
    H = np.diag([1.0, 1e-12, 1.0])
    validate_homography(H, NumericTolerances(max_condition_number=1e13))
    assert code_of(validate_homography, np.diag([1.0, 1e-3, 1.0]), NumericTolerances(max_condition_number=10.0)) == "ILL_CONDITIONED"


def test_default_tolerances_are_derived_from_machine_epsilon():
    t = NumericTolerances()
    assert t.denominator_rel_tol == pytest.approx(EPS ** 0.5) and t.max_condition_number == pytest.approx(EPS ** -0.5)


@pytest.mark.parametrize("kw", [dict(max_condition_number=0), dict(denominator_rel_tol=-1.0), dict(rank_rel_tol=float("nan")),
                                dict(region_boundary_tol_px=-1.0), dict(size_rounding_px=float("inf"))])
def test_invalid_tolerances_rejected(kw):
    with pytest.raises(ConfigError):
        NumericTolerances(**kw)


# ---------------------------------------------------------- 8-9 point problems
def test_non_finite_points_flagged_not_repaired():
    pts = [[1.0, 2.0], [np.nan, 2.0], [np.inf, 0.0], [3.0, 4.0]]
    out = apply_homography(I3, pts)
    assert out.valid.tolist() == [True, False, False, True]
    assert out.reasons == ["ok", "non_finite_input", "non_finite_input", "ok"]
    assert np.isnan(out.xy[1]).all() and out.xy[3].tolist() == [3.0, 4.0]


def test_denominator_instability_and_zero():
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]])   # w = x - 1
    out = apply_homography(H, [[1.0, 0.0], [1.0 + 1e-12, 0.0], [2.0, 0.0]])
    assert out.reasons[:2] == ["denominator_unstable"] * 2 and out.reasons[2] == "ok" and out.valid.tolist() == [False, False, True]
    assert out.xy[2].tolist() == [2.0, 0.0]


def test_denominator_tolerance_is_configurable():
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]])
    strict = apply_homography(H, [[1.0 + 1e-6, 0.0]], NumericTolerances(denominator_rel_tol=1e-3))
    loose = apply_homography(H, [[1.0 + 1e-6, 0.0]], NumericTolerances(denominator_rel_tol=1e-12))
    assert strict.reasons == ["denominator_unstable"] and loose.reasons == ["ok"]


def test_behind_horizon_sign_check():
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]])
    out = apply_homography(H, [[0.5, 0.0], [2.0, 0.0]], reference_sign=1)
    assert out.reasons == ["behind_horizon", "ok"]


def test_overflowing_output_flagged_not_returned():
    H = np.diag([10.0, 10.0, 1.0])
    out = apply_homography(H, [[1e308, 1.0], [1.0, 1.0]])
    assert out.reasons == ["non_finite_output", "ok"] and out.valid.tolist() == [False, True]
    assert np.isnan(out.xy[0]).all()


@pytest.mark.parametrize("pts", [np.zeros((3, 3)), [["a", "b"]], np.zeros((2, 2, 2)), [[None, 1.0]]])
def test_invalid_point_arrays_rejected(pts):
    with pytest.raises(CalibrationError):
        apply_homography(I3, pts)


def test_input_arrays_not_modified():
    pts = np.array([[1.0, 2.0], [np.nan, 1.0]])
    H = np.eye(3) * 2
    H_before, p_before = H.copy(), pts.copy()
    apply_homography(H, pts)
    assert np.array_equal(H, H_before) and np.array_equal(pts, p_before, equal_nan=True)


# ---------------------------------------------------------- 18 fitting
SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_fit_recovers_known_homography():
    H = np.array([[0.02, 0.001, -3.0], [0.0005, 0.025, -2.0], [1e-5, 2e-5, 1.0]])
    src = np.array([[300, 200], [1600, 220], [1500, 900], [350, 880], [960, 540], [700, 400]], dtype=float)
    dst = apply_homography(H, src).xy
    Hf = fit_homography(src, dst)
    probe = [[100.0, 100.0], [1800.0, 1000.0], [1000.0, 300.0]]
    assert np.allclose(apply_homography(Hf, probe).xy, apply_homography(H, probe).xy, rtol=1e-9, atol=1e-9)


def test_fit_is_deterministic():
    src = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], dtype=float)
    dst = src * 2 + 1
    assert np.array_equal(fit_homography(src, dst), fit_homography(src, dst))


@pytest.mark.parametrize("src,code", [
    (SQUARE[:3], "INSUFFICIENT_CORRESPONDENCES"),
    ([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], "COLLINEAR"),
    ([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 5.0)], "DEGENERATE"),     # three of four collinear
    ([(1.0, 1.0)] * 4, "COLLINEAR"),                                       # coincident points
])
def test_insufficient_or_degenerate_correspondences_rejected(src, code):
    dst = [(x * 2.0, y * 3.0) for x, y in src]
    assert code_of(fit_homography, src, dst) == code


def test_correspondence_input_errors():
    assert code_of(check_correspondences, SQUARE, SQUARE[:3]) == "BAD_POINT_SHAPE"
    bad = [list(p) for p in SQUARE]
    bad[0][0] = np.nan
    assert code_of(check_correspondences, bad, SQUARE) == "NON_FINITE"
    assert code_of(check_correspondences, SQUARE + [(5.0, 5.0)], SQUARE + [(5.0, 5.0)], min_points=6) == "INSUFFICIENT_CORRESPONDENCES"
    check_correspondences(SQUARE, SQUARE)


def test_points_are_collinear():
    assert points_are_collinear([(0, 0), (1, 1), (2, 2)]) and not points_are_collinear(SQUARE)
    assert points_are_collinear([(1.0, 1.0)])


# ------------------------------------------------------------- polygons
BOX = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_point_in_polygon_inside_outside_boundary():
    pts = [[5, 5], [-1, 5], [11, 5], [5, 11], [0, 5], [10, 10], [5, 0]]
    assert points_in_polygon(pts, BOX).tolist() == [True, False, False, False, True, True, True]


def test_point_in_non_convex_polygon():
    L = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    assert points_in_polygon([[2, 8], [8, 2], [8, 8]], L).tolist() == [True, True, False]


def test_boundary_tolerance_for_slanted_edges():
    tri = [(0.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    p = [[5.0, 5.0 + 1e-9]]
    assert points_in_polygon(p, tri, boundary_tol=1e-6).all()


def test_polygon_simplicity_and_area():
    assert polygon_is_simple(BOX) and polygon_signed_area(BOX) == 100.0
    assert not polygon_is_simple([(0, 0), (10, 10), (10, 0), (0, 10)])      # bow-tie
    assert polygon_signed_area(BOX[::-1]) == -100.0
