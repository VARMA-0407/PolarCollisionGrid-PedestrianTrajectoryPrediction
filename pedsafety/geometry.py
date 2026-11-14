"""Planar-homography mathematics (Stage 3). Pure geometry: no calibration
claims, no units, no dataset knowledge.

A 3x3 matrix applied here is just a projective map between two planes. That
the matrix is invertible, finite and well conditioned says nothing about
whether its output is physical; that decision lives in ``calibration.py``.

Numerical tolerances are collected in :class:`NumericTolerances`. Their
defaults are derived from double-precision machine epsilon (documented per
field), not tuned to any dataset, and every one can be overridden.
"""

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .errors import ConfigError, PedsafetyError

EPS = float(np.finfo(float).eps)


class CalibrationError(PedsafetyError):
    """A calibration/geometry input is unusable. ``code`` is a stable identifier."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__("[%s] %s" % (code, message))


@dataclass(frozen=True)
class NumericTolerances:
    """Documented numerical tolerances.

    max_condition_number: reject a homography whose 2-norm condition number
        exceeds this. Default 1/sqrt(eps) ~ 6.7e7: beyond it more than half of
        the 16 significant digits of a solve are lost.
    denominator_rel_tol: a point is rejected when |w| <= tol * (|h31 x|+|h32 y|+|h33|)
        for the homogeneous denominator w, i.e. when cancellation leaves fewer
        than ~half the digits. Default sqrt(eps) ~ 1.5e-8.
    rank_rel_tol: DLT design-matrix / point-set degeneracy guard,
        sigma_8 / sigma_1 (or sigma_min / sigma_max for a point set) must exceed
        this. Default sqrt(eps). This catches numerical collinearity only; the
        QUALITY of a correspondence set is judged by held-out residuals.
    region_boundary_tol_px: points within this distance of a polygon edge count
        as inside. Default 0.0 (exact arithmetic; non-axis-aligned edges need a
        positive value to include their boundary reliably).
    size_rounding_px: allowed difference between an expected resized size and the
        observed raster size (integer rounding of resized dimensions). Default 1.0.
    """

    max_condition_number: float = 1.0 / math.sqrt(EPS)
    denominator_rel_tol: float = math.sqrt(EPS)
    rank_rel_tol: float = math.sqrt(EPS)
    region_boundary_tol_px: float = 0.0
    size_rounding_px: float = 1.0

    def __post_init__(self):
        for name in ("max_condition_number", "denominator_rel_tol", "rank_rel_tol"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ConfigError("%s must be a positive finite number, got %r" % (name, v))
        for name in ("region_boundary_tol_px", "size_rounding_px"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                raise ConfigError("%s must be a non-negative finite number, got %r" % (name, v))


DEFAULT_TOLERANCES = NumericTolerances()


def as_float_array(x: Any, what: str) -> np.ndarray:
    """Numeric ndarray as float64. Strings, None, objects and bool are refused
    (NumPy would otherwise coerce numeric-looking strings silently)."""
    try:
        a = np.asarray(x)
    except Exception as exc:  # ragged input etc.
        raise CalibrationError("NON_NUMERIC", "%s cannot be converted to an array: %s" % (what, exc))
    if a.dtype.kind not in "iuf":
        raise CalibrationError("NON_NUMERIC", "%s must contain real numbers (got dtype %s)" % (what, a.dtype))
    return a.astype(float)


def validate_homography(H: Any, tol: NumericTolerances = DEFAULT_TOLERANCES) -> np.ndarray:
    """Return H as a float 3x3 array or raise :class:`CalibrationError`.

    Checks: exactly 3x3, finite, non-singular, condition number within tolerance.
    Does NOT assert anything about physical meaning.
    """
    a = as_float_array(H, "homography")
    if a.shape != (3, 3):
        raise CalibrationError("BAD_SHAPE", "homography must be exactly 3x3, got shape %s" % (a.shape,))
    if not np.isfinite(a).all():
        raise CalibrationError("NON_FINITE", "homography contains NaN or infinity")
    s = np.linalg.svd(a, compute_uv=False)
    if s[0] == 0.0 or s[-1] <= EPS * s[0]:
        raise CalibrationError("SINGULAR", "homography is numerically singular (smallest singular value %.3g is "
                               "within machine epsilon of the largest %.3g)" % (s[-1], s[0]))
    cond = float(s[0] / s[-1])
    if cond > tol.max_condition_number:
        raise CalibrationError("ILL_CONDITIONED", "homography condition number %.3g exceeds the limit %.3g"
                               % (cond, tol.max_condition_number))
    return a.copy()


@dataclass
class ProjectionOutput:
    xy: np.ndarray            # (N, 2); NaN where invalid
    valid: np.ndarray         # (N,) bool
    reasons: List[str]        # 'ok' or a failure reason per point


REASON_OK = "ok"
REASON_NON_FINITE_INPUT = "non_finite_input"
REASON_DENOMINATOR = "denominator_unstable"
REASON_BEHIND_HORIZON = "behind_horizon"
REASON_NON_FINITE_OUTPUT = "non_finite_output"


def homogeneous_denominator(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points[:, 0] * H[2, 0] + points[:, 1] * H[2, 1] + H[2, 2]


def apply_homography(H: Any, points: Any, tol: NumericTolerances = DEFAULT_TOLERANCES,
                     reference_sign: Optional[int] = None) -> ProjectionOutput:
    """Map ``points`` (N,2) through ``H``. Invalid points are flagged, never
    clipped or repaired; their output is NaN.

    A point is invalid when its input is non-finite, when the homogeneous
    denominator is unstable (see ``NumericTolerances.denominator_rel_tol``), when
    its denominator has the opposite sign to ``reference_sign`` (the point lies
    on the far side of the vanishing line), or when the output is non-finite.
    """
    Hm = validate_homography(H, tol)
    pts = as_float_array(points, "points")
    if pts.shape in ((0,), (0, 2)):   # an empty batch is valid and yields an empty result
        return ProjectionOutput(xy=np.empty((0, 2)), valid=np.zeros(0, dtype=bool), reasons=[])
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise CalibrationError("BAD_POINT_SHAPE", "points must have shape (N, 2), got %s" % (pts.shape,))
    n = pts.shape[0]
    finite = np.isfinite(pts).all(axis=1)
    safe = np.where(finite[:, None], pts, 0.0)
    with np.errstate(all="ignore"):
        w = homogeneous_denominator(Hm, safe)
        scale = np.abs(safe[:, 0] * Hm[2, 0]) + np.abs(safe[:, 1] * Hm[2, 1]) + abs(Hm[2, 2])
        stable = np.abs(w) > tol.denominator_rel_tol * scale
        num_x = safe[:, 0] * Hm[0, 0] + safe[:, 1] * Hm[0, 1] + Hm[0, 2]
        num_y = safe[:, 0] * Hm[1, 0] + safe[:, 1] * Hm[1, 1] + Hm[1, 2]
        xy = np.column_stack([num_x / w, num_y / w])
    reasons = [REASON_OK] * n
    valid = np.ones(n, dtype=bool)
    for i in range(n):
        if not finite[i]:
            reasons[i] = REASON_NON_FINITE_INPUT
        elif not stable[i]:
            reasons[i] = REASON_DENOMINATOR
        elif reference_sign is not None and np.sign(w[i]) != reference_sign:
            reasons[i] = REASON_BEHIND_HORIZON
        elif not np.isfinite(xy[i]).all():
            reasons[i] = REASON_NON_FINITE_OUTPUT
        valid[i] = reasons[i] == REASON_OK
    xy[~valid] = np.nan
    return ProjectionOutput(xy=xy, valid=valid, reasons=reasons)


def transform_point(H: Any, x: float, y: float, tol: NumericTolerances = DEFAULT_TOLERANCES,
                    reference_sign: Optional[int] = None) -> Tuple[Optional[Tuple[float, float]], str]:
    out = apply_homography(H, [[x, y]], tol, reference_sign)
    return (None if not out.valid[0] else (float(out.xy[0, 0]), float(out.xy[0, 1]))), out.reasons[0]


# ------------------------------------------------------------------ fitting
def _normalise(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = pts.mean(axis=0)
    d = float(np.sqrt(((pts - c) ** 2).sum(axis=1)).mean())
    if d == 0.0:
        raise CalibrationError("DEGENERATE", "all correspondence points coincide")
    s = math.sqrt(2.0) / d
    T = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])
    return (pts - c) * s, T


def points_are_collinear(pts: Any, tol: NumericTolerances = DEFAULT_TOLERANCES) -> bool:
    """True when the centred point set is numerically rank <= 1."""
    a = as_float_array(pts, "points")
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 2:
        return True
    c = a - a.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    return bool(s[0] == 0.0 or s[-1] <= tol.rank_rel_tol * s[0])


def _design(src: np.ndarray, dst: np.ndarray):
    n1, T1 = _normalise(src)
    n2, T2 = _normalise(dst)
    n = src.shape[0]
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = n1[i]
        u, v = n2[i]
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
    return A, T1, T2


def check_correspondences(src: Any, dst: Any, tol: NumericTolerances = DEFAULT_TOLERANCES,
                          min_points: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a point-correspondence set for estimating a plane homography.

    Requires equal counts, >= ``min_points`` (and never fewer than 4), finite
    values, neither set collinear, and a full-rank (8) DLT design so that no
    three of a minimal set are collinear. Returns the validated float arrays.
    """
    s = as_float_array(src, "source points")
    d = as_float_array(dst, "destination points")
    if s.ndim != 2 or s.shape[1] != 2 or d.shape != s.shape:
        raise CalibrationError("BAD_POINT_SHAPE", "source/destination must both have shape (N, 2); got %s and %s"
                               % (s.shape, d.shape))
    if not (np.isfinite(s).all() and np.isfinite(d).all()):
        raise CalibrationError("NON_FINITE", "correspondences contain NaN or infinity")
    if s.shape[0] < max(4, min_points):
        raise CalibrationError("INSUFFICIENT_CORRESPONDENCES",
                               "need at least %d correspondences, got %d" % (max(4, min_points), s.shape[0]))
    if points_are_collinear(s, tol) or points_are_collinear(d, tol):
        raise CalibrationError("COLLINEAR", "correspondence points are collinear")
    A, _, _ = _design(s, d)
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[7] <= tol.rank_rel_tol * sv[0]:
        raise CalibrationError("DEGENERATE", "correspondence geometry is degenerate (rank-deficient; e.g. three "
                               "of four points are collinear)")
    return s, d


def fit_homography(src: Any, dst: Any, tol: NumericTolerances = DEFAULT_TOLERANCES) -> np.ndarray:
    """Normalised DLT estimate of H with dst ~ H src. Deterministic. The result
    is a projective map only; it carries no unit, scale or validity claim."""
    s, d = check_correspondences(src, dst, tol)
    A, T1, T2 = _design(s, d)
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    nrm = float(np.linalg.norm(H))
    H = H / H[2, 2] if abs(H[2, 2]) > math.sqrt(EPS) * nrm else H / nrm
    return validate_homography(H, tol)


# ----------------------------------------------------------------- polygons
def polygon_signed_area(poly: Sequence[Tuple[float, float]]) -> float:
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c) -> bool:
    return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])


def _segments_intersect(p1, p2, p3, p4) -> bool:
    o1, o2, o3, o4 = _orient(p1, p2, p3), _orient(p1, p2, p4), _orient(p3, p4, p1), _orient(p3, p4, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:  # proper crossing
        return True
    return ((o1 == 0 and _on_segment(p1, p2, p3)) or (o2 == 0 and _on_segment(p1, p2, p4)) or
            (o3 == 0 and _on_segment(p3, p4, p1)) or (o4 == 0 and _on_segment(p3, p4, p2)))


def polygon_is_simple(poly: Sequence[Tuple[float, float]]) -> bool:
    """True when no two non-adjacent edges touch or cross."""
    n = len(poly)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _segments_intersect(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]):
                return False
    return True


def points_in_polygon(points: Any, poly: Sequence[Tuple[float, float]], boundary_tol: float = 0.0) -> np.ndarray:
    """Even-odd inside test; points within ``boundary_tol`` of an edge count as inside."""
    pts = as_float_array(points, "points").reshape(-1, 2)
    P = np.asarray(poly, dtype=float)
    n = len(P)
    inside = np.zeros(len(pts), dtype=bool)
    near = np.zeros(len(pts), dtype=bool)
    with np.errstate(all="ignore"):
        for k in range(n):
            x1, y1 = P[k]
            x2, y2 = P[(k + 1) % n]
            x, y = pts[:, 0], pts[:, 1]
            cross = (y1 > y) != (y2 > y)
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            inside ^= cross & (x < xint)
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            t = np.clip(((x - x1) * dx + (y - y1) * dy) / L2, 0.0, 1.0)
            near |= np.hypot(x - (x1 + t * dx), y - (y1 + t * dy)) <= boundary_tol
    return inside | near
