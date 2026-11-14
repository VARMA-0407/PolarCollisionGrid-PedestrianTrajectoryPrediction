"""Calibration manifest, raster compatibility and metric-eligibility (Stage 3).

Two separate questions are answered here and never conflated:

1. Is the manifest INTERNALLY CONSISTENT and USABLE as a projective map? (the
   constructors raise ``CalibrationError`` otherwise)
2. Do its own claims and evidence satisfy the SOFTWARE checks for declaring the
   output metric? (:func:`evaluate_metric_eligibility`)

Answering "yes" to (2) means ``manifest_eligible``. It is NOT physical
verification: this code cannot survey a road. The verification claim
(who/when/reference) is read from the manifest and recorded, never confirmed.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .geometry import (
    DEFAULT_TOLERANCES, CalibrationError, NumericTolerances, apply_homography, as_float_array,
    check_correspondences, homogeneous_denominator, points_in_polygon, polygon_is_simple,
    polygon_signed_area, validate_homography,
)
from .io.csv_adapter import PHYSICAL_UNITS

PIXEL_CONVENTIONS = ("pixel_center_half", "pixel_center_integer")   # centres at i+0.5 / at integer i
PIXEL_DOMAINS = ("as_captured", "undistorted", "unknown")
VERIFICATION_STATUSES = ("unverified", "self_reported", "independently_verified")
# Ground truth for calibration must be independent of agent tracks.
CONTROL_POINT_SOURCES = ("survey", "metric_map", "known_distance", "other_independent")
CONTROL_POINT_ROLES = ("fit", "holdout")


def _nonempty(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _pair(v: Any, what: str) -> Tuple[float, float]:
    a = as_float_array(v, what)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise CalibrationError("BAD_POINT", "%s must be two finite numbers, got %r" % (what, v))
    return (float(a[0]), float(a[1]))


# ------------------------------------------------------------------- rasters
@dataclass(frozen=True)
class RasterSpec:
    """The image raster that pixel coordinates refer to.

    ``pixel_domain``: 'as_captured' = lens distortion left in; 'undistorted' =
    already rectified by a model documented elsewhere; 'unknown'. No distortion
    model is implemented or assumed here.
    """

    raster_id: str
    width: int
    height: int
    coordinate_system: str
    pixel_convention: str
    pixel_domain: str
    origin: str = "top_left"

    def __post_init__(self):
        for n in ("raster_id", "coordinate_system"):
            if not _nonempty(getattr(self, n)):
                raise CalibrationError("BAD_RASTER", "%s must be a non-empty string" % n)
        for n in ("width", "height"):
            v = getattr(self, n)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise CalibrationError("BAD_RASTER", "%s must be a positive integer, got %r" % (n, v))
        if self.pixel_convention not in PIXEL_CONVENTIONS:
            raise CalibrationError("BAD_RASTER", "pixel_convention must be one of %s" % (PIXEL_CONVENTIONS,))
        if self.pixel_domain not in PIXEL_DOMAINS:
            raise CalibrationError("BAD_RASTER", "pixel_domain must be one of %s" % (PIXEL_DOMAINS,))
        if self.origin != "top_left":
            raise CalibrationError("BAD_RASTER", "only a top_left origin is supported, got %r" % (self.origin,))

    def domain(self) -> Tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max), inclusive."""
        if self.pixel_convention == "pixel_center_half":
            return (0.0, float(self.width), 0.0, float(self.height))
        return (-0.5, self.width - 0.5, -0.5, self.height - 0.5)

    def contains(self, pts: Any) -> np.ndarray:
        p = as_float_array(pts, "points").reshape(-1, 2)
        x0, x1, y0, y1 = self.domain()
        with np.errstate(invalid="ignore"):
            return np.isfinite(p).all(axis=1) & (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)

    def to_dict(self) -> Dict[str, Any]:
        return {"raster_id": self.raster_id, "width": self.width, "height": self.height,
                "coordinate_system": self.coordinate_system, "pixel_convention": self.pixel_convention,
                "pixel_domain": self.pixel_domain, "origin": self.origin}


@dataclass(frozen=True)
class RasterTransform:
    """Documented crop -> resize -> pad that maps a calibration raster to an
    observed raster: ``u' = scale * (u - crop) + pad`` (pixel-edge coordinates)."""

    scale_x: float
    scale_y: float
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_width: Optional[float] = None
    crop_height: Optional[float] = None
    pad_left: float = 0.0
    pad_top: float = 0.0
    pad_right: float = 0.0
    pad_bottom: float = 0.0

    def __post_init__(self):
        for n in ("scale_x", "scale_y"):
            v = getattr(self, n)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0):
                raise CalibrationError("BAD_RASTER_TRANSFORM", "%s must be a positive finite number, got %r" % (n, v))
        for n in ("crop_x", "crop_y", "pad_left", "pad_top", "pad_right", "pad_bottom"):
            v = getattr(self, n)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0):
                raise CalibrationError("BAD_RASTER_TRANSFORM", "%s must be a non-negative finite number" % n)
        for n in ("crop_width", "crop_height"):
            v = getattr(self, n)
            if v is not None and not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise CalibrationError("BAD_RASTER_TRANSFORM", "%s must be positive when given" % n)

    def expected_size(self, source: RasterSpec) -> Tuple[float, float]:
        cw = self.crop_width if self.crop_width is not None else source.width - self.crop_x
        ch = self.crop_height if self.crop_height is not None else source.height - self.crop_y
        if cw <= 0 or ch <= 0 or self.crop_x + cw > source.width or self.crop_y + ch > source.height:
            raise CalibrationError("BAD_CROP", "crop region lies outside the %dx%d calibration raster"
                                   % (source.width, source.height))
        return (cw * self.scale_x + self.pad_left + self.pad_right, ch * self.scale_y + self.pad_top + self.pad_bottom)

    def matrix(self, pixel_convention: str) -> np.ndarray:
        """3x3 A with x' = A x, expressed in the raster's own pixel convention."""
        A = np.array([[self.scale_x, 0.0, -self.scale_x * self.crop_x + self.pad_left],
                      [0.0, self.scale_y, -self.scale_y * self.crop_y + self.pad_top],
                      [0.0, 0.0, 1.0]])
        if pixel_convention == "pixel_center_half":
            return A
        Tp = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        Tm = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])
        return Tm @ A @ Tp


# ------------------------------------------------------------ manifest parts
@dataclass(frozen=True)
class CalibrationProvenance:
    calibration_id: str
    version: str
    description: str = ""
    created_by: Optional[str] = None
    created_on: Optional[str] = None

    def __post_init__(self):
        for n in ("calibration_id", "version"):
            if not _nonempty(getattr(self, n)):
                raise CalibrationError("BAD_PROVENANCE", "%s must be a non-empty string" % n)


@dataclass(frozen=True)
class VerificationClaim:
    """What the manifest ASSERTS about verification. The software only records it."""

    status: str = "unverified"
    verified_by: Optional[str] = None
    verified_on: Optional[str] = None
    reference: Optional[str] = None

    def __post_init__(self):
        if self.status not in VERIFICATION_STATUSES:
            raise CalibrationError("BAD_VERIFICATION", "status must be one of %s" % (VERIFICATION_STATUSES,))


@dataclass(frozen=True)
class ControlPoint:
    point_id: str
    image_xy: Tuple[float, float]
    plane_xy: Tuple[float, float]
    role: str
    source: str

    def __post_init__(self):
        if not _nonempty(self.point_id):
            raise CalibrationError("BAD_CONTROL_POINT", "point_id must be non-empty")
        object.__setattr__(self, "image_xy", _pair(self.image_xy, "image_xy of %s" % self.point_id))
        object.__setattr__(self, "plane_xy", _pair(self.plane_xy, "plane_xy of %s" % self.point_id))
        if self.role not in CONTROL_POINT_ROLES:
            raise CalibrationError("BAD_CONTROL_POINT", "role must be one of %s" % (CONTROL_POINT_ROLES,))
        if self.source not in CONTROL_POINT_SOURCES:
            raise CalibrationError("BAD_CONTROL_POINT_SOURCE", "control point source %r is not allowed; allowed: %s. "
                                   "Agent trajectories or tracker output are never calibration ground truth."
                                   % (self.source, CONTROL_POINT_SOURCES))

    def to_dict(self) -> Dict[str, Any]:
        return {"point_id": self.point_id, "image_xy": list(self.image_xy), "plane_xy": list(self.plane_xy),
                "role": self.role, "source": self.source}


@dataclass(frozen=True)
class CalibrationManifest:
    """Image-to-plane homography plus everything needed to judge it.

    ``homography`` maps calibration-raster pixel coordinates to plane coordinates
    (direction ``image_to_plane`` only). ``plane_unit`` None means the plane
    coordinates are relative/projective with no declared scale.
    """

    provenance: CalibrationProvenance
    raster: RasterSpec
    homography: Any
    direction: str = "image_to_plane"
    plane_coordinate_system: Optional[str] = None
    plane_unit: Optional[str] = None
    plane_origin: Optional[str] = None
    plane_axes: Optional[str] = None
    handedness: Optional[str] = None
    scale_basis: Optional[str] = None
    valid_region: Optional[Tuple[Tuple[float, float], ...]] = None
    control_points: Tuple[ControlPoint, ...] = ()
    holdout_error_threshold: Optional[float] = None
    verification: VerificationClaim = VerificationClaim()
    distortion_note: Optional[str] = None

    def __post_init__(self):
        if self.direction != "image_to_plane":
            raise CalibrationError("BAD_DIRECTION", "direction must be 'image_to_plane', got %r; a plane_to_image "
                                   "matrix must be inverted explicitly by the caller" % (self.direction,))
        H = validate_homography(self.homography)
        object.__setattr__(self, "homography", tuple(tuple(float(v) for v in row) for row in H))
        if self.plane_unit is not None and self.plane_unit not in PHYSICAL_UNITS:
            raise CalibrationError("BAD_UNIT", "plane_unit must be one of %s or None, got %r"
                                   % (sorted(PHYSICAL_UNITS), self.plane_unit))
        if self.handedness not in (None, "right", "left"):
            raise CalibrationError("BAD_HANDEDNESS", "handedness must be 'right', 'left' or None")
        t = self.holdout_error_threshold
        if t is not None and not (isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t > 0):
            raise CalibrationError("BAD_THRESHOLD", "holdout_error_threshold must be a positive finite number or None")
        cps = tuple(self.control_points)
        object.__setattr__(self, "control_points", cps)
        ids = [c.point_id for c in cps]
        if len(set(ids)) != len(ids):
            raise CalibrationError("BAD_CONTROL_POINT", "control point ids must be unique")
        if cps and not self.raster.contains([c.image_xy for c in cps]).all():
            raise CalibrationError("BAD_CONTROL_POINT", "a control point lies outside the calibration raster")
        if self.valid_region is not None:
            reg = tuple(_pair(v, "valid_region vertex") for v in self.valid_region)
            object.__setattr__(self, "valid_region", reg)
            if len(reg) < 3:
                raise CalibrationError("BAD_REGION", "valid_region needs at least 3 vertices")
            if not self.raster.contains(reg).all():
                raise CalibrationError("BAD_REGION", "valid_region extends outside the calibration raster")
            if not polygon_is_simple(reg) or polygon_signed_area(reg) == 0.0:
                raise CalibrationError("BAD_REGION", "valid_region must be a simple polygon with non-zero area")
            w = homogeneous_denominator(np.array(self.homography), np.array(reg))
            scale = (np.abs(np.array(reg)[:, 0] * self.homography[2][0]) + np.abs(np.array(reg)[:, 1] * self.homography[2][1])
                     + abs(self.homography[2][2]))
            if not (np.abs(w) > DEFAULT_TOLERANCES.denominator_rel_tol * scale).all() or not (np.sign(w) == np.sign(w[0])).all():
                raise CalibrationError("REGION_CROSSES_HORIZON", "the homogeneous denominator is zero/unstable or "
                                       "changes sign inside valid_region (region touches the vanishing line)")

    @property
    def H(self) -> np.ndarray:
        return np.array(self.homography, dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provenance": {"calibration_id": self.provenance.calibration_id, "version": self.provenance.version,
                           "description": self.provenance.description, "created_by": self.provenance.created_by,
                           "created_on": self.provenance.created_on},
            "raster": self.raster.to_dict(),
            "homography": [list(r) for r in self.homography],
            "direction": self.direction,
            "plane_coordinate_system": self.plane_coordinate_system, "plane_unit": self.plane_unit,
            "plane_origin": self.plane_origin, "plane_axes": self.plane_axes, "handedness": self.handedness,
            "scale_basis": self.scale_basis,
            "valid_region": None if self.valid_region is None else [list(v) for v in self.valid_region],
            "control_points": [c.to_dict() for c in self.control_points],
            "holdout_error_threshold": self.holdout_error_threshold,
            "verification": {"status": self.verification.status, "verified_by": self.verification.verified_by,
                             "verified_on": self.verification.verified_on, "reference": self.verification.reference},
            "distortion_note": self.distortion_note,
        }

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CalibrationManifest":
        """Strict loader: unknown or missing keys are errors. Loading a file does
        not make it valid; call :func:`evaluate_metric_eligibility` for that."""
        allowed = {"provenance", "raster", "homography", "direction", "plane_coordinate_system", "plane_unit",
                   "plane_origin", "plane_axes", "handedness", "scale_basis", "valid_region", "control_points",
                   "holdout_error_threshold", "verification", "distortion_note"}
        if not isinstance(d, Mapping):
            raise CalibrationError("BAD_MANIFEST", "manifest must be a mapping")
        extra = sorted(set(d) - allowed)
        missing = sorted({"provenance", "raster", "homography"} - set(d))
        if extra or missing:
            raise CalibrationError("BAD_MANIFEST", "unknown keys %s / missing keys %s" % (extra, missing))
        try:
            kw = {k: v for k, v in d.items() if k not in ("provenance", "raster", "control_points", "verification", "valid_region")}
            return cls(
                provenance=CalibrationProvenance(**d["provenance"]),
                raster=RasterSpec(**d["raster"]),
                control_points=tuple(ControlPoint(**{**c, "image_xy": tuple(c["image_xy"]), "plane_xy": tuple(c["plane_xy"])})
                                     for c in d.get("control_points", [])),
                verification=VerificationClaim(**d["verification"]) if "verification" in d else VerificationClaim(),
                valid_region=None if d.get("valid_region") is None else tuple(tuple(v) for v in d["valid_region"]),
                **kw)
        except TypeError as exc:
            raise CalibrationError("BAD_MANIFEST", "malformed manifest section: %s" % exc)


def reference_sign(manifest: CalibrationManifest, tol: NumericTolerances = DEFAULT_TOLERANCES) -> int:
    """Sign of the homogeneous denominator at the region centroid (or raster centre)."""
    if manifest.valid_region is not None:
        pt = np.array(manifest.valid_region).mean(axis=0)
    else:
        x0, x1, y0, y1 = manifest.raster.domain()
        pt = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    out = apply_homography(manifest.H, [pt], tol)
    if not out.valid[0]:
        raise CalibrationError("UNSTABLE_REFERENCE", "homogeneous denominator is unstable at the reference point (%s)" % out.reasons[0])
    return int(np.sign(homogeneous_denominator(manifest.H, pt.reshape(1, 2))[0]))


# -------------------------------------------------------------- eligibility
@dataclass(frozen=True)
class EligibilityPolicy:
    min_fit_points: int = 4          # mathematical minimum for a planar homography
    min_holdout_points: int = 1      # at least one point not used to fit; more is better
    require_independent_verification: bool = True
    tolerances: NumericTolerances = DEFAULT_TOLERANCES

    def __post_init__(self):
        if isinstance(self.min_fit_points, bool) or not isinstance(self.min_fit_points, int) or self.min_fit_points < 4:
            raise CalibrationError("BAD_POLICY", "min_fit_points must be an integer >= 4")
        if isinstance(self.min_holdout_points, bool) or not isinstance(self.min_holdout_points, int) or self.min_holdout_points < 1:
            raise CalibrationError("BAD_POLICY", "min_holdout_points must be an integer >= 1")


@dataclass(frozen=True)
class EligibilityResult:
    """Outcome of the SOFTWARE checks. ``manifest_eligible`` is not physical
    verification; ``physical_verification_established_by_software`` is always False."""

    manifest_eligible: bool
    reasons: Tuple[str, ...]
    checks: Tuple[Tuple[str, bool], ...]
    n_fit: int
    n_holdout: int
    fit_max_error: Optional[float]
    holdout_rms_error: Optional[float]
    holdout_max_error: Optional[float]
    error_unit: Optional[str]
    threshold: Optional[float]
    basis: str = "software_manifest_checks_only"
    physical_verification_established_by_software: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"manifest_eligible": self.manifest_eligible, "reasons": list(self.reasons),
                "checks": dict(self.checks), "n_fit": self.n_fit, "n_holdout": self.n_holdout,
                "fit_max_error": self.fit_max_error, "holdout_rms_error": self.holdout_rms_error,
                "holdout_max_error": self.holdout_max_error, "error_unit": self.error_unit,
                "threshold": self.threshold, "basis": self.basis,
                "physical_verification_established_by_software": self.physical_verification_established_by_software}


def _errors(manifest: CalibrationManifest, pts: Sequence[ControlPoint], tol: NumericTolerances) -> Optional[np.ndarray]:
    if not pts:
        return None
    out = apply_homography(manifest.H, [c.image_xy for c in pts], tol, reference_sign(manifest, tol))
    if not out.valid.all():
        return None
    return np.linalg.norm(out.xy - np.array([c.plane_xy for c in pts]), axis=1)


def evaluate_metric_eligibility(manifest: CalibrationManifest, policy: EligibilityPolicy = EligibilityPolicy()) -> EligibilityResult:
    """Run every software check and report ALL failures (no short-circuit).

    Control-point residuals are RECOMPUTED from the matrix; reported residuals are
    not trusted. Agent trajectories can never be control points (rejected at
    ControlPoint construction).
    """
    tol = policy.tolerances
    checks: List[Tuple[str, bool]] = []
    reasons: List[str] = []

    def check(name: str, ok: bool, reason: str):
        checks.append((name, bool(ok)))
        if not ok:
            reasons.append(reason)

    m = manifest
    fit = [c for c in m.control_points if c.role == "fit"]
    hold = [c for c in m.control_points if c.role == "holdout"]
    check("named_destination_frame", _nonempty(m.plane_coordinate_system), "no named destination coordinate frame")
    check("physical_unit_declared", m.plane_unit in PHYSICAL_UNITS, "no declared physical unit (plane coordinates are relative/projective)")
    check("frame_definition_documented", _nonempty(m.plane_origin) and _nonempty(m.plane_axes) and m.handedness is not None,
          "frame origin, axes and handedness are not all documented")
    check("scale_basis_documented", _nonempty(m.scale_basis), "no documented basis for the physical scale")
    check("valid_region_documented", m.valid_region is not None, "no documented valid projection region")
    check("raster_pixel_domain_known", m.raster.pixel_domain != "unknown",
          "calibration raster distortion status (pixel_domain) is unknown")
    try:
        if len(fit) < policy.min_fit_points:
            raise CalibrationError("INSUFFICIENT_CORRESPONDENCES", "%d fit correspondence(s); need >= %d" % (len(fit), policy.min_fit_points))
        check_correspondences([c.image_xy for c in fit], [c.plane_xy for c in fit], tol, policy.min_fit_points)
        check("fit_correspondences_sufficient_and_nondegenerate", True, "")
    except CalibrationError as exc:
        check("fit_correspondences_sufficient_and_nondegenerate", False, "fit correspondences: %s" % exc)
    check("holdout_present", len(hold) >= policy.min_holdout_points,
          "%d held-out point(s); need >= %d" % (len(hold), policy.min_holdout_points))
    fit_img = {c.image_xy for c in fit}
    fit_plane = {c.plane_xy for c in fit}
    check("holdout_disjoint_from_fit", not any(c.image_xy in fit_img or c.plane_xy in fit_plane for c in hold),
          "a held-out point coincides with a fit point")
    if m.valid_region is not None and m.control_points:
        inside = points_in_polygon([c.image_xy for c in m.control_points], m.valid_region, tol.region_boundary_tol_px).all()
        check("control_points_inside_region", inside, "a control point lies outside the documented valid region")
    else:
        check("control_points_inside_region", False, "cannot confirm control points lie inside a valid region")
    thr = m.holdout_error_threshold
    check("error_threshold_declared", thr is not None, "no acceptance threshold (in destination units) declared")
    try:
        fe, he = _errors(m, fit, tol), _errors(m, hold, tol)
    except CalibrationError as exc:
        fe = he = None
        reasons.append("residual computation failed: %s" % exc)
    check("fit_reproduced_by_homography", fe is not None and thr is not None and float(fe.max()) <= thr,
          "the homography does not reproduce its fit correspondences within the threshold")
    check("holdout_error_within_threshold", he is not None and thr is not None and float(he.max()) <= thr,
          "held-out error is missing or exceeds the threshold")
    v = m.verification
    if policy.require_independent_verification:
        check("independent_verification_claimed",
              v.status == "independently_verified" and _nonempty(v.verified_by) and _nonempty(v.verified_on) and _nonempty(v.reference),
              "no complete independent-verification claim (status/verified_by/verified_on/reference)")
    return EligibilityResult(
        manifest_eligible=all(ok for _, ok in checks), reasons=tuple(reasons), checks=tuple(checks),
        n_fit=len(fit), n_holdout=len(hold),
        fit_max_error=None if fe is None else float(fe.max()),
        holdout_rms_error=None if he is None else float(np.sqrt(np.mean(he ** 2))),
        holdout_max_error=None if he is None else float(he.max()),
        error_unit=m.plane_unit, threshold=thr)


# ------------------------------------------------------ raster compatibility
@dataclass(frozen=True)
class EffectiveCalibration:
    """Homography usable on ``observed_raster`` pixels."""

    homography: np.ndarray                    # maps observed-raster pixels to plane
    to_calibration_raster: Optional[np.ndarray]   # A^-1 (observed -> calibration raster), None if identical
    adapted: bool
    calibration_raster: RasterSpec
    observed_raster: RasterSpec
    homography_sha256: str


def _h_hash(H: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(H, dtype=np.float64).tobytes()).hexdigest()


def resolve_effective_calibration(manifest: CalibrationManifest, observed: RasterSpec,
                                  transform: Optional[RasterTransform] = None,
                                  tol: NumericTolerances = DEFAULT_TOLERANCES) -> EffectiveCalibration:
    """Return the homography valid for ``observed`` pixels or raise
    ``CalibrationError('RASTER_MISMATCH')``. Without a documented transform the
    rasters must be identical; with one, size must match the transform's
    prediction and the pixel domain/convention must be unchanged."""
    cal = manifest.raster
    if transform is None:
        diffs = [n for n in ("raster_id", "width", "height", "coordinate_system", "pixel_convention", "pixel_domain")
                 if getattr(cal, n) != getattr(observed, n)]
        if diffs:
            raise CalibrationError("RASTER_MISMATCH", "observed raster differs from the calibration raster in %s and no "
                                   "documented RasterTransform was supplied (calibration %r %dx%d, observed %r %dx%d)"
                                   % (diffs, cal.raster_id, cal.width, cal.height, observed.raster_id, observed.width, observed.height))
        return EffectiveCalibration(manifest.H, None, False, cal, observed, _h_hash(manifest.H))
    if cal.pixel_domain != observed.pixel_domain:
        raise CalibrationError("RASTER_MISMATCH", "pixel_domain differs (%s vs %s); a resize/crop cannot change "
                               "distortion state and no distortion model is implemented" % (cal.pixel_domain, observed.pixel_domain))
    if cal.pixel_convention != observed.pixel_convention:
        raise CalibrationError("RASTER_MISMATCH", "pixel_convention differs (%s vs %s)" % (cal.pixel_convention, observed.pixel_convention))
    ew, eh = transform.expected_size(cal)
    if abs(ew - observed.width) > tol.size_rounding_px or abs(eh - observed.height) > tol.size_rounding_px:
        raise CalibrationError("RASTER_MISMATCH", "the documented RasterTransform predicts a %.2fx%.2f raster but the "
                               "observed raster is %dx%d" % (ew, eh, observed.width, observed.height))
    A = transform.matrix(cal.pixel_convention)
    A_inv = np.linalg.inv(A)
    H_eff = manifest.H @ A_inv
    return EffectiveCalibration(H_eff, A_inv, True, cal, observed, _h_hash(H_eff))
