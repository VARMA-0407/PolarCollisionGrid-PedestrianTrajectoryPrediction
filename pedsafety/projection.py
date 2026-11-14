"""Stage 3 integration: project Stage 2 agent-state records onto a calibration plane.

Inputs are the Stage 2 canonical records and their provenance sidecar; Stage 2
behaviour is untouched. Outputs:

* a NEW list of records (inputs are never mutated). Canonical ``world_*`` fields
  are filled, and ``world_position_valid`` set, ONLY when all of these hold:
  the manifest passed the software eligibility checks (``manifest_eligible``),
  the config explicitly enables promotion, and this point projected cleanly.
  Otherwise the canonical world fields stay null / false.
* a parallel sidecar of :class:`ProjectionRecord` holding the transformed planar
  coordinates, frame, unit, calibration identity/hash and the reason for every
  non-projection. Projective (unverified-scale) coordinates live ONLY here.

``manifest_eligible`` is a statement about the manifest's own claims and
evidence. It is not physical verification and the sidecar says so.
"""

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import (
    CalibrationManifest, EligibilityPolicy, EligibilityResult, RasterSpec, RasterTransform,
    evaluate_metric_eligibility, reference_sign, resolve_effective_calibration,
)
from .errors import ConfigError, PedsafetyError
from .geometry import (
    REASON_BEHIND_HORIZON, REASON_DENOMINATOR, REASON_NON_FINITE_INPUT, REASON_NON_FINITE_OUTPUT,
    CalibrationError, apply_homography, points_in_polygon,
)
from .agent_state import RecordProvenance
from .schema import collect_violations

PROJECTION_VERSION = "projection-0.3.0-stage3"
POSITION_SOURCES = ("image_xy", "footpoint")
KIND_PROJECTIVE = "projective_unverified_scale"
KIND_ELIGIBLE = "metric_eligible_by_manifest_checks"


class ProjectionStatus(str, Enum):
    PROJECTED = "projected"
    NO_CALIBRATION = "no_calibration"
    SKIPPED_NOT_IMAGE_SPACE = "skipped_not_image_space"
    NO_IMAGE_POSITION = "no_image_position"
    COORDINATE_SYSTEM_MISMATCH = "coordinate_system_mismatch"
    FOOTPOINT_MISMATCH = "footpoint_mismatch"
    NON_FINITE_INPUT = "non_finite_input"
    OUTSIDE_IMAGE = "outside_image"
    OUTSIDE_CALIBRATION_REGION = "outside_calibration_region"
    DENOMINATOR_UNSTABLE = "denominator_unstable"
    BEHIND_HORIZON = "behind_horizon"
    NON_FINITE_OUTPUT = "non_finite_output"
    RASTER_MISMATCH = "raster_mismatch"
    CALIBRATION_UNUSABLE = "calibration_unusable"
    SCHEMA_VIOLATION = "schema_violation"


_NOT_FAILURES = {ProjectionStatus.PROJECTED, ProjectionStatus.NO_CALIBRATION,
                 ProjectionStatus.SKIPPED_NOT_IMAGE_SPACE, ProjectionStatus.NO_IMAGE_POSITION}
_REASON_TO_STATUS = {
    REASON_NON_FINITE_INPUT: ProjectionStatus.NON_FINITE_INPUT,
    REASON_DENOMINATOR: ProjectionStatus.DENOMINATOR_UNSTABLE,
    REASON_BEHIND_HORIZON: ProjectionStatus.BEHIND_HORIZON,
    REASON_NON_FINITE_OUTPUT: ProjectionStatus.NON_FINITE_OUTPUT,
}


@dataclass(frozen=True)
class TransformConfig:
    """``position_source``: 'image_xy' (default: the record's image_x/image_y, the
    Task 3 footpoint convention) or 'footpoint' (the bbox-derived footpoint).
    ``footpoint_tolerance_px``: if set and both exist, image_xy farther than this
    from the footpoint is rejected (None = no cross-check).
    ``promote_eligible_to_canonical``: must be True before ANY canonical world
    field is written, even for an eligible manifest."""

    position_source: str = "image_xy"
    footpoint_tolerance_px: Optional[float] = None
    promote_eligible_to_canonical: bool = False
    policy: EligibilityPolicy = EligibilityPolicy()
    strict: bool = False

    def __post_init__(self):
        if self.position_source not in POSITION_SOURCES:
            raise ConfigError("position_source must be one of %s, got %r" % (POSITION_SOURCES, self.position_source))
        t = self.footpoint_tolerance_px
        if t is not None and not (isinstance(t, (int, float)) and not isinstance(t, bool) and t >= 0 and t == t and t != float("inf")):
            raise ConfigError("footpoint_tolerance_px must be a non-negative finite number or None")


@dataclass(frozen=True)
class ProjectionRecord:
    """Sidecar for one record (parallel to the records list)."""

    input_index: int
    track_id: str
    frame_index: int
    status: ProjectionStatus
    reason: Optional[str]
    plane_xy: Optional[Tuple[float, float]]
    plane_coordinate_system: Optional[str]
    plane_unit: Optional[str]
    coordinate_kind: Optional[str]
    promoted_to_canonical: bool
    calibration_id: Optional[str]
    calibration_version: Optional[str]
    manifest_sha256: Optional[str]
    effective_homography_sha256: Optional[str]
    raster_adapted: bool
    calibration_raster_id: Optional[str]
    observed_raster_id: Optional[str]
    source_point: Optional[str]
    image_xy_used: Optional[Tuple[float, float]]
    manifest_eligible: Optional[bool]
    eligibility_basis: str = "software_manifest_checks_only"
    physical_verification_established_by_software: bool = False
    projection_version: str = PROJECTION_VERSION


@dataclass(frozen=True)
class Preflight:
    calibration_id: str
    calibration_version: str
    manifest_sha256: str
    eligibility: EligibilityResult
    effective_homography_sha256: Optional[str]
    raster_adapted: bool
    error: Optional[str]


@dataclass
class TransformResult:
    records: List[Dict[str, Any]]
    projections: List[ProjectionRecord]
    preflight: Optional[Preflight]
    counters: Dict[str, int] = field(default_factory=dict)


class ProjectionError(PedsafetyError):
    def __init__(self, result: TransformResult):
        self.result = result
        bad = {k: v for k, v in result.counters.items() if k != "promoted" and k not in {s.value for s in _NOT_FAILURES} and v}
        super().__init__("projection failed for %d record(s): %s" % (sum(bad.values()), dict(sorted(bad.items()))))


def _select_point(r: Dict[str, Any], p: RecordProvenance, cfg: TransformConfig):
    if p.position_semantics != "image_pixel":
        return ProjectionStatus.SKIPPED_NOT_IMAGE_SPACE, "position_semantics is %r, not image_pixel" % (p.position_semantics,), None
    if cfg.position_source == "footpoint":
        pt = r.get("footpoint")
    else:
        pt = None if r.get("image_x") is None or r.get("image_y") is None else [r["image_x"], r["image_y"]]
    if pt is None:
        return ProjectionStatus.NO_IMAGE_POSITION, "record has no %s" % cfg.position_source, None
    x, y = float(pt[0]), float(pt[1])
    fp = r.get("footpoint")
    if cfg.position_source == "image_xy" and cfg.footpoint_tolerance_px is not None and fp is not None:
        d = float(np.hypot(x - fp[0], y - fp[1]))
        if d > cfg.footpoint_tolerance_px:
            return ProjectionStatus.FOOTPOINT_MISMATCH, "image_x/y is %.6g px from the bbox footpoint (limit %.6g)" % (d, cfg.footpoint_tolerance_px), None
    return None, None, (x, y)


def transform_agent_states(records: Sequence[Dict[str, Any]], provenance: Sequence[RecordProvenance],
                           manifest: Optional[CalibrationManifest], observed_raster: RasterSpec,
                           config: Optional[TransformConfig] = None,
                           raster_transform: Optional[RasterTransform] = None) -> TransformResult:
    cfg = config or TransformConfig()
    tol = cfg.policy.tolerances
    recs, provs = list(records), list(provenance)
    if len(recs) != len(provs):
        raise ValueError("records and provenance must be parallel (got %d and %d)" % (len(recs), len(provs)))
    for i, (r, p) in enumerate(zip(recs, provs)):
        if r.get("frame_index") != p.frame_index or r.get("source_dataset") != p.source_dataset:
            raise ValueError("provenance[%d] does not describe records[%d]" % (i, i))
    out = copy.deepcopy(recs)

    preflight, eff, ref_sign, calib_error, calib_status = None, None, None, None, None
    elig = None
    if manifest is not None:
        elig = evaluate_metric_eligibility(manifest, cfg.policy)
        try:
            eff = resolve_effective_calibration(manifest, observed_raster, raster_transform, tol)
            ref_sign = reference_sign(manifest, tol)
        except CalibrationError as exc:
            calib_error = str(exc)
            calib_status = ProjectionStatus.RASTER_MISMATCH if exc.code == "RASTER_MISMATCH" else ProjectionStatus.CALIBRATION_UNUSABLE
        preflight = Preflight(manifest.provenance.calibration_id, manifest.provenance.version, manifest.sha256(), elig,
                              None if eff is None else eff.homography_sha256, bool(eff and eff.adapted), calib_error)

    n = len(recs)
    status: List[Optional[ProjectionStatus]] = [None] * n
    reason: List[Optional[str]] = [None] * n
    obs_pts: Dict[int, Tuple[float, float]] = {}
    for i, (r, p) in enumerate(zip(recs, provs)):
        if manifest is None:
            status[i], reason[i] = ProjectionStatus.NO_CALIBRATION, "no calibration manifest supplied"
            continue
        st, why, pt = _select_point(r, p, cfg)
        if st is not None:
            status[i], reason[i] = st, why
            continue
        if r.get("image_coordinate_system") != observed_raster.coordinate_system:
            status[i], reason[i] = ProjectionStatus.COORDINATE_SYSTEM_MISMATCH, (
                "record image_coordinate_system %r != observed raster %r" % (r.get("image_coordinate_system"), observed_raster.coordinate_system))
            continue
        if calib_status is not None:
            status[i], reason[i] = calib_status, calib_error
            continue
        if not np.isfinite(pt).all():
            status[i], reason[i] = ProjectionStatus.NON_FINITE_INPUT, "image position is not finite"
            continue
        if not observed_raster.contains([pt])[0]:
            status[i], reason[i] = ProjectionStatus.OUTSIDE_IMAGE, "point %r is outside the observed %dx%d raster" % (pt, observed_raster.width, observed_raster.height)
            continue
        pc = np.array(pt) if eff.to_calibration_raster is None else (eff.to_calibration_raster @ np.array([pt[0], pt[1], 1.0]))[:2]
        in_cal = manifest.raster.contains([pc])[0]
        if in_cal and manifest.valid_region is not None:
            in_cal = bool(points_in_polygon([pc], manifest.valid_region, tol.region_boundary_tol_px)[0])
        if not in_cal:
            status[i], reason[i] = ProjectionStatus.OUTSIDE_CALIBRATION_REGION, "point maps outside the calibrated %s" % (
                "region" if manifest.valid_region is not None else "raster")
            continue
        obs_pts[i] = pt

    plane: Dict[int, Tuple[float, float]] = {}
    if obs_pts:
        idx = sorted(obs_pts)
        res = apply_homography(eff.homography, [obs_pts[i] for i in idx], tol, ref_sign)
        for k, i in enumerate(idx):
            if res.valid[k]:
                plane[i] = (float(res.xy[k, 0]), float(res.xy[k, 1]))
                status[i] = ProjectionStatus.PROJECTED
            else:
                status[i], reason[i] = _REASON_TO_STATUS[res.reasons[k]], res.reasons[k]

    eligible = bool(elig and elig.manifest_eligible)
    promoted = [False] * n
    for i in plane:
        if eligible and cfg.promote_eligible_to_canonical:
            before = out[i]
            cand = dict(before)
            cand.update(world_x=plane[i][0], world_y=plane[i][1], world_coordinate_system=manifest.plane_coordinate_system,
                        world_coordinate_unit=manifest.plane_unit, world_position_valid=True)
            viol = collect_violations(cand)
            if viol:
                status[i], reason[i] = ProjectionStatus.SCHEMA_VIOLATION, "; ".join(str(v) for v in viol[:3])
            else:
                out[i], promoted[i] = cand, True

    projections: List[ProjectionRecord] = []
    for i, (r, p) in enumerate(zip(recs, provs)):
        ok = status[i] == ProjectionStatus.PROJECTED
        projections.append(ProjectionRecord(
            input_index=i, track_id=r["track_id"], frame_index=r["frame_index"], status=status[i], reason=reason[i],
            plane_xy=plane.get(i) if ok else None,
            plane_coordinate_system=manifest.plane_coordinate_system if (ok and manifest) else None,
            plane_unit=manifest.plane_unit if (ok and manifest) else None,
            coordinate_kind=(KIND_ELIGIBLE if eligible else KIND_PROJECTIVE) if ok else None,
            promoted_to_canonical=promoted[i],
            calibration_id=manifest.provenance.calibration_id if manifest else None,
            calibration_version=manifest.provenance.version if manifest else None,
            manifest_sha256=preflight.manifest_sha256 if preflight else None,
            effective_homography_sha256=preflight.effective_homography_sha256 if preflight else None,
            raster_adapted=bool(preflight and preflight.raster_adapted),
            calibration_raster_id=manifest.raster.raster_id if manifest else None,
            observed_raster_id=observed_raster.raster_id if manifest else None,
            source_point=cfg.position_source if i in obs_pts or ok else None,
            image_xy_used=obs_pts.get(i), manifest_eligible=eligible if manifest else None))
    counters: Dict[str, int] = {}
    for s in status:
        counters[s.value] = counters.get(s.value, 0) + 1
    counters["promoted"] = sum(promoted)
    result = TransformResult(out, projections, preflight, dict(sorted(counters.items())))
    if cfg.strict and any(s not in _NOT_FAILURES for s in status):
        raise ProjectionError(result)
    return result
