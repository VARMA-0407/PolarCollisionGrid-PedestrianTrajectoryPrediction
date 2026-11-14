"""Stage 2: RawObservation -> canonical agent-state record (approved schema).

Design decisions (all documented in docs/phase2/task7_implementation.md):

* The approved schema is used unchanged. A record is built only from an
  observation that passed every check; each record is validated against the
  schema before it is returned.
* Identity (follows the Task 3 examples): ``track_id =
  "<dataset>:<sequence>:<source_track>"`` where ``<dataset>`` is the lower-cased
  ``source_dataset``; ``sequence_id``/``scene_id`` get the same ``<dataset>:``
  prefix. ``agent_id`` is null by default (no cross-scene identity is invented)
  or the raw source track id when the config declares ``source_track_id``.
* ``agent_class`` is a schema enum value chosen through an explicit, configurable
  label map. Unknown labels are an issue, never a default. The original label is
  kept in ``source_label``.
* ``world_position_valid`` is ALWAYS False in Stage 2. The schema cannot tell a
  declared world coordinate from a calibration-verified one, so declared world
  values and unverified planar values are withheld from the canonical record
  and kept only in the provenance sidecar. Stage 3 owns that decision.
* Raw ``vel_x``/``vel_y`` are withheld unless the config declares an image-space
  velocity contract (pixel, seconds). Nothing is estimated here.
* The schema has no field for "causal estimate"; every Stage 2 record is an
  observed value and the sidecar says so (``state_origin='observed'``).
* Nothing is sorted, deduplicated, clipped or repaired. Problem observations are
  quarantined (no record) and reported with a structured issue.
"""

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .errors import ConfigError
from .io.base import RawObservation
from .io.csv_adapter import (
    PHYSICAL_UNITS, SEMANTICS_IMAGE_PIXEL, SEMANTICS_UNVERIFIED, SEMANTICS_WORLD,
)
from .schema import collect_violations, load_schema, schema_sha256
from .validation import (
    ObservationIssueCode as C, ObservationValidationError, ValidationIssue, ValidationReport,
    bbox_center, bbox_footpoint, bbox_problem, is_real_number,
)

BUILDER_VERSION = "agent-state-0.2.0-stage2"
AGENT_ID_POLICIES = ("none", "source_track_id")

# Convenience default only; every dataset must review it. 'bike' is ambiguous
# (bicycle vs motorbike) and is mapped to 'bicyclist' here by assumption.
DEFAULT_CLASS_MAP: Mapping[str, str] = MappingProxyType({
    "pedestrian": "pedestrian", "ped": "pedestrian",
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "vehicle": "vehicle",
    "bike": "bicyclist", "bicycle": "bicyclist", "bicyclist": "bicyclist",
    "motorcycle": "motorcyclist", "motorbike": "motorcyclist", "motorcyclist": "motorcyclist",
})


@dataclass(frozen=True)
class VelocityContract:
    """Explicit declaration that raw ``vel_x``/``vel_y`` are image-space,
    pixel/second. World-space contracts are refused in Stage 2 because world
    kinematics need the Stage 3 calibration gate."""

    space: str = "image"
    position_unit: str = "pixel"
    time_unit: str = "s"

    def __post_init__(self):
        if self.space == "world":
            raise ConfigError("world-space velocity contracts are not accepted in Stage 2: world kinematics "
                              "require the Stage 3 calibration gate")
        if self.space != "image" or self.position_unit != "pixel" or self.time_unit != "s":
            raise ConfigError("only an image-space contract in pixel/s is supported in Stage 2, got "
                              "space=%r unit=%r time=%r" % (self.space, self.position_unit, self.time_unit))


@dataclass(frozen=True)
class AgentStateConfig:
    class_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_MAP))
    agent_id_policy: str = "none"
    velocity_contract: Optional[VelocityContract] = None
    strict: bool = False
    _norm_map: Dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.agent_id_policy not in AGENT_ID_POLICIES:
            raise ConfigError("agent_id_policy must be one of %s, got %r" % (AGENT_ID_POLICIES, self.agent_id_policy))
        allowed = set(load_schema()["properties"]["agent_class"]["enum"])
        norm: Dict[str, str] = {}
        for label, target in self.class_map.items():
            if not isinstance(label, str) or not label.strip():
                raise ConfigError("class_map keys must be non-empty strings, got %r" % (label,))
            if target not in allowed:
                raise ConfigError("class_map[%r] = %r is not a schema agent_class; allowed: %s"
                                  % (label, target, sorted(allowed)))
            key = label.strip().lower()
            if key in norm and norm[key] != target:
                raise ConfigError("class_map has conflicting entries for label %r after case/space normalisation" % key)
            norm[key] = target
        if self.velocity_contract is not None and not isinstance(self.velocity_contract, VelocityContract):
            raise ConfigError("velocity_contract must be a VelocityContract or None")
        object.__setattr__(self, "_norm_map", norm)


@dataclass(frozen=True)
class RecordProvenance:
    """Sidecar for one canonical record (the schema cannot carry these)."""

    input_index: int
    source_dataset: str
    raw_sequence_id: str
    raw_scene_id: str
    raw_track_id: str
    frame_index: int
    source_line: int
    source_label: str
    mapped_agent_class: str
    state_origin: str                      # always 'observed' in Stage 2
    position_semantics: Optional[str]
    position_unit: Optional[str]
    position_coordinate_system: Optional[str]
    withheld_position_status: Optional[str]    # e.g. 'declared_not_calibration_verified'
    withheld_position_xy: Optional[Tuple[float, float]]
    raw_velocity: Optional[Tuple[float, float]]
    velocity_status: Optional[str]         # 'emitted_under_declared_contract' | 'withheld_no_unit_contract'
    schema_sha256: str
    builder_version: str = BUILDER_VERSION


@dataclass
class BuildResult:
    records: List[Dict[str, Any]]
    provenance: List[RecordProvenance]
    report: ValidationReport


# ------------------------------------------------------------------ helpers
def dataset_key(source_dataset: str) -> str:
    return source_dataset.strip().lower()


def canonical_ids(o: RawObservation) -> Tuple[str, str, str]:
    """(sequence_id, scene_id, track_id) in canonical, dataset-namespaced form."""
    k = dataset_key(o.source_dataset)
    seq = "%s:%s" % (k, o.sequence_id)
    return seq, "%s:%s" % (k, o.scene_id), "%s:%s" % (seq, o.track_id)


def classify_label(label: Optional[str], cfg: AgentStateConfig) -> Optional[str]:
    """Schema agent_class for ``label`` or None if missing/unmapped (never a default)."""
    if not isinstance(label, str) or not label.strip():
        return None
    return cfg._norm_map.get(label.strip().lower())


def _ids_ok(o: RawObservation) -> bool:
    return all(isinstance(v, str) and v.strip() for v in (o.source_dataset, o.sequence_id, o.scene_id, o.track_id))


def _frame_ok(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _ts_ok(v: Any) -> bool:
    return is_real_number(v) and math.isfinite(v) and v >= 0


def _num_or_none_ok(v: Any) -> bool:
    return v is None or (is_real_number(v) and math.isfinite(v))


# ------------------------------------------------------- per-observation build
def _issue(o_idx: int, o: RawObservation, code: C, msg: str, field_name: Optional[str] = None, **details) -> ValidationIssue:
    def s(v):
        return v if isinstance(v, str) else None
    return ValidationIssue(
        code=code, message=msg, input_index=o_idx, source_line=getattr(o, "source_line", 0) or 0,
        source_dataset=s(o.source_dataset), sequence_id=s(o.sequence_id), track_id=s(o.track_id),
        frame_index=o.frame_index if _frame_ok(o.frame_index) else None, field=field_name,
        details=tuple(sorted((k, str(v)) for k, v in details.items())),
    )


def _build_one(idx: int, o: RawObservation, cfg: AgentStateConfig):
    """Return (record, provenance, issues). record/provenance are None if issues."""
    issues: List[ValidationIssue] = []
    add = lambda code, msg, f=None, **d: issues.append(_issue(idx, o, code, msg, f, **d))

    if not _ids_ok(o):
        add(C.INVALID_IDENTIFIER, "source_dataset, sequence_id, scene_id and track_id must be non-empty strings", "identity")
    if not _frame_ok(o.frame_index):
        add(C.INVALID_FRAME_INDEX, "frame_index must be a non-negative integer, got %r" % (o.frame_index,), "frame_index")
    if not _ts_ok(o.timestamp):
        add(C.INVALID_TIMESTAMP, "timestamp must be a finite number of seconds >= 0 from the sequence origin, got %r"
            % (o.timestamp,), "timestamp")

    # class
    agent_class = None
    if not isinstance(o.source_label, str) or not o.source_label.strip():
        add(C.MISSING_CLASS_LABEL, "source_label is missing; no default class is assumed", "source_label")
    else:
        agent_class = classify_label(o.source_label, cfg)
        if agent_class is None:
            add(C.UNMAPPED_CLASS, "source_label %r is not in the class map; add an explicit mapping" % (o.source_label,),
                "source_label", label=o.source_label)

    # confidence
    conf = o.detection_confidence
    if conf is not None and not (is_real_number(conf) and math.isfinite(conf) and 0.0 <= conf <= 1.0):
        add(C.INVALID_CONFIDENCE, "detection_confidence must be a finite number in [0, 1], got %r" % (conf,), "detection_confidence")

    # coordinate declaration
    sem, unit, cs = o.position_semantics, o.position_unit, o.position_coordinate_system
    has_bbox = o.bbox_xyxy is not None
    has_pos = o.pos_x is not None or o.pos_y is not None
    if (o.pos_x is None) != (o.pos_y is None):
        add(C.INCOMPLETE_POSITION, "pos_x and pos_y must both be present or both absent", "pos")
    if not (_num_or_none_ok(o.pos_x) and _num_or_none_ok(o.pos_y)):
        add(C.NON_FINITE_VALUE, "pos_x/pos_y must be finite numbers, got (%r, %r)" % (o.pos_x, o.pos_y), "pos")
    decl_ok = True
    if has_bbox or has_pos:
        if sem is None:
            decl_ok = False
            add(C.COORDINATE_DECLARATION_INVALID, "position geometry present but position_semantics is not declared", "position_semantics")
        elif sem == SEMANTICS_IMAGE_PIXEL:
            if unit != "pixel":
                decl_ok = False
                add(C.COORDINATE_DECLARATION_INVALID, "image_pixel requires unit 'pixel', got %r" % (unit,), "position_unit")
        elif sem == SEMANTICS_WORLD:
            if unit not in PHYSICAL_UNITS or not (isinstance(cs, str) and cs.strip()):
                decl_ok = False
                add(C.COORDINATE_DECLARATION_INVALID, "world_coordinate requires a physical unit %s and a named "
                    "coordinate system, got unit=%r system=%r" % (sorted(PHYSICAL_UNITS), unit, cs), "position_unit")
        elif sem == SEMANTICS_UNVERIFIED:
            if unit is not None:
                decl_ok = False
                add(C.COORDINATE_DECLARATION_INVALID, "unverified_planar must not carry a unit, got %r" % (unit,), "position_unit")
        else:
            decl_ok = False
            add(C.COORDINATE_DECLARATION_INVALID, "unknown position_semantics %r" % (sem,), "position_semantics")

    # bbox
    bbox_t = None
    if has_bbox:
        problem = bbox_problem(o.bbox_xyxy)
        if problem is not None:
            add(problem[0], problem[1], "bbox_xyxy")
        else:
            bbox_t = tuple(float(v) for v in o.bbox_xyxy)
        if decl_ok and sem != SEMANTICS_IMAGE_PIXEL:
            add(C.BBOX_SEMANTICS_MISMATCH, "bbox_xyxy is image-pixel geometry but position_semantics is %r" % (sem,), "bbox_xyxy")

    # image-space coordinate system name is required by the schema when image geometry exists
    image_geometry = decl_ok and sem == SEMANTICS_IMAGE_PIXEL and (has_bbox or has_pos)
    if image_geometry and not (isinstance(cs, str) and cs.strip()):
        add(C.MISSING_COORDINATE_SYSTEM, "image geometry present but position_coordinate_system is not named "
            "(the schema requires image_coordinate_system); none is invented", "position_coordinate_system")

    # velocity
    vx, vy = o.vel_x, o.vel_y
    raw_velocity, vel_status, emit_velocity = None, None, False
    if (vx is None) != (vy is None):
        add(C.INCOMPLETE_VELOCITY, "vel_x and vel_y must both be present or both absent", "velocity")
    elif not (_num_or_none_ok(vx) and _num_or_none_ok(vy)):
        add(C.NON_FINITE_VALUE, "vel_x/vel_y must be finite numbers, got (%r, %r)" % (vx, vy), "velocity")
    elif vx is not None:
        raw_velocity = (float(vx), float(vy))
        if cfg.velocity_contract is None:
            vel_status = "withheld_no_unit_contract"
        elif sem != SEMANTICS_IMAGE_PIXEL:
            add(C.VELOCITY_CONTRACT_MISMATCH, "image-space velocity contract declared but position_semantics is %r" % (sem,), "velocity")
        else:
            vel_status, emit_velocity = "emitted_under_declared_contract", True

    if issues:
        return None, None, issues

    seq, scene, track = canonical_ids(o)
    center = foot = None
    if bbox_t is not None:
        center, foot = bbox_center(bbox_t), bbox_footpoint(bbox_t)
    image_x = image_y = None
    if image_geometry and has_pos:
        image_x, image_y = float(o.pos_x), float(o.pos_y)
    rec: Dict[str, Any] = {
        "track_id": track,
        "agent_id": o.track_id if cfg.agent_id_policy == "source_track_id" else None,
        "agent_class": agent_class,
        "timestamp": float(o.timestamp),
        "frame_index": o.frame_index,
        "sequence_id": seq,
        "scene_id": scene,
        "bbox_xyxy": list(bbox_t) if bbox_t is not None else None,
        "bbox_center": list(center) if center else None,
        "footpoint": list(foot) if foot else None,
        "detection_confidence": float(conf) if conf is not None else None,
        "image_x": image_x,
        "image_y": image_y,
        "image_coordinate_system": cs if image_geometry else None,
        "image_coordinate_unit": "pixel" if image_geometry else None,
        "world_x": None, "world_y": None,
        "world_coordinate_system": None, "world_coordinate_unit": None,
        "world_position_valid": False,
        "velocity_x": float(vx) if emit_velocity else None,
        "velocity_y": float(vy) if emit_velocity else None,
        "speed": None, "acceleration_x": None, "acceleration_y": None,
        "kinematic_coordinate_space": "image" if emit_velocity else None,
        "kinematic_position_unit": "pixel" if emit_velocity else None,
        "kinematic_time_unit": "s" if emit_velocity else None,
        "heading": None, "visibility": None, "occlusion_status": None, "tracking_confidence": None,
        "observation_valid": True,
        "interpolation_flag": False,
        "source_dataset": o.source_dataset,
        "source_label": o.source_label,
    }
    violations = collect_violations(rec)
    if violations:
        return None, None, [_issue(idx, o, C.SCHEMA_VIOLATION, "constructed record violates the approved schema: %s"
                                   % "; ".join(str(v) for v in violations[:5]))]

    withheld_status = withheld_xy = None
    if has_pos and sem == SEMANTICS_WORLD:
        withheld_status, withheld_xy = "declared_world_not_calibration_verified", (float(o.pos_x), float(o.pos_y))
    elif has_pos and sem == SEMANTICS_UNVERIFIED:
        withheld_status, withheld_xy = "unverified_planar_not_in_record", (float(o.pos_x), float(o.pos_y))
    prov = RecordProvenance(
        input_index=idx, source_dataset=o.source_dataset, raw_sequence_id=o.sequence_id, raw_scene_id=o.scene_id,
        raw_track_id=o.track_id, frame_index=o.frame_index, source_line=o.source_line, source_label=o.source_label,
        mapped_agent_class=agent_class, state_origin="observed", position_semantics=sem, position_unit=unit,
        position_coordinate_system=cs, withheld_position_status=withheld_status, withheld_position_xy=withheld_xy,
        raw_velocity=raw_velocity, velocity_status=vel_status, schema_sha256=schema_sha256(),
    )
    return rec, prov, []


# -------------------------------------------------------------- batch checks
def _batch_issues(obs: List[RawObservation]) -> List[ValidationIssue]:
    """Duplicate, track-id-collision and temporal-order checks. No sorting: the
    input order is the evidence, and it is evaluated exactly as given."""
    out: List[ValidationIssue] = []
    usable = [i for i, o in enumerate(obs) if _ids_ok(o)]

    # duplicates: same (dataset, sequence, track, frame). ALL members are quarantined:
    # which one is right is unknowable, and nothing is averaged or picked.
    groups: Dict[Tuple[str, str, str, int], List[int]] = {}
    for i in usable:
        o = obs[i]
        if _frame_ok(o.frame_index):
            groups.setdefault((dataset_key(o.source_dataset), o.sequence_id, o.track_id, o.frame_index), []).append(i)
    for key, members in groups.items():
        if len(members) > 1:
            for i in members:
                out.append(_issue(i, obs[i], C.DUPLICATE_OBSERVATION,
                                  "observation shares (dataset, sequence, track, frame) with %d other input(s)" % (len(members) - 1),
                                  "identity", members=members, source_lines=[obs[m].source_line for m in members]))

    # canonical track_id collisions between distinct raw identities
    by_canon: Dict[str, Dict[Tuple[str, str], List[int]]] = {}
    for i in usable:
        o = obs[i]
        by_canon.setdefault(canonical_ids(o)[2], {}).setdefault((o.sequence_id, o.track_id), []).append(i)
    for canon, raws in by_canon.items():
        if len(raws) > 1:
            for members in raws.values():
                for i in members:
                    out.append(_issue(i, obs[i], C.TRACK_ID_COLLISION,
                                      "distinct (sequence, track) identities produce the same canonical track_id %r" % canon,
                                      "identity", raw_identities=sorted(raws)))

    # order: per track, frames must strictly increase in input order; timestamps too
    last: Dict[Tuple[str, str, str], Tuple[int, int, Optional[float]]] = {}
    for i in usable:
        o = obs[i]
        if not _frame_ok(o.frame_index):
            continue
        key = (dataset_key(o.source_dataset), o.sequence_id, o.track_id)
        prev = last.get(key)
        if prev is None:
            last[key] = (o.frame_index, i, float(o.timestamp) if _ts_ok(o.timestamp) else None)
            continue
        p_frame, p_idx, p_ts = prev
        if o.frame_index < p_frame:
            out.append(_issue(i, o, C.OUT_OF_ORDER_FRAME,
                              "frame %d appears after frame %d for the same track; input is not sorted or repaired"
                              % (o.frame_index, p_frame), "frame_index", previous_frame=p_frame, previous_input=p_idx))
        elif o.frame_index > p_frame:
            if p_ts is not None and _ts_ok(o.timestamp) and float(o.timestamp) <= p_ts:
                out.append(_issue(i, o, C.TIMESTAMP_NOT_MONOTONIC,
                                  "timestamp %r does not increase with frame_index (previous %r at frame %d)"
                                  % (o.timestamp, p_ts, p_frame), "timestamp", previous_frame=p_frame, previous_input=p_idx))
            else:
                last[key] = (o.frame_index, i, float(o.timestamp) if _ts_ok(o.timestamp) else p_ts)
    return out


def build_agent_states(observations: Iterable[RawObservation], config: Optional[AgentStateConfig] = None) -> BuildResult:
    """Convert observations to schema-valid records.

    Strict mode raises :class:`ObservationValidationError` (carrying the full
    report) if ANY issue exists. Lenient mode returns records for the
    observations without issues plus the structured report. Output order equals
    input order. Deterministic for identical input and configuration.
    """
    cfg = config or AgentStateConfig()
    obs = list(observations)
    built = [_build_one(i, o, cfg) for i, o in enumerate(obs)]
    issues: List[ValidationIssue] = []
    for _, _, iss in built:
        issues.extend(iss)
    issues.extend(_batch_issues(obs))
    issues.sort(key=lambda x: x.input_index)  # stable: per-observation issues precede batch issues

    bad = {i.input_index for i in issues}
    records, provenance = [], []
    counters = {"raw_velocity_withheld": 0, "declared_world_position_withheld": 0, "unverified_planar_position_withheld": 0}
    for i, (rec, prov, _) in enumerate(built):
        if i in bad or rec is None:
            continue
        records.append(rec)
        provenance.append(prov)
        if prov.velocity_status == "withheld_no_unit_contract":
            counters["raw_velocity_withheld"] += 1
        if prov.withheld_position_status == "declared_world_not_calibration_verified":
            counters["declared_world_position_withheld"] += 1
        if prov.withheld_position_status == "unverified_planar_not_in_record":
            counters["unverified_planar_position_withheld"] += 1
    report = ValidationReport(n_input=len(obs), n_records=len(records), issues=issues, counters=counters)
    if cfg.strict and issues:
        raise ObservationValidationError(report)
    return BuildResult(records, provenance, report)
