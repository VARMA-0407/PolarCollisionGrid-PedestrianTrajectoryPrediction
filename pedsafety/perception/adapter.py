"""Track records -> approved agent-state records (Stage 2A).

The adapter produces ``RawObservation`` objects and hands them to the EXISTING Stage 2 builder
(``pedsafety.agent_state.build_agent_states``), so every record is validated against the unchanged approved schema
and the identity, class and box rules are the ones already tested. The schema is not modified.

Mapping decisions:
* ``track_id`` (positive int) -> source track id string; the builder namespaces it as ``<dataset>:<sequence>:<id>``.
* ``scene_id`` is REQUIRED configuration: a perception pipeline has no scene concept and none is invented.
* ``source_label`` keeps the supported category (``car``/``truck``/``bus`` stay distinguishable); the schema class comes
  from the Stage 2 class map (car/truck/bus -> vehicle, bike -> bicyclist [an assumption recorded in Stage 2]).
* Position = the footpoint (bottom-centre of the box) in image pixels of the named raster; this is the convention the
  Stage 3/4 code expects for ``image_x``/``image_y``. It is a ground-contact proxy, not a measurement.
* ``detection_confidence`` = confidence of the matched detection; ``tracking_confidence`` is filled only if the tracker
  supplied one (the builder leaves it null; the adapter sets it and re-validates against the schema).
* Predicted-only track rows (no matched detection) are DROPPED and counted: they are estimates, and the schema has no way
  to mark a causal estimate, so writing one as an observation would be wrong (recorded Stage 4 safeguard #2).
* Output of MOCK components is only accepted when ``source_dataset`` contains "mock" (case-insensitive), so it cannot be
  mistaken for real data downstream.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..agent_state import AgentStateConfig, RecordProvenance, build_agent_states
from ..errors import ConfigError
from ..io.base import RawObservation
from ..schema import collect_violations
from ..validation import ValidationReport
from .pipeline import PerceptionResult
from .types import ComponentInfo, IssueCode, PerceptionError, PerceptionIssue, TrackRecord

ADAPTER_VERSION = "perception-adapter-0.1.0-stage2a"


@dataclass(frozen=True)
class AdapterConfig:
    source_dataset: str
    scene_id: str
    agent_state: Optional[AgentStateConfig] = None

    def __post_init__(self):
        for n in ("source_dataset", "scene_id"):
            v = getattr(self, n)
            if not (isinstance(v, str) and v.strip()):
                raise ConfigError("AdapterConfig.%s must be an explicit non-empty string (nothing is defaulted or invented)" % n)


@dataclass(frozen=True)
class PerceptionProvenance:
    """Sidecar parallel to the records (the schema cannot carry it)."""

    input_index: int
    track_id: int
    frame_index: int
    matched_detection_id: Optional[int]
    detector: ComponentInfo
    tracker: ComponentInfo
    frame_source: ComponentInfo
    is_mock: bool
    pipeline_version: str
    adapter_version: str = ADAPTER_VERSION


@dataclass
class AdapterResult:
    records: List[Dict[str, Any]]
    provenance: List[RecordProvenance]                 # Stage 2 sidecar (parallel to records)
    perception_provenance: List[PerceptionProvenance]  # perception sidecar (parallel to records)
    report: ValidationReport
    dropped_predicted: int
    is_mock: bool


def track_to_observation(t: TrackRecord, cfg: AdapterConfig) -> RawObservation:
    fx, fy = t.footpoint
    return RawObservation(
        source_dataset=cfg.source_dataset, sequence_id=t.sequence_id, scene_id=cfg.scene_id, track_id=str(t.track_id), frame_index=t.frame_index,
        timestamp=t.timestamp_s, source_label=t.category.value, bbox_xyxy=t.bbox_xyxy, pos_x=fx, pos_y=fy, position_semantics="image_pixel",
        position_unit="pixel", position_coordinate_system=t.coordinate_system, detection_confidence=t.detection_confidence, source_line=0)


def tracks_to_agent_states(result: PerceptionResult, cfg: AdapterConfig) -> AdapterResult:
    if result.is_mock and "mock" not in cfg.source_dataset.lower():
        raise ConfigError("the result comes from MOCK components; source_dataset %r must contain 'mock' so the records cannot be mistaken "
                          "for real detections or tracks" % (cfg.source_dataset,))
    tracks: List[TrackRecord] = []
    dropped = 0
    for fr in result.frames:
        for t in fr.tracks:
            if t.is_predicted:
                dropped += 1
            else:
                tracks.append(t)
    obs = [track_to_observation(t, cfg) for t in tracks]
    built = build_agent_states(obs, cfg.agent_state)
    records, provs, pprov = [], [], []
    for rec, prov in zip(built.records, built.provenance):
        t = tracks[prov.input_index]
        if t.tracking_confidence is not None:
            rec["tracking_confidence"] = float(t.tracking_confidence)
            bad = collect_violations(rec)
            if bad:
                raise PerceptionError(PerceptionIssue("adapter", IssueCode.INVALID_TRACK, "record violates the schema after setting "
                                                      "tracking_confidence: %s" % "; ".join(str(v) for v in bad[:3]), t.frame_index))
        records.append(rec)
        provs.append(prov)
        pprov.append(PerceptionProvenance(prov.input_index, t.track_id, t.frame_index, t.matched_detection_id, t.detector, t.tracker, result.source,
                                          result.is_mock, result.pipeline_version))
    return AdapterResult(records, provs, pprov, built.report, dropped, result.is_mock)
