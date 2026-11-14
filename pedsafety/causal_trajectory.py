"""CAUSAL trajectory features: only the current and past observations are used.

This is the module to import in a real-time inference path. It does not import
``offline_trajectory`` and cannot interpolate or use centred smoothing; both are
refused at configuration time. Frames that are absent simply stay absent: the
next sample reports ``gap_before`` and has no velocity (nothing is filled using a
later observation).

Streaming and batch use the same engine. The batch helper feeds records in input
order, one at a time, through the streaming class.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .agent_state import RecordProvenance
from .errors import ConfigError, PedsafetyError
from .projection import ProjectionRecord
from .trajectory import (
    ORIGIN_OBSERVED, VIEW_CAUSAL, FeatureEngine, FeatureSample, Grouping, ObsSample, TrajectoryConfig, TrajectoryFeatures,
    TrajectoryIssue, TrajectorySetResult, assemble_arrays, check_parallel, group_tracks, rejected_row, sampling_report,
    select_position, trajectory_metadata,
)


class NonCausalInputError(PedsafetyError, TypeError):
    """An OFFLINE / non-causal object was given to a causal-only consumer."""


class CausalTrajectoryFeatures(TrajectoryFeatures):
    VIEW = VIEW_CAUSAL


def assert_causal_config(config: TrajectoryConfig) -> None:
    if config.max_interpolation_gap_frames != 0 or config.max_interpolation_gap_seconds is not None:
        raise ConfigError("interpolation uses future observations and is OFFLINE-only; a causal pipeline needs "
                          "max_interpolation_gap_frames=0 and no seconds limit")
    if config.smoothing.offline_only:
        raise ConfigError("%s is OFFLINE/NON-CAUSAL (it reads future samples); use causal_ema or none" % config.smoothing.describe())


def require_causal(obj: Any) -> Any:
    """Guard for real-time consumers: reject anything that is not a causal view."""
    if not isinstance(obj, CausalTrajectoryFeatures) or getattr(obj, "VIEW", None) != VIEW_CAUSAL:
        raise NonCausalInputError("expected a CAUSAL trajectory view, got %s (view=%r). Offline features may use future "
                                  "observations and must not enter a real-time inference path."
                                  % (type(obj).__name__, getattr(obj, "VIEW", None)))
    return obj


class CausalFeatureStream:
    """Feed observations one at a time, in time order, per track."""

    def __init__(self, config: Optional[TrajectoryConfig] = None):
        self.config = config or TrajectoryConfig()
        assert_causal_config(self.config)
        self._engines: Dict[str, FeatureEngine] = {}
        self._last: Dict[str, Tuple[int, float]] = {}
        self.issues: List[TrajectoryIssue] = []
        self._count = 0

    def update(self, record: Mapping[str, Any], provenance: RecordProvenance, projection: Optional[ProjectionRecord] = None,
               input_index: Optional[int] = None) -> FeatureSample:
        i = self._count if input_index is None else input_index
        self._count += 1
        tid, f, t = record["track_id"], record["frame_index"], record["timestamp"]
        eng = self._engines.setdefault(tid, FeatureEngine(self.config))
        last = self._last.get(tid)
        if last is not None and (f <= last[0] or t <= last[1]):
            # A stream cannot sort or retract: the late/duplicate sample is rejected and reported, state is untouched.
            code = "duplicate_or_out_of_order" if f <= last[0] else "timestamp_not_monotonic"
            self.issues.append(TrajectoryIssue(code, tid, i, f, "frame %d / t=%r arrives after frame %d / t=%r" % (f, t, last[0], last[1])))
            return rejected_row(tid, i, f, t, "rejected_" + code)
        pos, why = select_position(record, provenance, projection, self.config.coordinate_mode)
        raw = (pos.x, pos.y) if pos else None
        sample = eng.update(ObsSample(i, tid, f, t, ORIGIN_OBSERVED, pos, why, raw))
        self._last[tid] = (f, t)
        return sample


def process_causal(records: Sequence[Mapping[str, Any]], provenance: Sequence[RecordProvenance],
                   projections: Optional[Sequence[ProjectionRecord]] = None, config: Optional[TrajectoryConfig] = None) -> TrajectorySetResult:
    """Batch convenience over :class:`CausalFeatureStream`. Duplicate/out-of-order
    handling matches Stage 2 (see ``group_tracks``); accepted rows are fed in input order."""
    cfg = config or TrajectoryConfig()
    assert_causal_config(cfg)
    recs, provs = list(records), list(provenance)
    projs = None if projections is None else list(projections)
    check_parallel(recs, provs, projs)
    grouping: Grouping = group_tracks(recs, cfg.frame_step)
    accepted = sorted(i for idxs in grouping.groups.values() for i in idxs)
    stream = CausalFeatureStream(cfg)
    rows: Dict[str, List[FeatureSample]] = {t: [] for t in grouping.groups}
    infos: Dict[str, list] = {t: [] for t in grouping.groups}
    for i in accepted:
        rows[recs[i]["track_id"]].append(stream.update(recs[i], provs[i], None if projs is None else projs[i], input_index=i))
        pos, _ = select_position(recs[i], provs[i], None if projs is None else projs[i], cfg.coordinate_mode)
        if pos is not None:
            infos[recs[i]["track_id"]].append(pos)
    issues = list(grouping.issues) + stream.issues
    out: List[TrajectoryFeatures] = []
    excluded: List[str] = []
    for tid, idxs in grouping.groups.items():
        r = rows[tid]
        n_valid = sum(1 for s in r if s.position_state == "ok")
        short = cfg.min_valid_observations is not None and n_valid < cfg.min_valid_observations
        if short and cfg.short_track_policy == "exclude":
            excluded.append(tid)
            issues.append(TrajectoryIssue("track_too_short", tid, idxs[0], recs[idxs[0]]["frame_index"],
                                          "%d valid observation(s) < min_valid_observations=%d; excluded" % (n_valid, cfg.min_valid_observations)))
            continue
        meta = trajectory_metadata(tid, idxs[0], recs, provs, infos[tid], cfg)
        out.append(CausalTrajectoryFeatures(
            **meta, smoothing=cfg.smoothing.describe(), interpolation="none (causal)",
            sampling=sampling_report([s.frame_index for s in r], [s.timestamp for s in r], [s.position_state == "ok" for s in r], cfg),
            n_observed=n_valid, n_imputed=0, n_missing=0, too_short=bool(short), arrays=assemble_arrays(r)))
    counters = {"tracks": len(out), "excluded": len(excluded), "issues": len(issues)}
    return TrajectorySetResult(out, issues, excluded, counters)
