"""OFFLINE / NON-CAUSAL trajectory preparation. NOT for real-time inference.

Everything produced here is labelled ``OFFLINE_NON_CAUSAL``. It may use FUTURE
observations in two explicit, configurable ways:

* bounded linear interpolation of absent frames between two valid observations;
* optional centred moving-average smoothing.

Both are disabled by default. Interpolated samples are marked ``imputed`` in the
``origin`` array; raw observations and raw coordinates (``x``/``y``) are never
overwritten, and the interpolated values live only in the processed rows. Causal
code (``causal_trajectory.py``) does not import this module, and its
``require_causal`` guard rejects the objects returned here.

Interpolation never crosses: a gap larger than the configured maximum (frames
and optionally seconds), a row whose coordinate is unavailable, a track boundary
(rows are grouped per canonical track id), or a change of coordinate frame,
calibration or agent class (compared through ``PositionInfo.identity``).
"""

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .agent_state import RecordProvenance
from .projection import ProjectionRecord
from .trajectory import (
    ORIGIN_IMPUTED, ORIGIN_MISSING, ORIGIN_OBSERVED, VIEW_OFFLINE, FeatureEngine, FeatureSample, ObsSample, PositionInfo, TrajectoryConfig,
    TrajectoryFeatures, TrajectoryIssue, TrajectorySetResult, assemble_arrays, check_parallel, group_tracks, missing_row,
    sampling_report, select_position, trajectory_metadata,
)

INTERPOLATION_METHOD = "linear_offline_non_causal"


class OfflineTrajectoryFeatures(TrajectoryFeatures):
    VIEW = VIEW_OFFLINE


def _interpolate(entries: List[Dict[str, Any]], cfg: TrajectoryConfig) -> None:
    """Fill absent-frame runs in place under the configured limits (OFFLINE)."""
    g = cfg.max_interpolation_gap_frames
    if g <= 0:
        return
    i = 0
    n = len(entries)
    while i < n:
        if entries[i]["origin"] != ORIGIN_MISSING:
            i += 1
            continue
        j = i
        while j < n and entries[j]["origin"] == ORIGIN_MISSING:
            j += 1
        left, right = entries[i - 1], entries[j] if j < n else None
        run = j - i
        if right is not None and run <= g and left["pos"] is not None and right["pos"] is not None \
                and left["pos"].identity == right["pos"].identity and left["origin"] == ORIGIN_OBSERVED and right["origin"] == ORIGIN_OBSERVED \
                and (cfg.max_interpolation_gap_seconds is None or right["ts"] - left["ts"] <= cfg.max_interpolation_gap_seconds):
            for k in range(1, run + 1):
                a = k / (run + 1)
                p = left["pos"]
                x = p.x + a * (right["pos"].x - p.x)
                y = p.y + a * (right["pos"].y - p.y)
                entries[i + k - 1].update(origin=ORIGIN_IMPUTED, ts=left["ts"] + a * (right["ts"] - left["ts"]),
                                          pos=replace(p, x=x, y=y), reason="ok", raw=(x, y))
        i = j


def _smooth_centered(entries: List[Dict[str, Any]], cfg: TrajectoryConfig) -> List[Optional[PositionInfo]]:
    """Centred moving average over a COMPLETE window of usable samples, else unavailable."""
    w = cfg.smoothing.window
    h = (w - 1) // 2
    usable = (ORIGIN_OBSERVED, ORIGIN_IMPUTED) if cfg.allow_imputed_support else (ORIGIN_OBSERVED,)
    out: List[Optional[PositionInfo]] = []
    for i, e in enumerate(entries):
        if e["origin"] == ORIGIN_MISSING or e["pos"] is None:
            out.append(None)
            continue
        lo, hi = i - h, i + h
        if lo < 0 or hi >= len(entries):
            out.append(None)
            continue
        win = entries[lo:hi + 1]
        if all(x["origin"] in usable and x["pos"] is not None and x["pos"].identity == e["pos"].identity for x in win):
            out.append(replace(e["pos"], x=float(np.mean([x["pos"].x for x in win])), y=float(np.mean([x["pos"].y for x in win]))))
        else:
            out.append(None)
    return out


def process_offline(records: Sequence[Mapping[str, Any]], provenance: Sequence[RecordProvenance],
                    projections: Optional[Sequence[ProjectionRecord]] = None, config: Optional[TrajectoryConfig] = None) -> TrajectorySetResult:
    """Batch OFFLINE preparation. Returns ``OfflineTrajectoryFeatures`` (view
    OFFLINE_NON_CAUSAL) on a full frame grid: observed, imputed and missing rows."""
    cfg = config or TrajectoryConfig()
    recs, provs = list(records), list(provenance)
    projs = None if projections is None else list(projections)
    check_parallel(recs, provs, projs)
    grouping = group_tracks(recs, cfg.frame_step)
    issues = list(grouping.issues)
    out: List[TrajectoryFeatures] = []
    excluded: List[str] = []
    for tid, idxs in grouping.groups.items():
        obs_rows = []
        for i in idxs:
            pos, why = select_position(recs[i], provs[i], None if projs is None else projs[i], cfg.coordinate_mode)
            obs_rows.append(dict(i=i, frame=recs[i]["frame_index"], ts=recs[i]["timestamp"], pos=pos, reason=why))
        first, last = obs_rows[0]["frame"], obs_rows[-1]["frame"]
        by_frame = {r["frame"]: r for r in obs_rows}
        entries: List[Dict[str, Any]] = []
        for f in range(first, last + 1, cfg.frame_step):
            r = by_frame.get(f)
            if r is None:
                entries.append(dict(frame=f, i=-1, ts=float("nan"), origin=ORIGIN_MISSING, pos=None, reason="missing_position", raw=None))
            else:
                entries.append(dict(frame=f, i=r["i"], ts=r["ts"], origin=ORIGIN_OBSERVED, pos=r["pos"], reason=r["reason"],
                                    raw=None if r["pos"] is None else (r["pos"].x, r["pos"].y)))
        _interpolate(entries, cfg)
        smoothed = _smooth_centered(entries, cfg) if cfg.smoothing.offline_only else None
        engine = FeatureEngine(cfg)
        rows: List[FeatureSample] = []
        for k, e in enumerate(entries):
            if e["origin"] == ORIGIN_MISSING:
                rows.append(missing_row(tid, e["frame"]))
                continue
            pos, reason = e["pos"], e["reason"]
            if smoothed is not None and e["pos"] is not None:
                pos = smoothed[k]
                if pos is None:
                    reason = "smoothing_window_incomplete"
            rows.append(engine.update(ObsSample(e["i"], tid, e["frame"], e["ts"], e["origin"], pos, reason, e["raw"])))
        observed_valid = [r for r in obs_rows if r["pos"] is not None]
        short = cfg.min_valid_observations is not None and len(observed_valid) < cfg.min_valid_observations
        if short and cfg.short_track_policy == "exclude":
            excluded.append(tid)
            issues.append(TrajectoryIssue("track_too_short", tid, idxs[0], first, "%d valid observation(s) < min_valid_observations=%d; excluded"
                                          % (len(observed_valid), cfg.min_valid_observations)))
            continue
        meta = trajectory_metadata(tid, idxs[0], recs, provs, [r["pos"] for r in observed_valid], cfg)
        interp = "none" if cfg.max_interpolation_gap_frames == 0 else "%s(max_gap_frames=%d, max_gap_seconds=%s)" % (
            INTERPOLATION_METHOD, cfg.max_interpolation_gap_frames, cfg.max_interpolation_gap_seconds)
        out.append(OfflineTrajectoryFeatures(
            **meta, smoothing=cfg.smoothing.describe(), interpolation=interp,
            sampling=sampling_report([r["frame"] for r in obs_rows], [r["ts"] for r in obs_rows], [r["pos"] is not None for r in obs_rows], cfg),
            n_observed=len(observed_valid), n_imputed=sum(1 for r in rows if r.origin == ORIGIN_IMPUTED),
            n_missing=sum(1 for r in rows if r.origin == ORIGIN_MISSING), too_short=bool(short), arrays=assemble_arrays(rows)))
    return TrajectorySetResult(out, issues, excluded, {"tracks": len(out), "excluded": len(excluded), "issues": len(issues)})
