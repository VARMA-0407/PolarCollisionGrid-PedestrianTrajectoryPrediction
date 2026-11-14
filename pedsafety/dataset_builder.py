"""Stage 5 orchestration: records -> groups -> partitions -> per-partition
trajectories -> windows -> isolation proof -> report.

The order is what prevents leakage:

1. every record gets a group (explicit policy; unreliable grouping is an error);
2. whole groups are assigned to train / validation / test;
3. each partition's records are processed SEPARATELY by Stage 4, so interpolation,
   smoothing and trajectory construction can never read another partition;
4. windows are generated inside each partition;
5. isolation of groups, tracks, sequences and source frames is re-verified on the
   generated windows;
6. normalisation is fitted afterwards from the training windows only.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .agent_state import RecordProvenance
from .causal_trajectory import process_causal
from .errors import ConfigError
from .normalization import NormalizationConfig, NormalizationState
from .offline_trajectory import process_offline
from .projection import ProjectionRecord
from .splitting import (
    PARTITIONS, GroupingPolicy, SplitConfig, SplitError, SplitPlan, compute_groups, dataset_key, make_plan, verify_window_isolation,
)
from .trajectory import TrajectoryConfig, TrajectoryIssue, check_parallel, select_position
from .windows import SKIP_REASONS, SplitWindow, WindowConfig, generate_windows

DATASET_BUILDER_VERSION = "dataset-builder-0.5.0-stage5"
VIEWS = ("causal", "offline")


@dataclass
class PartitionData:
    name: str
    groups: List[str]
    n_rows: int
    trajectories: list
    windows: List[SplitWindow]
    trajectory_issues: List[TrajectoryIssue]
    skipped: Dict[str, int]
    record_indices: List[int]


@dataclass
class SplitDataset:
    plan: SplitPlan
    partitions: Dict[str, PartitionData]
    report: Dict[str, Any]
    view: str
    window_config: WindowConfig
    trajectory_config: TrajectoryConfig

    @property
    def windows(self) -> Dict[str, List[SplitWindow]]:
        return {p: self.partitions[p].windows for p in PARTITIONS}

    def fit_normalization(self, config: Optional[NormalizationConfig] = None) -> NormalizationState:
        """Fit from the TRAIN partition only, recording the split identity."""
        info = {
            "split_plan_sha256": self.plan.sha256(), "split_seed": self.plan.seed, "split_method": self.plan.method,
            "grouping_unit": self.plan.grouping_unit, "view": self.view, "window_config": self.window_config.to_dict(),
            "trajectory_config": {"coordinate_mode": self.trajectory_config.coordinate_mode.value, "frame_step": self.trajectory_config.frame_step,
                                  "max_interpolation_gap_frames": self.trajectory_config.max_interpolation_gap_frames,
                                  "smoothing": self.trajectory_config.smoothing.describe(),
                                  "allow_imputed_support": self.trajectory_config.allow_imputed_support},
        }
        return NormalizationState.fit(self.windows["train"], config, info)


def build_split_dataset(records: Sequence[Mapping[str, Any]], provenance: Sequence[RecordProvenance],
                        projections: Optional[Sequence[ProjectionRecord]] = None, *, grouping: GroupingPolicy, split: SplitConfig,
                        windows: WindowConfig, trajectory: Optional[TrajectoryConfig] = None, view: str = "causal") -> SplitDataset:
    if view not in VIEWS:
        raise ConfigError("view must be one of %s, got %r" % (VIEWS, view))
    if view == "offline" and not windows.allow_offline_view:
        raise ConfigError("the offline view may use future observations in its history features; set "
                          "WindowConfig.allow_offline_view=True to accept that explicitly")
    tcfg = trajectory or TrajectoryConfig()
    recs, provs = list(records), list(provenance)
    projs = None if projections is None else list(projections)
    check_parallel(recs, provs, projs)
    keys, infos = compute_groups(recs, provs, grouping)
    plan = make_plan(infos, grouping, split)
    runner = process_offline if view == "offline" else process_causal

    parts: Dict[str, PartitionData] = {}
    for name in PARTITIONS:
        idx = [i for i, k in enumerate(keys) if plan.assignments[k] == name]
        sub_r, sub_p = [recs[i] for i in idx], [provs[i] for i in idx]
        sub_j = None if projs is None else [projs[i] for i in idx]
        res = runner(sub_r, sub_p, sub_j, tcfg)           # ONLY this partition's records are visible
        identities = {}
        for j, i in enumerate(idx):
            pos, _ = select_position(recs[i], provs[i], None if projs is None else projs[i], tcfg.coordinate_mode)
            identities[j] = None if pos is None else pos.identity
        group_of_track = {recs[i]["track_id"]: keys[i] for i in idx}
        wins, skipped = generate_windows(res.trajectories, partition=name, group_of_track=group_of_track, group_unit=plan.grouping_unit,
                                         wcfg=windows, identities=identities, global_index=idx)
        stray = [t.track_id for t in res.trajectories if plan.assignments[group_of_track[t.track_id]] != name]
        if stray:
            raise SplitError("PARTITION_LEAKAGE", "trajectories computed in %s belong to other partitions" % name, stray)
        parts[name] = PartitionData(name, plan.groups_in(name), len(idx), res.trajectories, wins, res.issues, skipped, idx)
    verify_window_isolation({n: p.windows for n, p in parts.items()})
    return SplitDataset(plan, parts, _report(plan, parts, recs, provs, keys, grouping), view, windows, tcfg)


def _count(items) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return dict(sorted(out.items()))


def _report(plan, parts, recs, provs, keys, grouping) -> Dict[str, Any]:
    per = {}
    for name, p in parts.items():
        issue_codes = _count(i.code for i in p.trajectory_issues)
        per[name] = {
            "n_groups": len(p.groups), "groups": p.groups, "n_observations": p.n_rows, "n_trajectories": len(p.trajectories),
            "n_windows": len(p.windows), "windows_by_agent_class": _count(w.agent_class for w in p.windows),
            "windows_by_dataset": _count(w.dataset for w in p.windows), "windows_by_coordinate_mode": _count(w.coordinate_mode for w in p.windows),
            "windows_skipped": {r: p.skipped.get(r, 0) for r in SKIP_REASONS}, "trajectory_issue_counts": issue_codes,
        }
    warnings: List[str] = []
    for name, d in per.items():
        if d["n_windows"] == 0:
            warnings.append("partition %r produced no windows" % name)
    scene_parts: Dict[str, set] = {}
    for k, r in zip(keys, recs):
        scene_parts.setdefault(r["scene_id"], set()).add(plan.assignments[k])
    crossing = sorted(s for s, ps in scene_parts.items() if len(ps) > 1)
    if crossing:
        msg = "scenes spanning more than one partition: %s" % crossing
        if grouping.unit == "scene" or grouping.isolate_scenes:
            raise SplitError("SCENE_CROSSES_PARTITIONS", msg)
        warnings.append(msg + " (allowed: grouping unit is %r and isolate_scenes is False)" % grouping.unit)
    return {
        "builder_version": DATASET_BUILDER_VERSION, "split_plan_sha256": plan.sha256(), "seed": plan.seed, "method": plan.method,
        "grouping_unit": plan.grouping_unit, "independence_basis": plan.independence_basis, "n_groups": len(plan.groups),
        "ratios_target": dict(plan.ratios), "partitions": per, "warnings": warnings,
        "row_fractions_achieved": {p: v["row_fraction"] for p, v in plan.to_dict()["partition_summary"].items()},
    }
