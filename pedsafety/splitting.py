"""Stage 5: leakage-aware, group-level train / validation / test splitting.

Rules enforced here:

* The unit of splitting is a GROUP (a scene, a recording, or a sequence) chosen by
  an explicit :class:`GroupingPolicy` that must state WHY those groups are
  independent. There is no default and nothing is inferred.
* Individual observations, tracks or windows are never assigned to partitions;
  whole groups are. A track or a sequence that maps to more than one group is an
  error (it would cross a boundary), as is missing grouping information.
* Group ids are namespaced by dataset, so equal ids in different datasets never
  collide, and sequence ids are never assumed to be globally unique.
* Assignment is deterministic and library-independent: groups are ordered by
  ``sha256(seed|group)`` and placed greedily by weighted relative deficit, so the
  result does not depend on input order or on a RNG implementation.
* Explicit assignments are validated for duplicates, conflicts, unknown groups and
  completeness.

Chronological (within-group) splitting is intentionally NOT implemented: it needs
a purge derived from the actual dependency intervals and is a protocol decision
that has not been approved. Asking for it is impossible through this API.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .agent_state import RecordProvenance
from .errors import ConfigError, PedsafetyError

PARTITIONS = ("train", "validation", "test")
UNITS = ("scene", "recording", "sequence")
SPLIT_PLAN_VERSION = "pedsafety-split-plan-v1"
METHOD_HASH_GREEDY = "hash_greedy_v1"
METHOD_EXPLICIT = "explicit"


class SplitError(PedsafetyError):
    """A split cannot be made safely. ``code`` is a stable identifier."""

    def __init__(self, code: str, message: str, details: Optional[Sequence[str]] = None):
        self.code = code
        self.details = list(details or [])
        shown = ""
        if self.details:
            shown = " [" + "; ".join(self.details[:8]) + (" ..." if len(self.details) > 8 else "") + "]"
        super().__init__("[%s] %s%s" % (code, message, shown))


def _nonempty(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def dataset_key(source_dataset: str) -> str:
    return source_dataset.strip().lower()


# ------------------------------------------------------------------ policy
@dataclass(frozen=True)
class GroupingPolicy:
    """Which key defines an independent group, and the stated reason.

    unit 'scene': group = the record's (dataset-namespaced) scene id.
    unit 'sequence': group = the record's sequence id; use only if you can state
        why separate sequences are independent recordings.
    unit 'recording': group = ``recording_map[(source_dataset, raw_sequence_id)]``;
        the map is mandatory and every sequence must appear in it.
    ``independence_basis`` is mandatory free text recorded in the manifest.
    ``isolate_scenes``: for units finer than a scene, also merge groups that share
        a scene id so that no scene crosses partitions.
    """

    unit: str
    independence_basis: str
    recording_map: Any = None
    isolate_scenes: bool = False

    def __post_init__(self):
        if self.unit not in UNITS:
            raise ConfigError("grouping unit must be one of %s, got %r" % (UNITS, self.unit))
        if not _nonempty(self.independence_basis):
            raise ConfigError("independence_basis is required: state why the chosen groups are independent recordings/scenes")
        if self.unit == "recording":
            m = self.recording_map
            if m is None or len(m) == 0:
                raise ConfigError("unit 'recording' requires an explicit recording_map {(source_dataset, raw_sequence_id): recording_id}")
            pairs = list(m.items()) if isinstance(m, Mapping) else list(m)
            norm: Dict[Tuple[str, str], str] = {}
            for item in pairs:
                try:
                    (ds, seq), rec = item
                except (TypeError, ValueError):
                    raise ConfigError("recording_map entries must be ((source_dataset, raw_sequence_id), recording_id)")
                if not (_nonempty(ds) and _nonempty(seq) and _nonempty(rec)):
                    raise ConfigError("recording_map entries must be non-empty strings, got %r" % (item,))
                key = (ds, seq)
                if key in norm and norm[key] != rec:
                    raise SplitError("CONFLICTING_RECORDING_MAP", "sequence %r maps to both %r and %r" % (key, norm[key], rec))
                if key in norm:
                    raise SplitError("DUPLICATE_RECORDING_MAP", "sequence %r is listed more than once" % (key,))
                norm[key] = rec
            object.__setattr__(self, "recording_map", norm)
        elif self.recording_map is not None:
            raise ConfigError("recording_map is only meaningful for unit 'recording'")


@dataclass(frozen=True)
class SplitConfig:
    """seed + ratios drive the deterministic hash-greedy assignment; alternatively
    ``explicit_assignments`` ((group, partition) pairs) must cover every group."""

    seed: int
    ratios: Mapping[str, float] = None
    explicit_assignments: Optional[Sequence[Tuple[str, str]]] = None

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ConfigError("seed must be an explicit integer, got %r" % (self.seed,))
        r = dict(self.ratios) if self.ratios is not None else {"train": 0.7, "validation": 0.15, "test": 0.15}
        if set(r) != set(PARTITIONS):
            raise ConfigError("ratios must have exactly the keys %s" % (PARTITIONS,))
        for k, v in r.items():
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0):
                raise ConfigError("ratio for %s must be a positive finite number, got %r" % (k, v))
        if abs(sum(r.values()) - 1.0) > 1e-9:
            raise ConfigError("ratios must sum to 1 (to 1e-9), got %r" % (sum(r.values()),))
        object.__setattr__(self, "ratios", {k: float(r[k]) for k in PARTITIONS})


# --------------------------------------------------------------- group maths
@dataclass(frozen=True)
class GroupInfo:
    key: str
    n_rows: int
    n_tracks: int
    datasets: Tuple[str, ...]
    scenes: Tuple[str, ...]
    sequences: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"n_rows": self.n_rows, "n_tracks": self.n_tracks, "datasets": list(self.datasets),
                "scenes": list(self.scenes), "sequences": list(self.sequences)}


def _base_group(record: Mapping[str, Any], prov: RecordProvenance, policy: GroupingPolicy) -> Optional[str]:
    if policy.unit == "scene":
        return "scene:%s" % record["scene_id"]
    if policy.unit == "sequence":
        return "sequence:%s" % record["sequence_id"]
    rec = policy.recording_map.get((prov.source_dataset, prov.raw_sequence_id))
    return None if rec is None else "recording:%s:%s" % (dataset_key(prov.source_dataset), rec)


def compute_groups(records: Sequence[Mapping[str, Any]], provenance: Sequence[RecordProvenance],
                   policy: GroupingPolicy) -> Tuple[List[str], Dict[str, GroupInfo]]:
    """Group key per record plus per-group counts. Raises :class:`SplitError` when the
    grouping is unreliable: missing mapping, a track or a sequence that spans groups."""
    if not isinstance(policy, GroupingPolicy):
        raise ConfigError("a GroupingPolicy is required; no default grouping exists")
    if len(records) != len(provenance):
        raise ValueError("records and provenance must be parallel (got %d and %d)" % (len(records), len(provenance)))
    keys: List[Optional[str]] = [_base_group(r, p, policy) for r, p in zip(records, provenance)]
    missing = sorted({"%s/%s" % (p.source_dataset, p.raw_sequence_id) for k, p in zip(keys, provenance) if k is None})
    if missing:
        raise SplitError("MISSING_GROUPING_INFO", "no recording is mapped for %d sequence(s); an explicit entry is required for each"
                         % len(missing), missing)
    if not keys:
        raise SplitError("NO_DATA", "no records to split")
    if policy.isolate_scenes and policy.unit != "scene":
        parent: Dict[str, str] = {k: k for k in keys}

        def find(a: str) -> str:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        first_group_of_scene: Dict[str, str] = {}
        for k, r in zip(keys, records):
            s = r["scene_id"]
            if s in first_group_of_scene:
                ra, rb = find(first_group_of_scene[s]), find(k)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            else:
                first_group_of_scene[s] = k
        keys = ["scenegroup:" + find(k) for k in keys]
    # containment: a track or a sequence must live in exactly one group
    track_groups: Dict[str, set] = {}
    seq_groups: Dict[Tuple[str, str], set] = {}
    for k, r, p in zip(keys, records, provenance):
        track_groups.setdefault(r["track_id"], set()).add(k)
        seq_groups.setdefault((dataset_key(p.source_dataset), p.raw_sequence_id), set()).add(k)
    bad_tracks = sorted("%s -> %s" % (t, sorted(g)) for t, g in track_groups.items() if len(g) > 1)
    if bad_tracks:
        raise SplitError("TRACK_SPANS_GROUPS", "%d track(s) map to more than one group; they cannot be placed in a single partition"
                         % len(bad_tracks), bad_tracks)
    bad_seqs = sorted("%s -> %s" % ("/".join(s), sorted(g)) for s, g in seq_groups.items() if len(g) > 1)
    if bad_seqs:
        raise SplitError("SEQUENCE_SPANS_GROUPS", "%d sequence(s) appear under more than one group (e.g. one sequence listed under "
                         "two scenes); the same video would cross partitions" % len(bad_seqs), bad_seqs)
    acc: Dict[str, Dict[str, set]] = {}
    rows: Dict[str, int] = {}
    for k, r, p in zip(keys, records, provenance):
        a = acc.setdefault(k, {"d": set(), "s": set(), "q": set(), "t": set()})
        a["d"].add(p.source_dataset)
        a["s"].add(r["scene_id"])
        a["q"].add(r["sequence_id"])
        a["t"].add(r["track_id"])
        rows[k] = rows.get(k, 0) + 1
    infos = {k: GroupInfo(k, rows[k], len(a["t"]), tuple(sorted(a["d"])), tuple(sorted(a["s"])), tuple(sorted(a["q"])))
             for k, a in sorted(acc.items())}
    return keys, infos


# ---------------------------------------------------------------------- plan
def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class SplitPlan:
    assignments: Mapping[str, str]
    groups: Mapping[str, GroupInfo]
    seed: int
    ratios: Mapping[str, float]
    method: str
    grouping_unit: str
    independence_basis: str
    isolate_scenes: bool

    def partition_of(self, group: str) -> str:
        return self.assignments[group]

    def groups_in(self, partition: str) -> List[str]:
        return sorted(g for g, p in self.assignments.items() if p == partition)

    def row_counts(self) -> Dict[str, int]:
        return {p: sum(self.groups[g].n_rows for g in self.groups_in(p)) for p in PARTITIONS}

    def to_dict(self) -> Dict[str, Any]:
        total = sum(g.n_rows for g in self.groups.values()) or 1
        counts = self.row_counts()
        return {
            "schema_version": SPLIT_PLAN_VERSION, "method": self.method, "seed": self.seed, "ratios": dict(self.ratios),
            "grouping": {"unit": self.grouping_unit, "independence_basis": self.independence_basis, "isolate_scenes": self.isolate_scenes},
            "assignments": {g: self.assignments[g] for g in sorted(self.assignments)},
            "groups": {g: self.groups[g].to_dict() for g in sorted(self.groups)},
            "partition_summary": {p: {"n_groups": len(self.groups_in(p)), "n_rows": counts[p], "row_fraction": counts[p] / total}
                                  for p in PARTITIONS},
        }

    def sha256(self) -> str:
        return hashlib.sha256(_canon(self.to_dict()).encode("utf-8")).hexdigest()


def validate_plan(plan: SplitPlan, observed_groups: Iterable[str]) -> None:
    """Completeness (every observed group assigned), exclusivity (exactly one valid
    partition each) and no phantom groups."""
    observed = set(observed_groups)
    assigned = set(plan.assignments)
    if observed - assigned:
        raise SplitError("INCOMPLETE_SPLIT", "groups without a partition", sorted(observed - assigned))
    if assigned - observed:
        raise SplitError("UNKNOWN_GROUP", "assignments for groups that do not exist in the data", sorted(assigned - observed))
    bad = sorted("%s -> %r" % (g, p) for g, p in plan.assignments.items() if p not in PARTITIONS)
    if bad:
        raise SplitError("INVALID_PARTITION", "partition must be one of %s" % (PARTITIONS,), bad)
    empty = [p for p in PARTITIONS if not plan.groups_in(p)]
    if empty:
        raise SplitError("EMPTY_PARTITION", "partition(s) %s received no group" % empty)


def _hash_order(groups: Iterable[str], seed: int) -> List[str]:
    return sorted(groups, key=lambda g: (hashlib.sha256(("%d|%s" % (seed, g)).encode("utf-8")).hexdigest(), g))


def make_plan(infos: Mapping[str, GroupInfo], policy: GroupingPolicy, cfg: SplitConfig) -> SplitPlan:
    """Assign whole groups to partitions (deterministic for equal inputs)."""
    if len(infos) < len(PARTITIONS):
        raise SplitError("INSUFFICIENT_GROUPS", "need at least %d independent groups for train/validation/test but found %d (%s); "
                         "choose a finer grouping unit with an honest independence basis, or add data" % (len(PARTITIONS), len(infos), sorted(infos)))
    if cfg.explicit_assignments is not None:
        seen: Dict[str, str] = {}
        dup, conflict = [], []
        for item in cfg.explicit_assignments:
            try:
                g, p = item
            except (TypeError, ValueError):
                raise ConfigError("explicit_assignments entries must be (group, partition) pairs, got %r" % (item,))
            if g in seen and seen[g] != p:
                conflict.append("%s -> %s and %s" % (g, seen[g], p))
            elif g in seen:
                dup.append(g)
            seen.setdefault(g, p)
        if conflict:
            raise SplitError("CONFLICTING_ASSIGNMENT", "a group is assigned to more than one partition", conflict)
        if dup:
            raise SplitError("DUPLICATE_ASSIGNMENT", "a group is listed more than once", dup)
        assignments, method = seen, METHOD_EXPLICIT
    else:
        order = _hash_order(infos, cfg.seed)
        total = float(sum(infos[g].n_rows for g in order))
        cur = {p: 0.0 for p in PARTITIONS}
        assignments = {}
        for g in order:
            deficit = {p: (cfg.ratios[p] * total - cur[p]) / (cfg.ratios[p] * total) for p in PARTITIONS}
            p = max(PARTITIONS, key=lambda q: (deficit[q], -PARTITIONS.index(q)))
            assignments[g] = p
            cur[p] += infos[g].n_rows
        method = METHOD_HASH_GREEDY
    plan = SplitPlan({g: assignments[g] for g in sorted(assignments)}, dict(sorted(infos.items())), cfg.seed, dict(cfg.ratios), method,
                     policy.unit, policy.independence_basis, policy.isolate_scenes)
    validate_plan(plan, infos)
    return plan


# ------------------------------------------------------------- isolation checks
def verify_window_isolation(windows_by_partition: Mapping[str, Sequence[Any]]) -> None:
    """Post-hoc check on the ACTUAL source data of generated windows, independent of
    how the split was made: no group, track, sequence or source frame may appear in
    more than one partition, and no two partitions may touch the same sequence."""
    seen_group: Dict[str, str] = {}
    seen_track: Dict[str, str] = {}
    seen_seq: Dict[Tuple[str, str], str] = {}
    seen_frame: Dict[Tuple[str, str, str, int], str] = {}
    problems: List[str] = []
    for part, windows in windows_by_partition.items():
        for w in windows:
            for registry, key, what in ((seen_group, w.group_id, "group"), (seen_track, w.track_id, "track"),
                                        (seen_seq, (w.dataset.strip().lower(), w.raw_sequence_id), "sequence")):
                prev = registry.setdefault(key, part)
                if prev != part:
                    problems.append("%s %r appears in %s and %s" % (what, key, prev, part))
            for f in tuple(w.history_frame_index) + tuple(w.target_frame_index):
                fk = (w.dataset.strip().lower(), w.raw_sequence_id, w.raw_track_id, int(f))
                prev = seen_frame.setdefault(fk, part)
                if prev != part:
                    problems.append("source frame %r appears in %s and %s" % (fk, prev, part))
    if problems:
        raise SplitError("PARTITION_LEAKAGE", "%d isolation violation(s) in the generated windows" % len(problems), sorted(set(problems)))
