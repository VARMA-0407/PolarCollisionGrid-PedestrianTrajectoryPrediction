import copy
import dataclasses
import json

import pytest

from pedsafety.errors import ConfigError
from pedsafety.splitting import (
    PARTITIONS, GroupingPolicy, SplitConfig, SplitError, compute_groups, make_plan, validate_plan, verify_window_isolation,
)
from stage5_fixtures import GROUP, SPLIT, WIN, corpus, dataset


def err(fn, *a, **k):
    with pytest.raises(SplitError) as ei:
        fn(*a, **k)
    return ei.value


# ============================================ 1-3 groups, scenes, overlapping windows
def test_recording_scene_groups_never_cross_partitions():
    ds = dataset()
    scenes_by_part = {p: {w.raw_scene_id for w in ds.windows[p]} for p in PARTITIONS}
    assert all(scenes_by_part[p] for p in PARTITIONS)
    for a in PARTITIONS:
        for b in PARTITIONS:
            if a < b:
                assert not scenes_by_part[a] & scenes_by_part[b], (a, b)
    assert set().union(*scenes_by_part.values()) == {"sc%d" % i for i in range(6)}
    assert sorted(g for p in PARTITIONS for g in ds.plan.groups_in(p)) == sorted(ds.plan.groups)       # complete and exclusive
    validate_plan(ds.plan, ds.plan.groups)


def test_trajectories_of_one_scene_all_land_in_one_partition():
    ds = dataset()
    recs, prov = corpus()
    keys, _ = compute_groups(recs, prov, GROUP)
    part_of_record = [ds.plan.assignments[k] for k in keys]
    for scene in {r["scene_id"] for r in recs}:
        assert len({p for p, r in zip(part_of_record, recs) if r["scene_id"] == scene}) == 1, scene
    for p in PARTITIONS:
        assert all(w.group_id in ds.plan.groups_in(p) for w in ds.windows[p])
        assert set(ds.partitions[p].record_indices) == {i for i, q in enumerate(part_of_record) if q == p}


def test_overlapping_windows_stay_together_and_isolation_check_detects_tampering():
    ds = dataset(windows=dataclasses.replace(WIN, window_stride=1))
    by_track = {}
    for p in PARTITIONS:
        for w in ds.windows[p]:
            by_track.setdefault(w.track_id, set()).add(p)
    assert all(len(v) == 1 for v in by_track.values())
    n_tracks_with_overlap = sum(1 for p in PARTITIONS for t in ds.partitions[p].trajectories
                                if len([w for w in ds.windows[p] if w.track_id == t.track_id]) > 1)
    assert n_tracks_with_overlap > 0                                             # the overlap really exists
    verify_window_isolation(ds.windows)
    train = ds.windows["train"]
    leaked = err(verify_window_isolation, {"train": train, "validation": train[:1]})
    assert leaked.code == "PARTITION_LEAKAGE"
    relabelled = dataclasses.replace(train[0], group_id="other", track_id="other:track")      # same source frames, new labels
    assert "source frame" in " ".join(err(verify_window_isolation, {"train": [train[0]], "test": [relabelled]}).details)
    assert err(verify_window_isolation, {"train": [train[0]], "test": [dataclasses.replace(train[0], group_id="g2", track_id="t2", raw_track_id="x")]}).code \
        == "PARTITION_LEAKAGE"                                                  # same sequence in two partitions


# ================================================================ 8 determinism
def test_split_is_deterministic_and_seed_dependent_and_order_independent():
    a, b = dataset(), dataset()
    assert a.plan.sha256() == b.plan.sha256() and a.report == b.report
    for p in PARTITIONS:
        assert [w.window_id for w in a.windows[p]] == [w.window_id for w in b.windows[p]]
    plans = {s: dataset(split=SplitConfig(seed=s)).plan.assignments for s in range(8)}
    assert len({json.dumps(v, sort_keys=True) for v in plans.values()}) > 1
    recs, prov = corpus()
    f_keys, f_infos = compute_groups(recs, prov, GROUP)
    r_keys, r_infos = compute_groups(recs[::-1], prov[::-1], GROUP)
    assert make_plan(f_infos, GROUP, SPLIT).assignments == make_plan(r_infos, GROUP, SPLIT).assignments


def test_ratios_and_counts_are_reported():
    ds = dataset()
    rep = ds.report
    total = sum(d["n_observations"] for d in rep["partitions"].values())
    assert total == 720 and sum(d["n_groups"] for d in rep["partitions"].values()) == 6
    assert {p: rep["partitions"][p]["n_groups"] for p in PARTITIONS} == {"train": 4, "validation": 1, "test": 1}
    assert sum(rep["row_fractions_achieved"].values()) == pytest.approx(1.0)
    for p in PARTITIONS:
        d = rep["partitions"][p]
        assert d["n_windows"] == len(ds.windows[p]) > 0 and sum(d["windows_by_agent_class"].values()) == d["n_windows"]
        assert set(d["windows_skipped"]) >= {"unusable_target", "mixed_coordinate_identity"}
    assert rep["grouping_unit"] == "scene" and rep["independence_basis"] and rep["seed"] == 7 and len(rep["split_plan_sha256"]) == 64
    json.dumps(rep)


def test_plan_manifest_records_the_decisions():
    d = dataset().plan.to_dict()
    assert d["schema_version"] == "pedsafety-split-plan-v1" and d["method"] == "hash_greedy_v1" and d["seed"] == 7
    assert d["grouping"]["independence_basis"] and set(d["assignments"].values()) == set(PARTITIONS)
    assert dataset(split=SplitConfig(seed=8)).plan.sha256() != dataset().plan.sha256()          # the seed is part of the identity


# ============================================ 9 duplicate / conflicting assignments
def _plan_inputs():
    recs, prov = corpus()
    _, infos = compute_groups(recs, prov, GROUP)
    return infos


def test_explicit_assignments_valid_and_validated():
    infos = _plan_inputs()
    g = sorted(infos)
    ok = [(g[0], "train"), (g[1], "train"), (g[2], "train"), (g[3], "train"), (g[4], "validation"), (g[5], "test")]
    plan = make_plan(infos, GROUP, SplitConfig(seed=1, explicit_assignments=ok))
    assert plan.method == "explicit" and plan.groups_in("test") == [g[5]]


@pytest.mark.parametrize("mutate,code", [
    (lambda a, g: a + [(g[0], "train")], "DUPLICATE_ASSIGNMENT"),
    (lambda a, g: a + [(g[0], "test")], "CONFLICTING_ASSIGNMENT"),
    (lambda a, g: a[:-1], "INCOMPLETE_SPLIT"),
    (lambda a, g: a + [("scene:SYN:ghost", "train")], "UNKNOWN_GROUP"),
    (lambda a, g: a[:-1] + [(g[5], "holdout")], "INVALID_PARTITION"),
    (lambda a, g: [(x, "train") for x, _ in a], "EMPTY_PARTITION"),
])
def test_bad_explicit_assignments_are_detected(mutate, code):
    infos = _plan_inputs()
    g = sorted(infos)
    base = [(g[0], "train"), (g[1], "train"), (g[2], "train"), (g[3], "train"), (g[4], "validation"), (g[5], "test")]
    assert err(make_plan, infos, GROUP, SplitConfig(seed=1, explicit_assignments=mutate(base, g))).code == code


def test_recording_map_duplicates_and_conflicts_are_detected():
    base = {"unit": "recording", "independence_basis": "x"}
    assert err(GroupingPolicy, recording_map=[(("SYN", "s0_0"), "r1"), (("SYN", "s0_0"), "r2")], **base).code == "CONFLICTING_RECORDING_MAP"
    assert err(GroupingPolicy, recording_map=[(("SYN", "s0_0"), "r1"), (("SYN", "s0_0"), "r1")], **base).code == "DUPLICATE_RECORDING_MAP"


def test_config_validation():
    for kw in (dict(seed=1.5), dict(seed=True), dict(seed=1, ratios={"train": 0.5, "validation": 0.5}), dict(seed=1, ratios={"train": 0.5, "validation": 0.3, "test": 0.3}),
               dict(seed=1, ratios={"train": 1.0, "validation": 0.0, "test": 0.0}), dict(seed=1, ratios={"train": 0.7, "validation": 0.15, "tset": 0.15})):
        with pytest.raises(ConfigError):
            SplitConfig(**kw)
    for kw in (dict(unit="country", independence_basis="x"), dict(unit="scene", independence_basis=""), dict(unit="scene", independence_basis="  "),
               dict(unit="scene", independence_basis="x", recording_map={("a", "b"): "c"}), dict(unit="recording", independence_basis="x"),
               dict(unit="recording", independence_basis="x", recording_map={("a", ""): "c"})):
        with pytest.raises(ConfigError):
            GroupingPolicy(**kw)


# ============================================== 10 missing or unreliable grouping metadata
def test_a_grouping_policy_is_mandatory():
    recs, prov = corpus()
    with pytest.raises(ConfigError, match="GroupingPolicy"):
        compute_groups(recs, prov, None)
    from pedsafety.dataset_builder import build_split_dataset
    with pytest.raises(TypeError):                                      # no default grouping exists
        build_split_dataset(recs, prov, split=SPLIT, windows=WIN)
    with pytest.raises(ConfigError, match="GroupingPolicy"):
        build_split_dataset(recs, prov, grouping="scene", split=SPLIT, windows=WIN)


def test_too_few_groups_fails_explicitly_instead_of_splitting_inside_a_group():
    assert err(dataset, n_scenes=2).code == "INSUFFICIENT_GROUPS"
    e = err(dataset, n_scenes=1, seqs_per_scene=4)                     # one scene, four sequences: sequences are NOT assumed independent
    assert e.code == "INSUFFICIENT_GROUPS" and "scene:syn:sc0" in str(e)


def test_recording_unit_requires_every_sequence_to_be_mapped():
    recs, prov = corpus()
    seqs = sorted({(p.source_dataset, p.raw_sequence_id) for p in prov})
    partial = {s: "rec_%s" % s[1] for s in seqs[:-2]}
    e = err(compute_groups, recs, prov, GroupingPolicy("recording", "x", recording_map=partial))
    assert e.code == "MISSING_GROUPING_INFO" and len(e.details) == 2 and "SYN/s5_1" in e.details


def test_recording_unit_groups_follow_the_explicit_map_and_preserve_provenance():
    recs, prov = corpus()
    rmap = {("SYN", "s%d_%d" % (s, q)): "rec%d" % s for s in range(6) for q in range(2)}
    pol = GroupingPolicy("recording", "each recording is a separate drive (synthetic)", recording_map=rmap)
    keys, infos = compute_groups(recs, prov, pol)
    assert sorted(infos) == ["recording:syn:rec%d" % i for i in range(6)]
    assert infos["recording:syn:rec0"].sequences == ("syn:s0_0", "syn:s0_1") and infos["recording:syn:rec0"].n_tracks == 6
    ds = dataset(grouping=pol)
    assert all(w.group_unit == "recording" and w.group_id.startswith("recording:syn:rec") for p in PARTITIONS for w in ds.windows[p])


def test_a_track_that_spans_two_groups_is_an_error():
    recs, prov = corpus()
    idx = [i for i, r in enumerate(recs) if r["track_id"] == recs[0]["track_id"]]
    for i in idx[10:]:
        recs[i]["scene_id"] = "syn:sc_other"
        prov[i] = dataclasses.replace(prov[i], raw_scene_id="sc_other")
    e = err(compute_groups, recs, prov, GROUP)
    assert e.code == "TRACK_SPANS_GROUPS" and recs[0]["track_id"] in str(e)


def test_a_sequence_under_two_scenes_is_an_error_because_the_same_video_would_cross_partitions():
    recs, prov = corpus()
    for i, r in enumerate(recs):
        if r["sequence_id"] == "syn:s0_0" and r["track_id"].endswith(":t0"):
            r["scene_id"] = "syn:sc_x"
            prov[i] = dataclasses.replace(prov[i], raw_scene_id="sc_x")
    e = err(compute_groups, recs, prov, GROUP)
    assert e.code == "SEQUENCE_SPANS_GROUPS" and "syn/s0_0" in str(e)


def test_group_ids_are_namespaced_by_dataset_so_equal_ids_never_collide():
    from pedsafety.agent_state import build_agent_states
    from stage5_fixtures import corpus_observations
    obs = corpus_observations(n_scenes=3, dataset="AAA") + corpus_observations(n_scenes=3, dataset="BBB")
    res = build_agent_states(obs)
    _, infos = compute_groups(res.records, res.provenance, GROUP)
    assert len(infos) == 6 and {k for k in infos if "sc0" in k} == {"scene:aaa:sc0", "scene:bbb:sc0"}


def test_sequence_unit_with_scene_isolation_merges_sequences_of_a_scene():
    recs, prov = corpus()
    pol = GroupingPolicy("sequence", "sequences are separate drives", isolate_scenes=True)
    keys, infos = compute_groups(recs, prov, pol)
    assert len(infos) == 6 and all(k.startswith("scenegroup:") for k in infos) and all(len(g.sequences) == 2 for g in infos.values())
    ds = dataset(grouping=pol)
    assert ds.report["warnings"] == [] or not any("scenes spanning" in w for w in ds.report["warnings"])


def test_sequence_unit_may_split_a_scene_but_the_report_warns_without_forbidding_it():
    recs, prov = corpus()
    pol = GroupingPolicy("sequence", "sequences are separate drives")
    _, infos = compute_groups(recs, prov, pol)
    g = sorted(infos)                                                    # 12 sequences; scene sc0 = first two keys
    assign = [(x, "train") for x in g]
    assign[0] = (g[0], "validation")
    assign[11] = (g[11], "test")
    ds = dataset(grouping=pol, split=SplitConfig(seed=1, explicit_assignments=assign))
    assert any("scenes spanning more than one partition" in w for w in ds.report["warnings"])
    isolated = GroupingPolicy("sequence", "x", isolate_scenes=True)
    ok = dataset(grouping=isolated)
    assert not any("scenes spanning" in w for w in ok.report["warnings"])
