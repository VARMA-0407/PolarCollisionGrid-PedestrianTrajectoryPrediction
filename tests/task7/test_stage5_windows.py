import copy
import dataclasses

import numpy as np
import pytest

import pedsafety.dataset_builder as builder
from pedsafety.agent_state import build_agent_states
from pedsafety.errors import ConfigError
from pedsafety.splitting import PARTITIONS, GroupingPolicy, SplitConfig, compute_groups
from pedsafety.trajectory import TrajectoryConfig
from pedsafety.windows import WindowConfig
from stage4_fixtures import make_track
from stage5_fixtures import GROUP, SPLIT, WIN, corpus, corpus_observations, dataset


def all_windows(ds):
    return [w for p in PARTITIONS for w in ds.windows[p]]


def same(a, b):
    return np.array_equal(a, b, equal_nan=np.asarray(a).dtype.kind == "f")


# ====================================================== 11 provenance + temporal order
def test_windows_preserve_provenance_and_temporal_order():
    ds = dataset()
    recs, prov = corpus()
    for w in all_windows(ds):
        hf, tf = w.history_frame_index, w.target_frame_index
        assert (np.diff(hf) == 1).all() and (np.diff(tf) == 1).all() and tf[0] == hf[-1] + 1 and not set(hf) & set(tf)
        assert (np.diff(w.history_timestamp) > 0).all() and (np.diff(w.target_timestamp) > 0).all() and w.target_timestamp[0] > w.history_timestamp[-1]
        assert (w.dataset, w.view, w.history_may_use_future, w.partition in PARTITIONS) == ("SYN", "CAUSAL", False, True)
        assert w.track_id == "syn:%s:%s" % (w.raw_sequence_id, w.raw_track_id) and w.raw_scene_id == "sc" + w.raw_sequence_id[1]
        # every step points back at the exact source record
        for src, fr in zip(np.concatenate([w.history_source_index, w.target_source_index]), np.concatenate([hf, tf])):
            assert recs[src]["track_id"] == w.track_id and recs[src]["frame_index"] == fr and prov[src].raw_track_id == w.raw_track_id
        assert w.obs_len == 4 and w.pred_len == 3 and len(w.history["xs"]) == 4 and w.target_xy.shape == (3, 2)
    ids = [w.window_id for w in all_windows(ds)]
    assert len(ids) == len(set(ids))


def test_windows_are_ordered_by_track_then_start_frame():
    ds = dataset()
    for p in PARTITIONS:
        starts = {}
        for w in ds.windows[p]:
            starts.setdefault(w.track_id, []).append(int(w.history_frame_index[0]))
        assert all(s == sorted(s) for s in starts.values())


def test_window_arrays_are_read_only():
    w = dataset().windows["train"][0]
    for arr in (w.history["xs"], w.target_xy, w.history_frame_index, w.history_valid, w.history_available["vx"]):
        with pytest.raises(ValueError):
            arr[...] = 0


# ============================================================ configured sampling
def test_window_count_matches_the_configured_stride():
    ds = dataset(windows=WindowConfig(obs_len=4, pred_len=3, window_stride=1))
    per_track = [w for w in ds.windows["train"] if w.track_id == ds.windows["train"][0].track_id]
    assert len(per_track) == 20 - 7 + 1
    ds3 = dataset(windows=WindowConfig(obs_len=4, pred_len=3, window_stride=3))
    t3 = [w for w in ds3.windows["train"] if w.track_id == ds3.windows["train"][0].track_id]
    assert [int(w.history_frame_index[0]) for w in t3] == [0, 3, 6, 9, 12]


def test_frame_stride_spaces_steps_and_keeps_native_rate_features():
    ds = dataset(windows=WindowConfig(obs_len=3, pred_len=2, frame_stride=2))
    w = ds.windows["train"][0]
    assert list(np.diff(w.history_frame_index)) == [2, 2] and list(np.diff(w.target_frame_index)) == [2] and w.target_frame_index[0] == w.history_frame_index[-1] + 2
    assert w.frame_stride == 2 and np.diff(w.history_timestamp) == pytest.approx([0.2, 0.2])
    base = dataset().windows["train"][0]
    assert w.history["vx"][-1] == pytest.approx(base.history["vx"][-1], rel=1e-9)             # velocity is still the native 0.1 s difference


def test_gaps_that_violate_the_policy_never_produce_windows():
    ds = dataset(drop_by_track={0: (8, 9), 1: (12,)})
    skipped = {r: sum(ds.partitions[p].skipped[r] for p in PARTITIONS) for r in ds.partitions["train"].skipped}
    assert sum(skipped.values()) > 0
    for w in all_windows(ds):
        if w.raw_track_id == "t0":
            assert not ({8, 9} & set(w.history_frame_index) | {8, 9} & set(w.target_frame_index))
        if w.raw_track_id == "t1":
            assert 12 not in set(w.history_frame_index) | set(w.target_frame_index)
        assert (np.diff(np.concatenate([w.history_frame_index, w.target_frame_index])) == 1).all()


def test_frame_stride_cannot_straddle_a_gap_unless_contiguity_is_waived():
    obs = []                                                              # 4 scenes x 1 track, frame 5 missing in all
    for s in range(4):
        obs += make_track("t0", frames=range(20), seq="s%d" % s, scene="sc%d" % s, drop=(5,), x0=200.0 + 50 * s)
    res = build_agent_states(obs)
    strict = dataset(res.records, res.provenance, windows=WindowConfig(obs_len=3, pred_len=2, frame_stride=2))
    loose = dataset(res.records, res.provenance, windows=WindowConfig(obs_len=3, pred_len=2, frame_stride=2, require_contiguous_native_frames=False))
    # a stride-2 window starting at frame 1 uses frames 1,3,5(missing)...: strict rejects, and frame 5 is never touched implicitly
    straddle = lambda ds: [w for p in PARTITIONS for w in ds.windows[p] if w.history_frame_index[0] < 5 < w.target_frame_index[-1]]
    assert straddle(strict) == []
    assert sum(strict.partitions[p].skipped["gap_in_native_frames"] for p in PARTITIONS) > 0
    assert len(straddle(loose)) > 0 or sum(loose.partitions[p].skipped["unusable_target"] for p in PARTITIONS) > 0


# ============================================= validity / availability / left padding
def test_feature_availability_masks_and_nan_not_zero():
    w = dataset().windows["train"][0]
    assert int(w.history_frame_index[0]) == 0
    assert w.history_available["xs"].all() and not w.history_available["vx"][0] and w.history_available["vx"][1:].all()
    assert np.isnan(w.history["vx"][0]) and not np.isnan(w.history["vx"][1:]).any()
    assert not w.history_available["ax"][:2].any() and w.history_available["ax"][2:].all() and np.isnan(w.history["ax"][:2]).all()
    assert w.history_valid.all() and (w.history_origin == "observed").all() and not w.history_imputed.any() and not w.target_imputed.any()


def test_left_padding_requires_an_explicit_minimum_and_is_masked():
    obs = []
    for s in range(4):
        t = make_track("t0", frames=range(14), seq="s%d" % s, scene="sc%d" % s, x0=200.0 + 50 * s)
        t[0] = dataclasses.replace(t[0], pos_x=None, pos_y=None, bbox_xyxy=None)
        t[1] = dataclasses.replace(t[1], pos_x=None, pos_y=None, bbox_xyxy=None)
        obs += t
    res = build_agent_states(obs)
    default = dataset(res.records, res.provenance)
    assert all(int(w.history_frame_index[0]) >= 2 for w in all_windows(default))
    assert sum(default.partitions[p].skipped["insufficient_history"] for p in PARTITIONS) > 0
    padded = dataset(res.records, res.provenance, windows=WindowConfig(obs_len=4, pred_len=3, min_valid_history_steps=2))
    first = [w for w in all_windows(padded) if int(w.history_frame_index[0]) == 0][0]
    assert list(first.history_valid) == [False, False, True, True] and list(first.history_origin) == ["masked", "masked", "observed", "observed"]
    assert np.isnan(first.history["xs"][:2]).all() and np.isnan(first.history_timestamp[:2]).all() and not first.history_available["xs"][:2].any()
    # masked steps keep their source record (the row exists but has no usable coordinate), so the evidence is traceable
    assert all(res.records[i]["image_x"] is None and res.records[i]["track_id"] == first.track_id for i in first.history_source_index[:2])


def test_required_last_step_features_reject_windows_without_them():
    ds = dataset(windows=WindowConfig(obs_len=2, pred_len=2, require_last_step_features=("ax",)))
    assert all(w.history_available["ax"][-1] for w in all_windows(ds)) and min(int(w.history_frame_index[0]) for w in all_windows(ds)) == 1
    assert sum(ds.partitions[p].skipped["required_feature_unavailable"] for p in PARTITIONS) > 0


# ============================================ 12 coordinate spaces and units stay consistent
def test_units_and_coordinate_metadata_are_attached_and_consistent():
    for w in all_windows(dataset()):
        assert (w.coordinate_mode, w.unit_status, w.position_unit, w.velocity_unit, w.acceleration_unit, w.physical) == \
            ("image_pixel", "pixel_image_space", "pixel", "pixel/s", "pixel/s^2", False)
        assert w.channel_units()["heading"] == "rad" and w.channel_units()["speed"] == "pixel/s" and w.frame_name == "synthetic_camera_image"


def test_a_window_never_mixes_coordinate_identities():
    recs, prov = corpus()
    victim = recs[0]["track_id"]
    for r in recs:
        if r["track_id"] == victim and r["frame_index"] >= 10:
            r["image_coordinate_system"] = "other_image_system"
    ds = dataset(recs, prov)
    spans = [w for w in all_windows(ds) if w.track_id == victim]
    assert spans and all(not (w.history_frame_index[0] < 10 <= w.target_frame_index[-1]) for w in spans)
    assert sum(ds.partitions[p].skipped["mixed_coordinate_identity"] for p in PARTITIONS) > 0


def test_invalid_window_config_is_rejected():
    for kw in (dict(obs_len=0, pred_len=1), dict(obs_len=1, pred_len=0), dict(obs_len=2, pred_len=2, frame_stride=0), dict(obs_len=2, pred_len=2, window_stride=0),
               dict(obs_len=2, pred_len=2, min_valid_history_steps=3), dict(obs_len=2, pred_len=2, min_valid_history_steps=0),
               dict(obs_len=True, pred_len=1), dict(obs_len=2, pred_len=2, require_last_step_features=("bogus",))):
        with pytest.raises(ConfigError):
            WindowConfig(**kw)


# ====================================================== 4 no future in observed history
def test_target_period_changes_never_reach_history_features():
    base = dataset()
    w0 = base.windows["train"][5]
    recs, prov = corpus()
    cut = int(w0.target_frame_index[0])
    for r in recs:
        if r["track_id"] == w0.track_id and r["frame_index"] >= cut:
            r["image_x"] += 400.0
            r["image_y"] -= 300.0
    pert = dataset(recs, prov)
    w1 = {w.window_id: w for w in all_windows(pert)}[w0.window_id]
    assert w1.partition == w0.partition                                           # same plan: values do not influence grouping
    for ch in w0.history:
        assert same(w0.history[ch], w1.history[ch]), ch
        assert same(w0.history_available[ch], w1.history_available[ch])
    assert not same(w0.target_xy, w1.target_xy) and w0.history_frame_index[-1] < w0.target_frame_index[0]


def test_history_features_equal_those_computed_from_a_truncated_record_set():
    from pedsafety.causal_trajectory import process_causal
    base = dataset()
    w = base.windows["validation"][3]
    recs, prov = corpus()
    keep = [i for i, r in enumerate(recs) if r["track_id"] == w.track_id and r["frame_index"] <= int(w.history_frame_index[-1])]
    traj = process_causal([recs[i] for i in keep], [prov[i] for i in keep], None, TrajectoryConfig()).trajectories[0]
    rows = {int(f): k for k, f in enumerate(traj["frame_index"])}
    for ch in ("xs", "ys", "vx", "vy", "speed", "ax", "ay", "heading"):
        expected = np.array([traj[ch][rows[int(f)]] for f in w.history_frame_index])
        got = np.where(w.history_available[ch], w.history[ch], np.nan)
        assert np.allclose(got[w.history_available[ch]], expected[w.history_available[ch]], rtol=1e-12, atol=1e-12), ch


# ============================================== 5 no interpolation across partition boundaries
def test_each_partition_is_processed_separately_so_nothing_can_cross():
    seen = []
    real = builder.process_causal

    def spy(records, provenance, projections, config):
        seen.append({r["track_id"] for r in records})
        return real(records, provenance, projections, config)

    builder.process_causal = spy
    try:
        ds = dataset()
    finally:
        builder.process_causal = real
    assert len(seen) == 3
    recs, prov = corpus()
    keys, _ = compute_groups(recs, prov, GROUP)
    for p, tracks in zip(PARTITIONS, seen):
        groups = set(ds.plan.groups_in(p))
        allowed = {r["track_id"] for r, k in zip(recs, keys) if k in groups}
        assert tracks == allowed
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])


def test_offline_view_needs_explicit_consent_and_interpolates_inside_partitions_only():
    with pytest.raises(ConfigError, match="allow_offline_view"):
        dataset(view="offline")
    seen = []
    real = builder.process_offline

    def spy(records, provenance, projections, config):
        seen.append({r["track_id"] for r in records})
        return real(records, provenance, projections, config)

    builder.process_offline = spy
    try:
        ds = dataset(drop_by_track={0: (6, 7), 1: (10,)}, view="offline",
                     windows=WindowConfig(obs_len=4, pred_len=3, allow_offline_view=True, allow_imputed=True),
                     trajectory=TrajectoryConfig(max_interpolation_gap_frames=2, allow_imputed_support=True))
    finally:
        builder.process_offline = real
    assert len(seen) == 3 and not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    imputed = [t for p in PARTITIONS for t in ds.partitions[p].trajectories if t.n_imputed > 0]
    assert imputed and all(t.view == "OFFLINE_NON_CAUSAL" for t in imputed)
    for p in PARTITIONS:
        part_tracks = {t.track_id for t in ds.partitions[p].trajectories}
        for t in ds.partitions[p].trajectories:
            assert t.track_id in part_tracks
        assert all(w.history_may_use_future and w.view == "OFFLINE_NON_CAUSAL" for w in ds.windows[p])
    assert any(w.history_imputed.any() or w.target_imputed.any() for w in all_windows(ds))


def test_imputed_rows_are_not_usable_unless_the_window_config_allows_them():
    kw = dict(drop_by_track={0: (6, 7)}, view="offline", trajectory=TrajectoryConfig(max_interpolation_gap_frames=2, allow_imputed_support=True))
    strict = dataset(windows=WindowConfig(obs_len=4, pred_len=3, allow_offline_view=True), **kw)
    loose = dataset(windows=WindowConfig(obs_len=4, pred_len=3, allow_offline_view=True, allow_imputed=True), **kw)
    assert not any(w.history_imputed.any() or w.target_imputed.any() for w in all_windows(strict))
    assert any(w.history_imputed.any() or w.target_imputed.any() for w in all_windows(loose))
    assert len(all_windows(loose)) > len(all_windows(strict))


def test_view_validation():
    with pytest.raises(ConfigError, match="view"):
        dataset(view="realtime")
