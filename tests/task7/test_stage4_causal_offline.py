import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest

import pedsafety.causal_trajectory as causal_module
from pedsafety.causal_trajectory import (
    CausalFeatureStream, CausalTrajectoryFeatures, NonCausalInputError, assert_causal_config, process_causal, require_causal,
)
from pedsafety.errors import ConfigError, PedsafetyError
from pedsafety.offline_trajectory import OfflineTrajectoryFeatures, process_offline
from pedsafety.trajectory import ARRAY_KEYS, SmoothingPolicy, TrajectoryConfig, prepare_windows, trajectory_features_equal
from stage4_fixtures import col, make_track, same, stage2

CFG = TrajectoryConfig()
FEATURE_KEYS = [k for k in ARRAY_KEYS if k not in ("input_index",)]


def build(obs):
    return stage2(obs)


def causal(obs, cfg=CFG):
    recs, prov = build(obs)
    return process_causal(recs, prov, None, cfg).trajectories[0]


def offline(obs, cfg=CFG):
    recs, prov = build(obs)
    return process_offline(recs, prov, None, cfg).trajectories[0]


def prefix_equal(full, part, n):
    return all(same(full.arrays[k][:n], part.arrays[k][:n]) for k in ARRAY_KEYS)


# =================================================== interpolation (OFFLINE only)
def test_interpolation_fills_short_gap_linearly_and_labels_it():
    cfg = dataclasses.replace(CFG, max_interpolation_gap_frames=2)
    t = offline(make_track(frames=range(10), drop=(4, 5)), cfg)
    assert isinstance(t, OfflineTrajectoryFeatures) and t.view == "OFFLINE_NON_CAUSAL"
    assert list(col(t, "origin")) == ["observed"] * 4 + ["imputed"] * 2 + ["observed"] * 4
    assert (t.n_observed, t.n_imputed, t.n_missing) == (8, 2, 0)
    assert col(t, "x")[4:6] == pytest.approx([312.0, 315.0]) and col(t, "y")[4:6] == pytest.approx([400.0 - 4.0, 400.0 - 5.0])
    assert col(t, "timestamp")[4:6] == pytest.approx([0.4, 0.5])
    assert list(col(t, "input_index")[4:6]) == [-1, -1]
    assert t.interpolation.startswith("linear_offline_non_causal(max_gap_frames=2")


def test_interpolation_respects_maximum_gap_and_default_is_off():
    obs = make_track(frames=range(10), drop=(3, 4, 5))
    long_gap = offline(obs, dataclasses.replace(CFG, max_interpolation_gap_frames=2))
    assert list(col(long_gap, "origin")[3:6]) == ["missing"] * 3 and long_gap.n_imputed == 0 and long_gap.n_missing == 3
    assert np.isnan(col(long_gap, "x")[3:6]).all() and set(col(long_gap, "velocity_state")[3:6]) == {"missing_position"}
    ok = offline(obs, dataclasses.replace(CFG, max_interpolation_gap_frames=3))
    assert ok.n_imputed == 3
    default = offline(obs, CFG)
    assert default.n_imputed == 0 and default.n_missing == 3 and default.interpolation == "none"


def test_interpolation_seconds_limit():
    obs = make_track(frames=range(10), drop=(4, 5))                   # anchors at t=0.3 and 0.6
    assert offline(obs, dataclasses.replace(CFG, max_interpolation_gap_frames=5, max_interpolation_gap_seconds=0.25)).n_imputed == 0
    assert offline(obs, dataclasses.replace(CFG, max_interpolation_gap_frames=5, max_interpolation_gap_seconds=0.35)).n_imputed == 2


def test_no_interpolation_across_an_unavailable_coordinate_row():
    obs = make_track(frames=range(8), drop=(3,))
    obs[3] = dataclasses.replace(obs[3], pos_x=None, pos_y=None, bbox_xyxy=None)               # frame 4 observed but has no position
    t = offline(obs, dataclasses.replace(CFG, max_interpolation_gap_frames=3))
    assert t.n_imputed == 0 and col(t, "origin")[3] == "missing" and col(t, "origin")[4] == "observed" and np.isnan(col(t, "xs")[4])


def test_no_interpolation_across_identity_change_inside_a_track():
    obs = make_track(frames=range(4), label="pedestrian") + make_track(frames=range(6, 10), label="car")
    t = offline(obs, dataclasses.replace(CFG, max_interpolation_gap_frames=5))
    assert t.n_imputed == 0 and t.n_missing == 2


def test_tracks_are_never_bridged_to_each_other():
    obs = make_track("a", frames=range(0, 4)) + make_track("b", frames=range(6, 10))
    recs, prov = build(obs)
    res = process_offline(recs, prov, None, dataclasses.replace(CFG, max_interpolation_gap_frames=9))
    a, b = res.trajectories
    assert (len(a), len(b)) == (4, 4) and a.n_imputed == b.n_imputed == 0


def test_imputed_support_is_excluded_from_features_by_default_and_flagged_when_allowed():
    obs = make_track(frames=range(10), drop=(4, 5), vx=30.0, vy=0.0)
    off = offline(obs, dataclasses.replace(CFG, max_interpolation_gap_frames=2))
    vs = col(off, "velocity_state")
    assert list(vs[4:7]) == ["imputed_support_disallowed"] * 3 and np.isnan(col(off, "vx")[4:7]).all()
    assert vs[7] == "valid"
    on = offline(obs, dataclasses.replace(CFG, max_interpolation_gap_frames=2, allow_imputed_support=True))
    assert list(col(on, "velocity_state")[4:7]) == ["valid"] * 3 and col(on, "vx")[4:7] == pytest.approx([30.0] * 3, abs=1e-9)
    assert list(col(on, "support_imputed")[4:7]) == [True] * 3 and not col(on, "support_imputed")[7]


def test_observed_raw_values_are_identical_with_and_without_interpolation():
    obs = make_track(frames=range(10), drop=(4, 5))
    base = offline(obs, CFG)
    interp = offline(obs, dataclasses.replace(CFG, max_interpolation_gap_frames=2))
    seen = col(base, "origin") == "observed"
    assert same(col(base, "x")[seen], col(interp, "x")[col(interp, "origin") == "observed"])
    assert same(col(base, "timestamp")[seen], col(interp, "timestamp")[col(interp, "origin") == "observed"])


# ======================================================= causal vs offline separation
def test_causal_pipeline_refuses_offline_options():
    for cfg in (dataclasses.replace(CFG, max_interpolation_gap_frames=1), dataclasses.replace(CFG, max_interpolation_gap_seconds=1.0),
                dataclasses.replace(CFG, smoothing=SmoothingPolicy("offline_centered_moving_average", window=3))):
        with pytest.raises(ConfigError, match="OFFLINE"):
            assert_causal_config(cfg)
        with pytest.raises(ConfigError):
            CausalFeatureStream(cfg)
        with pytest.raises(ConfigError):
            process_causal([], [], None, cfg)
    assert_causal_config(dataclasses.replace(CFG, smoothing=SmoothingPolicy("causal_ema", alpha=0.3)))


def test_offline_results_are_labelled_and_rejected_by_the_causal_guard():
    obs = make_track(frames=range(6))
    c, o = causal(obs), offline(obs)
    assert isinstance(c, CausalTrajectoryFeatures) and c.view == "CAUSAL" and require_causal(c) is c
    assert o.view == "OFFLINE_NON_CAUSAL"
    with pytest.raises(NonCausalInputError, match="OFFLINE|offline|Offline"):
        require_causal(o)
    with pytest.raises(TypeError):
        require_causal(o)
    with pytest.raises(NonCausalInputError):
        require_causal({"vx": [1.0]})


def test_causal_module_does_not_depend_on_the_offline_module():
    src = Path(causal_module.__file__).read_text(encoding="utf-8")
    assert "offline_trajectory" not in src.replace("``offline_trajectory``", "").replace("(``offline_trajectory.py``", "")
    import re
    assert not re.search(r"^\s*(from|import)\s+.*offline_trajectory", src, re.M)


def test_offline_without_interpolation_or_smoothing_equals_causal_on_observed_rows():
    obs = make_track(frames=range(12), drop=(5, 6), vx=7.0, vy=3.0, ax=4.0, ay=-2.0)
    c, o = causal(obs), offline(obs)
    seen = col(o, "origin") == "observed"
    for k in FEATURE_KEYS:
        assert same(col(c, k), col(o, k)[seen]), k


def test_offline_interpolation_changes_what_causal_cannot_know():
    obs = make_track(frames=range(10), drop=(4, 5))
    c = causal(obs)
    o = offline(obs, dataclasses.replace(CFG, max_interpolation_gap_frames=2, allow_imputed_support=True))
    assert c.n_imputed == 0 and len(c) == 8 and len(o) == 10 and set(col(c, "origin")) == {"observed"}


# ============================================= leakage: causal ignores the future
@pytest.mark.parametrize("cfg", [CFG, dataclasses.replace(CFG, smoothing=SmoothingPolicy("causal_ema", alpha=0.4))])
def test_causal_features_are_identical_when_all_future_rows_are_removed(cfg):
    obs = make_track(frames=range(14), drop=(6, 7), ax=5.0, ay=2.0)
    recs, prov = build(obs)
    full = process_causal(recs, prov, None, cfg).trajectories[0]
    for k in range(1, len(recs) + 1):
        part = process_causal(recs[:k], prov[:k], None, cfg).trajectories[0]
        assert prefix_equal(full, part, len(part)), k


def test_causal_features_are_unchanged_by_arbitrary_future_values():
    obs = make_track(frames=range(12), ax=3.0)
    recs, prov = build(obs)
    cut = 6
    base = process_causal(recs, prov, None, CFG).trajectories[0]
    rng = np.random.default_rng(0)
    for _ in range(5):
        pert = copy.deepcopy(recs)
        for r in pert[cut:]:
            r["image_x"] += float(rng.normal(0, 500))
            r["image_y"] = float("nan") if rng.random() < 0.3 else r["image_y"] + float(rng.normal(0, 500))
        res = process_causal(pert, prov, None, CFG).trajectories[0]
        assert prefix_equal(base, res, cut)
        assert not same(col(base, "vx")[cut:], col(res, "vx")[cut:])             # the perturbation really did change later rows


def test_streaming_equals_batch_and_never_needs_future_input():
    obs = make_track(frames=range(10), drop=(4,), ax=2.0)
    recs, prov = build(obs)
    batch = process_causal(recs, prov, None, CFG).trajectories[0]
    stream = CausalFeatureStream(CFG)
    rows = [stream.update(r, p, None) for r, p in zip(recs, prov)]
    assert same(np.array([r.vx for r in rows]), col(batch, "vx")) and same(np.array([r.ax for r in rows]), col(batch, "ax"))
    # a stream fed only a prefix produces the same values for that prefix
    s2 = CausalFeatureStream(CFG)
    head = [s2.update(r, p, None) for r, p in zip(recs[:5], prov[:5])]
    assert same(np.array([r.speed for r in head]), np.array([r.speed for r in rows[:5]]))


def test_stream_rejects_late_and_duplicate_samples_without_corrupting_state():
    recs, prov = build(make_track(frames=range(6)))
    s = CausalFeatureStream(CFG)
    out = [s.update(recs[i], prov[i], None) for i in range(4)]
    late = s.update(recs[1], prov[1], None)                                         # arrives after frame 3
    dup = s.update(recs[3], prov[3], None)
    assert late.position_state == "rejected_duplicate_or_out_of_order" and dup.velocity_state == "rejected" and np.isnan(late.vx)
    nxt = s.update(recs[4], prov[4], None)
    assert nxt.velocity_state == "valid" and nxt.vx == pytest.approx(30.0, abs=1e-9)
    assert [i.code for i in s.issues] == ["duplicate_or_out_of_order"] * 2


def test_offline_features_DO_depend_on_the_future_which_is_why_they_are_labelled():
    cfg = dataclasses.replace(CFG, smoothing=SmoothingPolicy("offline_centered_moving_average", window=3))
    obs = make_track(frames=range(10), vx=0.0, vy=0.0)
    recs, prov = build(obs)
    base = process_offline(recs, prov, None, cfg).trajectories[0]
    pert = copy.deepcopy(recs)
    pert[6]["image_x"] += 90.0                                                      # future of row 5
    res = process_offline(pert, prov, None, cfg).trajectories[0]
    assert not same(col(base, "xs")[5], col(res, "xs")[5])                          # row 5 changed because of row 6
    c_base, c_res = process_causal(recs, prov, None, CFG).trajectories[0], process_causal(pert, prov, None, CFG).trajectories[0]
    assert prefix_equal(c_base, c_res, 6)                                           # the causal pipeline is immune


def test_offline_interpolation_depends_on_the_right_anchor():
    cfg = dataclasses.replace(CFG, max_interpolation_gap_frames=2)
    recs, prov = build(make_track(frames=range(10), drop=(4, 5)))
    a = process_offline(recs, prov, None, cfg).trajectories[0]
    pert = copy.deepcopy(recs)
    pert[4]["image_x"] += 60.0                                                      # frame 6, the right anchor
    b = process_offline(pert, prov, None, cfg).trajectories[0]
    assert not same(col(a, "x")[4:6], col(b, "x")[4:6])


# ====================================================== offline centred smoothing
def test_centered_moving_average_known_values_and_edges():
    cfg = dataclasses.replace(CFG, smoothing=SmoothingPolicy("offline_centered_moving_average", window=3))
    t = offline(make_track(frames=range(6), vx=0.0, vy=0.0, x0=100.0), cfg)
    # constant position: the average equals the position wherever the window is complete
    assert np.isnan(col(t, "xs")[[0, 5]]).all() and col(t, "xs")[1:5] == pytest.approx([100.0] * 4)
    assert col(t, "position_state")[0] == "smoothing_window_incomplete" and col(t, "x")[0] == pytest.approx(100.0)   # raw preserved
    lin = offline(make_track(frames=range(7), vx=30.0, vy=0.0), cfg)
    assert col(lin, "xs")[1:6] == pytest.approx(col(lin, "x")[1:6])                                  # symmetric window on a line
    assert lin.smoothing == "offline_centered_moving_average(window=3)"


def test_centered_moving_average_does_not_bridge_gaps():
    cfg = dataclasses.replace(CFG, smoothing=SmoothingPolicy("offline_centered_moving_average", window=3))
    t = offline(make_track(frames=range(9), drop=(4,)), cfg)
    assert np.isnan(col(t, "xs")[[3, 4, 5]]).all() and np.isfinite(col(t, "xs")[[2, 6]]).all()


# ================================================================ windows
def windows_for(obs, **kw):
    t = causal(obs)
    return t, prepare_windows(t, **kw)


def test_windows_separate_history_from_target_and_carry_identity():
    t, ws = windows_for(make_track("w", frames=range(10)), obs_len=4, pred_len=3)
    assert len(ws.windows) == 4 and sum(ws.skipped.values()) == 0
    w = ws.windows[1]
    assert w.obs_frames == (1, 2, 3, 4) and w.pred_frames == (5, 6, 7) and not set(w.obs_frames) & set(w.pred_frames)
    assert w.target_xy == pytest.approx(np.column_stack([col(t, "x")[5:8], col(t, "y")[5:8]]))
    assert len(w.history["xs"]) == 4 and w.history["xs"] == pytest.approx(col(t, "x")[1:5])
    assert (w.track_id, w.raw_track_id, w.raw_sequence_id, w.dataset, w.view, w.history_may_use_future) == \
        ("syn:s1:w", "w", "s1", "SYN", "CAUSAL", False)
    assert w.velocity_unit == "pixel/s" and w.target_origin == ("observed",) * 3
    assert not hasattr(w, "split") and not hasattr(w, "partition")                  # split-agnostic by design (Stage 5)


def test_history_does_not_change_when_only_the_target_period_changes():
    recs, prov = build(make_track(frames=range(10)))
    pert = copy.deepcopy(recs)
    for r in pert[7:]:
        r["image_x"] += 300.0
    a = prepare_windows(process_causal(recs, prov, None, CFG).trajectories[0], 4, 3).windows[-1]
    b = prepare_windows(process_causal(pert, prov, None, CFG).trajectories[0], 4, 3).windows[-1]
    assert a.obs_range == b.obs_range == (3, 7) and a.pred_range == (7, 10)
    for k in a.history:
        assert same(a.history[k], b.history[k]), k
    assert not np.array_equal(a.target_xy, b.target_xy)


def test_stride_and_gap_handling():
    t, ws = windows_for(make_track(frames=range(12), drop=(6,)), obs_len=3, pred_len=2, stride=2)
    assert [w.obs_range for w in ws.windows] == [(0, 3), (6, 9)] or all(not set(range(6, 7)) & set(w.obs_frames + w.pred_frames) for w in ws.windows)
    assert ws.skipped["not_contiguous"] >= 1
    for w in ws.windows:
        assert 6 not in w.obs_frames + w.pred_frames


def test_offline_views_are_refused_for_windowing_unless_explicit():
    o = offline(make_track(frames=range(10)))
    with pytest.raises(PedsafetyError, match="OFFLINE_NON_CAUSAL"):
        prepare_windows(o, 4, 2)
    ws = prepare_windows(o, 4, 2, allow_offline_view=True)
    assert ws.windows and all(w.history_may_use_future and w.view == "OFFLINE_NON_CAUSAL" for w in ws.windows)


def test_imputed_rows_are_unusable_in_windows_unless_allowed():
    cfg = dataclasses.replace(CFG, max_interpolation_gap_frames=2)
    o = offline(make_track(frames=range(10), drop=(4, 5)), cfg)
    strict = prepare_windows(o, 3, 2, allow_offline_view=True)
    loose = prepare_windows(o, 3, 2, allow_offline_view=True, allow_imputed=True)
    assert len(loose.windows) > len(strict.windows) and strict.skipped["unusable_history"] + strict.skipped["unusable_target"] > 0
    assert all("imputed" not in w.target_origin for w in strict.windows)
    assert any("imputed" in w.target_origin or "imputed" in tuple(w.history["origin"]) for w in loose.windows)


def test_require_valid_history_features():
    t = causal(make_track(frames=range(8)))
    assert len(prepare_windows(t, 3, 2).windows) == 4
    assert len(prepare_windows(t, 3, 2, require_valid_history_features=True).windows) == 3        # first window starts at the feature-less row


def test_window_parameter_validation_and_short_tracks():
    t = causal(make_track(frames=range(5)))
    for kw in (dict(obs_len=0, pred_len=1), dict(obs_len=1, pred_len=0), dict(obs_len=2, pred_len=2, stride=0), dict(obs_len=True, pred_len=1)):
        with pytest.raises(ConfigError):
            prepare_windows(t, **kw)
    assert prepare_windows(t, 4, 4).windows == []


def test_unusable_positions_block_windows():
    recs, prov = build(make_track(frames=range(8)))
    bad = copy.deepcopy(recs)
    bad[2]["image_x"] = float("nan")
    t = process_causal(bad, prov, None, CFG).trajectories[0]
    ws = prepare_windows(t, 3, 2)
    assert all(2 not in w.obs_frames + w.pred_frames for w in ws.windows) and ws.skipped["unusable_history"] + ws.skipped["unusable_target"] >= 1
