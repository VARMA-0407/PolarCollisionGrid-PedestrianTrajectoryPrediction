import copy
import dataclasses
import math

import numpy as np
import pytest

from pedsafety.causal_trajectory import process_causal
from pedsafety.errors import ConfigError
from pedsafety.offline_trajectory import process_offline
from pedsafety.trajectory import (
    CoordinateMode, FeatureEngine, ObsSample, PositionInfo, SmoothingPolicy, TrajectoryConfig, angular_difference, group_tracks,
    wrap_angle,
)
from stage4_fixtures import col, make_track, same, stage2

CFG = TrajectoryConfig()


def one(obs, cfg=CFG, fn=process_causal):
    recs, prov = stage2(obs)
    res = fn(recs, prov, None, cfg)
    assert len(res.trajectories) == 1
    return res.trajectories[0]


def codes(res):
    return [i.code for i in res.issues]


# ----------------------------------------------------------------- 1 grouping
def test_grouping_by_canonical_track_identity_without_cross_sequence_joins():
    obs = []
    for f in range(3):
        obs += make_track("a", [f], seq="s1") + make_track("a", [f], seq="s2", scene="sc2") + make_track("a", [f], dataset="OTHER") \
               + make_track("b", [f], seq="s1")
    recs, prov = stage2(obs)
    res = process_causal(recs, prov, None, CFG)
    assert [t.track_id for t in res.trajectories] == ["syn:s1:a", "syn:s2:a", "other:s1:a", "syn:s1:b"]   # first-appearance order
    assert all(len(t) == 3 for t in res.trajectories) and res.issues == []
    t = res.trajectories[1]
    assert (t.raw_sequence_id, t.raw_track_id, t.dataset, t.sequence_id) == ("s2", "a", "SYN", "syn:s2")


def test_interleaved_tracks_do_not_leak_state_into_each_other():
    a, b = make_track("a", vx=30.0, vy=0.0), make_track("b", vx=-50.0, vy=0.0, x0=900.0)
    inter = [o for pair in zip(a, b) for o in pair]
    both = process_causal(*stage2(inter), None, CFG).trajectories
    alone = [one(a), one(b)]
    for x, y in zip(both, alone):
        assert same(x["vx"], y["vx"]) and same(x["velocity_state"], y["velocity_state"])


# --------------------------------------------------------- 2 ordering / 5 duplicates
def test_out_of_order_frames_are_quarantined_not_sorted():
    recs, prov = stage2(make_track(frames=range(6)))
    order = [0, 1, 3, 2, 4, 5]
    r2, p2 = [recs[i] for i in order], [prov[i] for i in order]
    res = process_causal(r2, p2, None, CFG)
    assert codes(res) == ["out_of_order_frame"] and res.issues[0].input_index == 3
    assert list(col(res.trajectories[0], "frame_index")) == [0, 1, 3, 4, 5]          # input order kept; offender dropped, not re-sorted


def test_duplicate_frames_quarantine_every_copy():
    recs, prov = stage2(make_track(frames=range(4)))
    r2, p2 = recs[:2] + [recs[1]] + recs[2:], prov[:2] + [prov[1]] + prov[2:]
    res = process_causal(r2, p2, None, CFG)
    assert codes(res) == ["duplicate_observation"] * 2
    assert list(col(res.trajectories[0], "frame_index")) == [0, 2, 3]


def test_timestamp_must_increase_with_frame_and_frames_must_be_on_grid():
    recs, prov = stage2(make_track(frames=range(4)))
    bad = copy.deepcopy(recs)
    bad[2]["timestamp"] = bad[1]["timestamp"]                      # equal time, later frame
    res = process_causal(bad, prov, None, CFG)
    assert codes(res) == ["timestamp_not_monotonic"] and res.issues[0].input_index == 2
    grid = process_causal(recs, prov, None, dataclasses.replace(CFG, frame_step=2))
    assert codes(grid) == ["off_grid_frame"] * 2 and list(col(grid.trajectories[0], "frame_index")) == [0, 2]


def test_invalid_time_values_are_quarantined():
    recs, prov = stage2(make_track(frames=range(4)))
    bad = copy.deepcopy(recs)
    bad[1]["timestamp"] = float("nan")
    bad[2]["timestamp"] = "0.2"
    res = process_causal(bad, prov, None, CFG)
    assert codes(res) == ["invalid_time"] * 2 and list(col(res.trajectories[0], "frame_index")) == [0, 3]


def test_group_tracks_keeps_input_order_and_reports():
    recs, _ = stage2(make_track("x", [5, 6]) + make_track("y", [0]))
    g = group_tracks(recs, 1)
    assert list(g.groups) == ["syn:s1:x", "syn:s1:y"] and g.groups["syn:s1:x"] == [0, 1] and g.issues == []


# --------------------------------------------------- 3 irregular, 4 missing frames
def test_irregular_sampling_is_reported_and_dt_is_actual():
    times = [0.0, 0.1, 0.25, 0.3, 0.55]
    vx = 40.0
    t = one(make_track(frames=range(5), vx=vx, vy=0.0, times=times))
    s = t.sampling
    assert [(a, b) for a, b, _ in s.irregular_intervals] == [(1, 2), (2, 3), (3, 4)] or len(s.irregular_intervals) >= 2
    assert s.min_dt_per_step == pytest.approx(0.05) and s.max_dt_per_step == pytest.approx(0.25)
    assert col(t, "dt")[1:] == pytest.approx(np.diff(times))                       # actual dt, not the nominal 0.1
    assert col(t, "vx")[1:] == pytest.approx([vx] * 4, rel=1e-9)                   # exact for a linear track


def test_regular_sampling_reports_no_irregularity():
    s = one(make_track()).sampling
    assert s.irregular_intervals == () and s.missing_frames == () and s.median_dt_per_step == pytest.approx(0.1)


def test_missing_frames_detected_and_features_not_computed_across_the_gap():
    t = one(make_track(frames=range(10), drop=(4, 5)))
    assert t.sampling.missing_frames == (4, 5)
    fr, vs = col(t, "frame_index"), col(t, "velocity_state")
    assert list(fr) == [0, 1, 2, 3, 6, 7, 8, 9]
    assert vs[list(fr).index(6)] == "gap_before" and np.isnan(col(t, "vx")[list(fr).index(6)])
    assert vs[list(fr).index(7)] == "valid"                                          # resumes after the gap
    assert col(t, "acceleration_state")[list(fr).index(7)] == "previous_velocity_unavailable"
    assert col(t, "acceleration_state")[list(fr).index(8)] == "valid"
    assert t.n_missing == 0 and t.n_observed == 8                                    # causal view never invents rows


# ------------------------------------------------- 9-10 known kinematics, units
def test_velocity_matches_known_constant_velocity_track():
    t = one(make_track(vx=30.0, vy=-10.0))
    assert np.isnan(col(t, "vx")[0]) and col(t, "velocity_state")[0] == "insufficient_samples"
    assert col(t, "vx")[1:] == pytest.approx([30.0] * 9, abs=1e-9) and col(t, "vy")[1:] == pytest.approx([-10.0] * 9, abs=1e-9)
    assert col(t, "speed")[1:] == pytest.approx([math.hypot(30, 10)] * 9, abs=1e-9)
    assert col(t, "dx")[1:] == pytest.approx([3.0] * 9, abs=1e-9) and col(t, "dy")[1:] == pytest.approx([-1.0] * 9, abs=1e-9)


def test_acceleration_matches_known_constant_acceleration_track():
    ax, ay, vx0, vy0 = 12.0, -6.0, 5.0, 2.0
    t = one(make_track(vx=vx0, vy=vy0, ax=ax, ay=ay, frames=range(12)))
    ts = col(t, "timestamp")
    # backward difference of a quadratic is the velocity at the interval midpoint
    assert col(t, "vx")[1:] == pytest.approx(vx0 + ax * (ts[1:] - 0.05), abs=1e-9)
    assert list(col(t, "acceleration_state")[:2]) == ["previous_velocity_unavailable"] * 2 or col(t, "acceleration_state")[1] == "previous_velocity_unavailable"
    assert col(t, "acceleration_state")[0] == "insufficient_samples"
    assert np.isnan(col(t, "ax")[:2]).all()
    assert col(t, "ax")[2:] == pytest.approx([ax] * 10, abs=1e-6) and col(t, "ay")[2:] == pytest.approx([ay] * 10, abs=1e-6)


def test_image_space_units_are_pixels_and_never_physical():
    t = one(make_track())
    assert (t.position_unit, t.velocity_unit, t.acceleration_unit) == ("pixel", "pixel/s", "pixel/s^2")
    assert t.physical is False and t.physical_basis is None and t.unit_status == "pixel_image_space"
    assert t.coordinate_mode == "image_pixel" and t.frame_name == "synthetic_camera_image"
    assert "clockwise" in t.heading_convention and "not a world heading" in t.heading_convention


# --------------------------------------------------------------- 11 heading
@pytest.mark.parametrize("vx,vy,expected", [(30.0, 0.0, 0.0), (0.0, 30.0, math.pi / 2), (0.0, -30.0, -math.pi / 2),
                                            (-30.0, 0.0, -math.pi), (30.0, 30.0, math.pi / 4), (-30.0, -30.0, -3 * math.pi / 4)])
def test_heading_directions_and_range(vx, vy, expected):
    t = one(make_track(vx=vx, vy=vy))
    h = col(t, "heading")[1:]
    assert h == pytest.approx([expected] * 9, abs=1e-9) and (h >= -math.pi).all() and (h < math.pi).all()


def test_wrap_angle_and_difference():
    assert wrap_angle(math.pi) == -math.pi and wrap_angle(-math.pi) == -math.pi
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2) and wrap_angle(2 * math.pi) == pytest.approx(0.0, abs=1e-12)
    assert wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2) and wrap_angle(7.0) == pytest.approx(7.0 - 2 * math.pi)
    assert angular_difference(math.radians(179), math.radians(-179)) == pytest.approx(math.radians(-2))
    assert angular_difference(math.radians(-179), math.radians(179)) == pytest.approx(math.radians(2))


def test_stationary_agent_has_no_heading_not_zero():
    t = one(make_track(vx=0.0, vy=0.0))
    assert np.isnan(col(t, "heading")[1:]).all() and set(col(t, "heading_state")[1:]) == {"below_min_speed"}
    assert col(t, "vx")[1:] == pytest.approx([0.0] * 9) and set(col(t, "velocity_state")[1:]) == {"valid"}    # velocity IS zero here: measured


def test_min_heading_speed_is_configurable():
    obs = make_track(vx=3.0, vy=0.0)
    low = one(obs, dataclasses.replace(CFG, min_heading_speed=1.0))
    high = one(obs, dataclasses.replace(CFG, min_heading_speed=5.0))
    assert col(low, "heading_state")[1] == "valid" and col(high, "heading_state")[1] == "below_min_speed" and np.isnan(col(high, "heading")[1])


# ----------------------------------------------- 12 zero / negative time intervals
def _engine_sample(frame, t, x=0.0, y=0.0, ident=("i",)):
    return ObsSample(0, "tid", frame, t, "observed", PositionInfo(x, y, ident, "pixel", "pixel/s", "pixel/s^2", "u", False, "f"), "ok", (x, y))


def test_zero_and_negative_dt_are_rejected_not_divided():
    e = FeatureEngine(CFG)
    e.update(_engine_sample(0, 1.0))
    z = e.update(_engine_sample(1, 1.0, 5.0))
    n = e.update(_engine_sample(1, 0.5, 9.0))                       # next frame, but time runs backwards
    for s in (z, n):
        assert s.velocity_state == "nonpositive_dt" and math.isnan(s.vx) and math.isnan(s.speed) and math.isnan(s.heading)
    ok = e.update(_engine_sample(1, 1.1, 5.0))                      # state was not advanced by the rejected samples
    assert ok.velocity_state == "valid" and ok.vx == pytest.approx(50.0)


def test_non_finite_timestamp_is_rejected_by_engine():
    e = FeatureEngine(CFG)
    e.update(_engine_sample(0, 0.0))
    s = e.update(_engine_sample(1, float("nan"), 1.0))
    assert s.velocity_state == "non_finite_dt" and math.isnan(s.vx)


def test_overflowing_differences_become_explicit_non_finite_states_not_zero():
    e = FeatureEngine(CFG)
    e.update(_engine_sample(0, 0.0, x=-1e308))
    s = e.update(_engine_sample(1, 0.1, x=1e308))                    # dx overflows to inf
    assert s.velocity_state == "non_finite_result" and s.acceleration_state == "non_finite_result"
    for v in (s.dx, s.vx, s.vy, s.speed, s.ax, s.ay, s.heading):
        assert math.isnan(v)
    tiny = FeatureEngine(CFG)
    tiny.update(_engine_sample(0, 0.0, x=0.0))
    s2 = tiny.update(_engine_sample(1, 1e-320, x=1.0))               # dt so small that dx/dt overflows
    assert s2.velocity_state == "non_finite_result" and math.isnan(s2.vx)
    nxt = tiny.update(_engine_sample(2, 1.0, x=2.0))                 # chain restarted: no stale velocity is reused
    assert nxt.velocity_state == "valid" and nxt.acceleration_state == "previous_velocity_unavailable"


def test_frame_going_backwards_in_engine_is_rejected():
    e = FeatureEngine(CFG)
    e.update(_engine_sample(5, 0.5))
    assert e.update(_engine_sample(4, 0.6, 1.0)).velocity_state == "frame_not_increasing"


def test_max_dt_is_a_break_not_a_value():
    t = one(make_track(frames=range(4), times=[0.0, 0.1, 2.0, 2.1]), dataclasses.replace(CFG, max_dt_seconds=0.5))
    assert list(col(t, "velocity_state")) == ["insufficient_samples", "valid", "dt_too_large", "valid"]


# ------------------------------------------------------------ 13 non-finite inputs
def test_non_finite_position_gives_explicit_unavailable_state_never_zero():
    recs, prov = stage2(make_track(frames=range(6)))
    bad = copy.deepcopy(recs)
    bad[3]["image_x"] = float("nan")
    bad[4]["image_y"] = float("inf")
    res = process_causal(bad, prov, None, CFG)
    t = res.trajectories[0]
    st = col(t, "velocity_state")
    assert list(st[:3]) == ["insufficient_samples", "valid", "valid"] and list(st[3:5]) == ["position_unavailable"] * 2
    assert st[5] == "insufficient_samples"                           # chain restarts after the bad samples
    for k in ("x", "y", "xs", "vx", "vy", "speed", "heading"):
        assert np.isnan(col(t, k)[3:5]).all(), k
    assert set(col(t, "position_state")[3:5]) == {"non_finite_position"}
    assert t.sampling.n_valid_positions == 4


def test_samples_without_an_image_position_are_unavailable_in_image_mode():
    obs = make_track(frames=range(3))
    obs[1] = dataclasses.replace(obs[1], pos_x=None, pos_y=None, bbox_xyxy=None)
    t = one(obs)
    assert col(t, "position_state")[1].startswith("no_image_position") and np.isnan(col(t, "xs")[1]) and col(t, "origin")[1] == "observed"


# -------------------------------------------------------- 14 insufficient observations
def test_single_observation_has_no_features():
    t = one(make_track(frames=[0]))
    assert len(t) == 1 and col(t, "velocity_state")[0] == "insufficient_samples" and np.isnan(col(t, "vx")[0])
    assert col(t, "acceleration_state")[0] == "insufficient_samples" and col(t, "heading_state")[0] == "insufficient_samples"


def test_short_track_policy_flag_and_exclude():
    obs = make_track("short", frames=range(2)) + make_track("long", frames=range(8))
    recs, prov = stage2(obs)
    flag = process_causal(recs, prov, None, dataclasses.replace(CFG, min_valid_observations=5))
    assert {t.track_id: t.too_short for t in flag.trajectories} == {"syn:s1:short": True, "syn:s1:long": False}
    drop = process_causal(recs, prov, None, dataclasses.replace(CFG, min_valid_observations=5, short_track_policy="exclude"))
    assert [t.track_id for t in drop.trajectories] == ["syn:s1:long"] and drop.excluded_tracks == ["syn:s1:short"]
    assert "track_too_short" in codes(drop)                          # reported, not silent
    keep = process_causal(recs, prov, None, CFG)
    assert not any(t.too_short for t in keep.trajectories)


# ---------------------------------------------------------------- smoothing
def test_causal_ema_known_values_and_raw_positions_preserved():
    cfg = dataclasses.replace(CFG, smoothing=SmoothingPolicy("causal_ema", alpha=0.5))
    t = one(make_track(vx=30.0, vy=0.0, frames=range(4)), cfg)
    x = col(t, "x")
    xs = [x[0], 0.5 * x[1] + 0.5 * x[0], None, None]
    xs[2] = 0.5 * x[2] + 0.5 * xs[1]
    xs[3] = 0.5 * x[3] + 0.5 * xs[2]
    assert col(t, "xs") == pytest.approx(xs, abs=1e-9)
    assert col(t, "x") == pytest.approx([300.0, 303.0, 306.0, 309.0])                # raw untouched
    assert col(t, "vx")[1] == pytest.approx((xs[1] - xs[0]) / 0.1) and t.smoothing == "causal_ema(alpha=0.5)"


def test_ema_restarts_after_a_gap_instead_of_bridging_it():
    cfg = dataclasses.replace(CFG, smoothing=SmoothingPolicy("causal_ema", alpha=0.5))
    t = one(make_track(vx=30.0, vy=0.0, frames=range(6), drop=(2, 3)), cfg)
    assert col(t, "xs")[2] == col(t, "x")[2]


def test_config_validation():
    for kw in (dict(frame_step=0), dict(max_interpolation_gap_frames=-1), dict(max_interpolation_gap_seconds=0.0), dict(min_heading_speed=-1.0),
               dict(coordinate_mode="metres"), dict(short_track_policy="drop"), dict(min_valid_observations=0), dict(dt_rel_tol=0),
               dict(max_dt_seconds=float("nan")), dict(smoothing="ema")):
        with pytest.raises(ConfigError):
            TrajectoryConfig(**kw)
    for kw in (dict(kind="gaussian"), dict(kind="causal_ema"), dict(kind="causal_ema", alpha=0.0), dict(kind="causal_ema", alpha=1.5),
               dict(kind="offline_centered_moving_average", window=4), dict(kind="offline_centered_moving_average", window=1)):
        with pytest.raises(ConfigError):
            SmoothingPolicy(**kw)
    assert TrajectoryConfig(coordinate_mode="image_pixel").coordinate_mode is CoordinateMode.IMAGE_PIXEL


# ---------------------------------------------------- 21-22 inputs unchanged, deterministic
def test_inputs_are_unchanged_and_outputs_are_read_only_and_deterministic():
    recs, prov = stage2(make_track(frames=range(8), drop=(3,)))
    before_r, before_p = copy.deepcopy(recs), list(prov)
    for fn in (process_causal, process_offline):
        a = fn(recs, prov, None, CFG)
        b = fn(recs, prov, None, CFG)
        assert recs == before_r and prov == before_p
        from pedsafety.trajectory import trajectory_features_equal
        assert all(trajectory_features_equal(x, y) for x, y in zip(a.trajectories, b.trajectories))
        with pytest.raises(ValueError):
            a.trajectories[0].arrays["vx"][0] = 1.0                    # read-only arrays


def test_parallel_input_contract():
    recs, prov = stage2(make_track(frames=range(3)))
    with pytest.raises(ValueError, match="parallel"):
        process_causal(recs, prov[:-1], None, CFG)
    with pytest.raises(ValueError, match="does not describe"):
        process_causal(recs, list(reversed(prov)), None, CFG)
    with pytest.raises(ValueError, match="parallel"):
        process_causal(recs, prov, [], CFG)


def test_empty_input():
    for fn in (process_causal, process_offline):
        r = fn([], [], None, CFG)
        assert r.trajectories == [] and r.issues == []
