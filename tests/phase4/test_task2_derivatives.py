"""Velocity, acceleration and heading. Every expected value is worked out by hand from the synthetic positions/times."""

import math

import pytest

from pedsafety.image_kinematics import process_causal_records
from task2_helpers import cfg, lines, rec, track


def run(records, **over):
    return process_causal_records(records, lines(records), cfg(**over)).rows


def uniform(n, dx, dy, x0=100.0, y0=500.0):
    return track([(f, f / 30.0, x0 + dx * f, y0 + dy * f) for f in range(n)])


def test_constant_horizontal_motion():
    rows = run(uniform(5, 3.0, 0.0))                     # 3 px/frame at 30 fps -> 90 px/s
    assert rows[0]["velocity_state"] == "insufficient_samples" and rows[0]["velocity_x"] is None
    for r in rows[1:]:
        assert r["velocity_state"] == "valid"
        assert r["velocity_x"] == pytest.approx(90.0, rel=1e-9) and r["velocity_y"] == 0.0 and r["speed"] == pytest.approx(90.0, rel=1e-9)
        assert r["heading"] == 0.0 and r["dt_s"] == pytest.approx(1 / 30.0)
    assert rows[1]["acceleration_state"] == "previous_velocity_unavailable" and rows[1]["acceleration_x"] is None
    for r in rows[2:]:
        assert r["acceleration_state"] == "valid" and abs(r["acceleration_x"]) < 1e-6 and abs(r["acceleration_y"]) < 1e-6


def test_constant_vertical_motion_down_and_up():
    down = run(uniform(4, 0.0, 2.0))                     # +v is DOWN in image coordinates -> heading +pi/2
    assert down[2]["velocity_y"] == pytest.approx(60.0, rel=1e-9) and down[2]["velocity_x"] == 0.0
    assert down[2]["heading"] == pytest.approx(math.pi / 2)
    up = run(uniform(4, 0.0, -2.0))
    assert up[2]["velocity_y"] == pytest.approx(-60.0, rel=1e-9) and up[2]["heading"] == pytest.approx(-math.pi / 2)


def test_leftward_heading_wraps_to_minus_pi():
    rows = run(uniform(3, -3.0, 0.0))
    assert rows[2]["velocity_x"] == pytest.approx(-90.0, rel=1e-9)
    assert rows[2]["heading"] == pytest.approx(-math.pi)    # atan2(0, -90) = pi, wrapped into [-pi, pi)
    assert -math.pi <= rows[2]["heading"] < math.pi


def test_diagonal_motion():
    rows = run(uniform(3, 3.0, 4.0))                     # (90, 120) px/s, speed 150
    r = rows[2]
    assert r["velocity_x"] == pytest.approx(90.0, rel=1e-9) and r["velocity_y"] == pytest.approx(120.0, rel=1e-9)
    assert r["speed"] == pytest.approx(150.0, rel=1e-9) and r["heading"] == pytest.approx(math.atan2(120.0, 90.0))


def test_heading_range_over_all_directions():
    for k in range(-16, 17):
        a = k * math.pi / 16
        rows = run(uniform(2, 5 * math.cos(a), 5 * math.sin(a)))
        h = rows[1]["heading"]
        assert -math.pi <= h < math.pi
        assert abs(math.atan2(math.sin(h - a), math.cos(h - a))) < 1e-9


def test_constant_acceleration_uniform_sampling():
    a = 60.0                                             # x = a t^2 / 2  ->  backward-difference acceleration is exactly a
    rows = run(track([(f, f / 30.0, 100.0 + 0.5 * a * (f / 30.0) ** 2, 500.0) for f in range(8)]))
    for k, r in enumerate(rows[1:], start=1):
        t1, t0 = k / 30.0, (k - 1) / 30.0
        assert r["velocity_x"] == pytest.approx(0.5 * a * (t1 + t0), rel=1e-9)   # secant slope = v at the midpoint
    for r in rows[2:]:
        assert r["acceleration_x"] == pytest.approx(a, rel=1e-6) and abs(r["acceleration_y"]) < 1e-9


def test_irregular_timestamps_use_actual_dt():
    pts = [(0, 0.00, 0.0, 0.0), (1, 0.03, 3.0, 0.0), (2, 0.08, 8.0, 0.0), (3, 0.10, 12.0, 0.0)]
    rows = run(track([(f, t, 100 + x, 500 + y) for f, t, x, y in pts]))
    assert [r["velocity_x"] for r in rows[1:]] == pytest.approx([100.0, 100.0, 200.0], rel=1e-9)   # 3/0.03, 5/0.05, 4/0.02
    assert [r["dt_s"] for r in rows[1:]] == pytest.approx([0.03, 0.05, 0.02])
    assert rows[2]["acceleration_x"] == pytest.approx(0.0, abs=1e-6)          # (100-100)/0.05
    assert rows[3]["acceleration_x"] == pytest.approx(5000.0, rel=1e-9)       # (200-100)/0.02  (nominal 1/30 would give 3000)


def test_near_zero_displacement_has_no_heading():
    rows = run(uniform(3, 0.0, 0.0))
    assert rows[2]["velocity_state"] == "valid" and rows[2]["speed"] == 0.0
    assert rows[2]["heading"] is None and rows[2]["heading_state"] == "below_min_speed"


def test_heading_threshold_boundaries():
    # dt = 0.5 s exactly, so speeds are exact: 13 px -> 26 px/s, 15 px -> 30 px/s, 16 px -> 32 px/s
    def one(dx):
        return run(track([(0, 0.0, 100.0, 500.0), (1, 0.5, 100.0 + dx, 500.0)]), min_heading_speed_px_per_s=30.0)[1]
    assert one(13.0)["heading_state"] == "below_min_speed" and one(13.0)["heading"] is None
    assert one(15.0)["heading_state"] == "valid" and one(15.0)["heading"] == 0.0      # equal to the minimum: allowed
    assert one(16.0)["heading_state"] == "valid"
    assert one(13.0)["velocity_state"] == "valid"                                     # velocity itself is still valid


def test_invalid_velocity_gives_null_heading_and_acceleration():
    recs = uniform(4, 3.0, 0.0)
    recs[2]["image_x"] = None                                                         # position unavailable
    rows = run(recs)
    assert rows[2]["velocity_state"] == "position_unavailable" and rows[2]["heading"] is None and rows[2]["acceleration_x"] is None
    assert rows[3]["velocity_state"] == "insufficient_samples"                        # engine restarted, nothing bridged


def test_agent_class_change_breaks_differencing():
    recs = uniform(4, 3.0, 0.0)
    recs[2]["agent_class"] = "bike"
    rows = run(recs)
    assert rows[2]["velocity_state"] == "frame_identity_changed" and rows[2]["velocity_x"] is None
    assert rows[3]["velocity_state"] == "frame_identity_changed"                      # class changes back


@pytest.mark.parametrize("t2", [0.1, 0.05])
def test_zero_and_negative_dt_are_quarantined(t2):
    recs = track([(0, 0.0, 100.0, 500.0), (1, 0.1, 103.0, 500.0), (2, t2, 106.0, 500.0), (3, 0.2, 109.0, 500.0)])
    rows = run(recs)
    assert rows[2]["observation_type"] == "rejected" and rows[2]["velocity_state"] == "rejected_timestamp_not_monotonic"
    assert rows[2]["velocity_x"] is None
    assert rows[3]["velocity_state"] == "gap_before"                                  # frame 2 was removed: no bridging


def test_non_finite_timestamp_is_quarantined():
    recs = track([(0, 0.0, 100.0, 500.0), (1, float("nan"), 103.0, 500.0), (2, 2 / 30.0, 106.0, 500.0)])
    res = process_causal_records(recs, lines(recs), cfg())
    assert [q["reason"] for q in res.quarantined] == ["timestamp_missing_or_non_finite"]
    assert len(res.rows) == 2 and res.rows[1]["velocity_state"] == "gap_before"
