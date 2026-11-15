"""P5.4 baselines B0/B1/B2: hand-calculable behaviour, causality, masks, input isolation."""

import numpy as np
import pytest

from pedsafety.phase5.baselines import (HistoryInput, HistoryLinearRegression, LastValidCausalVelocity, Predictor, ZeroVelocity,
                                        make_predictor)
from pedsafety.phase5.errors import WindowContractError
from p54_helpers import F, H, TAU, linear_history, make_hist


def B2(**kw):
    p = dict(min_valid_samples=3, min_time_span_s=0.2, min_centered_time_variance_s2=1e-6, center_timestamps=True)
    p.update(kw)
    return HistoryLinearRegression(**p)


# ---------------- B0 ----------------
def test_b0_returns_last_position_every_step():
    uv = linear_history([500.0, 300.0], [40.0, -10.0])
    pr = ZeroVelocity().predict(make_hist(uv))
    assert pr.positions.shape == (1, F, 2)
    assert np.array_equal(pr.positions[0], np.repeat(uv[0, -1][None], F, axis=0))
    assert pr.window_valid.all() and pr.valid.all() and pr.baseline_id == "B0_zero_velocity"


def test_b0_invalid_anchor_is_nan_not_zero():
    pv = np.ones((1, H), bool)
    pv[0, -1] = False
    pr = ZeroVelocity().predict(make_hist(np.full((1, H, 2), 50.0), pvalid=pv))
    assert not pr.window_valid[0] and np.isnan(pr.positions).all()
    assert pr.invalid_counts() == {"anchor_position_unavailable": 1}


# ---------------- B1 ----------------
def test_b1_extrapolates_with_anchor_causal_velocity():
    uv = np.full((1, H, 2), 100.0)
    vel = np.zeros((1, H, 2))
    vel[0, -1] = [10.0, -20.0]
    vv = np.zeros((1, H), bool)
    vv[0, -1] = True
    pr = LastValidCausalVelocity(5).predict(make_hist(uv, vel=vel, vvalid=vv))
    exp = np.array([100.0, 100.0]) + np.array([10.0, -20.0]) * TAU[:, None]
    assert np.allclose(pr.positions[0], exp, atol=1e-12)
    assert pr.aux["velocity_source_index"][0] == 5


def test_b1_searches_backward_for_latest_valid_velocity():
    uv = np.full((2, H, 2), 100.0)
    vel = np.zeros((2, H, 2))
    vel[:, 3] = [1.0, 2.0]
    vel[:, 4] = [5.0, 6.0]
    vv = np.zeros((2, H), bool)
    vv[0, [3, 4]] = True          # window 0: latest valid is index 4
    vv[1, 3] = True               # window 1: only index 3
    pr = LastValidCausalVelocity(5).predict(make_hist(uv, vel=vel, vvalid=vv))
    assert pr.aux["velocity_source_index"].tolist() == [4, 3]
    assert np.allclose(pr.positions[0, -1], [100 + 5 * 1.5, 100 + 6 * 1.5])
    assert np.allclose(pr.positions[1, -1], [100 + 1 * 1.5, 100 + 2 * 1.5])
    # extrapolation is always from the LAST observed position, not from the velocity's sample
    assert np.allclose(pr.positions[1, 0], [100.1, 100.2])


def test_b1_respects_lookback_limit_and_never_substitutes_zero():
    uv = np.full((1, H, 2), 100.0)
    vel = np.zeros((1, H, 2))
    vel[0, 0] = [3.0, 3.0]
    vv = np.zeros((1, H), bool)
    vv[0, 0] = True
    assert LastValidCausalVelocity(5).predict(make_hist(uv, vel=vel, vvalid=vv)).window_valid[0]
    pr = LastValidCausalVelocity(2).predict(make_hist(uv, vel=vel, vvalid=vv))      # index 0 out of reach
    assert not pr.window_valid[0] and np.isnan(pr.positions).all()
    assert pr.invalid_counts()["no_valid_causal_velocity_in_history"] == 1
    none = LastValidCausalVelocity(5).predict(make_hist(uv))                        # no valid velocity at all
    assert not none.window_valid[0] and np.isnan(none.positions).all()


def test_b1_ignores_velocity_values_flagged_invalid():
    uv = np.full((1, H, 2), 100.0)
    vel = np.full((1, H, 2), 999.0)                     # garbage under invalid masks (e.g. a non-causal value)
    vel[0, 2] = [4.0, 0.0]
    vv = np.zeros((1, H), bool)
    vv[0, 2] = True
    pr = LastValidCausalVelocity(5).predict(make_hist(uv, vel=vel, vvalid=vv))
    assert np.allclose(pr.positions[0, -1], [106.0, 100.0])


# ---------------- B2 ----------------
def test_b2_recovers_linear_trajectory_exactly():
    p0, v = np.array([800.0, 450.0]), np.array([12.0, -7.0])
    pr = B2().predict(make_hist(linear_history(p0, v)))
    assert np.allclose(pr.positions[0], p0 + v * TAU[:, None], atol=1e-9)
    assert np.allclose(pr.aux["fitted_velocity_px_s"][0], v, atol=1e-9)


def test_b2_uses_actual_timestamps():
    t = np.array([-0.5, -0.45, -0.3, -0.2, -0.05, 0.0])   # irregular sampling
    p0, v = np.array([100.0, 200.0]), np.array([30.0, 10.0])
    pr = B2().predict(make_hist(linear_history(p0, v, t), t=t))
    assert np.allclose(pr.aux["fitted_velocity_px_s"][0], v, atol=1e-9)
    # an index-based fit would give a different slope on irregular times
    idx_slope = np.polyfit(np.arange(H) * 0.1, linear_history(p0, v, t)[0, :, 0], 1)[0]
    assert abs(idx_slope - v[0]) > 1.0


def test_b2_anchors_at_last_observation_not_intercept():
    uv = linear_history([100.0, 100.0], [10.0, 0.0]).copy()
    uv[0, -1, 1] += 4.0                                   # noisy last observation
    pr = B2().predict(make_hist(uv))
    slope = pr.aux["fitted_velocity_px_s"][0]
    assert np.allclose(pr.positions[0, 0], uv[0, -1] + slope * 0.1)


def test_b2_missing_history_without_interpolation():
    uv = linear_history([300.0, 300.0], [20.0, 5.0]).copy()
    pv = np.ones((1, H), bool)
    pv[0, [1, 3]] = False
    uv[0, [1, 3]] = 1e6                                  # garbage under the invalid mask must be ignored
    pr = B2().predict(make_hist(uv, pvalid=pv))
    assert pr.window_valid[0] and np.allclose(pr.aux["fitted_velocity_px_s"][0], [20.0, 5.0], atol=1e-9)
    assert pr.aux["valid_history_samples"][0] == 4


def test_b2_reliability_rules():
    uv = linear_history([300.0, 300.0], [20.0, 5.0])
    pv = np.zeros((1, H), bool)
    pv[0, [4, 5]] = True
    assert B2().predict(make_hist(uv, pvalid=pv)).invalid_counts()["too_few_valid_history_samples"] == 1
    pv[0, 3] = True
    ok = B2().predict(make_hist(uv, pvalid=pv))              # 3 samples, span 0.2 s: accepted
    assert ok.window_valid[0]
    short = B2(min_time_span_s=0.3).predict(make_hist(uv, pvalid=pv))
    assert short.invalid_counts()["history_time_span_too_short"] == 1 and np.isnan(short.positions).all()
    ill = B2(min_centered_time_variance_s2=1.0).predict(make_hist(uv))
    assert ill.invalid_counts()["ill_conditioned_fit"] == 1
    with pytest.raises(WindowContractError):
        B2(center_timestamps=False)
    with pytest.raises(WindowContractError):
        B2(min_valid_samples=1)


# ---------------- interface / isolation ----------------
def test_predictors_accept_only_history_input():
    for p in (ZeroVelocity(), LastValidCausalVelocity(5), B2()):
        with pytest.raises(WindowContractError, match="PREDICTOR_INPUT_TYPE"):
            p.predict({"history_uv_px": np.zeros((1, H, 2)), "target_future_uv_px": np.zeros((1, F, 2))})


def test_history_input_has_no_future_fields_and_is_immutable():
    hist = make_hist(np.full((1, H, 2), 10.0))
    assert not any("future" in s or "target_future" in s for s in HistoryInput.__slots__)
    with pytest.raises(AttributeError):
        hist.target_future_uv_px = np.zeros((1, F, 2))
    with pytest.raises(ValueError):
        hist.history_uv_px[0, 0, 0] = 5.0                       # read-only arrays


def test_history_input_copies_caller_arrays():
    uv = np.full((1, H, 2), 10.0)
    pv = np.ones((1, H), bool)
    hist = make_hist(uv, pvalid=pv)
    uv[0, -1] = 999.0                                            # caller edits after construction ...
    pv[0, 0] = False                                             # ... stay possible and do not leak into the input
    assert hist.history_uv_px[0, -1, 0] == 10.0 and hist.history_position_valid[0, 0]


def test_history_input_validation():
    with pytest.raises(WindowContractError, match="COORDINATE_SYSTEM_MISMATCH"):
        make_hist(np.zeros((1, H, 2)), cs="world_metric")
    with pytest.raises(WindowContractError, match="COORDINATE_SYSTEM_MISMATCH"):
        make_hist(np.zeros((1, H, 2)), unit="metre")
    bad = np.zeros((1, H, 2))
    bad[0, 2, 0] = np.nan
    with pytest.raises(WindowContractError, match="NONFINITE_VALID_VALUE"):
        make_hist(bad)
    pv = np.ones((1, H), bool)
    pv[0, 2] = False
    make_hist(bad, pvalid=pv)                                    # NaN under an invalid mask is fine
    vel = np.zeros((1, H, 2))
    vel[0, 1] = np.inf
    vv = np.zeros((1, H), bool)
    vv[0, 1] = True
    with pytest.raises(WindowContractError, match="NONFINITE_VALID_VALUE"):
        make_hist(np.zeros((1, H, 2)), vel=vel, vvalid=vv)
    with pytest.raises(WindowContractError, match="HISTORY_SHAPE"):
        make_hist(np.zeros((1, H - 1, 2)))
    with pytest.raises(WindowContractError, match="HORIZON_OFFSETS"):
        make_hist(np.zeros((1, H, 2)), tau=np.array([0.2, 0.1]))


def test_prediction_dimensions_validated():
    class Broken(Predictor):
        BASELINE_ID = "broken"

        def config(self):
            return {}

        def _predict(self, hist):
            return np.zeros((hist.n, hist.f - 1, 2)), np.ones(hist.n, bool), np.full(hist.n, -1), {}
    with pytest.raises(WindowContractError, match="PREDICTION_SHAPE"):
        Broken().predict(make_hist(np.zeros((1, H, 2))))


def test_deterministic_and_traceable():
    uv = np.random.default_rng(0).normal(500, 20, size=(4, H, 2))
    h = make_hist(uv)
    a, b = B2().predict(h), B2().predict(h)
    assert np.array_equal(a.positions, b.positions)
    assert a.window_id.tolist() == [100, 101, 102, 103]


def test_registry_and_config_errors():
    assert isinstance(make_predictor({"baseline_id": "B0_zero_velocity", "params": {}}), ZeroVelocity)
    with pytest.raises(WindowContractError, match="UNKNOWN_BASELINE"):
        make_predictor({"baseline_id": "B9"})
    with pytest.raises(WindowContractError, match="BASELINE_CONFIG"):
        make_predictor({"baseline_id": "B1_last_valid_causal_velocity", "params": {"oops": 1}})
