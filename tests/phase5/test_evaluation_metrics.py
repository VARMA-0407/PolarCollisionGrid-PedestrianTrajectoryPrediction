"""P5.4 evaluator: hand-calculated ADE/FDE, masks, aggregation, oracle metrics, miss rate, validation."""

import numpy as np
import pytest

from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.evaluation import EvalConfig, common_window_mask, evaluate_predictions
from p54_helpers import F, TAU

CS, UNIT = "captured_1920x1080", "pixel"


def ev(gt, gv, pr, pv, cfg=None, tracks=None, probs=None, wid=None, cs=CS, unit=UNIT):
    n = gt.shape[0]
    return evaluate_predictions(gt, gv, pr, pv, np.arange(n) if wid is None else wid,
                                tracks if tracks is not None else ["t%d" % i for i in range(n)], coordinate_system=cs, unit=unit,
                                model_id="m", horizon_offsets_s=TAU, cfg=cfg or EvalConfig(), mode_probabilities=probs)


def gt1(n=1):
    g = np.zeros((n, F, 2))
    g[:, :, 0] = 100.0 + 10.0 * TAU
    g[:, :, 1] = 200.0
    return g, np.ones((n, F), bool)


def test_perfect_prediction_zero_error():
    g, gv = gt1()
    r = ev(g, gv, g.copy(), gv.copy())
    assert r.ade[0] == 0.0 and r.fde[0] == 0.0


def test_constant_horizontal_and_vertical_error():
    g, gv = gt1()
    rh = ev(g, gv, g + [3.0, 0.0], gv)
    rv = ev(g, gv, g + [0.0, -4.0], gv)
    assert rh.ade[0] == pytest.approx(3.0) and rh.fde[0] == pytest.approx(3.0)
    assert rv.ade[0] == pytest.approx(4.0) and rv.fde[0] == pytest.approx(4.0)


def test_diagonal_euclidean_and_growing_error():
    g, gv = gt1()
    assert ev(g, gv, g + [3.0, 4.0], gv).ade[0] == pytest.approx(5.0)
    grow = g.copy()
    grow[0, :, 0] += np.arange(1, F + 1)                    # error t px at step t
    r = ev(g, gv, grow, gv)
    assert r.ade[0] == pytest.approx(8.0) and r.fde[0] == pytest.approx(15.0)
    assert r.step_error[0].tolist() == pytest.approx(list(range(1, F + 1)))


def test_invalid_prediction_not_zero_and_counted():
    g, gv = gt1(2)
    p = g.copy()
    pv = np.ones((2, F), bool)
    pv[1] = False
    p[1] = np.nan
    r = ev(g, gv, p, pv)
    assert r.evaluated.tolist() == [True, False] and np.isnan(r.ade[1]) and np.isnan(r.fde[1])
    s = r.summary
    assert s["windows_evaluated"] == 1 and s["not_evaluated_reasons"]["prediction_invalid"] == 1
    assert s["window_weighted"]["ade_px"]["n"] == 1 and s["window_weighted"]["ade_px"]["mean"] == 0.0


def test_invalid_target_not_zero_and_counted():
    g, gv = gt1(2)
    pred = g + 1.0
    gv[0, 7] = False
    g[0, 7] = np.nan
    r = ev(g, gv, pred, np.ones((2, F), bool))
    assert not r.evaluated[0] and r.invalid_reason[0] == 0 and np.isnan(r.ade[0])
    assert r.summary["not_evaluated_reasons"]["target_incomplete"] == 1


def test_partial_prediction_mask_excluded_strict_policy():
    g, gv = gt1()
    pv = np.ones((1, F), bool)
    pv[0, -1] = False                                       # 14 of 15 valid
    r = ev(g, gv, g.copy(), pv)
    assert not r.evaluated[0] and r.invalid_reason[0] == 1
    with pytest.raises(WindowContractError):
        EvalConfig(partial_target_policy="PARTIAL_ALLOWED")


def test_dimension_validation():
    g, gv = gt1()
    with pytest.raises(WindowContractError, match="EVAL_SHAPE"):
        ev(g, gv, g[:, :-1], gv[:, :-1])
    with pytest.raises(WindowContractError, match="EVAL_SHAPE"):
        ev(g, gv, g, gv[:, :-1])
    with pytest.raises(WindowContractError, match="EVAL_SHAPE"):
        ev(g[0], gv, g, gv)


def test_coordinate_metadata_validation():
    g, gv = gt1()
    with pytest.raises(WindowContractError, match="COORDINATE_SYSTEM_MISMATCH"):
        ev(g, gv, g, gv, cs="world_utm")
    with pytest.raises(WindowContractError, match="COORDINATE_SYSTEM_MISMATCH"):
        ev(g, gv, g, gv, unit="metre")


def test_nan_and_inf_rejected_where_valid_allowed_where_masked():
    g, gv = gt1()
    p = g.copy()
    p[0, 3, 1] = np.inf
    with pytest.raises(WindowContractError, match="NONFINITE_VALID_VALUE"):
        ev(g, gv, p, gv)
    g2 = g.copy()
    g2[0, 2, 0] = np.nan
    with pytest.raises(WindowContractError, match="NONFINITE_VALID_VALUE"):
        ev(g2, gv, g, gv)
    pv = gv.copy()
    pv[0, 3] = False
    r = ev(g, gv, p, pv)                                    # inf under an invalid mask: allowed, window excluded
    assert not r.evaluated[0]


def test_window_ids_traceable_and_unique():
    g, gv = gt1(3)
    r = ev(g, gv, g, gv, wid=np.array([7, 3, 9]))
    assert r.window_id.tolist() == [7, 3, 9]
    with pytest.raises(WindowContractError, match="DUPLICATE_WINDOW_ID"):
        ev(g, gv, g, gv, wid=np.array([1, 1, 2]))


def test_window_vs_track_weighted_aggregation():
    g, gv = gt1(4)
    p = g.copy()
    p[:3, :, 0] += 1.0                                      # track A: 3 windows, ADE 1
    p[3, :, 0] += 5.0                                       # track B: 1 window, ADE 5
    r = ev(g, gv, p, gv, tracks=["A", "A", "A", "B"])
    s = r.summary
    assert s["window_weighted"]["ade_px"]["mean"] == pytest.approx(2.0)
    assert s["track_weighted"]["ade_px_mean_of_track_means"] == pytest.approx(3.0)
    assert s["window_weighted"]["ade_px"]["median"] == pytest.approx(1.0)
    assert s["window_weighted"]["ade_px"]["std"] == pytest.approx(np.std([1, 1, 1, 5]))
    assert [t["track_id"] for t in s["per_track"]] == ["A", "B"]


def test_k1_min_metrics_equal_ade_fde():
    g, gv = gt1()
    r = ev(g, gv, g + [3.0, 4.0], gv)
    assert r.k == 1 and r.min_ade[0] == r.ade[0] and r.min_fde[0] == r.fde[0]
    assert r.summary["oracle_best_of_k"]["identical_to_ade_fde"] is True


def test_k_gt_1_oracle_and_primary_mode():
    g, gv = gt1()
    modes = np.stack([g + [10.0, 0.0], g + [1.0, 0.0]], axis=1)       # [1, 2, F, 2]
    mv = np.ones((1, 2, F), bool)
    r = ev(g, gv, modes, mv, probs=np.array([[0.8, 0.2]]))
    assert r.min_ade[0] == pytest.approx(1.0)                          # oracle picks the closest
    assert r.ade[0] == pytest.approx(10.0)                             # primary = most probable, never the closest
    assert "ORACLE" in r.summary["oracle_best_of_k"]["label"]
    r2 = ev(g, gv, modes, mv)                                          # K > 1 without probabilities: no primary metric
    assert not r2.evaluated.any() and r2.min_ade[0] == pytest.approx(1.0)


def test_miss_rate_not_configured_unless_threshold_given():
    g, gv = gt1(2)
    p = g.copy()
    p[1, :, 0] += 20.0
    assert ev(g, gv, p, gv).summary["miss_rate"]["status"] == "NOT_CONFIGURED"
    s = ev(g, gv, p, gv, cfg=EvalConfig(miss_rate_threshold_px=10.0)).summary["miss_rate"]
    assert s["status"] == "CONFIGURED" and s["miss_rate"] == pytest.approx(0.5)
    with pytest.raises(WindowContractError):
        EvalConfig(miss_rate_threshold_px=-3.0)


def test_per_horizon_report_and_common_mask():
    g, gv = gt1(2)
    grow = g.copy()
    grow[:, :, 0] += np.arange(1, F + 1)
    r = ev(g, gv, grow, gv)
    at = r.summary["per_horizon_mean_displacement_error_px"]["at_report_horizons"]
    assert at == {"0.5s": pytest.approx(5.0), "1.0s": pytest.approx(10.0), "1.5s": pytest.approx(15.0)}
    pv = gv.copy()
    pv[0] = False
    r2 = ev(g, gv, np.where(pv[..., None], grow, np.nan), pv)
    assert common_window_mask([r, r2]).tolist() == [False, True]


def test_deterministic():
    g, gv = gt1(5)
    p = g + np.random.default_rng(1).normal(0, 3, size=g.shape)
    a, b = ev(g, gv, p, gv), ev(g, gv, p, gv)
    assert np.array_equal(a.ade, b.ade) and a.summary == b.summary
