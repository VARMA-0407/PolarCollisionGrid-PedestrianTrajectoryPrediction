import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pedsafety.phase6 import (ConflictConfig, ConflictContractError, ConflictLabelError, ConflictLabelSet, ConflictTrainingBlocked, LogisticConflictHead,
                              compute_pair_features, evaluate_supervised, validate_label_set, FEATURE_SPECS)
from pedsafety.phase6.metrics import average_precision, candidate_indicator_agreement, roc_auc
from pedsafety.phase6.scoring import ConflictScores, group_split

REPO = Path(__file__).resolve().parents[2]
HZ = np.arange(1, 16) / 10.0


def feats(P=2, K=2, F=15, ped=None, oth=None, ped_valid=None, oth_valid=None, probs=None, **kw):
    ped = np.zeros((P, K, F, 2)) + np.array([100.0, 0.0]) if ped is None else ped
    oth = np.zeros((P, F, 2)) if oth is None else oth
    pv = np.ones((P, K, F), bool) if ped_valid is None else ped_valid
    ov = np.ones((P, F), bool) if oth_valid is None else oth_valid
    pr = np.full((P, K), 1.0 / K) if probs is None else probs
    args = dict(min_valid_steps=3, kappa=2.0, sens_thresholds=[0.5, 1.0], eps=1e-9)
    args.update(kw)
    return compute_pair_features(ped, pv, pr, oth, ov, np.tile([100.0, 0.0], (P, 1)), np.zeros((P, 2)), np.tile([20.0, 100.0], (P, 1)),
                                 np.tile([40.0, 100.0], (P, 1)), HZ, **args)


def test_separation_units_and_scale():
    r = feats()
    assert np.allclose(r["features"]["sep_px"][r["masks"]["step_valid"]], 100.0)
    assert np.allclose(r["features"]["sep_over_h"][r["masks"]["step_valid"]], 1.0)
    assert set(v[1] for v in FEATURE_SPECS.values()) <= {"px", "1", "px/s", "s", "steps"}


def test_deterministic():
    a, b = feats(), feats()
    for k in a["features"]:
        assert np.array_equal(a["features"][k], b["features"][k], equal_nan=True)


def test_converging_closing_rate_and_los():
    ped = np.zeros((1, 1, 15, 2)) + np.array([100.0, 0.0]) + HZ[None, None, :, None] * np.array([-60.0, 0.0])
    r = feats(P=1, K=1, ped=ped)
    assert r["features"]["max_closing_rate_px_s"][0, 0] > 0
    assert r["features"]["approach_fraction"][0, 0] == 1.0
    assert r["features"]["los_convergence_cos"][0, 0] == pytest.approx(1.0)
    assert r["features"]["t_min_sep_s"][0, 0] == pytest.approx(1.5)


def test_mask_propagation_and_missing():
    pv = np.ones((2, 2, 15), bool)
    pv[0, :, :] = False
    ov = np.ones((2, 15), bool)
    ov[1, 4:] = False
    r = feats(ped_valid=pv, oth_valid=ov)
    assert not r["masks"]["pair_valid"][0] and np.isnan(r["features"]["mode_weighted_score"][0])
    assert r["masks"]["pair_valid"][1] and r["features"]["n_valid_steps"][1].tolist() == [4, 4]
    assert np.isnan(r["features"]["sep_px"][1, :, 4:]).all()
    ov[1, :] = False
    ov[1, :2] = True
    assert not feats(ped_valid=pv, oth_valid=ov)["masks"]["pair_valid"][1]


def test_invalid_numbers_rejected_only_where_valid():
    ped = np.zeros((1, 1, 15, 2)) + 100.0
    ped[0, 0, 3, 0] = np.nan
    pv = np.ones((1, 1, 15), bool)
    with pytest.raises(ConflictContractError):
        feats(P=1, K=1, ped=ped, ped_valid=pv)
    pv[0, 0, 3] = False
    assert feats(P=1, K=1, ped=ped, ped_valid=pv)["masks"]["pair_valid"][0]
    with pytest.raises(ConflictContractError):
        feats(probs=np.array([[0.9, 0.9], [0.5, 0.5]]))
    with pytest.raises(ConflictContractError):
        feats(probs=np.array([[np.nan, 0.5], [0.5, 0.5]]))


def test_mode_pair_handling():
    ped = np.zeros((1, 2, 15, 2))
    ped[0, 0] = [300.0, 0.0]
    ped[0, 1] = [10.0, 0.0]
    near = feats(P=1, K=2, ped=ped, probs=np.array([[0.1, 0.9]]))["features"]
    far = feats(P=1, K=2, ped=ped, probs=np.array([[0.9, 0.1]]))["features"]
    assert near["most_probable_mode"][0] == 1 and far["most_probable_mode"][0] == 0
    assert near["mode_weighted_score"][0] > far["mode_weighted_score"][0]
    assert near["worst_case_min_sep_over_h"][0] == far["worst_case_min_sep_over_h"][0]


def test_horizon_alignment_enforced():
    with pytest.raises(ConflictContractError):
        compute_pair_features(np.zeros((1, 1, 3, 2)), np.ones((1, 1, 3), bool), np.ones((1, 1)), np.zeros((1, 3, 2)), np.ones((1, 3), bool), np.zeros((1, 2)),
                              np.zeros((1, 2)), np.ones((1, 2)), np.ones((1, 2)), np.array([0.3, 0.2, 0.1]), min_valid_steps=1, kappa=2.0,
                              sens_thresholds=[1.0], eps=1e-9)


def test_config_coordinate_enforcement_and_unknown_keys():
    d = json.loads((REPO / "configs/phase6/conflict_default.json").read_text())
    cfg = ConflictConfig.from_dict(d)
    assert cfg.sha256() == ConflictConfig.from_dict(copy.deepcopy(d)).sha256()
    for k, v in (("coordinate_system", "world_m"), ("coordinate_unit", "meter"), ("candidate_label_policy", "USE_AS_TRUTH"), ("label", "VALIDATED")):
        bad = dict(d)
        bad[k] = v
        with pytest.raises(ConflictContractError):
            ConflictConfig.from_dict(bad)
    with pytest.raises(ConflictContractError):
        ConflictConfig.from_dict({**d, "extra": 1})


def _labels(kind="VERIFIED_NEAR_CONFLICT", verified=True, n=60, groups=("a", "b", "c"), start_offset=1):
    anchor = np.arange(n, dtype=np.int64)
    pos = np.arange(n) % 2 == 0
    return ConflictLabelSet(kind, verified, {"name": "x", "type": "official_dataset_annotation", "version": "1", "sha256": "0" * 64, "protocol_reference": "doc"},
                            np.arange(n), np.zeros(n, np.int32), np.ones(n, np.int32), anchor, anchor + start_offset, anchor + 10, pos, np.ones(n, bool),
                            np.array([groups[i % len(groups)] for i in range(n)]), event_frame=np.where(pos, anchor + 5, -1))


def test_label_validation():
    assert validate_label_set(_labels())["supervision_eligible"]
    with pytest.raises(ConflictLabelError):
        validate_label_set(_labels("CANDIDATE_HEURISTIC_PERCEIVED_TRACK_PROXIMITY", False))
    with pytest.raises(ConflictLabelError):
        validate_label_set(_labels(verified=False))
    with pytest.raises(ConflictLabelError) as e:
        validate_label_set(_labels(start_offset=0))
    assert e.value.code == "LABEL_TIME_LEAK"
    with pytest.raises(ConflictLabelError) as e:
        validate_label_set(_labels(groups=("a",)))
    assert e.value.code == "LABEL_GROUPS_INSUFFICIENT"
    lab = _labels()
    lab.source["sha256"] = ""
    with pytest.raises(ConflictLabelError):
        validate_label_set(lab)


def test_training_blocked_without_verified_labels():
    with pytest.raises(ConflictTrainingBlocked):
        LogisticConflictHead()._score(np.zeros((1, 9)), np.ones((1, 9), bool))
    with pytest.raises(ConflictTrainingBlocked):
        group_split(["only"])


def test_logistic_arithmetic_synthetic_and_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] > 0
    M = np.ones_like(X, bool)
    a, b = LogisticConflictHead(columns=("a", "b", "c")), LogisticConflictHead(columns=("a", "b", "c"))
    a._fit_arrays(X, M, y)
    b._fit_arrays(X, M, y)
    assert np.array_equal(a.w, b.w)
    assert roc_auc(y, a._score(X, M)) > 0.95


def test_metrics_and_gating():
    y = np.array([1, 1, 0, 0, 0], bool)
    s = np.array([.9, .4, .5, .2, .1])
    assert roc_auc(y, s) == pytest.approx(5 / 6)
    assert average_precision(y, s) == pytest.approx((1 + 2 / 3) / 2)
    n = 60
    lab = _labels(n=n)
    sc = ConflictScores(np.where(lab.positive, .8, .2), np.ones(n, bool), "UNCALIBRATED_HEURISTIC_PROXIMITY_SCORE", window_id=lab.window_id,
                        ped_track_index=lab.ped_track_index, other_track_index=lab.other_track_index)
    out = evaluate_supervised(sc, lab, threshold=0.5, fps=30.0)
    assert out["f1"] == 1.0 and out["score_calibrated"] is False
    with pytest.raises(ConflictLabelError):
        evaluate_supervised(sc, _labels("CANDIDATE_HEURISTIC_PERCEIVED_TRACK_PROXIMITY", False), threshold=0.5)
    with pytest.raises(ConflictContractError):
        evaluate_supervised(sc, lab, threshold=float("nan"))
    assert "precision" not in candidate_indicator_agreement(np.array([1, 0], bool), np.array([1, 1], bool), np.ones(2, bool))


def test_scores_never_probabilities():
    with pytest.raises(ConflictContractError):
        ConflictScores(np.zeros(1), np.ones(1, bool), "x", calibrated=True)


# --------------------------- integration on real Phase 4/5 artifacts ---------------------------
W = REPO / "experiments/phase5_task3_windows_pie_video_0004/20261115T035825Z_pie_video_0004/primary"
P = REPO / "experiments/phase5_task7_neural_evaluation/20261115T191319Z_pie_video_0004/predictions_neural.npz"
real = pytest.mark.skipif(not (W.exists() and P.exists()), reason="real Phase 5 artifacts not present")


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    from pedsafety.phase6.runner import run_conflict_pipeline
    out = tmp_path_factory.mktemp("p6")
    return run_conflict_pipeline(W, P, REPO / "configs/phase6/conflict_default.json", REPO / "configs/phase5/window_generation_default.json", out), out


@real
def test_real_integration(real_run):
    rep, out = real_run
    assert rep["exclusions"]["pairs"] == rep["summary"]["n_pairs"] > 0
    assert rep["provenance"]["future_observations_used"] is False and rep["provenance"]["labels_used"] is False
    assert rep["provenance"]["predictions_sha256"] == "c070134fb67ef42753ecca347260c9f25e8b7ad37ae3333b690f3f008ee99748"
    assert rep["supervision"]["candidate_labels_generated"] is False
    assert (out / "conflict_report.json").is_file()


@real
def test_real_determinism_identity_and_no_future_access():
    from pedsafety.phase5.evaluation.experiment import load_baseline_config
    from pedsafety.phase5.window_config import WindowConfig
    from pedsafety.phase5.window_storage import read_dataset
    from pedsafety.phase6.pairs import build_conflict_pairs
    from pedsafety.phase6.runner import load_predictions
    cfg = ConflictConfig.load(REPO / "configs/phase6/conflict_default.json")
    wcfg = WindowConfig.load(REPO / "configs/phase5/window_generation_default.json")
    ws, bank, extra = read_dataset(W)
    pred = load_predictions(P)
    bcfg, _ = load_baseline_config(REPO / cfg.baseline_evaluation_config_path)
    idx = np.arange(0, ws.n_windows, 25)
    args = (ws, bank, extra["history_target_bank_node"], wcfg, cfg, bcfg)
    b1 = build_conflict_pairs(pred, *args, window_indices=idx)
    b2 = build_conflict_pairs(pred, *args, window_indices=idx)
    assert np.array_equal(b1.features["mode_weighted_score"], b2.features["mode_weighted_score"], equal_nan=True)
    assert np.array_equal(b1.ped_track_index, ws.target_track_index[b1.window_index])
    assert (b1.pair_type == 0).all() and b1.coordinate_unit == "pixel"
    # corrupting every context node outside the history frames of the selected windows must not change the features
    used = np.unique(ws.history_frames[idx])
    outside = ~np.isin(bank.nodes["frame"], used)
    assert outside.any()
    saved = bank.nodes["features"].copy()
    bank.nodes["features"][outside] = 1e6
    try:
        b3 = build_conflict_pairs(pred, *args, window_indices=idx)
    finally:
        bank.nodes["features"][:] = saved
    assert np.array_equal(b1.features["min_sep_px"], b3.features["min_sep_px"], equal_nan=True)
    bad = load_predictions(P)
    bad.coordinate_system = "world_m"
    with pytest.raises(ConflictContractError):
        build_conflict_pairs(bad, *args, window_indices=idx)
    bad = load_predictions(P)
    bad.anchor_frame = bad.anchor_frame + 3
    with pytest.raises(ConflictContractError):
        build_conflict_pairs(bad, *args, window_indices=idx)
