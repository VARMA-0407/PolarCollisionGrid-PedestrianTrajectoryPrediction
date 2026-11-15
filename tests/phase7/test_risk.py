import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pedsafety.phase6 import ConflictLabelSet, compute_pair_features
from pedsafety.phase6.pairs import ConflictPairBatch
from pedsafety.phase6.scoring import ConflictScores, HeuristicProximityScorer
from pedsafety.phase7 import (CalibrationArtifact, CalibrationBlocked, ProvenanceMismatch, RiskConfig, RiskContractError, assess_risk, evaluate_calibrated,
                              fit_calibrator, summarize_risk)
from pedsafety.phase7.calibration import fit_isotonic, apply_method, reliability, brier, nll, group_bootstrap_ci

REPO = Path(__file__).resolve().parents[2]
CFG = RiskConfig.load(REPO / "configs/phase7/risk_default.json")
HZ = np.arange(1, 16) / 10.0
CK, MS, P6 = "a" * 64, "b" * 64, "c" * 64


def make_batch(P=4, dists=(5.0, 40.0, 100.0, 400.0), ped_valid=None, ctx_valid=None, scale=1.0):
    ped = np.zeros((P, 1, 15, 2))
    ped[:, 0, :, 0] = np.array(dists[:P])[:, None]
    pv = np.ones((P, 1, 15), bool) if ped_valid is None else ped_valid
    r = compute_pair_features(ped, pv, np.ones((P, 1)), np.zeros((P, 15, 2)), np.ones((P, 15), bool), ped[:, 0, 0, :], np.zeros((P, 2)),
                              np.tile([20.0, 100.0], (P, 1)), np.tile([40.0, 100.0 * scale], (P, 1)), HZ, min_valid_steps=3, kappa=2.0, sens_thresholds=[1.0], eps=1e-9)
    ctx = np.ones(P, bool) if ctx_valid is None else ctx_valid
    ids = np.arange(P)
    return ConflictPairBatch(window_id=ids.astype(np.int64), ped_track_index=np.zeros(P, np.int32), ped_track_id=np.array(["p0"] * P), other_track_index=ids.astype(np.int32),
                             other_track_id=np.array(["v%d" % i for i in ids]), other_type=np.ones(P, np.int8), pair_type=np.zeros(P, np.int8),
                             anchor_frame=(ids * 3).astype(np.int64), mode_probabilities=np.ones((P, 1)), horizon_s=HZ, features=r["features"], masks=r["masks"],
                             context_prediction_valid=ctx, window_index=ids, geometry={}, coordinate_system="captured_1920x1080", coordinate_unit="pixel",
                             config_sha256=P6, provenance={"labels_used": False})


def risk(batch, **kw):
    return assess_risk(batch, HeuristicProximityScorer().predict(batch), CFG, **kw)


def test_config_validation():
    d = json.loads((REPO / "configs/phase7/risk_default.json").read_text())
    assert CFG.category_status == "PROVISIONAL_ENGINEERING"
    for k, v in (("moderate_min", 0.9), ("category_status", "VALIDATED"), ("urgency_weight_floor", 0.0)):
        with pytest.raises(RiskContractError):
            RiskConfig.from_dict({**d, k: v})
    with pytest.raises(RiskContractError):
        RiskConfig.from_dict({**d, "category_status": "EMPIRICALLY_DERIVED"})          # needs cited evidence
    with pytest.raises(RiskContractError):
        RiskConfig.from_dict({**d, "extra": 1})


def test_uncalibrated_semantics_and_risk_score():
    b = make_batch()
    o = risk(b)
    assert o.calibration_status == "NOT_CALIBRATED" and np.isnan(o.calibrated_probability).all()
    assert o.risk_basis == "raw_conflict_score_uncalibrated" and np.isnan(o.severity).all()
    assert o.severity_status.startswith("SEVERITY_NOT_SUPPORTED")
    # nearest pair has highest score; risk = score * (w + (1-w)*urgency), urgency = 1 - tbar/T
    assert o.risk_score[0] > o.risk_score[1] > o.risk_score[2] >= o.risk_score[3]
    w = CFG.urgency_weight_floor
    t0 = b.features["t_min_sep_s"][0, 0]
    exp = o.conflict_score[0] * (w + (1 - w) * (1 - t0 / HZ[-1]))
    assert o.risk_score[0] == pytest.approx(exp)
    assert np.all((o.risk_score[np.isfinite(o.risk_score)] >= 0) & (o.risk_score[np.isfinite(o.risk_score)] <= 1))
    rec = o.to_records()[0]
    assert rec["calibrated_probability"] is None and rec["calibration_status"] == "NOT_CALIBRATED" and rec["severity"] is None
    assert rec["pedestrian_id"] == "p0" and rec["vehicle_id"] == "v0" and rec["anchor_frame"] == 0


def test_category_thresholds_and_provisional_status():
    o = risk(make_batch())
    assert o.risk_category[0] in ("HIGH", "CRITICAL") and o.risk_category[3] == "LOW"
    assert set(o.category_status[o.masks["category_assigned"]]) <= {"PROVISIONAL_ENGINEERING", "PROVISIONAL_ENGINEERING_CONFLICTING_INDICATORS"}
    strict = RiskConfig.from_dict({**json.loads((REPO / "configs/phase7/risk_default.json").read_text()), "moderate_min": 0.95, "high_min": 0.97, "critical_min": 0.99})
    o2 = assess_risk(make_batch(), HeuristicProximityScorer().predict(make_batch()), strict)
    assert (o2.risk_category[o2.masks["category_assigned"]] == "LOW").all()
    assert o.provenance["risk_config_sha256"] != o2.provenance["risk_config_sha256"]


def test_insufficient_evidence_and_masks():
    pv = np.ones((4, 1, 15), bool)
    pv[1] = False
    ctx = np.array([True, True, False, True])
    o = risk(make_batch(ped_valid=pv, ctx_valid=ctx))
    assert o.risk_category[1] == "INSUFFICIENT_EVIDENCE" and o.risk_category[2] == "INSUFFICIENT_EVIDENCE"
    assert np.isnan(o.risk_score[1]) and np.isnan(o.risk_score[2]) and not o.masks["conflict_valid"][[1, 2]].any()
    assert o.reasons[1] == "INVALID_OR_MISSING_CONFLICT_PREDICTION"
    assert o.risk_category[0] != "INSUFFICIENT_EVIDENCE"
    ood = risk(make_batch(scale=10.0))                                                  # agent 10x taller -> depth mismatch
    assert (ood.risk_category == "INSUFFICIENT_EVIDENCE").all() and ood.masks["out_of_distribution"].all()


def test_deterministic_and_provenance():
    b = make_batch()
    a1, a2 = risk(b, provenance={"dataset": "x"}), risk(b, provenance={"dataset": "x"})
    assert np.array_equal(a1.risk_score, a2.risk_score, equal_nan=True) and a1.to_records() == a2.to_records()
    assert a1.provenance["dataset"] == "x" and a1.provenance["phase6_config_sha256"] == P6 and a1.provenance["future_observations_used"] is False
    assert summarize_risk(a1)["evaluation"] == "DESCRIPTIVE_ONLY_NO_VERIFIED_LABELS"


def test_coordinate_and_identity_enforced():
    b = make_batch()
    s = HeuristicProximityScorer().predict(b)
    b.coordinate_system = "world_m"
    with pytest.raises(RiskContractError):
        assess_risk(b, s, CFG)
    b = make_batch()
    bad = HeuristicProximityScorer().predict(b)
    bad.window_id = bad.window_id + 1
    with pytest.raises(RiskContractError):
        assess_risk(b, bad, CFG)


# ------------------------ calibration (synthetic verified labels) ------------------------
def labelled(n_per_group=150, groups=("g1", "g2", "g3", "g4", "g5", "g6"), seed=0, kind="VERIFIED_NEAR_CONFLICT", verified=True):
    rng = np.random.default_rng(seed)
    n = n_per_group * len(groups)
    g = np.array([groups[i // n_per_group] for i in range(n)])
    score = rng.uniform(0, 1, n)
    pos = rng.uniform(0, 1, n) < 0.1 + 0.8 * score ** 2             # true P(y|score) is smooth and monotone
    anchor = np.arange(n, dtype=np.int64)
    lab = ConflictLabelSet(kind, verified, {"name": "x", "type": "official_dataset_annotation", "version": "1", "sha256": "0" * 64, "protocol_reference": "doc"},
                           np.arange(n), np.zeros(n, np.int32), np.arange(n, dtype=np.int32), anchor, anchor + 1, anchor + 10, pos, np.ones(n, bool), g)
    keys = list(zip(range(n), [0] * n, range(n)))
    return score, np.ones(n, bool), lab, keys


SRC = {"checkpoint_file_sha256": CK, "model_state_sha256": MS, "phase6_config_sha256": P6, "coordinate_system": "captured_1920x1080", "scorer_id": "unit"}
SPLIT = dict(train_groups=["g1", "g2"], calibration_groups=["g3", "g4"], test_groups=["g5", "g6"])
KIND = "UNCALIBRATED_HEURISTIC_PROXIMITY_SCORE"


def fitted(method="platt", **over):
    s, v, lab, keys = labelled()
    return fit_calibrator(s, v, lab, keys, score_kind=KIND, source=SRC, method=method, **{**SPLIT, **over}), (s, v, lab, keys)


def test_platt_fit_improves_and_evaluates_on_test_only(tmp_path):
    art, (s, v, lab, keys) = fitted()
    assert art.payload["split"]["test_groups"] == ["g5", "g6"]
    assert art.payload["n_calibration_rows"] == 300
    ev = evaluate_calibrated(art, s, v, lab, keys)
    assert ev["evaluated_on"] == "test_groups" and ev["calibrated"]["n"] == 300 and ev["groups"] == ["g5", "g6"]
    assert ev["calibrated"]["ece"] < 0.12 and 0 <= ev["calibrated"]["brier"] <= 1
    assert ev["calibrated"]["ci_brier"]["available"] is False                               # only 2 test groups -> no cluster CI claimed
    p = art.apply(s, v)
    assert np.all((p > 0) & (p < 1))


def test_isotonic_gated_by_data_and_monotone():
    with pytest.raises(CalibrationBlocked) as e:
        fitted("isotonic")                                                                 # 300 rows < 1000
    assert e.value.code == "ISOTONIC_DATA"
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 500)
    y = rng.uniform(0, 1, 500) < s
    p = apply_method("isotonic", fit_isotonic(s, y), np.sort(s))
    assert np.all(np.diff(p) >= -1e-12)


def test_partition_isolation():
    for over in (dict(calibration_groups=["g2", "g3"]), dict(test_groups=["g4", "g5"]), dict(train_groups=["g1", "g6"]), dict(test_groups=[])):
        with pytest.raises(CalibrationBlocked):
            fitted(**over)
    # calibration rows never include test or train groups: perturbing their labels must not change the fitted parameters
    s, v, lab, keys = labelled()
    a = fit_calibrator(s, v, lab, keys, score_kind=KIND, source=SRC, **SPLIT)
    lab2 = copy.deepcopy(lab)
    other = np.isin(lab2.group_id, ["g1", "g2", "g5", "g6"])
    lab2.positive = np.where(other, ~lab2.positive, lab2.positive)
    b = fit_calibrator(s, v, lab2, keys, score_kind=KIND, source=SRC, **SPLIT)
    assert a.payload["params"] == b.payload["params"] and a.payload["n_calibration_rows"] == b.payload["n_calibration_rows"]


def test_blocked_without_verified_labels_or_groups():
    s, v, lab, keys = labelled(kind="CANDIDATE_HEURISTIC_PERCEIVED_TRACK_PROXIMITY", verified=False)
    with pytest.raises(CalibrationBlocked) as e:
        fit_calibrator(s, v, lab, keys, score_kind=KIND, source=SRC, **SPLIT)
    assert e.value.code == "LABELS_NOT_VERIFIED"
    s, v, lab, keys = labelled(groups=("g1", "g2"))
    with pytest.raises(CalibrationBlocked):
        fit_calibrator(s, v, lab, keys, score_kind=KIND, source=SRC, train_groups=["g1"], calibration_groups=["g2"], test_groups=["g3"])
    s, v, lab, keys = labelled(n_per_group=10)
    with pytest.raises(CalibrationBlocked):
        fit_calibrator(s, v, lab, keys, score_kind=KIND, source=SRC, **SPLIT)               # too few rows / groups of positives
    with pytest.raises(CalibrationBlocked):
        fit_calibrator(*labelled()[:2], labelled()[2], labelled()[3], score_kind=KIND, source={**SRC, "checkpoint_file_sha256": ""}, **SPLIT)


def test_artifact_serialization_and_hash_verification(tmp_path):
    art, _ = fitted()
    path = tmp_path / "cal.json"
    h = art.save(path)
    assert CalibrationArtifact.load(path, expected_sha256=h).payload == art.payload
    with pytest.raises(ProvenanceMismatch):
        CalibrationArtifact.load(path, expected_sha256="0" * 64)
    d = json.loads(path.read_text())
    d["params"]["b"] += 1.0
    path.write_text(json.dumps(d))
    with pytest.raises(ProvenanceMismatch) as e:
        CalibrationArtifact.load(path)
    assert e.value.code == "ARTIFACT_HASH"


def test_calibrated_risk_output_and_provenance_mismatch():
    art, _ = fitted()
    b = make_batch()
    s = HeuristicProximityScorer().predict(b)
    o = assess_risk(b, s, CFG, art, checkpoint_file_sha256=CK, phase6_config_sha256=P6)
    ok = o.masks["conflict_valid"]
    assert o.calibration_status == "CALIBRATED" and o.risk_basis == "calibrated_probability"
    assert np.all((o.calibrated_probability[ok] > 0) & (o.calibrated_probability[ok] < 1)) and np.isnan(o.calibrated_probability[~ok]).all()
    assert o.provenance["calibration_artifact_sha256"] == art.sha256 and o.provenance["source_checkpoint_file_sha256"] == CK
    with pytest.raises(ProvenanceMismatch) as e:
        assess_risk(b, s, CFG, art, checkpoint_file_sha256="d" * 64, phase6_config_sha256=P6)
    assert e.value.code == "PROVENANCE_SOURCE_CHECKPOINT_FILE_SHA256"
    with pytest.raises(ProvenanceMismatch):
        assess_risk(b, s, CFG, art, checkpoint_file_sha256=CK, phase6_config_sha256="e" * 64)
    other = ConflictScores(s.scores, s.valid, "UNCALIBRATED_CLASSIFIER_SCORE", window_id=s.window_id, ped_track_index=s.ped_track_index, other_track_index=s.other_track_index)
    with pytest.raises(ProvenanceMismatch):
        assess_risk(b, other, CFG, art, checkpoint_file_sha256=CK, phase6_config_sha256=P6)
    with pytest.raises(RiskContractError):
        assess_risk(b, s, CFG, art)                                                       # hashes required


def test_metrics_correctness():
    p, y = np.array([0.9, 0.1, 0.8, 0.2]), np.array([1, 0, 1, 0], bool)
    assert brier(p, y) == pytest.approx(np.mean([0.01, 0.01, 0.04, 0.04]))
    assert nll(p, y) == pytest.approx(-np.mean(np.log([0.9, 0.9, 0.8, 0.8])))
    r = reliability(p, y)
    assert r["ece"] == pytest.approx(0.15) and sum(b["n"] for b in r["bins"]) == 4
    ci = group_bootstrap_ci(np.tile(p, 3), np.tile(y, 3), np.repeat(list("abcdef"), 2), brier)
    assert ci["available"] and ci["lo"] <= ci["hi"]


# ------------------------ integration on real artifacts ------------------------
W = REPO / "experiments/phase5_task3_windows_pie_video_0004/20261115T035825Z_pie_video_0004/primary"
N = REPO / "experiments/phase5_task7_neural_evaluation/20261115T191319Z_pie_video_0004"


@pytest.mark.skipif(not (W.exists() and N.exists()), reason="real artifacts not present")
def test_real_integration(tmp_path):
    from pedsafety.phase7.runner import run_risk_pipeline
    kw = (W, N, REPO / "configs/phase6/conflict_default.json", REPO / "configs/phase7/risk_default.json", REPO / "configs/phase5/window_generation_default.json")
    r1 = run_risk_pipeline(*kw, out_dir=tmp_path)
    r2 = run_risk_pipeline(*kw)
    assert r1["summary"] == r2["summary"]
    assert r1["summary"]["calibration_status"] == "NOT_CALIBRATED" and r1["calibration"]["status"].startswith("BLOCKED")
    assert r1["claims"] == {"calibrated_probability": False, "validated_risk_categories": False, "physical_ttc_pet": False, "severity": False, "deployment_ready": False}
    assert r1["provenance"]["source_checkpoint_file_sha256"] == "40aff17f1a40eb2da24d0e0b63245eecefa5bba4e3f6e44c272dc07f3ab71b44"
    arr = np.load(tmp_path / "risk_arrays.npz")
    assert np.isnan(arr["calibrated_probability"]).all()
    assert "INSUFFICIENT_EVIDENCE" in r1["summary"]["category_counts"]
