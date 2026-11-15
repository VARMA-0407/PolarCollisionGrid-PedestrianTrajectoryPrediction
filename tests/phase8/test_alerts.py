import copy
import gzip
import itertools
import json
from pathlib import Path

import pytest

from pedsafety.phase8 import (AlertConfig, AlertContractError, AlertManager, InputContractError, InvalidTransition, LeadTimeBlocked, Observation, TimestampOrderError,
                              parse_record, validate_event)
from pedsafety.phase8.lead_time import blocked_status, compute_lead_times
from pedsafety.phase8.manager import ACTIVE, ALLOWED, CLEARED, ESCALATED, INSUFFICIENT, NO_ALERT, PENDING, transition_allowed
from pedsafety.phase8.schema import STATES, schema_sha256

REPO = Path(__file__).resolve().parents[2]
CFG = AlertConfig.load(REPO / "configs/phase8/alert_default.json")
SCORE = {"LOW": 0.05, "MODERATE": 0.2, "HIGH": 0.5, "CRITICAL": 0.8}
STEP = 3


def ob(frame, cat, ped="pie:video_0004:1", veh="7", prob=None, status="NOT_CALIBRATED", conflicting=False, idx=None):
    valid = cat in SCORE
    return Observation(video_id="pie:video_0004", pedestrian_id=ped, vehicle_id=veh, frame=frame, window_id=0, risk_category=cat,
                       category_status="PROVISIONAL_ENGINEERING" + ("_CONFLICTING_INDICATORS" if conflicting else ""), risk_score=SCORE.get(cat), conflict_score=SCORE.get(cat),
                       conflict_score_kind="UNCALIBRATED_HEURISTIC_PROXIMITY_SCORE", calibrated_probability=prob, calibration_status=status, temporal_urgency=0.5,
                       masks={"conflict_valid": valid, "category_assigned": valid, "out_of_distribution": False}, reason=None if valid else "X", valid=valid,
                       invalid_reason=None if valid else "X", conflicting_indicators=conflicting, source_index=idx)


def run(cats, cfg=CFG, **kw):
    """Feed one pair a list of categories at anchors 0, 3, 6, ...; None = pair absent. Returns (manager, [state after each step])."""
    m = AlertManager(cfg, {"dataset": "unit"}, source_records_sha256="s" * 64, hashes={"phase7_risk_config_sha256": "r" * 64, "source_checkpoint_file_sha256": "c" * 64})
    states = []
    for i, c in enumerate(cats):
        m.step(i * STEP, [] if c is None else [ob(i * STEP, c, **kw)])
        ps = m.pairs.get(("pie:video_0004", "pie:video_0004:1", "7"))
        states.append(ps.state if ps else NO_ALERT)
    return m, states


def trans(m):
    return [(e["frame"], e["from_state"], e["to_state"], e["trigger_reason"]) for e in m.events if e["event_type"] == "STATE_TRANSITION"]


# ------------------------------------------------------------------ state machine
def test_transition_table_is_exactly_the_documented_one():
    expect = {(NO_ALERT, PENDING), (NO_ALERT, INSUFFICIENT), (PENDING, ACTIVE), (PENDING, NO_ALERT), (PENDING, INSUFFICIENT), (ACTIVE, ESCALATED), (ACTIVE, CLEARED),
              (ACTIVE, INSUFFICIENT), (ESCALATED, ACTIVE), (ESCALATED, CLEARED), (ESCALATED, INSUFFICIENT), (INSUFFICIENT, NO_ALERT), (INSUFFICIENT, PENDING),
              (INSUFFICIENT, ACTIVE), (INSUFFICIENT, ESCALATED), (INSUFFICIENT, CLEARED), (CLEARED, NO_ALERT)}
    got = {(a, b) for a, b in itertools.product(STATES, STATES) if transition_allowed(a, b)}
    assert got == expect and set(ALLOWED) == set(STATES)
    # an insufficient-evidence pair may only resume an alert if it has one, and may only start pending if it has none
    assert not transition_allowed(INSUFFICIENT, ACTIVE, has_alert=False) and not transition_allowed(INSUFFICIENT, PENDING, has_alert=True)
    assert transition_allowed(INSUFFICIENT, CLEARED, has_alert=True) and not transition_allowed(INSUFFICIENT, CLEARED, has_alert=False)


def test_invalid_transitions_raise():
    m, _ = run(["HIGH", "HIGH", "HIGH"])
    ps = next(iter(m.pairs.values()))
    assert ps.state == ACTIVE
    for bad in (NO_ALERT, PENDING, "ACTIVE_WARNING"):
        with pytest.raises(InvalidTransition):
            m._move(ps, bad, 99, None, "x", [])
    ps.state = CLEARED
    with pytest.raises(InvalidTransition):
        m._move(ps, ACTIVE, 99, None, "x", [])


def test_full_lifecycle_with_persistence_escalation_deescalation_clearance():
    cats = ["LOW", "HIGH", "HIGH", "HIGH", "CRITICAL", "CRITICAL", "MODERATE", "MODERATE", "LOW", "LOW", "LOW", "LOW"]
    m, st = run(cats)
    assert st == [NO_ALERT, PENDING, PENDING, ACTIVE, ACTIVE, ESCALATED, ESCALATED, ACTIVE, ACTIVE, ACTIVE, CLEARED, NO_ALERT]
    t = trans(m)
    assert [(a, b) for _, a, b, _ in t] == [(NO_ALERT, PENDING), (PENDING, ACTIVE), (ACTIVE, ESCALATED), (ESCALATED, ACTIVE), (ACTIVE, CLEARED), (CLEARED, NO_ALERT)]
    assert t[1][3] == "PERSISTENCE_SATISFIED" and t[2][3] == "ESCALATION_PERSISTENCE_SATISFIED" and t[3][3] == "DEESCALATION_PERSISTENCE_SATISFIED"
    assert t[4][3] == "CLEAR_PERSISTENCE_SATISFIED"
    a = m.alert_table()[0]
    assert a["first_detection_frame"] == 3 and a["activated_frame"] == 9 and a["clearance_frame"] == 30 and a["clearance_kind"] == "RISK_SUBSIDED"
    assert a["peak_state"] == ESCALATED and a["peak_risk_category"] == "CRITICAL"
    assert a["first_detection_time_s"] == pytest.approx(3 / 30.0) and a["clearance_time_s"] == pytest.approx(1.0)


def test_persistence_required_and_pending_resets():
    _, st = run(["HIGH", "HIGH", "LOW", "HIGH", "HIGH", "HIGH"])
    assert st == [PENDING, PENDING, NO_ALERT, PENDING, PENDING, ACTIVE]
    _, st = run(["CRITICAL", "LOW", "CRITICAL", "LOW"])                                      # abrupt single spikes never alert
    assert ACTIVE not in st and ESCALATED not in st


def test_hysteresis_no_oscillation():
    # MODERATE neither enters nor clears: alternating HIGH / MODERATE never reaches ACTIVE...
    _, st = run(["HIGH", "MODERATE"] * 6)
    assert ACTIVE not in st
    # ...and once ACTIVE, MODERATE holds the alert while LOW does not
    m, st = run(["HIGH", "HIGH", "HIGH"] + ["MODERATE", "HIGH"] * 5)
    assert set(st[2:]) == {ACTIVE} and len(m.alert_table()) == 1 and not trans(m)[2:]
    # a single LOW between MODERATE observations does not clear (needs 3 consecutive)
    _, st = run(["HIGH"] * 3 + ["LOW", "LOW", "MODERATE", "LOW", "LOW"])
    assert st[-1] == ACTIVE


def test_escalation_needs_persistence_and_deescalation_hysteresis():
    _, st = run(["HIGH"] * 3 + ["CRITICAL", "HIGH", "CRITICAL", "HIGH"])
    assert ESCALATED not in st                                                                # never two consecutive CRITICAL
    # ESCALATED holds while HIGH (escalated_hold) and only de-escalates after 2 consecutive < HIGH
    _, st = run(["HIGH"] * 3 + ["CRITICAL"] * 2 + ["HIGH", "MODERATE", "HIGH", "MODERATE", "MODERATE"])
    assert st[4:] == [ESCALATED, ESCALATED, ESCALATED, ESCALATED, ESCALATED, ACTIVE]


def test_escalated_alert_deescalates_then_clears_on_sustained_low_risk():
    m, st = run(["HIGH"] * 3 + ["CRITICAL"] * 2 + ["LOW"] * 3)
    # deescalation (2 consecutive < HIGH) fires before clearance (3 consecutive < MODERATE); the clear streak is kept across the de-escalation
    assert st[4:] == [ESCALATED, ESCALATED, ACTIVE, CLEARED]
    assert [(a, b) for _, a, b, _ in trans(m)][-2:] == [(ESCALATED, ACTIVE), (ACTIVE, CLEARED)]


def test_deduplication_single_alert_per_pair():
    m, _ = run(["HIGH"] * 12)
    assert len(m.alert_table()) == 1
    creations = [e for e in m.events if e["to_state"] == ACTIVE and e["event_type"] == "STATE_TRANSITION"]
    assert len(creations) == 1 and len(m.events) == 2                                         # NO_ALERT->PENDING, PENDING->ACTIVE only
    # identical repeated frame is ignored; a different repeat is an error
    m = AlertManager(CFG)
    m.step(0, [ob(0, "HIGH")])
    assert m.step(0, [ob(0, "HIGH")]) == [] and m.stats["duplicate_batches_ignored"] == 1
    with pytest.raises(TimestampOrderError):
        m.step(0, [ob(0, "LOW")])
    with pytest.raises(AlertContractError):
        AlertManager(CFG).step(0, [ob(0, "HIGH"), ob(0, "LOW")])                              # same pair twice in a frame


def test_cooldown_suppresses_reentry_then_allows_it():
    m, _ = run(["HIGH"] * 3 + ["LOW"] * 3 + ["LOW"] + ["HIGH"] * 3)                              # cleared at frame 15, cooldown until 45; HIGH at frames 21..27
    assert len(m.alert_table()) == 1 and m.stats["suppressed_by_cooldown"] >= 3
    m, _ = run(["HIGH"] * 3 + ["LOW"] * 4 + ["HIGH"] * 3 + ["HIGH"] * 12)
    assert len(m.alert_table()) == 2
    a1, a2 = m.alert_table()
    assert a1["alert_id"] != a2["alert_id"] and a2["activated_frame"] - a1["clearance_frame"] >= CFG.cooldown_frames
    zero = AlertConfig.from_dict({**CFG.to_dict(), "cooldown_frames": 0})
    m, _ = run(["HIGH"] * 3 + ["LOW"] * 4 + ["HIGH"] * 3, cfg=zero)
    assert len(m.alert_table()) == 2


# ------------------------------------------------------------------ invalid / missing evidence
def test_invalid_observation_is_never_low_and_never_carried_forward():
    m, st = run(["HIGH"] * 3 + ["LOW", "LOW", "INSUFFICIENT_EVIDENCE", "LOW", "LOW"])
    assert st[5] == INSUFFICIENT and st[-1] == ACTIVE                                         # invalid observation did not complete the clear streak
    m, st = run(["HIGH"] * 3 + ["LOW", "LOW", "INSUFFICIENT_EVIDENCE", "LOW", "LOW", "LOW"])
    assert st[-1] == CLEARED
    ev = [e for e in m.events if e["to_state"] == INSUFFICIENT][0]
    assert ev["alert_id"] is not None and ev["warning_level"] == "UNDETERMINED_EVIDENCE_LOST" and ev["evidence_valid"] is False
    assert ev["phase7"]["risk_category"] == "INSUFFICIENT_EVIDENCE"


def test_evidence_loss_clears_with_distinct_kind_and_recovery_resumes():
    m, st = run(["HIGH"] * 3 + ["INSUFFICIENT_EVIDENCE"] * 6)
    assert st[-1] == CLEARED
    a = m.alert_table()[0]
    assert a["clearance_kind"] == "EVIDENCE_LOST" and a["evidence_lost_steps"] >= 5
    m, st = run(["HIGH"] * 3 + ["CRITICAL"] * 2 + ["INSUFFICIENT_EVIDENCE", "CRITICAL"])
    assert st[-2:] == [INSUFFICIENT, ESCALATED] and len(m.alert_table()) == 1               # same alert resumed, same identity
    _, st = run(["HIGH"] * 3 + ["CRITICAL"] * 2 + ["INSUFFICIENT_EVIDENCE", "MODERATE"])
    assert st[-1] == ACTIVE


def test_pre_alert_invalid_returns_to_no_alert_and_restarts_counting():
    _, st = run(["HIGH", "HIGH", "INSUFFICIENT_EVIDENCE", "HIGH", "HIGH", "HIGH"])
    assert st[2] == INSUFFICIENT and st[3] == PENDING and st[-1] == ACTIVE
    m, st = run(["INSUFFICIENT_EVIDENCE"] * 8 + ["LOW"])
    assert st[-1] == NO_ALERT and not m.alert_table()


def test_absent_pair_and_track_disappearance():
    m, st = run(["HIGH"] * 3 + [None, None])
    assert st[3] == INSUFFICIENT
    assert [e["trigger_reason"] for e in m.events if e["to_state"] == INSUFFICIENT] == ["PAIR_NOT_OBSERVED"]
    m, st = run(["HIGH"] * 3 + [None] * 8)
    assert st[-1] in (CLEARED, NO_ALERT) and m.alert_table()[0]["clearance_kind"] == "EVIDENCE_LOST"


def test_missing_frames_gap_breaks_the_sequence():
    m = AlertManager(CFG)
    for f in (0, 3, 6):
        m.step(f, [ob(f, "HIGH")])
    assert next(iter(m.pairs.values())).state == ACTIVE
    m.step(30, [ob(30, "HIGH")])                                                              # 24-frame gap > max_gap_frames
    reasons = [(e["from_state"], e["to_state"], e["trigger_reason"]) for e in m.events if e["frame"] == 30]
    assert reasons == [(ACTIVE, INSUFFICIENT, "MISSING_FRAMES"), (INSUFFICIENT, ACTIVE, "EVIDENCE_RESTORED")]
    m = AlertManager(CFG)
    for f in (0, 3):
        m.step(f, [ob(f, "HIGH")])
    m.step(30, [ob(30, "HIGH")])                                                              # pending sequence cannot span the gap
    assert next(iter(m.pairs.values())).state == PENDING and next(iter(m.pairs.values())).enter == 1


def test_track_identities_are_never_merged():
    m = AlertManager(CFG)
    for i in range(6):
        f = i * STEP
        obs = [ob(f, "HIGH", ped="pie:video_0004:1")] if i < 3 else [ob(f, "HIGH", ped="pie:video_0004:2")]
        m.step(f, obs)
    states = {k[1]: ps.state for k, ps in m.pairs.items()}
    assert states["pie:video_0004:1"] == INSUFFICIENT                                         # old identity lost evidence; it was not credited to the new one
    assert states["pie:video_0004:2"] == ACTIVE                                               # the new id earned ACTIVE through its own 3 observations (frames 9, 12, 15)
    alerts = {a["pedestrian_id"]: a for a in m.alert_table()}
    assert alerts["pie:video_0004:1"]["activated_frame"] == 6 and alerts["pie:video_0004:2"]["activated_frame"] == 15
    assert alerts["pie:video_0004:1"]["alert_id"] != alerts["pie:video_0004:2"]["alert_id"]
    assert alerts["pie:video_0004:1"]["final_state"] == INSUFFICIENT                          # not credited with, and not closed by, the new identity
    assert len({a["alert_id"] for a in m.alert_table()}) == len(m.alert_table())
    # the vehicle id is part of the identity as well
    m = AlertManager(CFG)
    for i in range(3):
        m.step(i * STEP, [ob(i * STEP, "HIGH", veh="7" if i < 2 else "8")])
    assert not m.alert_table()


def test_out_of_order_and_future_information_rejected():
    m = AlertManager(CFG)
    m.step(30, [ob(30, "LOW")])
    with pytest.raises(TimestampOrderError) as e:
        m.step(27, [ob(27, "LOW")])
    assert e.value.code == "OUT_OF_ORDER"
    with pytest.raises(TimestampOrderError) as e:
        m.step(33, [ob(36, "HIGH")])                                                          # an observation from the future in a past batch
    assert e.value.code == "FRAME_MISMATCH"
    with pytest.raises(AlertContractError):
        m.step(-1, [])


def test_decisions_do_not_depend_on_later_frames():
    cats = ["LOW", "HIGH", "HIGH", "HIGH", "CRITICAL", "CRITICAL", "MODERATE", "LOW", "LOW", "LOW", "HIGH", "HIGH", "HIGH"]
    full, _ = run(cats)
    for k in range(2, len(cats)):
        prefix, _ = run(cats[:k])
        assert full.events[:len(prefix.events)] == prefix.events                              # prefix property: later input never rewrites earlier output


def test_multiple_pairs_and_simultaneous_alerts_in_one_frame():
    m = AlertManager(CFG)
    for i in range(4):
        f = i * STEP
        m.step(f, [ob(f, "HIGH", ped="pie:video_0004:1", veh="7"), ob(f, "CRITICAL", ped="pie:video_0004:1", veh="8"), ob(f, "LOW", ped="pie:video_0004:2", veh="7"),
                   ob(f, "HIGH", ped="pie:video_0004:3", veh="7")])
    alerts = m.alert_table()
    assert len(alerts) == 3 and len({a["alert_id"] for a in alerts}) == 3
    same_frame = [e for e in m.events if e["to_state"] == ACTIVE and e["frame"] == 6]
    assert len(same_frame) == 3 and [e["dedup_key"] for e in same_frame] == sorted(e["dedup_key"] for e in same_frame)
    assert {a["dedup_key"] for a in alerts} == {"pie:video_0004|pie:video_0004:1|7", "pie:video_0004|pie:video_0004:1|8", "pie:video_0004|pie:video_0004:3|7"}
    assert not any(e["pedestrian_id"].endswith(":2") for e in m.events)                       # the LOW pair produced no events at all


def test_end_of_stream_closes_alerts_without_claiming_subsidence():
    m = AlertManager(CFG)
    for i in range(5):
        m.step(i * STEP, [ob(i * STEP, "HIGH")])
    m.step(15, [ob(15, "HIGH"), ob(15, "HIGH", ped="pie:video_0004:2")])                      # the alert continues; a second pair is pending at the end
    out = m.finalize()
    a = m.alert_table()[0]
    assert a["clearance_kind"] == "END_OF_STREAM" and a["clearance_frame"] == 15
    assert {(e["from_state"], e["to_state"], e["trigger_reason"]) for e in out} == {(ACTIVE, CLEARED, "END_OF_STREAM"), (PENDING, NO_ALERT, "END_OF_STREAM")}
    with pytest.raises(AlertContractError):
        m.step(18, [])


# ------------------------------------------------------------------ determinism, schema, provenance, semantics
def test_deterministic_replay_and_input_order_independence():
    cats = ["HIGH"] * 3 + ["CRITICAL"] * 2 + ["INSUFFICIENT_EVIDENCE", "LOW", "LOW", "LOW", "LOW"]
    a, _ = run(cats)
    b, _ = run(cats)
    assert a.events == b.events and a.alert_table() == b.alert_table()
    obs = [ob(f, c, ped="pie:video_0004:%d" % p) for f, c in zip(range(0, 15, 3), ["HIGH"] * 5) for p in (1, 2, 3)]
    m1, m2 = AlertManager(CFG), AlertManager(CFG)
    m1.process_stream(obs)
    m2.process_stream(list(reversed(obs)))
    assert m1.events == m2.events


def test_every_event_validates_and_schema_rejects_tampering():
    m, _ = run(["HIGH"] * 3 + ["CRITICAL"] * 2 + ["HIGH", "INSUFFICIENT_EVIDENCE", "LOW", "LOW", "LOW", "LOW", "LOW"])
    m.finalize()
    for e in m.events:
        validate_event(e)
    e = copy.deepcopy(m.events[-1])
    for mutate in (lambda x: x.update(extra=1), lambda x: x.update(to_state="MAYBE"), lambda x: x.pop("hashes"), lambda x: x["verified_event_time"].update(value=1.0),
                   lambda x: x.update(prototype_notice="validated"), lambda x: x.update(schema_version="x"), lambda x: x["severity"].update(value=3.0),
                   lambda x: x.update(clearance_kind="RISK_SUBSIDED", to_state="ACTIVE_WARNING", from_state="NO_ALERT")):
        bad = copy.deepcopy(e)
        mutate(bad)
        with pytest.raises(AlertContractError):
            validate_event(bad)


def test_provenance_hashes_and_unavailable_fields_are_explicit():
    m, _ = run(["HIGH"] * 4)
    e = m.events[-1]
    assert e["hashes"]["alert_config_sha256"] == CFG.sha256() and e["hashes"]["event_schema_sha256"] == schema_sha256()
    assert e["hashes"]["phase7_risk_config_sha256"] == "r" * 64 and e["hashes"]["source_checkpoint_file_sha256"] == "c" * 64 and e["hashes"]["calibration_artifact_sha256"] is None
    assert e["provenance"] == {"dataset": "unit"} and e["source"]["phase7_records_sha256"] == "s" * 64
    assert e["verified_event_time"] == {"status": "BLOCKED_INSUFFICIENT_GROUND_TRUTH", "value": None}
    assert e["predicted_conflict_time"]["value"] is None and e["severity"]["value"] is None
    assert e["prototype_notice"].startswith("RESEARCH PROTOTYPE") and e["pair_type"] == "pedestrian_vehicle"
    assert e["observation_time_s"] == pytest.approx(e["frame"] / 30.0) and e["warning_generation_time_s"] == e["observation_time_s"]


def test_heuristic_scores_are_never_turned_into_probabilities():
    m, _ = run(["HIGH"] * 4)
    assert all(e["phase7"]["calibrated_probability"] is None and e["phase7"]["calibration_status"] == "NOT_CALIBRATED" for e in m.events)
    # decisions ignore even a (hypothetically) calibrated probability: a LOW category with p=0.99 raises no alert
    m, _ = run(["LOW"] * 8, prob=0.99, status="CALIBRATED")
    assert not m.alert_table() and not m.events
    # and a calibrated probability, when upstream supplies one, is passed through untouched
    m, _ = run(["HIGH"] * 3, prob=0.37, status="CALIBRATED")
    assert m.events[-1]["phase7"]["calibrated_probability"] == 0.37 and "PROBABILITY_NOT_CALIBRATED" not in m.events[-1]["evidence_limitations"]
    assert "PROBABILITY_NOT_CALIBRATED" in run(["HIGH"] * 3)[0].events[-1]["evidence_limitations"]


def test_conflicting_indicator_policy():
    rec = {"pedestrian_id": "pie:video_0004:1", "vehicle_id": "7", "anchor_frame": 3, "window_id": 0, "conflict_score": 0.5, "conflict_score_kind": "K", "calibrated_probability": None,
           "calibration_status": "NOT_CALIBRATED", "temporal_urgency": 0.5, "severity": None, "severity_status": "S", "risk_score": 0.5, "risk_basis": "b", "risk_category": "HIGH",
           "category_status": "PROVISIONAL_ENGINEERING_CONFLICTING_INDICATORS", "reason": "R", "masks": {"conflict_valid": True, "category_assigned": True, "out_of_distribution": False}}
    flag = parse_record(rec)
    assert flag.valid and flag.conflicting_indicators
    strict = parse_record(rec, policy="treat_as_insufficient")
    assert not strict.valid and strict.invalid_reason == "CONFLICTING_INDICATORS_POLICY"
    m = AlertManager(CFG)
    m.step(3, [flag])
    assert "CONFLICTING_INDICATORS" in m.events[-1]["evidence_limitations"]


# ------------------------------------------------------------------ Phase 7 contract
def rec(**over):
    r = {"pedestrian_id": "pie:video_0004:1", "vehicle_id": "7", "anchor_frame": 3, "window_id": 0, "conflict_score": 0.5, "conflict_score_kind": "K", "calibrated_probability": None,
         "calibration_status": "NOT_CALIBRATED", "temporal_urgency": 0.5, "severity": None, "severity_status": "S", "risk_score": 0.5, "risk_basis": "b", "risk_category": "HIGH",
         "category_status": "PROVISIONAL_ENGINEERING", "reason": None, "masks": {"conflict_valid": True, "category_assigned": True, "out_of_distribution": False}}
    r.update(over)
    return r


def test_phase7_record_contract():
    o = parse_record(rec())
    assert o.valid and o.video_id == "pie:video_0004" and o.dedup_key == "pie:video_0004|pie:video_0004:1|7" and o.pair_key == ("pie:video_0004", "pie:video_0004:1", "7")
    ins = parse_record(rec(risk_category="INSUFFICIENT_EVIDENCE", risk_score=None, reason="OUT_OF_DISTRIBUTION", masks={"conflict_valid": True, "category_assigned": False, "out_of_distribution": True}))
    assert not ins.valid and ins.invalid_reason == "OUT_OF_DISTRIBUTION"
    inc = parse_record(rec(risk_score=None))
    assert not inc.valid and "INCONSISTENT" in inc.invalid_reason
    assert not parse_record(rec(risk_score=1.5)).valid
    for bad in (rec(risk_category="SEVERE"), {k: v for k, v in rec().items() if k != "masks"}, rec(anchor_frame=-1), rec(anchor_frame=True), rec(risk_score=float("nan")),
                rec(calibrated_probability=0.9), rec(calibrated_probability=0.9, calibration_status="CALIBRATED", risk_score=0.5) if False else rec(calibration_status="CALIBRATED"),
                rec(pedestrian_id="nocolon"), rec(vehicle_id=""), rec(masks={"conflict_valid": 1, "category_assigned": True, "out_of_distribution": False}), "not-a-dict"):
        with pytest.raises(InputContractError):
            parse_record(bad)
    ok = parse_record(rec(calibrated_probability=0.3, calibration_status="CALIBRATED"))
    assert ok.calibrated_probability == 0.3


def test_config_validation():
    d = CFG.to_dict()
    for k, v in (("config_status", "VALIDATED"), ("enter_persistence", 1), ("hold_min_category", "CRITICAL"), ("cooldown_frames", -1), ("source_fps", 0), ("clear_persistence", 0),
                 ("enter_min_category", "SEVERE"), ("conflicting_indicators_policy", "x"), ("escalated_hold_min_category", "LOW")):
        with pytest.raises(AlertContractError):
            AlertConfig.from_dict({**d, k: v})
    with pytest.raises(AlertContractError):
        AlertConfig.from_dict({**d, "extra": 1})
    assert CFG.config_status == "ENGINEERING_CONFIGURATION_NOT_VALIDATED"


def test_lead_time_is_blocked_without_verified_events():
    assert blocked_status()["status"] == "BLOCKED_INSUFFICIENT_GROUND_TRUTH"
    m, _ = run(["HIGH"] * 5)
    src = {"name": "n", "type": "official_dataset_annotation", "version": "1", "sha256": "0" * 64}
    ev = [{"verified": True, "event_frame": 30, "dedup_key": "pie:video_0004|pie:video_0004:1|7"}]
    with pytest.raises(LeadTimeBlocked):
        compute_lead_times(m.alert_table(), [], src, 30.0)
    with pytest.raises(LeadTimeBlocked):
        compute_lead_times(m.alert_table(), [{**ev[0], "verified": False}], src, 30.0)
    with pytest.raises(LeadTimeBlocked):
        compute_lead_times(m.alert_table(), ev, {**src, "type": "perceived_tracks"}, 30.0)
    r = compute_lead_times(m.alert_table(), ev, src, 30.0)
    assert r["rows"][0]["lead_time_s"] == pytest.approx((30 - 6) / 30.0)


# ------------------------------------------------------------------ integration on actual Phase 7 artifacts
W = REPO / "experiments/phase5_task3_windows_pie_video_0004/20261115T035825Z_pie_video_0004/primary"
N = REPO / "experiments/phase5_task7_neural_evaluation/20261115T191319Z_pie_video_0004"
P7 = REPO / "experiments/phase7_risk_pie_video_0004"
real = pytest.mark.skipif(not (W.exists() and N.exists() and P7.exists()), reason="real Phase 5/7 artifacts not present")


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    from pedsafety.phase8.__main__ import main
    out = tmp_path_factory.mktemp("p8")
    argv = ["from-phase7", "--windows-run", str(W), "--neural-eval-dir", str(N), "--phase7-dir", str(P7), "--conflict-config", str(REPO / "configs/phase6/conflict_default.json"),
            "--risk-config", str(REPO / "configs/phase7/risk_default.json"), "--window-config", str(REPO / "configs/phase5/window_generation_default.json"),
            "--alert-config", str(REPO / "configs/phase8/alert_default.json"), "--out", str(out)]
    assert main(argv) == 0
    return out


@real
def test_real_cli_outputs_and_phase7_cross_check(real_run):
    man = json.loads((real_run / "run_manifest.json").read_text())
    assert all(man["phase7_artifact_cross_check"]["fields_equal"].values()) and man["phase7_artifact_cross_check"]["n"] == 8233
    assert man["hashes"]["source_checkpoint_file_sha256"] == "40aff17f1a40eb2da24d0e0b63245eecefa5bba4e3f6e44c272dc07f3ab71b44"
    assert man["hashes"]["calibration_artifact_sha256"] is None and man["alert_config"]["sha256"] == CFG.sha256()
    assert "NOT an end-to-end latency" in man["component_throughput"]["caveat"]
    s = json.loads((real_run / "summary_report.json").read_text())
    assert s["n_records"] == 8233 and s["lead_time"]["status"] == "BLOCKED_INSUFFICIENT_GROUND_TRUTH" and not any(s["claims"].values())


@real
def test_real_event_log_lifecycle_invariants(real_run):
    with gzip.open(real_run / "alert_events.jsonl.gz", "rt", encoding="utf-8") as f:
        events = [json.loads(x) for x in f]
    alerts = json.loads((real_run / "alerts.json").read_text())["alerts"]
    assert [e["event_seq"] for e in events] == list(range(len(events)))
    for e in events:
        validate_event(e)
        assert e["phase7"]["calibrated_probability"] is None and e["hashes"]["source_checkpoint_file_sha256"].startswith("40aff17f")
        assert e["alert_id"] is None or e["alert_id"] in {a["alert_id"] for a in alerts}
    by_alert = {}
    for e in events:
        if e["alert_id"] and e["event_type"] == "STATE_TRANSITION":
            by_alert.setdefault(e["alert_id"], []).append(e)
    assert len(by_alert) == len(alerts) > 0
    open_by_key = {}
    for alert_id, evs in by_alert.items():
        states = [e["to_state"] for e in evs]
        assert evs[0]["to_state"] == ACTIVE and evs[0]["trigger_reason"] == "PERSISTENCE_SATISFIED"
        assert states.count(CLEARED) == 1 and states[-1] == NO_ALERT and states[-2] == CLEARED          # complete lifecycle, closed exactly once
        a = next(x for x in alerts if x["alert_id"] == alert_id)
        assert a["clearance_frame"] is not None and a["clearance_kind"] in ("RISK_SUBSIDED", "EVIDENCE_LOST", "END_OF_STREAM")
        assert a["activated_frame"] <= a["clearance_frame"] and a["first_detection_frame"] <= a["activated_frame"]
        open_by_key.setdefault(a["dedup_key"], []).append((a["activated_frame"], a["clearance_frame"]))
    for spans in open_by_key.values():                                                                  # deduplication: no two alerts of one pair overlap
        spans.sort()
        assert all(b[0] > a[1] for a, b in zip(spans, spans[1:]))


@real
def test_real_replay_is_deterministic_and_records_input_reproduces_events(real_run, tmp_path):
    from pedsafety.phase8.runner import read_records, run_alerts
    recs = read_records(real_run / "phase7_records_input.jsonl.gz")
    r1, r2 = run_alerts(recs, CFG), run_alerts(recs, CFG)
    assert r1["events_sha256"] == r2["events_sha256"] and r1["alerts"] == r2["alerts"]
    key = lambda ev: [(e["frame"], e["dedup_key"], e["from_state"], e["to_state"], e["trigger_reason"]) for e in ev]            # noqa: E731
    with gzip.open(real_run / "alert_events.jsonl.gz", "rt", encoding="utf-8") as f:
        saved = [json.loads(x) for x in f]
    assert key(r1["events"]) == key(saved)
    # no alert can come from a record that Phase 7 marked LOW / MODERATE / INSUFFICIENT only
    hot = {(r["pedestrian_id"], r["vehicle_id"]) for r in recs if r["risk_category"] in ("HIGH", "CRITICAL")}
    assert {(a["pedestrian_id"], a["vehicle_id"]) for a in r1["alerts"]} <= hot
