"""Deterministic warning state machine and alert manager (pure python, no wall clock, no randomness, no I/O).

One state machine per pedestrian-vehicle pair, keyed by ``video|pedestrian_id|vehicle_id``. Track identities are never merged: a new id is a new pair.
``step(frame, observations)`` consumes ONE anchor frame of Phase 7 observations and uses only that frame plus the machine's own past; a batch
whose observations belong to another frame is rejected, so a future observation can never influence an earlier decision.

Transitions (every other transition raises ``InvalidTransition``; at most one transition per pair per step, except that a frame gap may first move an alert to
INSUFFICIENT_EVIDENCE and the same frame's valid observation then moves it on):
    NO_ALERT          -> WARNING_PENDING   valid obs, category >= enter, not in cooldown
                      -> INSUFFICIENT_EVIDENCE  invalid / missing observation
    WARNING_PENDING   -> ACTIVE_WARNING    ``enter_persistence`` consecutive observations >= enter (alert created, id assigned)
                      -> NO_ALERT          category < enter before persistence is met (counter reset)
                      -> INSUFFICIENT_EVIDENCE
    ACTIVE_WARNING    -> ESCALATED_WARNING ``escalate_persistence`` consecutive observations >= escalate
                      -> CLEARED           ``clear_persistence`` consecutive valid observations < hold (hysteresis: hold < enter)
                      -> INSUFFICIENT_EVIDENCE
    ESCALATED_WARNING -> ACTIVE_WARNING    ``deescalate_persistence`` consecutive observations < escalated_hold
                      -> CLEARED           as above
                      -> INSUFFICIENT_EVIDENCE
    INSUFFICIENT_EVIDENCE -> (with alert) ACTIVE_WARNING | ESCALATED_WARNING  valid observation, restored to the pre-loss level allowed by the observation
                      -> (with alert) CLEARED  after ``max_insufficient_observations`` invalid steps (clearance_kind EVIDENCE_LOST, NOT "risk subsided")
                      -> (no alert)   WARNING_PENDING | NO_ALERT
    CLEARED           -> NO_ALERT          at the next observation or absence (consumes the step); the cooldown suppresses re-entry
Invalid observations are never treated as LOW and never carried forward; counters restart after them.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import CATEGORY_RANK, AlertConfig
from .errors import AlertContractError, InvalidTransition, TimestampOrderError
from .phase7_input import Observation
from .schema import EVENT_SCHEMA_VERSION, PROTOTYPE_NOTICE, schema_sha256, validate_event

NO_ALERT, PENDING, ACTIVE, ESCALATED, CLEARED, INSUFFICIENT = ("NO_ALERT", "WARNING_PENDING", "ACTIVE_WARNING", "ESCALATED_WARNING", "CLEARED", "INSUFFICIENT_EVIDENCE")

ALLOWED = {
    NO_ALERT: {PENDING, INSUFFICIENT},
    PENDING: {ACTIVE, NO_ALERT, INSUFFICIENT},
    ACTIVE: {ESCALATED, CLEARED, INSUFFICIENT},
    ESCALATED: {ACTIVE, CLEARED, INSUFFICIENT},
    INSUFFICIENT: {NO_ALERT, PENDING, ACTIVE, ESCALATED, CLEARED},
    CLEARED: {NO_ALERT},
}
WARNING_LEVEL = {NO_ALERT: "NONE", PENDING: "NONE", ACTIVE: "ADVISORY", ESCALATED: "ELEVATED", CLEARED: "NONE", INSUFFICIENT: "NONE"}


def transition_allowed(a: str, b: str, has_alert: Optional[bool] = None) -> bool:
    if b not in ALLOWED.get(a, ()):
        return False
    if a == INSUFFICIENT and has_alert is not None:
        return (b in (ACTIVE, ESCALATED, CLEARED)) if has_alert else (b in (NO_ALERT, PENDING))
    return True


@dataclass
class PairState:
    key: Tuple[str, str, str]
    state: str = NO_ALERT
    alert_id: Optional[str] = None
    enter: int = 0
    esc: int = 0
    deesc: int = 0
    clear: int = 0
    insuff: int = 0
    resume: Optional[str] = None
    first_detection_frame: Optional[int] = None
    activated_frame: Optional[int] = None
    last_update_frame: Optional[int] = None
    cooldown_until: int = -1
    last_category: Optional[str] = None
    last_conflicting: bool = False

    def reset_counters(self):
        self.enter = self.esc = self.deesc = self.clear = 0


class AlertManager:
    def __init__(self, cfg: AlertConfig, provenance: Optional[Mapping[str, Any]] = None, *, source_records_sha256: Optional[str] = None,
                 hashes: Optional[Mapping[str, Optional[str]]] = None, validate_events: bool = True):
        self.cfg = cfg
        self.ranks = cfg.ranks()
        self.provenance = dict(provenance or {})
        self.source_sha = source_records_sha256
        h = dict(hashes or {})
        self.hashes = {"alert_config_sha256": cfg.sha256(), "event_schema_sha256": schema_sha256(), "phase7_risk_config_sha256": h.get("phase7_risk_config_sha256"),
                       "source_checkpoint_file_sha256": h.get("source_checkpoint_file_sha256"), "calibration_artifact_sha256": h.get("calibration_artifact_sha256")}
        self.validate_events = validate_events
        self.pairs: Dict[Tuple[str, str, str], PairState] = {}
        self.alerts: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self.last_frame: Optional[int] = None
        self._last_fp: Optional[str] = None
        self.closed = False
        self.stats = {"steps": 0, "observations": 0, "valid_observations": 0, "invalid_observations": 0, "absent_pair_steps": 0, "duplicate_batches_ignored": 0,
                      "gap_events": 0, "suppressed_by_cooldown": 0}
        self._seq = 0

    # ------------------------------------------------------------------ public
    def step(self, frame: int, observations: Sequence[Observation]) -> List[Dict[str, Any]]:
        if self.closed:
            raise AlertContractError("CLOSED", "the manager was finalized; create a new one")
        if not (isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0):
            raise AlertContractError("FRAME", "frame must be a non-negative integer")
        seen = {}
        for o in observations:
            if o.frame != frame:
                raise TimestampOrderError("FRAME_MISMATCH", "observation for frame %d submitted in the batch of frame %d (future or stale information is rejected)" % (o.frame, frame))
            if o.pair_key in seen:
                raise AlertContractError("DUPLICATE_PAIR_IN_FRAME", "pair %s appears twice at frame %d" % (o.dedup_key, frame))
            seen[o.pair_key] = o
        fp = hashlib.sha256(json.dumps(sorted([o.pair_key, o.risk_category, o.risk_score, o.valid] for o in observations), sort_keys=True).encode()).hexdigest()
        if self.last_frame is not None:
            if frame < self.last_frame:
                raise TimestampOrderError("OUT_OF_ORDER", "frame %d arrives after frame %d" % (frame, self.last_frame))
            if frame == self.last_frame:
                if fp != self._last_fp:
                    raise TimestampOrderError("CONFLICTING_REPEAT", "frame %d was repeated with different content" % frame)
                self.stats["duplicate_batches_ignored"] += 1
                return []
        out: List[Dict[str, Any]] = []
        self.stats["steps"] += 1
        self.stats["observations"] += len(observations)
        for k in [k for k, ps in self.pairs.items() if ps.state == NO_ALERT and ps.cooldown_until <= frame and not (ps.enter or ps.esc or ps.deesc or ps.clear)]:
            del self.pairs[k]
        gap = self.last_frame is not None and frame - self.last_frame > self.cfg.max_gap_frames
        if gap:
            self.stats["gap_events"] += 1
        for key in sorted(self.pairs):
            ps = self.pairs[key]
            if gap and ps.state in (PENDING, ACTIVE, ESCALATED):
                self._invalid(ps, frame, None, "MISSING_FRAMES", out)
            elif gap and ps.state == INSUFFICIENT:
                ps.reset_counters()
        for key in sorted(set(seen) | set(self.pairs)):
            ps = self.pairs.get(key)
            obs = seen.get(key)
            if ps is None:
                ps = self.pairs[key] = PairState(key)
            if obs is None and ps.state == NO_ALERT:                                                       # nothing is open for an unobserved idle pair
                continue
            if obs is None:
                self.stats["absent_pair_steps"] += 1
                self._invalid(ps, frame, None, "PAIR_NOT_OBSERVED", out)
            elif obs.valid:
                self.stats["valid_observations"] += 1
                self._valid(ps, frame, obs, out)
            else:
                self.stats["invalid_observations"] += 1
                self._invalid(ps, frame, obs, "INVALID_OBSERVATION:" + str(obs.invalid_reason), out)
        self.last_frame, self._last_fp = frame, fp
        return out

    def finalize(self) -> List[Dict[str, Any]]:
        """End of stream: open alerts are closed with clearance_kind END_OF_STREAM (not 'risk subsided'); unconfirmed pairs return to NO_ALERT."""
        out: List[Dict[str, Any]] = []
        frame = self.last_frame if self.last_frame is not None else 0
        for key in sorted(self.pairs):
            ps = self.pairs[key]
            if ps.alert_id and ps.state in (ACTIVE, ESCALATED, INSUFFICIENT):
                self._move(ps, CLEARED, frame, None, "END_OF_STREAM", out, clearance_kind="END_OF_STREAM")
            elif ps.state in (PENDING, INSUFFICIENT):
                self._move(ps, NO_ALERT, frame, None, "END_OF_STREAM", out)
        self.closed = True
        return out

    def process_stream(self, observations: Iterable[Observation]) -> List[Dict[str, Any]]:
        """Replay a recorded stream: observations are grouped by frame and presented in ascending frame order, one batch at a time (a replay scheduler,
        not a lookahead: each batch is handed over only after all earlier ones were processed)."""
        by_frame: Dict[int, List[Observation]] = {}
        for o in observations:
            by_frame.setdefault(o.frame, []).append(o)
        out: List[Dict[str, Any]] = []
        for f in sorted(by_frame):
            out.extend(self.step(f, by_frame[f]))
        out.extend(self.finalize())
        return out

    def alert_table(self) -> List[Dict[str, Any]]:
        return [self.alerts[k] for k in sorted(self.alerts, key=lambda a: (self.alerts[a]["first_detection_frame"], a))]

    # ------------------------------------------------------------------ internals
    def _t(self, frame: Optional[int]) -> Optional[float]:
        return None if frame is None else frame / self.cfg.source_fps

    def _move(self, ps: PairState, to: str, frame: int, obs: Optional[Observation], reason: str, out: List, clearance_kind: Optional[str] = None):
        if not transition_allowed(ps.state, to, ps.alert_id is not None):
            raise InvalidTransition("INVALID_TRANSITION", "%s -> %s (alert=%s) is not allowed" % (ps.state, to, ps.alert_id))
        frm = ps.state
        if to == ACTIVE and ps.alert_id is None:
            ps.alert_id = "alert-" + hashlib.sha256(("%s|%d|%s" % ("|".join(ps.key), frame, self.hashes["alert_config_sha256"])).encode()).hexdigest()[:16]
            ps.activated_frame = frame
            self.alerts[ps.alert_id] = {"alert_id": ps.alert_id, "dedup_key": "|".join(ps.key), "video_id": ps.key[0], "pedestrian_id": ps.key[1], "vehicle_id": ps.key[2],
                                        "first_detection_frame": ps.first_detection_frame, "first_detection_time_s": self._t(ps.first_detection_frame),
                                        "activated_frame": frame, "activated_time_s": self._t(frame), "last_update_frame": frame, "last_update_time_s": self._t(frame),
                                        "clearance_frame": None, "clearance_time_s": None, "clearance_kind": None, "peak_state": ACTIVE, "peak_risk_category": None,
                                        "peak_risk_score": None, "n_events": 0, "evidence_lost_steps": 0, "final_state": ACTIVE}
        ps.state = to
        if to == NO_ALERT or to == PENDING:
            ps.resume = None
        if to == CLEARED:
            ps.cooldown_until = frame + self.cfg.cooldown_frames
        self._emit(ps, "STATE_TRANSITION", frm, to, frame, obs, reason, out, clearance_kind)
        if to == NO_ALERT:                                                                                 # the alert is closed: a later alert gets a new identity
            ps.alert_id = None
            ps.activated_frame = None

    def _emit(self, ps, etype, frm, to, frame, obs, reason, out, clearance_kind=None):
        a = self.alerts.get(ps.alert_id) if ps.alert_id else None
        if a is not None:
            a["last_update_frame"], a["last_update_time_s"], a["final_state"] = frame, self._t(frame), to
            a["n_events"] += 1
            if to == ESCALATED:
                a["peak_state"] = ESCALATED
            if to == CLEARED:
                a["clearance_frame"], a["clearance_time_s"], a["clearance_kind"] = frame, self._t(frame), clearance_kind
        limits = []
        if obs is None:
            limits.append("NO_OBSERVATION_AT_THIS_FRAME")
        elif not obs.valid:
            limits.append("INVALID_OBSERVATION:" + str(obs.invalid_reason))
        if obs is not None and obs.conflicting_indicators:
            limits.append("CONFLICTING_INDICATORS")
        if obs is not None and obs.calibration_status != "CALIBRATED":
            limits.append("PROBABILITY_NOT_CALIBRATED")
        limits.append("IMAGE_SPACE_HEURISTIC_NO_PHYSICAL_GEOMETRY")
        level = "UNDETERMINED_EVIDENCE_LOST" if (to == INSUFFICIENT and ps.alert_id) else WARNING_LEVEL[to]
        p7 = {"risk_score": None, "risk_category": None, "category_status": None, "conflict_score": None, "conflict_score_kind": None, "temporal_urgency": None,
              "calibrated_probability": None, "calibration_status": None, "masks": None, "reason": None, "conflicting_indicators": None}
        if obs is not None:
            p7.update({"risk_score": obs.risk_score, "risk_category": obs.risk_category, "category_status": obs.category_status, "conflict_score": obs.conflict_score,
                       "conflict_score_kind": obs.conflict_score_kind, "temporal_urgency": obs.temporal_urgency, "calibrated_probability": obs.calibrated_probability,
                       "calibration_status": obs.calibration_status, "masks": dict(obs.masks), "reason": obs.reason, "conflicting_indicators": obs.conflicting_indicators})
        eid = hashlib.sha256(("%s|%d|%s|%s|%s" % ("|".join(ps.key), self._seq, frm, to, reason)).encode()).hexdigest()[:20]
        ev = {"schema_version": EVENT_SCHEMA_VERSION, "event_id": eid, "event_seq": self._seq, "event_type": etype, "alert_id": ps.alert_id, "dedup_key": "|".join(ps.key),
              "video_id": ps.key[0], "pedestrian_id": ps.key[1], "vehicle_id": ps.key[2], "pair_type": "pedestrian_vehicle", "frame": frame,
              "observation_time_s": self._t(frame), "warning_generation_time_s": self._t(frame), "from_state": frm, "to_state": to, "warning_level": level,
              "warning_level_status": "PROVISIONAL_ENGINEERING_NOT_PHYSICAL_SEVERITY", "trigger_reason": reason, "phase7": p7,
              "evidence_valid": bool(obs is not None and obs.valid), "evidence_limitations": limits, "first_detection_frame": ps.first_detection_frame,
              "first_detection_time_s": self._t(ps.first_detection_frame), "last_update_frame": ps.last_update_frame, "last_update_time_s": self._t(ps.last_update_frame),
              "clearance_frame": frame if to == CLEARED else None, "clearance_time_s": self._t(frame) if to == CLEARED else None,
              "clearance_kind": clearance_kind if to == CLEARED else None,
              "predicted_conflict_time": {"status": "UNAVAILABLE_NOT_EXPOSED_BY_PHASE7_AND_NOT_PHYSICAL", "value": None},
              "verified_event_time": {"status": "BLOCKED_INSUFFICIENT_GROUND_TRUTH", "value": None},
              "severity": {"status": "NOT_SUPPORTED_NO_METRIC_GEOMETRY", "value": None},
              "source": {"phase7_record_index": obs.source_index if obs is not None else None, "phase7_records_sha256": self.source_sha},
              "hashes": dict(self.hashes), "provenance": self.provenance, "prototype_notice": PROTOTYPE_NOTICE}
        self._seq += 1
        if self.validate_events:
            validate_event(ev)
        self.events.append(ev)
        out.append(ev)

    def _touch(self, ps, frame, obs):
        ps.last_update_frame = frame
        a = self.alerts.get(ps.alert_id) if ps.alert_id else None
        if a is not None:
            a["last_update_frame"], a["last_update_time_s"] = frame, self._t(frame)
            if obs is not None and obs.valid:
                if a["peak_risk_score"] is None or obs.risk_score > a["peak_risk_score"]:
                    a["peak_risk_score"], a["peak_risk_category"] = obs.risk_score, obs.risk_category

    def _invalid(self, ps: PairState, frame: int, obs: Optional[Observation], reason: str, out: List):
        st = ps.state
        if st == CLEARED:
            ps.reset_counters()
            self._move(ps, NO_ALERT, frame, obs, "CLEARED_ACKNOWLEDGED_NEXT_OBSERVATION", out)
        elif st in (NO_ALERT, PENDING):
            ps.reset_counters()
            self._move(ps, INSUFFICIENT, frame, obs, reason, out)
            ps.insuff = 1
            ps.first_detection_frame = None
        elif st in (ACTIVE, ESCALATED):
            ps.resume = st
            ps.reset_counters()
            self._move(ps, INSUFFICIENT, frame, obs, reason, out)
            ps.insuff = 1
            self.alerts[ps.alert_id]["evidence_lost_steps"] += 1
        else:                                                                                              # already INSUFFICIENT
            ps.insuff += 1
            if ps.alert_id:
                self.alerts[ps.alert_id]["evidence_lost_steps"] += 1
            if ps.insuff > self.cfg.max_insufficient_observations:
                if ps.alert_id:
                    self._move(ps, CLEARED, frame, obs, "EVIDENCE_LOST", out, clearance_kind="EVIDENCE_LOST")
                else:
                    self._move(ps, NO_ALERT, frame, obs, "EVIDENCE_NOT_RECOVERED", out)

    def _valid(self, ps: PairState, frame: int, obs: Observation, out: List):
        r, rank, c = self.ranks, CATEGORY_RANK[obs.risk_category], self.cfg
        st = ps.state
        if st == CLEARED:
            ps.reset_counters()
            self._move(ps, NO_ALERT, frame, obs, "CLEARED_ACKNOWLEDGED_NEXT_OBSERVATION", out)
            return
        if st == NO_ALERT:
            if rank >= r["enter"]:
                if frame < ps.cooldown_until:
                    self.stats["suppressed_by_cooldown"] += 1
                    return
                ps.enter, ps.first_detection_frame = 1, frame
                self._touch(ps, frame, obs)
                self._move(ps, PENDING, frame, obs, "ENTER_THRESHOLD_MET", out)
            return
        if st == PENDING:
            if rank >= r["enter"]:
                ps.enter += 1
                self._touch(ps, frame, obs)
                if ps.enter >= c.enter_persistence:
                    ps.reset_counters()
                    self._move(ps, ACTIVE, frame, obs, "PERSISTENCE_SATISFIED", out)
                    self._touch(ps, frame, obs)
                    ps.last_category, ps.last_conflicting = obs.risk_category, obs.conflicting_indicators
            else:
                ps.reset_counters()
                ps.first_detection_frame = None
                self._move(ps, NO_ALERT, frame, obs, "RISK_BELOW_ENTER_BEFORE_PERSISTENCE", out)
            return
        if st == INSUFFICIENT:
            ps.insuff = 0
            if not ps.alert_id:
                ps.reset_counters()
                if rank >= r["enter"] and frame >= ps.cooldown_until:
                    ps.enter, ps.first_detection_frame = 1, frame
                    self._move(ps, PENDING, frame, obs, "EVIDENCE_RESTORED_ENTER_THRESHOLD_MET", out)
                else:
                    self._move(ps, NO_ALERT, frame, obs, "EVIDENCE_RESTORED", out)
                return
            target = ESCALATED if (ps.resume == ESCALATED and rank >= r["escalated_hold"]) else ACTIVE
            ps.reset_counters()
            ps.clear = 1 if rank < r["hold"] else 0
            self._move(ps, target, frame, obs, "EVIDENCE_RESTORED", out)
            self._touch(ps, frame, obs)
            ps.last_category, ps.last_conflicting = obs.risk_category, obs.conflicting_indicators
            return
        # ACTIVE / ESCALATED
        self._touch(ps, frame, obs)
        ps.clear = 0 if rank >= r["hold"] else ps.clear + 1
        if st == ACTIVE:
            ps.esc = ps.esc + 1 if rank >= r["escalate"] else 0
            if ps.clear >= c.clear_persistence:
                self._move(ps, CLEARED, frame, obs, "CLEAR_PERSISTENCE_SATISFIED", out, clearance_kind="RISK_SUBSIDED")
                return
            if ps.esc >= c.escalate_persistence:
                ps.esc = ps.deesc = 0
                self._move(ps, ESCALATED, frame, obs, "ESCALATION_PERSISTENCE_SATISFIED", out)
                ps.last_category, ps.last_conflicting = obs.risk_category, obs.conflicting_indicators
                return
        else:
            ps.deesc = ps.deesc + 1 if rank < r["escalated_hold"] else 0
            if ps.clear >= c.clear_persistence:
                self._move(ps, CLEARED, frame, obs, "CLEAR_PERSISTENCE_SATISFIED", out, clearance_kind="RISK_SUBSIDED")
                return
            if ps.deesc >= c.deescalate_persistence:
                ps.esc = ps.deesc = 0
                self._move(ps, ACTIVE, frame, obs, "DEESCALATION_PERSISTENCE_SATISFIED", out)
                ps.last_category, ps.last_conflicting = obs.risk_category, obs.conflicting_indicators
                return
        if obs.risk_category != ps.last_category or obs.conflicting_indicators != ps.last_conflicting:
            ps.last_category, ps.last_conflicting = obs.risk_category, obs.conflicting_indicators
            self._emit(ps, "ALERT_UPDATED", st, st, frame, obs, "CATEGORY_OR_EVIDENCE_FLAG_CHANGED", out)


__all__ = ["AlertManager", "ALLOWED", "transition_allowed", "PairState", "NO_ALERT", "PENDING", "ACTIVE", "ESCALATED", "CLEARED", "INSUFFICIENT"]
