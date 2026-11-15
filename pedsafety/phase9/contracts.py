"""Versioned stage contracts and validators of the Phase 3 -> Phase 8 pipeline (pure validation; no repair, no interpolation).

Every validator raises ``StageContractError`` with a stable code, or returns a statistics dict. Validators only read; they never modify an artifact.
Coordinates: the only authorised system is ``captured_1920x1080`` (image pixels, origin top-left, +v down); no stage may claim physical units.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .errors import StageContractError

CONTRACT_VERSION = "phase9-stage-contracts-1.0.0"
COORDINATE_SYSTEM = "captured_1920x1080"
STAGES = ("S3_PERCEPTION", "S4A_IMAGE_KINEMATICS", "S4B_INTERACTION", "S5A_WINDOWS", "S5B_FORECAST", "S6_CONFLICT", "S7_RISK", "S8_ALERTS", "S9_EVALUATION_GATES")

STAGE_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "S3_PERCEPTION": {"in": "video frames (not executable here) or precomputed tracked_agents.jsonl + agent_states.jsonl + run_manifest.json", "out": "tracked_agents.jsonl rows",
                      "checks": ["required fields", "coordinate_system", "finite ordered boxes", "frame/timestamp alignment", "unique (track, frame)", "single video id"]},
    "S4A_IMAGE_KINEMATICS": {"in": "agent_states.jsonl (Phase 3)", "out": "causal_kinematics.jsonl + run_manifest.json", "checks": ["input sha256 chain", "row count preserved", "no quarantine"]},
    "S4B_INTERACTION": {"in": "causal kinematics + Task 2 manifest + Phase 3 agent states", "out": "nodes / edges npz + run_manifest.json", "checks": ["manifest sha256 chain", "node count == causal rows"]},
    "S5A_WINDOWS": {"in": "Task 3 run", "out": "windows.npz + graph_bank.npz + run_manifest.json",
                    "checks": ["manifest sha256 chain", "history frames <= anchor < future frames", "timestamps == frame / fps", "identity preserved"]},
    "S5B_FORECAST": {"in": "windows (history inputs only) + verified checkpoint", "out": "predictions_neural.npz", "checks": ["window/track/anchor identity", "finite where valid", "mode probabilities sum to 1",
                                                                                                                       "checkpoint and window-run provenance verified", "future-target isolation"]},
    "S6_CONFLICT": {"in": "predictions + windows + graph bank", "out": "ConflictPairBatch", "checks": ["Phase 6 prediction contract", "history-only context motion", "mask propagation"]},
    "S7_RISK": {"in": "ConflictPairBatch + scores", "out": "phase7-risk-output-1.0.0 records", "checks": ["identity and frame provenance", "probabilities null unless calibrated", "INSUFFICIENT_EVIDENCE preserved"]},
    "S8_ALERTS": {"in": "Phase 7 records", "out": "phase8-alert-event-1.0.0 events + alert table", "checks": ["schema", "no alert from invalid evidence", "alerts only for HIGH/CRITICAL pairs", "timestamps == frame / fps"]},
    "S9_EVALUATION_GATES": {"in": "all stage outputs + independent-data manifests", "out": "evaluation status (computed or BLOCKED)", "checks": ["group-level frozen split", "verified labels", "no in-sample reuse"]},
}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(code: str, msg: str):
    raise StageContractError(code, msg)


# ---------------------------------------------------------------------------------------------------------------- Stage 3
_S3_REQUIRED = ("video_id", "frame_index", "timestamp_s", "track_id", "semantic_class", "bbox_xyxy", "image_width", "image_height", "coordinate_system", "tracking_validity")
SEMANTIC_CLASSES = ("pedestrian", "car", "truck", "bus", "bike")


def validate_phase3_rows(rows: Iterable[Mapping[str, Any]], fps: float, *, tol_s: float = 1e-3, max_violations: int = 20) -> Dict[str, Any]:
    """Validate Phase 3 tracked-agent rows; every violation is collected, none is repaired. Raises if any violation exists."""
    viol: List[str] = []
    seen = set()
    last_frame: Dict[Any, int] = {}
    videos, frames, tracks, n = set(), set(), set(), 0
    for i, r in enumerate(rows):
        n += 1
        miss = [k for k in _S3_REQUIRED if k not in r]
        if miss:
            viol.append("row %d missing %s" % (i, miss))
            continue
        f, t, b = r["frame_index"], r["track_id"], r["bbox_xyxy"]
        if not (isinstance(f, int) and not isinstance(f, bool) and f >= 0):
            viol.append("row %d bad frame_index %r" % (i, f))
            continue
        if r["coordinate_system"] != COORDINATE_SYSTEM:
            viol.append("row %d coordinate_system %r" % (i, r["coordinate_system"]))
        if r["semantic_class"] not in SEMANTIC_CLASSES:
            viol.append("row %d semantic_class %r" % (i, r["semantic_class"]))
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in b) and b[2] > b[0] and b[3] > b[1]):
            viol.append("row %d invalid box %r" % (i, b))
        elif not (0 <= b[0] and 0 <= b[1] and b[2] <= r["image_width"] + 1e-6 and b[3] <= r["image_height"] + 1e-6):
            viol.append("row %d box outside the %sx%s frame" % (i, r["image_width"], r["image_height"]))
        ts = r["timestamp_s"]
        if not (isinstance(ts, (int, float)) and math.isfinite(ts)) or abs(ts - f / fps) > tol_s:
            viol.append("row %d timestamp %r != frame/fps %.6f" % (i, ts, f / fps))
        if (t, f) in seen:
            viol.append("row %d duplicate (track %r, frame %d)" % (i, t, f))
        seen.add((t, f))
        if t in last_frame and f < last_frame[t]:
            viol.append("row %d track %r frame %d goes backwards (previous %d)" % (i, t, f, last_frame[t]))
        last_frame[t] = max(f, last_frame.get(t, f))
        videos.add(r["video_id"])
        frames.add(f)
        tracks.add(t)
        if len(viol) >= max_violations:
            break
    if len(videos) > 1:
        viol.append("rows of several videos: %s" % sorted(videos))
    if viol:
        fail("S3_CONTRACT", "%d violation(s), first: %s" % (len(viol), viol[:5]))
    return {"rows": n, "frames": len(frames), "tracks": len(tracks), "video_id": next(iter(videos)) if videos else None, "first_frame": min(frames) if frames else None,
            "last_frame": max(frames) if frames else None}


def validate_phase3_file(path, fps: float) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        from .errors import MissingInputError
        raise MissingInputError("S3_MISSING", "%s not found" % p)

    def gen():
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except ValueError:
                        fail("S3_CONTRACT", "line %d is not valid JSON" % ln)
    return validate_phase3_rows(gen(), fps)


# ------------------------------------------------------------------------------------------------------- manifest chains
def verify_outputs(run_dir, manifest: Mapping[str, Any], names: Sequence[str], key: str = "outputs") -> Dict[str, str]:
    """Check that the named files of a run exist and equal the hashes its manifest recorded."""
    out = {}
    for n in names:
        rec = manifest[key].get(n)
        if rec is None:
            fail("MANIFEST_ENTRY", "%s has no recorded output %s" % (run_dir, n))
        f = Path(run_dir) / n
        if not f.is_file():
            fail("ARTIFACT_MISSING", "%s missing" % f)
        got = sha256_file(f)
        if got != rec["sha256"]:
            fail("ARTIFACT_HASH", "%s sha256 %s differs from its manifest %s" % (f, got, rec["sha256"]))
        out[n] = got
    return out


def verify_chain(*, phase3_agent_states, task2_dir, task3_dir, windows_dir, t2: Mapping, t3: Mapping, w: Mapping) -> Dict[str, bool]:
    """Provenance chain Phase 3 file -> Task 2 manifest -> Task 3 manifest -> window manifest; every link is a sha256 recorded by the downstream stage."""
    links = {"task2_input_is_phase3_agent_states": t2["input"]["sha256_before"] == sha256_file(phase3_agent_states),
             "task3_references_task2_manifest": t3["inputs"]["task2_manifest"]["sha256"] == sha256_file(Path(task2_dir) / "run_manifest.json"),
             "task3_references_task2_causal_output": t3["inputs"]["causal_kinematics"]["sha256"] == t2["outputs"]["causal_kinematics.jsonl"]["sha256"],
             "windows_reference_task3_manifest": w["sources"]["task3_run_manifest"]["sha256"] == sha256_file(Path(task3_dir) / "run_manifest.json")}
    bad = [k for k, v in links.items() if not v]
    if bad:
        fail("CHAIN_BROKEN", "provenance chain broken at %s" % bad)
    return links


# ------------------------------------------------------------------------------------------------------- windows / inference
def validate_windows_history_only(ws, source_fps: float, tol_s: float = 1e-6) -> Dict[str, Any]:
    """History frames never exceed the anchor; future frames lie strictly after it; timestamps equal frame / fps (identity and time alignment)."""
    if not np.array_equal(ws.history_frames[:, -1], ws.anchor_frame):
        fail("ANCHOR", "the last history frame differs from the anchor frame")
    if (ws.history_frames > ws.anchor_frame[:, None]).any():
        fail("FUTURE_IN_HISTORY", "a history frame lies after its anchor")
    if (ws.future_frames <= ws.anchor_frame[:, None]).any():
        fail("FUTURE_ORDER", "a future frame is not strictly after the anchor")
    if (np.diff(ws.history_frames, axis=1) <= 0).any() or (np.diff(ws.future_frames, axis=1) <= 0).any():
        fail("FRAME_ORDER", "frames are not strictly increasing")
    if np.abs(ws.history_timestamps - ws.history_frames / source_fps).max() > tol_s or np.abs(ws.future_timestamps - ws.future_frames / source_fps).max() > tol_s:
        fail("TIMESTAMP_ALIGNMENT", "timestamps differ from frame / source_fps")
    return {"windows": int(ws.n_windows), "history_only": True, "min_future_gap_frames": int((ws.future_frames[:, 0] - ws.anchor_frame).min())}


def validate_predictions(pred, ws, window_cfg, conflict_cfg) -> None:
    """Prediction contract (identity, anchors, horizon alignment, coordinate system) via the Phase 6 checker, plus probability sanity."""
    from ..phase6.pairs import check_predictions
    check_predictions(pred, ws, window_cfg, conflict_cfg)
    if not np.isfinite(pred.mode_probabilities).all() or (pred.mode_probabilities < 0).any() or not np.allclose(pred.mode_probabilities.sum(axis=1), 1.0, atol=1e-4):
        fail("MODE_PROBABILITIES", "mode probabilities must be finite, non-negative and sum to 1")
    if pred.valid.any(axis=(1, 2)).sum() == 0:
        fail("NO_VALID_PREDICTION", "no window has a valid prediction")


def validate_risk_records(records: Sequence[Mapping[str, Any]], ws, source_fps: float) -> Dict[str, Any]:
    """Phase 7 records preserve window identity and anchor frames; probabilities exist only with calibration; invalid evidence stays explicit."""
    wid = {int(w): i for i, w in enumerate(ws.window_id.tolist())}
    n_bad_prob = n_ins = 0
    for i, r in enumerate(records):
        w = wid.get(r["window_id"])
        if w is None:
            fail("RECORD_WINDOW", "record %d references unknown window %r" % (i, r["window_id"]))
        if int(ws.anchor_frame[w]) != r["anchor_frame"]:
            fail("RECORD_ANCHOR", "record %d anchor %r differs from window %d anchor %d" % (i, r["anchor_frame"], r["window_id"], ws.anchor_frame[w]))
        if str(ws.target_track_id[w]) != r["pedestrian_id"]:
            fail("RECORD_IDENTITY", "record %d pedestrian %r differs from the window target %r" % (i, r["pedestrian_id"], ws.target_track_id[w]))
        if r["calibrated_probability"] is not None and r["calibration_status"] != "CALIBRATED":
            n_bad_prob += 1
        if r["risk_category"] == "INSUFFICIENT_EVIDENCE":
            n_ins += 1
            if r["risk_score"] is not None and r["masks"]["category_assigned"]:
                fail("RECORD_INSUFFICIENT", "record %d is INSUFFICIENT_EVIDENCE but carries an assigned risk" % i)
    if n_bad_prob:
        fail("RECORD_PROBABILITY", "%d record(s) carry a probability without calibration" % n_bad_prob)
    return {"records": len(records), "insufficient_evidence": n_ins, "source_fps": source_fps}


def validate_alert_outputs(events: Sequence[Mapping[str, Any]], alerts: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], source_fps: float) -> Dict[str, Any]:
    """No alert may arise from missing or invalid evidence; alerts only for pairs Phase 7 rated HIGH/CRITICAL; times are frame / fps; no alert is a probability claim."""
    hot = {(r["pedestrian_id"], r["vehicle_id"]) for r in records if r["risk_category"] in ("HIGH", "CRITICAL")}
    for e in events:
        if abs(e["observation_time_s"] - e["frame"] / source_fps) > 1e-9:
            fail("EVENT_TIME", "event %s time differs from frame / fps" % e["event_id"])
        if e["event_type"] == "STATE_TRANSITION" and e["to_state"] in ("ACTIVE_WARNING", "ESCALATED_WARNING") and e["from_state"] not in ("ACTIVE_WARNING", "ESCALATED_WARNING"):
            if not e["evidence_valid"]:
                fail("ALERT_FROM_INVALID_EVIDENCE", "event %s raises or restores a warning without valid evidence" % e["event_id"])
        if e["phase7"]["calibrated_probability"] is not None and e["phase7"]["calibration_status"] != "CALIBRATED":
            fail("EVENT_PROBABILITY", "event %s carries an uncalibrated probability" % e["event_id"])
    for a in alerts:
        if (a["pedestrian_id"], a["vehicle_id"]) not in hot:
            fail("ALERT_WITHOUT_HIGH_RISK", "alert %s has no HIGH/CRITICAL Phase 7 record" % a["alert_id"])
        if a["clearance_frame"] is None:
            fail("ALERT_NOT_CLOSED", "alert %s was never closed" % a["alert_id"])
    return {"events": len(events), "alerts": len(alerts)}


__all__ = ["CONTRACT_VERSION", "STAGES", "STAGE_CONTRACTS", "validate_phase3_rows", "validate_phase3_file", "verify_outputs", "verify_chain", "validate_windows_history_only",
           "validate_predictions", "validate_risk_records", "validate_alert_outputs", "sha256_file", "COORDINATE_SYSTEM"]
