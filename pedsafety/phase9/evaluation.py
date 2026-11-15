"""Independent, group-level scientific evaluation: gates, metric wrappers and alert-level metrics.

Nothing here computes a scientific result unless ``assess_independent_evaluation`` reports ``READY``: a frozen group-level split manifest with disjoint train / calibration /
validation / test groups, verified annotations with provenance, a verified identity mapping and timestamp alignment, and archived licence records. Otherwise every
evaluation raises ``EvaluationBlocked`` (status ``BLOCKED_INSUFFICIENT_INDEPENDENT_SCENES`` when fewer than 3 independent scene groups exist). No label or ground truth is
ever synthesised; in-sample pilot results are never routed through these functions as independent results.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..phase5.independent_data_status import MIN_INDEPENDENT_GROUPS, assess_independent_data
from .errors import EvaluationBlocked

SPLIT_VERSION = "phase9-group-split-1.0.0"
PARTITIONS = ("train", "calibration", "validation", "test")
BLOCKED_SCENES = "BLOCKED_INSUFFICIENT_INDEPENDENT_SCENES"
BLOCKED_SPLIT = "BLOCKED_NO_VALID_FROZEN_SPLIT_MANIFEST"
READY = "READY_FOR_INDEPENDENT_EVALUATION"
SOURCE_TYPES = ("official_dataset_annotation", "independent_human_annotation", "independent_sensor_event_log")


def validate_split_manifest(m: Mapping[str, Any]) -> List[str]:
    """Return the list of problems of a group-level split manifest (empty = valid). Pure; never repairs."""
    p: List[str] = []
    if m.get("manifest_version") != SPLIT_VERSION:
        p.append("manifest_version must be %s" % SPLIT_VERSION)
    if m.get("frozen") is not True or not m.get("frozen_utc"):
        p.append("the split must be frozen (frozen=true with frozen_utc) before any training")
    parts, groups = m.get("partitions"), m.get("groups")
    if not isinstance(parts, Mapping) or not isinstance(groups, Mapping) or not groups:
        return p + ["partitions and groups are required"]
    for name in ("train", "validation", "test"):
        if not parts.get(name):
            p.append("partition %s is empty or missing" % name)
    seen: Dict[str, str] = {}
    for name in PARTITIONS:
        for g in parts.get(name, []) or []:
            if g in seen:
                p.append("group %s is in both %s and %s" % (g, seen[g], name))
            seen[g] = name
    for g in groups:
        if g not in seen:
            p.append("group %s is not assigned to any partition" % g)
    for g in seen:
        if g not in groups:
            p.append("partition group %s has no group record" % g)
    video_owner: Dict[str, str] = {}
    for g, rec in groups.items():
        for v in rec.get("videos", []) or []:
            if v in video_owner and video_owner[v] != g:
                p.append("video %s belongs to groups %s and %s (scene/video overlap)" % (v, video_owner[v], g))
            video_owner[v] = g
        if not rec.get("videos"):
            p.append("group %s lists no videos" % g)
        ann = rec.get("annotation") or {}
        if not (ann.get("verified") is True and all(ann.get(k) for k in ("name", "type", "version", "sha256", "protocol_reference")) and ann.get("type") in SOURCE_TYPES
                and len(str(ann.get("sha256", ""))) == 64):
            p.append("group %s lacks verified annotation provenance (name, type in %s, version, 64-hex sha256, protocol_reference, verified=true)" % (g, SOURCE_TYPES))
        for key in ("identity_mapping", "timestamp_alignment"):
            if (rec.get(key) or {}).get("verified") is not True:
                p.append("group %s: %s is not verified" % (g, key))
        lic = rec.get("license") or {}
        if not (lic.get("name") and lic.get("archive_sha256") and len(str(lic["archive_sha256"])) == 64 and lic.get("acquisition_record")):
            p.append("group %s lacks a licence / acquisition record (name, archive_sha256, acquisition_record)" % g)
    if not (isinstance(m.get("purge_gap_frames"), int) and m["purge_gap_frames"] >= 0):
        p.append("purge_gap_frames (integer >= 0) is required")
    return p


def assess_independent_evaluation(repo_root, split_manifest_path=None) -> Dict[str, Any]:
    scan = assess_independent_data(repo_root)
    n_groups = scan["checks"]["independent_scene_groups_processed"]["found"]
    out: Dict[str, Any] = {"scan_status": scan["status"], "distinct_scene_groups_processed": n_groups, "required_groups": MIN_INDEPENDENT_GROUPS,
                           "annotation_xml_files": scan["found"]["annotation_xml_files"], "local_video_files": scan["found"]["local_video_files"],
                           "split_manifest": None, "problems": [], "fabricated_results": False}
    if n_groups < MIN_INDEPENDENT_GROUPS:
        out["status"] = BLOCKED_SCENES
        out["problems"].append("%d independent scene group(s) processed; at least %d are required" % (n_groups, MIN_INDEPENDENT_GROUPS))
        return out
    if not split_manifest_path or not Path(split_manifest_path).is_file():
        out["status"] = BLOCKED_SPLIT
        out["problems"].append("no frozen group-level split manifest supplied")
        return out
    m = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    probs = validate_split_manifest(m)
    out["split_manifest"] = str(split_manifest_path)
    out["problems"] = probs
    out["status"] = BLOCKED_SPLIT if probs else READY
    return out


def require_independent(assessment: Mapping[str, Any], split: Mapping[str, Any], *, evaluated_groups: Sequence[str], model_training_groups: Sequence[str],
                        required_partition: str = "test") -> None:
    """Gate for any independent metric. The evaluated groups must be whole groups of ``required_partition`` and disjoint from the groups the model was trained on."""
    if assessment.get("status") != READY:
        raise EvaluationBlocked(str(assessment.get("status", BLOCKED_SPLIT)), "independent evaluation is blocked: %s" % "; ".join(assessment.get("problems", [])) or "not ready")
    if validate_split_manifest(split):
        raise EvaluationBlocked(BLOCKED_SPLIT, "split manifest is invalid")
    part = set(split["partitions"][required_partition])
    ev, tr = set(evaluated_groups), set(model_training_groups)
    if not ev or not ev <= part:
        raise EvaluationBlocked("GROUPS_NOT_IN_PARTITION", "evaluated groups %s are not whole groups of partition %s" % (sorted(ev), required_partition))
    if ev & tr or not tr <= set(split["partitions"]["train"]):
        raise EvaluationBlocked("TRAINING_GROUP_LEAK", "the model must be trained on train groups only and never on the evaluated groups")


def evaluate_trajectories_independent(assessment, split, *, evaluated_groups, model_training_groups, gt_uv, gt_valid, pred_uv, pred_valid, window_id, track_id, mode_probabilities,
                                      horizon_offsets_s, eval_cfg, model_id: str) -> Dict[str, Any]:
    """ADE / FDE / horizon error with the most-probable mode as primary; minADE@K / minFDE@K stay in the separate oracle block (Phase 5 evaluator, unchanged)."""
    require_independent(assessment, split, evaluated_groups=evaluated_groups, model_training_groups=model_training_groups)
    from ..phase5.evaluation.metrics import evaluate_predictions, summarize
    res = evaluate_predictions(gt_uv, gt_valid, pred_uv, pred_valid, window_id, track_id, coordinate_system="captured_1920x1080", unit="pixel", model_id=model_id,
                               horizon_offsets_s=horizon_offsets_s, cfg=eval_cfg, mode_probabilities=mode_probabilities)
    return {"sample_status": "INDEPENDENT_HELD_OUT_GROUPS", "groups": sorted(evaluated_groups), "summary": summarize(res, horizon_offsets_s, eval_cfg)}


def evaluate_conflict_independent(assessment, split, *, evaluated_groups, model_training_groups, scores, labels, threshold, fps=None) -> Dict[str, Any]:
    require_independent(assessment, split, evaluated_groups=evaluated_groups, model_training_groups=model_training_groups)
    from ..phase6.metrics import evaluate_supervised
    return {"sample_status": "INDEPENDENT_HELD_OUT_GROUPS", "metrics": evaluate_supervised(scores, labels, threshold=threshold, fps=fps)}


def evaluate_calibration_independent(assessment, split, *, evaluated_groups, model_training_groups, artifact, scores, valid, labels, pair_keys) -> Dict[str, Any]:
    require_independent(assessment, split, evaluated_groups=evaluated_groups, model_training_groups=model_training_groups)
    cal = set(artifact.payload["split"]["calibration_groups"])
    if cal & set(evaluated_groups) or not cal <= set(split["partitions"]["calibration"]):
        raise EvaluationBlocked("CALIBRATION_PARTITION", "the calibration artifact must be fitted on the calibration partition and never on evaluated groups")
    from ..phase7.calibration import evaluate_calibrated
    return {"sample_status": "INDEPENDENT_HELD_OUT_GROUPS", "metrics": evaluate_calibrated(artifact, scores, valid, labels, pair_keys)}


def _q(vals: Sequence[float]) -> Optional[Dict[str, float]]:
    if not len(vals):
        return None
    a = np.asarray(vals, float)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def alert_level_metrics(alerts: Sequence[Mapping[str, Any]], verified_events: Sequence[Mapping[str, Any]], source: Mapping[str, Any], *, fps: float, observed_duration_s: float,
                        match_horizon_frames: int, assessment: Mapping[str, Any]) -> Dict[str, Any]:
    """Alert-level metrics against VERIFIED events (gated). Matching rule (declared by the caller via ``match_horizon_frames``, no default): an alert matches an event of the same
    pair when ``activated_frame <= event_frame <= activated_frame + match_horizon_frames``. Lead time = (event - activation) / fps; a duplicate is an additional alert matching an
    already matched event; coverage = share of events whose frame lies inside an alert's [activated, clearance] interval."""
    if assessment.get("status") != READY:
        raise EvaluationBlocked(str(assessment.get("status", BLOCKED_SPLIT)), "alert-level evaluation requires verified events on independent groups")
    from .errors import PipelineError
    from ..phase8.lead_time import VERIFIED_SOURCE_TYPES
    if source.get("type") not in VERIFIED_SOURCE_TYPES or not source.get("sha256") or len(str(source["sha256"])) != 64:
        raise EvaluationBlocked("EVENTS_NOT_VERIFIED", "verified-event source lacks an accepted type or sha256")
    if not verified_events or any(e.get("verified") is not True for e in verified_events):
        raise EvaluationBlocked("EVENTS_NOT_VERIFIED", "all events must be verified")
    if not (isinstance(match_horizon_frames, int) and match_horizon_frames > 0) or not (observed_duration_s > 0 and math.isfinite(observed_duration_s)):
        raise PipelineError("CONFIG", "match_horizon_frames > 0 and a positive observed duration are required")
    matched_ev, lead, true_alerts, dup, covered = {}, [], 0, 0, 0
    by_key: Dict[str, List[Mapping[str, Any]]] = {}
    for a in alerts:
        by_key.setdefault(a["dedup_key"], []).append(a)
    for i, e in enumerate(verified_events):
        c = sorted((a for a in by_key.get(e["dedup_key"], []) if a["activated_frame"] <= e["event_frame"] <= a["activated_frame"] + match_horizon_frames), key=lambda a: a["activated_frame"])
        if c:
            matched_ev[i] = c
            lead.append((e["event_frame"] - c[0]["activated_frame"]) / fps)
        if any(a["activated_frame"] <= e["event_frame"] <= a["clearance_frame"] for a in by_key.get(e["dedup_key"], [])):
            covered += 1
    matched_alert_ids = {a["alert_id"] for cs in matched_ev.values() for a in cs}
    for cs in matched_ev.values():
        dup += len(cs) - 1
    n_alerts, n_ev = len(alerts), len(verified_events)
    return {"sample_status": "INDEPENDENT_HELD_OUT_GROUPS", "n_alerts": n_alerts, "n_events": n_ev, "matching_rule": "activated <= event <= activated + %d frames, same pair" % match_horizon_frames,
            "alert_precision": (len(matched_alert_ids) / n_alerts) if n_alerts else None, "alert_recall": len(matched_ev) / n_ev, "missed_event_rate": 1 - len(matched_ev) / n_ev,
            "false_alerts_per_hour": (n_alerts - len(matched_alert_ids)) / (observed_duration_s / 3600.0), "duplicate_alert_rate": (dup / n_alerts) if n_alerts else None,
            "warning_coverage": covered / n_ev, "lead_time_s": _q(lead), "time_to_clearance_s": _q([(a["clearance_frame"] - a["activated_frame"]) / fps for a in alerts if a["clearance_frame"] is not None]),
            "time_definitions": "lead time uses warning-generation frame and verified event frame; observation time = frame / fps; predicted conflict time is not used"}


def blocked_evaluations(assessment: Mapping[str, Any]) -> Dict[str, Any]:
    reason = assessment.get("status")
    fams = {"trajectory_independent_ADE_FDE_horizon_minADE_minFDE": "needs independent held-out groups with verified trajectory ground truth",
            "conflict_precision_recall_F1_PR_ROC_AUC_false_alarm": "needs verified conflict labels on independent groups",
            "calibration_Brier_ECE_NLL_reliability": "needs verified labels and disjoint calibration / test groups",
            "alert_precision_recall_false_alerts_missed_duplicate_lead_time_coverage_clearance": "needs verified event annotations and timestamps on independent groups",
            "perception_quality_vs_annotations": "needs annotation files and an identity mapping"}
    return {k: {"status": reason if reason != READY else "READY_BUT_NOT_RUN", "needs": v} for k, v in fams.items()}


__all__ = ["validate_split_manifest", "assess_independent_evaluation", "require_independent", "evaluate_trajectories_independent", "evaluate_conflict_independent",
           "evaluate_calibration_independent", "alert_level_metrics", "blocked_evaluations", "BLOCKED_SCENES", "BLOCKED_SPLIT", "READY", "SPLIT_VERSION"]
