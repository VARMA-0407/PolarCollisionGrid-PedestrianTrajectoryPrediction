"""Conflict-label specification and validation.

Three concepts are kept apart (they must never be conflated):
  1. heuristic image-space INDICATORS (``features.py``): geometric functions of predicted / observed positions, no label involved;
  2. CANDIDATE labels (this module, ``CANDIDATE_HEURISTIC_*``): the same indicators evaluated on the REALISED perceived future of both agents.
     They are derived from perceived-track pseudo-observations by arbitrary exploratory thresholds, are NOT verified ground truth, and can
     never be used as supervision (``validate_label_set`` refuses them with ``LABELS_NOT_VERIFIED``);
  3. VERIFIED conflict labels (collision, near-conflict, annotated interaction events from an independent, documented annotation source).
     None is available in the repository (see the dataset feasibility report), so supervised training is blocked.

A label set that could support supervision must carry: a verified kind, a documented independent source (name, type, version, SHA-256, protocol
reference), identities of the pairs it refers to, label times strictly after the pair's anchor (no overlap with the history that produces the
features), independent scene-group ids (at least ``MIN_GROUPS``), both classes present, and no non-finite values.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .errors import ConflictLabelError

KIND_VERIFIED_COLLISION = "VERIFIED_COLLISION"
KIND_VERIFIED_NEAR_CONFLICT = "VERIFIED_NEAR_CONFLICT"
KIND_ANNOTATED_INTERACTION = "ANNOTATED_INTERACTION"
KIND_CANDIDATE = "CANDIDATE_HEURISTIC_PERCEIVED_TRACK_PROXIMITY"
SUPERVISABLE_KINDS = (KIND_VERIFIED_COLLISION, KIND_VERIFIED_NEAR_CONFLICT, KIND_ANNOTATED_INTERACTION)
LABEL_KINDS = SUPERVISABLE_KINDS + (KIND_CANDIDATE,)
VERIFIED_SOURCE_TYPES = ("official_dataset_annotation", "independent_human_annotation", "independent_sensor_event_log")
SOURCE_FIELDS = ("name", "type", "version", "sha256", "protocol_reference")
MIN_GROUPS = 3


@dataclass
class ConflictLabelSet:
    label_kind: str
    verified: bool
    source: Dict[str, Any]
    window_id: np.ndarray            # [L] pair identity (the window of the target pedestrian)
    ped_track_index: np.ndarray      # [L]
    other_track_index: np.ndarray    # [L]
    anchor_frame: np.ndarray         # [L] anchor of the pair's features (labels must lie after it)
    label_start_frame: np.ndarray    # [L] first frame of the labelled interval (must be > anchor_frame)
    label_end_frame: np.ndarray      # [L]
    positive: np.ndarray             # [L] bool
    valid: np.ndarray                # [L] bool
    group_id: np.ndarray             # [L] independent scene / video group identifier (str)
    event_frame: Optional[np.ndarray] = None   # [L] frame of the event for positives (NaN / -1 otherwise); needed for lead-time metrics
    definition: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.window_id.shape[0])

    def pair_keys(self):
        return list(zip(self.window_id.tolist(), self.ped_track_index.tolist(), self.other_track_index.tolist()))


def validate_label_set(labels: ConflictLabelSet, *, require_supervision: bool = True, pair_keys=None, anchor_frames: Optional[Mapping] = None,
                       min_groups: int = MIN_GROUPS) -> Dict[str, Any]:
    """Validate a label set. With ``require_supervision`` (default) candidate / unverified sets are refused. Returns a report on success."""
    if labels.label_kind not in LABEL_KINDS:
        raise ConflictLabelError("LABEL_KIND", "unknown label kind %r (known: %s)" % (labels.label_kind, LABEL_KINDS))
    L = labels.n
    for name in ("ped_track_index", "other_track_index", "anchor_frame", "label_start_frame", "label_end_frame", "positive", "valid", "group_id"):
        if getattr(labels, name).shape != (L,):
            raise ConflictLabelError("LABEL_SHAPE", "%s must have shape (%d,)" % (name, L))
    if labels.label_kind == KIND_CANDIDATE or not labels.verified:
        if require_supervision:
            raise ConflictLabelError("LABELS_NOT_VERIFIED", "labels of kind %s (verified=%s) are heuristic candidates derived from perceived tracks and cannot be used as "
                                     "supervision or as ground truth" % (labels.label_kind, labels.verified))
        return {"supervision_eligible": False, "label_kind": labels.label_kind, "n": L}
    src = labels.source
    missing = [k for k in SOURCE_FIELDS if not src.get(k)]
    if missing:
        raise ConflictLabelError("LABEL_PROVENANCE_MISSING", "label source lacks %s" % missing)
    if src["type"] not in VERIFIED_SOURCE_TYPES:
        raise ConflictLabelError("LABEL_SOURCE_TYPE", "source type %r is not an accepted independent source (%s)" % (src["type"], VERIFIED_SOURCE_TYPES))
    if len(str(src["sha256"])) != 64:
        raise ConflictLabelError("LABEL_PROVENANCE_MISSING", "source sha256 must be a 64-hex digest")
    for name in ("label_start_frame", "label_end_frame", "anchor_frame"):
        a = getattr(labels, name)
        if not np.issubdtype(a.dtype, np.integer):
            raise ConflictLabelError("LABEL_NONFINITE", "%s must be integer frame indices" % name)
    if not (labels.positive.dtype == bool and labels.valid.dtype == bool):
        raise ConflictLabelError("LABEL_SHAPE", "positive and valid must be boolean")
    v = labels.valid
    if v.any():
        if (labels.label_start_frame[v] <= labels.anchor_frame[v]).any():
            raise ConflictLabelError("LABEL_TIME_LEAK", "a labelled interval starts at or before the anchor of the features that predict it")
        if (labels.label_end_frame[v] < labels.label_start_frame[v]).any():
            raise ConflictLabelError("LABEL_TIME_ORDER", "label_end_frame precedes label_start_frame")
        if labels.event_frame is not None:
            ev = labels.event_frame
            pos = v & labels.positive
            if pos.any() and ((ev[pos] < labels.label_start_frame[pos]) | (ev[pos] > labels.label_end_frame[pos])).any():
                raise ConflictLabelError("LABEL_EVENT_OUTSIDE_INTERVAL", "an event frame lies outside its labelled interval")
    groups = set(labels.group_id[v].tolist()) if v.any() else set()
    if len(groups) < min_groups:
        raise ConflictLabelError("LABEL_GROUPS_INSUFFICIENT", "%d independent group(s) found, at least %d are required for a train/validation/test protocol" % (len(groups), min_groups))
    if not (labels.positive[v].any() and (~labels.positive[v]).any()):
        raise ConflictLabelError("LABEL_CLASS_ABSENT", "both positive and negative valid labels are required")
    if pair_keys is not None:
        known = set(pair_keys)
        unknown = [k for k in labels.pair_keys() if k not in known]
        if unknown:
            raise ConflictLabelError("LABEL_IDENTITY_MISMATCH", "%d label pair(s) are not in the pair batch (first: %s)" % (len(unknown), unknown[0]))
    if anchor_frames is not None:
        bad = [i for i, k in enumerate(labels.pair_keys()) if anchor_frames.get(k) is not None and anchor_frames[k] != int(labels.anchor_frame[i])]
        if bad:
            raise ConflictLabelError("LABEL_IDENTITY_MISMATCH", "label anchor frames differ from the pair batch for %d pair(s)" % len(bad))
    return {"supervision_eligible": True, "label_kind": labels.label_kind, "n": L, "n_valid": int(v.sum()), "n_positive": int((labels.positive & v).sum()), "groups": sorted(groups)}


def assert_trainable(labels: ConflictLabelSet, **kw) -> Dict[str, Any]:
    return validate_label_set(labels, require_supervision=True, **kw)


__all__ = ["ConflictLabelSet", "validate_label_set", "assert_trainable", "LABEL_KINDS", "SUPERVISABLE_KINDS", "KIND_CANDIDATE", "KIND_VERIFIED_COLLISION",
           "KIND_VERIFIED_NEAR_CONFLICT", "KIND_ANNOTATED_INTERACTION", "VERIFIED_SOURCE_TYPES", "MIN_GROUPS"]
