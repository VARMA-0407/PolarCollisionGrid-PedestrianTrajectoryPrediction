"""PIE ground-truth detection and tracking evaluation (Phase 3, Task 4).

Compares the ALREADY-PRODUCED ``detections.jsonl`` and ``tracked_agents.jsonl`` of a perception run with the official PIE
CVAT-style annotations of one video. It never runs YOLO or ByteTrack, never reads video, and works in image pixels only:
nothing here is a world-space or metric statement.

Evaluation policy (also written into the manifest and the Markdown report)
-------------------------------------------------------------------------
* **Identity** of a ground-truth object is the per-box ``id`` attribute, never the CVAT track id.
* **Classes**: pedestrian -> pedestrian; vehicle type car -> car; vehicle type bicycle -> bike; vehicle type train ->
  ``unsupported_train`` (NOT evaluated, never mapped to car/truck). Only pedestrian, car and bike are evaluated. Traffic lights, signs and
  crosswalks are ignored and only counted.
* **Boxes**: ``outside=1`` boxes are ignored; partially occluded boxes are retained; fully occluded boxes are retained by default
  (``exclude_fully_occluded=True`` drops them) and recall is also stratified by occlusion level. Malformed or zero-area boxes are discarded
  with counters. GT boxes are clipped to the image when its size is known (counted).
* **Evaluated frame set**: only frames in which the annotation file contains at least one valid non-outside box (any label) AND that lie inside the
  frame range actually covered by the predictions. A prediction in any other frame is NOT counted as a false positive, because the annotation
  file says nothing about those frames. A stricter sensitivity set (frames with at least one SUPPORTED GT box) is reported as well.
* **Matching**: class-aware, one-to-one per frame, maximum number of pairs first then maximum IoU, IoU >= threshold (inclusive), via
  ``scipy.optimize.linear_sum_assignment`` when available, else an exact pure-Python Hungarian solver. Deterministic.
* **Detection**: fixed operating point (all detections in the file, optionally filtered by ``min_confidence``); no confidence-ranked AP and no mAP.
  Predictions of classes without a PIE counterpart (truck, bus) are out of scope: they are counted and shown in the mismatch diagnostics but are not
  false positives of the evaluated classes.
* **Tracking**: CLEAR-MOT (per-frame matching that keeps the previous correspondence while it is still valid, ID switches against each GT identity's
  last matched predicted identity, persisting across unmatched frames and annotation gaps) and IDF1 (one GLOBAL identity assignment maximising the
  number of frames matched, not a sum of frame-wise matches). MOTP is reported as the mean IoU of matches (higher is better).
* **Tracking rows**: primary = ``tracking_validity`` true; diagnostics = detection-backed rows only (``is_predicted`` false) and all rows.
  ``is_predicted`` rows are counted and reported, never silently dropped.
"""

import hashlib
import json
import math
import platform
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..errors import ConfigError, PedsafetyError

EVALUATOR_VERSION = "pie-perception-evaluator-0.1.0-task4"

SUPPORTED_CLASSES = ("pedestrian", "car", "bike")
OUT_OF_SCOPE_PRED_CLASSES = ("truck", "bus")
UNSUPPORTED_TRAIN = "unsupported_train"
CLASS_MAPPING = {"pedestrian": "pedestrian", "vehicle:type=car": "car", "vehicle:type=bicycle": "bike", "vehicle:type=train": UNSUPPORTED_TRAIN,
                 "traffic_light": "ignored", "sign": "ignored", "crosswalk": "ignored"}
IGNORED_LABELS = ("traffic_light", "sign", "crosswalk")
FRAME_POLICIES = ("annotated_any", "supported_gt_only")
OCCLUSION_LEVELS = ("none", "partial", "full", "occluded_unspecified")
Box = Tuple[float, float, float, float]


class PieEvaluationError(PedsafetyError):
    """Unusable annotation or prediction input (missing file, wrong video, no usable rows)."""


# ---------------------------------------------------------------------------------------------------------------- data
@dataclass(frozen=True)
class GtBox:
    frame: int
    label: str                       # annotation label: pedestrian / vehicle / ...
    eval_class: str                  # pedestrian / car / bike / unsupported_train / ignored
    bbox: Box
    ident: Optional[str]             # the PIE per-box 'id' attribute (None when missing)
    occluded: int                    # CVAT occluded flag
    occlusion_level: str             # one of OCCLUSION_LEVELS
    keyframe: Optional[int]
    cvat_track_id: str
    attrs: Tuple[Tuple[str, str], ...] = ()


@dataclass
class GtParse:
    boxes: List[GtBox]
    counters: Dict[str, int]
    raw_counts_by_label: Dict[str, int]           # every <box>, including outside
    nonoutside_counts_by_label: Dict[str, int]
    vehicle_type_counts: Dict[str, int]           # non-outside valid vehicle boxes by type
    unsupported_counts: Dict[str, int]
    id_report: Dict[str, Any]
    meta: Dict[str, str]
    annotated_frames_any: List[int]
    supported_gt_frames: List[int]
    frame_range: Optional[Tuple[int, int]]
    discarded_examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Pred:
    frame: int
    cls: str                          # semantic class as exported
    bbox: Box
    conf: float
    ident: Optional[int] = None       # track_id for tracked rows, detection_id for detections
    is_predicted: bool = False
    tracking_validity: bool = True
    detection_validity: bool = True


@dataclass
class PredLoad:
    by_frame: Dict[int, List[Pred]]
    counters: Dict[str, int]
    frame_range: Optional[Tuple[int, int]]
    image_size: Optional[Tuple[int, int]]
    video_ids: Dict[str, int]


# ------------------------------------------------------------------------------------------------------------- geometry
def iou_xyxy(a: Box, b: Box) -> float:
    """Intersection over union of two (x_min, y_min, x_max, y_max) boxes in pixels; 0.0 for disjoint or degenerate boxes."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _finite(*vals: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


# ------------------------------------------------------------------------------------------------------- assignment
def _hungarian_min(cost: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """Exact minimum-cost assignment of every row of an n x m matrix (n <= m) to distinct columns (potentials / augmenting paths).
    Deterministic; used only when SciPy is unavailable."""
    n, m = len(cost), len(cost[0]) if cost else 0
    INF = float("inf")
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sorted((p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0)


def solve_assignment(cost: Sequence[Sequence[float]], prefer_scipy: bool = True) -> List[Tuple[int, int]]:
    """Minimum-cost one-to-one assignment on a rectangular matrix; returns sorted (row, col) pairs, ``min(n, m)`` of them."""
    if not cost or not cost[0]:
        return []
    if prefer_scipy:
        try:
            from scipy.optimize import linear_sum_assignment
            import numpy as np
            r, c = linear_sum_assignment(np.asarray(cost, dtype=float))
            return sorted((int(a), int(b)) for a, b in zip(r, c))
        except ImportError:
            pass
    n, m = len(cost), len(cost[0])
    if n <= m:
        return _hungarian_min(cost)
    return sorted((i, j) for j, i in _hungarian_min([[cost[i][j] for i in range(n)] for j in range(m)]))


_BIG = 10.0      # a valid pair is worth BIG + IoU, so more pairs always beat a higher summed IoU


def match_boxes(gt: Sequence[Tuple[str, Box]], pr: Sequence[Tuple[str, Box]], thr: float, prefer_scipy: bool = True) -> List[Tuple[int, int, float]]:
    """Class-aware one-to-one matching. ``gt``/``pr`` are (class, box). Returns (gt_index, pred_index, iou) with iou >= thr, sorted."""
    out: List[Tuple[int, int, float]] = []
    for cls in sorted({c for c, _ in gt} & {c for c, _ in pr}):
        gi = [i for i, (c, _) in enumerate(gt) if c == cls]
        pi = [j for j, (c, _) in enumerate(pr) if c == cls]
        iou = [[iou_xyxy(gt[i][1], pr[j][1]) for j in pi] for i in gi]
        if not any(x >= thr for row in iou for x in row):
            continue
        cost = [[-(_BIG + x) if x >= thr else 0.0 for x in row] for row in iou]
        for a, b in solve_assignment(cost, prefer_scipy):
            if iou[a][b] >= thr:
                out.append((gi[a], pi[b], iou[a][b]))
    return sorted(out)


# ------------------------------------------------------------------------------------------------------ GT parsing
def _attr_map(box: ET.Element) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for a in box.findall("attribute"):
        name = a.get("name")
        if name:
            attrs[name] = (a.text or "").strip()
    return attrs


def _int_or_none(s: Optional[str]) -> Optional[int]:
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


def _occlusion_level(label: str, attrs: Mapping[str, str], occluded: int) -> str:
    if label == "pedestrian" and "occlusion" in attrs:
        return {0: "none", 1: "partial", 2: "full"}.get(_int_or_none(attrs["occlusion"]), "occluded_unspecified" if occluded else "none")
    return "occluded_unspecified" if occluded else "none"


def parse_pie_annotations(path: Any, *, image_size: Optional[Tuple[int, int]] = None, exclude_fully_occluded: bool = False) -> GtParse:
    """Parse the CVAT-style PIE XML. See the module docstring for the policy; every discarded or ignored box is counted."""
    p = Path(path)
    if not p.is_file():
        raise PieEvaluationError("annotation file not found: %s" % p)
    try:
        root = ET.parse(str(p)).getroot()
    except (ET.ParseError, OSError) as exc:
        raise PieEvaluationError("cannot parse annotation XML %s: %s" % (p, exc))
    cnt: Counter = Counter()
    raw_by_label: Counter = Counter()
    non_out_by_label: Counter = Counter()
    veh_types: Counter = Counter()
    unsupported: Counter = Counter()
    boxes: List[GtBox] = []
    examples: List[Dict[str, Any]] = []
    meta: Dict[str, str] = {}
    task = root.find("meta/task")
    if task is not None:
        for tag in ("name", "size", "mode", "start_frame", "stop_frame", "source"):
            el = task.find(tag)
            if el is not None and el.text:
                meta[tag] = el.text.strip()
    for tr in root.iter("track"):
        label = (tr.get("label") or "").strip()
        tid = tr.get("id") or ""
        for bx in tr.findall("box"):
            cnt["boxes_total"] += 1
            raw_by_label[label] += 1
            if (bx.get("outside") or "0").strip() == "1":
                cnt["ignored_outside"] += 1
                continue
            non_out_by_label[label] += 1
            frame = _int_or_none(bx.get("frame"))
            try:
                x0, y0, x1, y1 = (float(bx.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            except (TypeError, ValueError):
                x0 = y0 = x1 = y1 = float("nan")
            if frame is None or frame < 0 or not _finite(x0, y0, x1, y1):
                cnt["discarded_malformed"] += 1
                if len(examples) < 10:
                    examples.append({"reason": "malformed", "track": tid, "frame": bx.get("frame")})
                continue
            if x1 <= x0 or y1 <= y0:
                cnt["discarded_zero_or_negative_area"] += 1
                if len(examples) < 10:
                    examples.append({"reason": "zero_or_negative_area", "track": tid, "frame": frame, "box": [x0, y0, x1, y1]})
                continue
            if image_size is not None:
                w, h = image_size
                cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
                if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
                    if cx1 <= cx0 or cy1 <= cy0:
                        cnt["discarded_outside_image"] += 1
                        continue
                    cnt["clipped_to_image"] += 1
                    x0, y0, x1, y1 = cx0, cy0, cx1, cy1
            attrs = _attr_map(bx)
            occluded = _int_or_none(bx.get("occluded")) or 0
            level = _occlusion_level(label, attrs, occluded)
            ident = (attrs.get("id") or bx.get("id") or "").strip() or None
            if label == "pedestrian":
                eval_class = "pedestrian"
            elif label == "vehicle":
                vt = (attrs.get("type") or "").strip().lower()
                veh_types[vt or "<missing>"] += 1
                eval_class = {"car": "car", "bicycle": "bike", "train": UNSUPPORTED_TRAIN}.get(vt, "unsupported_vehicle_type")
                if eval_class in (UNSUPPORTED_TRAIN, "unsupported_vehicle_type"):
                    unsupported[vt or "<missing>"] += 1
                    cnt["unsupported_vehicle_boxes"] += 1
            elif label in IGNORED_LABELS:
                eval_class = "ignored"
            else:
                eval_class = "ignored"
                cnt["unknown_label_boxes"] += 1
            if eval_class in SUPPORTED_CLASSES and exclude_fully_occluded and level == "full":
                cnt["excluded_fully_occluded"] += 1
                continue
            boxes.append(GtBox(frame, label, eval_class, (x0, y0, x1, y1), ident, occluded, level, _int_or_none(bx.get("keyframe")), tid,
                               tuple(sorted(attrs.items()))))
    cnt["valid_nonoutside_boxes"] = len(boxes)
    sup = [b for b in boxes if b.eval_class in SUPPORTED_CLASSES]
    cnt["supported_gt_boxes"] = len(sup)
    cnt["supported_partially_occluded_retained"] = sum(1 for b in sup if b.occlusion_level == "partial")
    cnt["supported_fully_occluded"] = sum(1 for b in sup if b.occlusion_level == "full")
    cnt["supported_interpolated_boxes(keyframe=0)"] = sum(1 for b in sup if b.keyframe == 0)
    anyf = sorted({b.frame for b in boxes})
    supf = sorted({b.frame for b in sup})
    return GtParse(boxes, dict(sorted(cnt.items())), dict(sorted(raw_by_label.items())), dict(sorted(non_out_by_label.items())), dict(sorted(veh_types.items())),
                   dict(sorted(unsupported.items())), _validate_ids(sup), meta, anyf, supf, (anyf[0], anyf[-1]) if anyf else None, examples)


def _validate_ids(sup: Sequence[GtBox]) -> Dict[str, Any]:
    """Identity checks on the evaluated (supported-class) boxes: missing ids, ids that change inside a CVAT track, ids shared by several tracks,
    ids with several classes, and ids that occur twice in one frame."""
    missing = [b for b in sup if b.ident is None]
    by_track: Dict[str, set] = defaultdict(set)
    by_ident_tracks: Dict[str, set] = defaultdict(set)
    by_ident_classes: Dict[str, set] = defaultdict(set)
    per_frame: Counter = Counter()
    for b in sup:
        if b.ident is None:
            continue
        by_track[b.cvat_track_id].add(b.ident)
        by_ident_tracks[b.ident].add(b.cvat_track_id)
        by_ident_classes[b.ident].add(b.eval_class)
        per_frame[(b.frame, b.ident)] += 1
    multi_id = {t: sorted(v) for t, v in by_track.items() if len(v) > 1}
    shared = {i: sorted(v) for i, v in by_ident_tracks.items() if len(v) > 1}
    multi_cls = {i: sorted(v) for i, v in by_ident_classes.items() if len(v) > 1}
    dup = {"%s@%d" % (i, f): n for (f, i), n in per_frame.items() if n > 1}
    return {"boxes_missing_id": len(missing), "missing_id_examples": [{"frame": b.frame, "track": b.cvat_track_id} for b in missing[:5]],
            "cvat_tracks_with_multiple_ids": len(multi_id), "multiple_id_examples": dict(sorted(multi_id.items())[:5]),
            "ids_spanning_multiple_cvat_tracks": len(shared), "shared_id_examples": dict(sorted(shared.items())[:5]),
            "ids_with_multiple_classes": len(multi_cls), "multiple_class_examples": dict(sorted(multi_cls.items())[:5]),
            "identity_duplicates_within_a_frame": len(dup), "duplicate_examples": dict(sorted(dup.items())[:5]),
            "distinct_identities": len(by_ident_tracks), "distinct_cvat_tracks": len(by_track),
            "ok": not (missing or multi_id or shared or multi_cls or dup)}


# ------------------------------------------------------------------------------------------------- prediction loading
def load_predictions(path: Any, *, video_id: Optional[str], tracked: bool, min_confidence: float = 0.0, frame_offset: int = 0) -> PredLoad:
    """Read a detections or tracked-agents JSONL. Rows of another ``video_id`` are skipped and counted; malformed rows are counted, not repaired."""
    p = Path(path)
    if not p.is_file():
        raise PieEvaluationError("prediction file not found: %s" % p)
    cnt: Counter = Counter()
    vids: Counter = Counter()
    by_frame: Dict[int, List[Pred]] = defaultdict(list)
    size: Optional[Tuple[int, int]] = None
    with open(p, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            if not line.strip():
                continue
            cnt["rows_total"] += 1
            try:
                r = json.loads(line)
                vid = str(r.get("video_id"))
                vids[vid] += 1
                if video_id is not None and vid != video_id:
                    cnt["rows_other_video_id"] += 1
                    continue
                frame = int(r["frame_index"])
                x0, y0, x1, y1 = (float(v) for v in r["bbox_xyxy"])
                conf = r.get("confidence")
                conf = 0.0 if conf is None else float(conf)
                cls = str(r["semantic_class"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                cnt["rows_malformed"] += 1
                continue
            if not _finite(x0, y0, x1, y1, conf) or x1 <= x0 or y1 <= y0 or frame < 0:
                cnt["rows_invalid_box"] += 1
                continue
            if size is None and isinstance(r.get("image_width"), int) and isinstance(r.get("image_height"), int):
                size = (r["image_width"], r["image_height"])
            if conf < min_confidence:
                cnt["rows_below_min_confidence"] += 1
                continue
            if tracked:
                ident = r.get("track_id")
                ident = ident if isinstance(ident, int) and not isinstance(ident, bool) else None
                pred = Pred(frame + frame_offset, cls, (x0, y0, x1, y1), conf, ident, bool(r.get("is_predicted", False)),
                            bool(r.get("tracking_validity", True)), bool(r.get("detection_validity", True)))
            else:
                did = r.get("detection_id")
                pred = Pred(frame + frame_offset, cls, (x0, y0, x1, y1), conf, did if isinstance(did, int) else None)
            by_frame[pred.frame].append(pred)
            cnt["rows_loaded"] += 1
    if cnt["rows_loaded"] == 0 and cnt["rows_total"] > 0:
        raise PieEvaluationError("no usable rows in %s for video_id %r (video ids present: %s)" % (p, video_id, dict(vids)))
    frames = sorted(by_frame)
    return PredLoad(dict(by_frame), dict(sorted(cnt.items())), (frames[0], frames[-1]) if frames else None, size, dict(sorted(vids.items())))


# --------------------------------------------------------------------------------------------------- frame sets
def evaluation_frames(gt: GtParse, policy: str, coverage: Optional[Tuple[int, int]]) -> Dict[str, Any]:
    if policy not in FRAME_POLICIES:
        raise ConfigError("frame policy must be one of %s, got %r" % (FRAME_POLICIES, policy))
    base = gt.annotated_frames_any if policy == "annotated_any" else gt.supported_gt_frames
    if coverage is None:
        return {"policy": policy, "frames": [], "dropped_outside_prediction_coverage": len(base), "coverage": None}
    lo, hi = coverage
    kept = [f for f in base if lo <= f <= hi]
    return {"policy": policy, "frames": kept, "dropped_outside_prediction_coverage": len(base) - len(kept), "coverage": [lo, hi]}


def _safe_div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


def _prf(tp: int, fp: int, fn: int, ious: Sequence[float]) -> Dict[str, Any]:
    p, r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)      # = 2PR/(P+R); 0.0 when nothing matched, null only when there is neither GT nor prediction
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "mean_iou_matched": (sum(ious) / len(ious)) if ious else None}


def _sorted_gt(gts: Iterable[GtBox]) -> List[GtBox]:
    return sorted(gts, key=lambda g: (g.eval_class, g.ident or "", g.bbox, g.cvat_track_id))


# -------------------------------------------------------------------------------------------- detection evaluation
def evaluate_detection(gt: GtParse, preds: Mapping[int, Sequence[Pred]], frames: Sequence[int], thr: float, prefer_scipy: bool = True) -> Dict[str, Any]:
    by_frame_gt: Dict[int, List[GtBox]] = defaultdict(list)
    for b in gt.boxes:
        by_frame_gt[b.frame].append(b)
    tp: Counter = Counter(); fp: Counter = Counter(); fn: Counter = Counter()
    ious: Dict[str, List[float]] = defaultdict(list)
    occ_gt: Counter = Counter(); occ_hit: Counter = Counter()
    mismatch: Counter = Counter(); fp_on_train: Counter = Counter(); fp_unmatched_other: Counter = Counter()
    out_of_scope: Counter = Counter(); unknown_cls: Counter = Counter()
    frames_with_gt = 0
    n_gt = n_pred_considered = 0
    for f in frames:
        gts = _sorted_gt(b for b in by_frame_gt.get(f, ()) if b.eval_class in SUPPORTED_CLASSES)
        trains = [b for b in by_frame_gt.get(f, ()) if b.eval_class == UNSUPPORTED_TRAIN]
        plist = sorted(preds.get(f, ()), key=lambda q: (q.cls, -q.conf, q.ident if q.ident is not None else -1, q.bbox))
        scope = [q for q in plist if q.cls in SUPPORTED_CLASSES]
        for q in plist:
            if q.cls in OUT_OF_SCOPE_PRED_CLASSES:
                out_of_scope[q.cls] += 1
            elif q.cls not in SUPPORTED_CLASSES:
                unknown_cls[q.cls] += 1
        frames_with_gt += 1 if gts else 0
        n_gt += len(gts)
        n_pred_considered += len(scope)
        pairs = match_boxes([(g.eval_class, g.bbox) for g in gts], [(q.cls, q.bbox) for q in scope], thr, prefer_scipy)
        mg, mp = {a for a, _, _ in pairs}, {b for _, b, _ in pairs}
        for a, b, v in pairs:
            tp[gts[a].eval_class] += 1
            ious[gts[a].eval_class].append(v)
        for i, g in enumerate(gts):
            occ_gt[g.occlusion_level] += 1
            if i in mg:
                occ_hit[g.occlusion_level] += 1
            else:
                fn[g.eval_class] += 1
        for j, q in enumerate(scope):
            if j not in mp:
                fp[q.cls] += 1
        # diagnostics: leftover predictions (any class, incl. truck/bus) against leftover GT (incl. unsupported trains), class-agnostic
        left_g = [(g.eval_class, g.bbox) for i, g in enumerate(gts) if i not in mg] + [(UNSUPPORTED_TRAIN, t.bbox) for t in trains]
        left_p = [(q.cls, q.bbox) for j, q in enumerate(scope) if j not in mp] + [(q.cls, q.bbox) for q in plist if q.cls not in SUPPORTED_CLASSES]
        if left_g and left_p:
            cost = [[-(_BIG + iou_xyxy(g[1], p_[1])) if iou_xyxy(g[1], p_[1]) >= thr else 0.0 for p_ in left_p] for g in left_g]
            for a, b in solve_assignment(cost, prefer_scipy):
                if iou_xyxy(left_g[a][1], left_p[b][1]) >= thr:
                    if left_g[a][0] == UNSUPPORTED_TRAIN:
                        fp_on_train[left_p[b][0]] += 1
                    else:
                        mismatch["pred:%s->gt:%s" % (left_p[b][0], left_g[a][0])] += 1
    per_class = {c: _prf(tp[c], fp[c], fn[c], ious[c]) for c in SUPPORTED_CLASSES}
    all_ious = [v for c in SUPPORTED_CLASSES for v in ious[c]]
    overall = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()), all_ious)
    strat = {lv: {"gt": occ_gt[lv], "matched": occ_hit[lv], "recall": _safe_div(occ_hit[lv], occ_gt[lv])} for lv in OCCLUSION_LEVELS if occ_gt[lv]}
    return {"iou_threshold": thr, "overall_supported_classes": overall, "per_class": per_class,
            "counts": {"evaluated_frames": len(frames), "frames_with_applicable_gt": frames_with_gt, "supported_gt_boxes": n_gt,
                       "supported_class_predictions_considered": n_pred_considered,
                       "out_of_scope_predictions_by_class(truck_bus)": dict(sorted(out_of_scope.items())),
                       "unknown_class_predictions": dict(sorted(unknown_cls.items()))},
            "recall_by_gt_occlusion_level": strat,
            "diagnostics": {"class_mismatch_pairs_iou_ge_threshold(leftovers)": dict(sorted(mismatch.items())),
                            "unmatched_predictions_overlapping_unsupported_train_gt": dict(sorted(fp_on_train.items()))}}


# --------------------------------------------------------------------------------------------- tracking evaluation
def _tracking_rows_variants(load: PredLoad) -> Dict[str, Dict[int, List[Pred]]]:
    variants: Dict[str, Dict[int, List[Pred]]] = {"primary_tracking_validity_true": defaultdict(list), "diagnostic_detection_backed_only": defaultdict(list),
                                                  "diagnostic_all_rows": defaultdict(list)}
    for f, rows in load.by_frame.items():
        for q in rows:
            variants["diagnostic_all_rows"][f].append(q)
            if q.tracking_validity:
                variants["primary_tracking_validity_true"][f].append(q)
                if not q.is_predicted:
                    variants["diagnostic_detection_backed_only"][f].append(q)
    return {k: dict(v) for k, v in variants.items()}


def _idf1_global(overlap: Mapping[Tuple[str, int], int], gt_total: int, pred_total: int, prefer_scipy: bool) -> Dict[str, Any]:
    """Global identity assignment maximising IDTP (frames matched), solved per connected component of the overlap graph."""
    adj_g: Dict[str, set] = defaultdict(set)
    adj_p: Dict[int, set] = defaultdict(set)
    for (g, p), n in overlap.items():
        if n > 0:
            adj_g[g].add(p)
            adj_p[p].add(g)
    seen_g: set = set()
    idtp = 0
    assigned = 0
    for g0 in sorted(adj_g):
        if g0 in seen_g:
            continue
        comp_g, comp_p, stack = set(), set(), [("g", g0)]
        while stack:
            kind, k = stack.pop()
            if kind == "g":
                if k in comp_g:
                    continue
                comp_g.add(k)
                stack.extend(("p", p) for p in adj_g[k])
            else:
                if k in comp_p:
                    continue
                comp_p.add(k)
                stack.extend(("g", g) for g in adj_p[k])
        seen_g |= comp_g
        gl, pl = sorted(comp_g), sorted(comp_p)
        cost = [[-overlap.get((g, p), 0) for p in pl] for g in gl]
        for a, b in solve_assignment(cost, prefer_scipy):
            n = overlap.get((gl[a], pl[b]), 0)
            if n > 0:
                idtp += n
                assigned += 1
    idfn, idfp = gt_total - idtp, pred_total - idtp
    idp, idr = _safe_div(idtp, idtp + idfp), _safe_div(idtp, idtp + idfn)
    idf1 = _safe_div(2 * idtp, 2 * idtp + idfp + idfn)
    return {"idtp": idtp, "idfp": idfp, "idfn": idfn, "id_precision": idp, "id_recall": idr, "idf1": idf1, "matched_identities_in_global_assignment": assigned}


def evaluate_tracking(gt: GtParse, rows: Mapping[int, Sequence[Pred]], frames: Sequence[int], thr: float, prefer_scipy: bool = True) -> Dict[str, Any]:
    by_frame_gt: Dict[int, List[GtBox]] = defaultdict(list)
    dropped_no_id = dropped_dup = 0
    for b in gt.boxes:
        if b.eval_class not in SUPPORTED_CLASSES:
            continue
        by_frame_gt[b.frame].append(b)
    pred_ids: set = set(); gt_ids: set = set()
    overlap: Counter = Counter()
    last_matched: Dict[str, int] = {}
    prev_frame_match: Dict[str, int] = {}
    tp = fp = fn = idsw = 0
    ious: List[float] = []
    gt_total = pred_total = 0
    gt_with_match: set = set()
    out_of_scope = no_track_id = 0
    for f in frames:
        gts_all = _sorted_gt(by_frame_gt.get(f, ()))
        gts, seen = [], set()
        for g in gts_all:
            if g.ident is None:
                dropped_no_id += 1
                continue
            if g.ident in seen:
                dropped_dup += 1
                continue
            seen.add(g.ident)
            gts.append(g)
        prs = []
        for q in sorted(rows.get(f, ()), key=lambda q: (q.cls, q.ident if q.ident is not None else -1, q.bbox)):
            if q.cls not in SUPPORTED_CLASSES:
                out_of_scope += 1
                continue
            if q.ident is None:
                no_track_id += 1
                continue
            prs.append(q)
        gt_total += len(gts)
        pred_total += len(prs)
        gt_ids.update(g.ident for g in gts)
        pred_ids.update(q.ident for q in prs)
        gtb, prb = [(g.eval_class, g.bbox) for g in gts], [(q.cls, q.bbox) for q in prs]
        for i, g in enumerate(gts):                                   # potential matches for the global IDF1 assignment
            for j, q in enumerate(prs):
                if g.eval_class == q.cls and iou_xyxy(g.bbox, q.bbox) >= thr:
                    overlap[(g.ident, q.ident)] += 1
        matched: Dict[int, int] = {}
        mp: set = set()
        for i, g in enumerate(gts):                                   # CLEAR-MOT: keep the previous correspondence while it is still valid
            pid = prev_frame_match.get(g.ident)
            if pid is None:
                continue
            for j, q in enumerate(prs):
                if j not in mp and q.ident == pid and g.eval_class == q.cls and iou_xyxy(g.bbox, q.bbox) >= thr:
                    matched[i] = j
                    mp.add(j)
                    break
        rest_g = [i for i in range(len(gts)) if i not in matched]
        rest_p = [j for j in range(len(prs)) if j not in mp]
        for a, b, _ in match_boxes([gtb[i] for i in rest_g], [prb[j] for j in rest_p], thr, prefer_scipy):
            matched[rest_g[a]] = rest_p[b]
            mp.add(rest_p[b])
        new_prev: Dict[str, int] = {}
        for i, j in sorted(matched.items()):
            g, q = gts[i], prs[j]
            if g.ident in last_matched and last_matched[g.ident] != q.ident:
                idsw += 1
            last_matched[g.ident] = q.ident
            new_prev[g.ident] = q.ident
            gt_with_match.add(g.ident)
            ious.append(iou_xyxy(g.bbox, q.bbox))
        prev_frame_match = new_prev
        tp += len(matched)
        fn += len(gts) - len(matched)
        fp += len(prs) - len(matched)
    idm = _idf1_global(overlap, gt_total, pred_total, prefer_scipy)
    if gt_total == 0:
        idm["idf1"] = None                      # undefined without ground truth (it would read 0.0 for any prediction set)
    mota = None if gt_total == 0 else 1.0 - (fn + fp + idsw) / gt_total
    notes = []
    if gt_total == 0:
        notes.append("no ground-truth boxes in the evaluated frames: MOTA, ID recall and IDF1 are null")
    if pred_total == 0:
        notes.append("no predicted rows in scope: ID precision is null")
    return {"iou_threshold": thr, **idm, "mota": mota, "motp_mean_iou": (sum(ious) / len(ious)) if ious else None,
            "false_positives": fp, "false_negatives": fn, "id_switches": idsw, "matches": tp,
            "gt_boxes": gt_total, "pred_boxes": pred_total, "gt_identities": len(gt_ids), "predicted_identities": len(pred_ids),
            "gt_identities_with_any_frame_match": len(gt_with_match),
            "excluded": {"gt_boxes_without_id": dropped_no_id, "gt_duplicate_identity_in_frame": dropped_dup,
                         "out_of_scope_class_rows(truck_bus_etc)": out_of_scope, "rows_without_track_id": no_track_id},
            "notes": notes}


# ------------------------------------------------------------------------------------------------------ driver
def sha256_file(path: Any) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _fmt(x: Any, nd: int = 4) -> str:
    return "n/a" if x is None else ("%.*f" % (nd, x) if isinstance(x, float) else str(x))


def run_evaluation(annotations: Any, detections: Any, tracked_agents: Any, output_dir: Any, *, video_id: str = "video_0004",
                   iou_thresholds: Sequence[float] = (0.50, 0.75), min_confidence: float = 0.0, frame_offset: int = 0,
                   frame_policy: str = "annotated_any", exclude_fully_occluded: bool = False, run_manifest: Optional[Any] = None,
                   prefer_scipy: bool = True) -> Dict[str, Any]:
    """Evaluate one video and write the four report files into a NEW (or empty) ``output_dir``. Returns the result dictionary."""
    thrs = sorted({float(t) for t in iou_thresholds})
    if not thrs or any(not 0.0 < t <= 1.0 for t in thrs):
        raise ConfigError("iou thresholds must be numbers in (0, 1], got %r" % (list(iou_thresholds),))
    if not 0.0 <= min_confidence <= 1.0:
        raise ConfigError("min_confidence must be in [0, 1]")
    if frame_policy not in FRAME_POLICIES:
        raise ConfigError("frame policy must be one of %s" % (FRAME_POLICIES,))
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ConfigError("output directory %s exists and is not empty; refusing to overwrite (choose a new directory)" % out)
    for pth, what in ((annotations, "annotations"), (detections, "detections"), (tracked_agents, "tracked agents")):
        if not Path(pth).is_file():
            raise PieEvaluationError("%s file not found: %s" % (what, pth))
    det = load_predictions(detections, video_id=video_id, tracked=False, min_confidence=min_confidence, frame_offset=frame_offset)
    trk = load_predictions(tracked_agents, video_id=video_id, tracked=True, min_confidence=0.0, frame_offset=frame_offset)
    size = det.image_size or trk.image_size
    gt = parse_pie_annotations(annotations, image_size=size, exclude_fully_occluded=exclude_fully_occluded)
    if gt.meta.get("name") and video_id not in gt.meta["name"] and video_id not in str(annotations):
        gt.counters["warning_video_id_not_found_in_xml_meta_or_path"] = 1
    # prediction coverage: the frame range the run actually processed (rows may be absent in frames with no detection)
    cov_lo = cov_hi = None
    ranges = [r for r in (det.frame_range, trk.frame_range) if r]
    man_info: Dict[str, Any] = {}
    if run_manifest and Path(run_manifest).is_file():
        try:
            mm = json.loads(Path(run_manifest).read_text(encoding="utf-8"))
            res = mm.get("results", {})
            man_info = {"first_frame_index": res.get("first_frame_index"), "last_frame_index": res.get("last_frame_index"),
                        "frames_seen": res.get("frames_seen"), "frames_processed": res.get("frames_processed"), "frames_skipped": res.get("frames_skipped"),
                        "detector": mm.get("components", {}).get("detector", {}).get("name"),
                        "detector_weights_sha256": mm.get("components", {}).get("detector", {}).get("parameters", {}).get("weights_sha256")}
            if isinstance(res.get("first_frame_index"), int) and isinstance(res.get("last_frame_index"), int):
                ranges.append((res["first_frame_index"] + frame_offset, res["last_frame_index"] + frame_offset))
        except (ValueError, OSError):
            man_info = {"error": "run manifest unreadable"}
    if ranges:
        cov_lo, cov_hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
    coverage = (cov_lo, cov_hi) if ranges else None
    results: Dict[str, Any] = {"detection": {}, "tracking": {}}
    for policy in FRAME_POLICIES:
        fs = evaluation_frames(gt, policy, coverage)
        key = "primary" if policy == frame_policy else "sensitivity_%s" % policy
        results["detection"][key] = {"frame_policy": policy, "frames": dict({k: v for k, v in fs.items() if k != "frames"}, count=len(fs["frames"])),
                                     "by_iou_threshold": {("%.2f" % t): evaluate_detection(gt, det.by_frame, fs["frames"], t, prefer_scipy) for t in thrs}}
        variants = _tracking_rows_variants(trk)
        results["tracking"][key] = {"frame_policy": policy, "frames": dict({k: v for k, v in fs.items() if k != "frames"}, count=len(fs["frames"])),
                                    "variants": {vn: {("%.2f" % t): evaluate_tracking(gt, rows, fs["frames"], t, prefer_scipy) for t in thrs}
                                                 for vn, rows in variants.items()}}
    # frame-alignment sensitivity (diagnostic only, never used for the reported numbers)
    prim_frames = evaluation_frames(gt, frame_policy, coverage)["frames"]
    align = {}
    for off in (-2, -1, 0, 1, 2):
        shifted = {f + off: v for f, v in det.by_frame.items()}
        r = evaluate_detection(gt, shifted, prim_frames, thrs[0], prefer_scipy)["overall_supported_classes"]
        align["%+d" % off] = {"recall": r["recall"], "precision": r["precision"], "f1": r["f1"]}
    results["detection"]["frame_alignment_sensitivity_at_lowest_iou_threshold"] = {
        "note": "prediction frames shifted by the given offset before matching; a clearly better offset other than +0 would indicate a frame-index mismatch",
        "by_offset": align}
    tracked_counts = {"rows_total": trk.counters.get("rows_total", 0), "rows_loaded": trk.counters.get("rows_loaded", 0),
                      "rows_is_predicted_true": sum(1 for rs in trk.by_frame.values() for q in rs if q.is_predicted),
                      "rows_tracking_validity_true": sum(1 for rs in trk.by_frame.values() for q in rs if q.tracking_validity),
                      "rows_tracking_validity_false": sum(1 for rs in trk.by_frame.values() for q in rs if not q.tracking_validity),
                      "rows_is_predicted_and_tracking_validity_true": sum(1 for rs in trk.by_frame.values() for q in rs if q.is_predicted and q.tracking_validity)}
    gt_info = {"counters": gt.counters, "raw_box_counts_by_label": gt.raw_counts_by_label, "nonoutside_box_counts_by_label": gt.nonoutside_counts_by_label,
               "vehicle_type_counts_nonoutside": gt.vehicle_type_counts, "unsupported_gt_counts": gt.unsupported_counts, "identity_validation": gt.id_report,
               "annotation_frame_range": list(gt.frame_range) if gt.frame_range else None, "annotated_frames_any_label": len(gt.annotated_frames_any),
               "frames_with_supported_gt": len(gt.supported_gt_frames), "xml_meta": gt.meta, "discarded_examples": gt.discarded_examples}
    pred_info = {"detections": {"counters": det.counters, "frame_range": det.frame_range, "video_ids": det.video_ids},
                 "tracked_agents": {"counters": trk.counters, "frame_range": trk.frame_range, "video_ids": trk.video_ids, **tracked_counts},
                 "image_size": size, "coverage_used": coverage, "run_manifest_info": man_info}
    limitations = [
        "One video (%s) only: nothing here supports a claim that the detector or tracker is accurate in general, robust, or production-ready." % video_id,
        "Evaluation is restricted to annotated frames inside the predicted frame range; predictions elsewhere are not scored (the annotation file does not say those frames are empty).",
        "Detection is a single operating point (all detections >= min_confidence); no confidence-ranked AP/mAP is computed.",
        "PIE has no GT for truck or bus: predictions of those classes are out of scope and a truck/bus on a GT car shows only as a class mismatch and a missed car.",
        "Trains are unsupported and neither scored nor mapped to another class; predictions on trains appear only in the diagnostics.",
        "Interpolated (keyframe=0) CVAT boxes and fully occluded boxes are retained by default; recall by occlusion level is reported separately.",
        "Frame alignment between XML frame numbers and predicted frame_index is assumed identical (offset %+d); the alignment-sensitivity block is a sanity check, not proof." % frame_offset,
        "MOTP is the mean IoU of matched pairs (higher is better); IoU matching only, no appearance or motion terms.",
        "Image-space pixels only; no world coordinates, no metric accuracy.",
    ]
    manifest = {
        "evaluator_version": EVALUATOR_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "video_id": video_id,
        "inputs": {"annotations": {"path": str(annotations), "sha256": sha256_file(annotations)}, "detections": {"path": str(detections), "sha256": sha256_file(detections)},
                   "tracked_agents": {"path": str(tracked_agents), "sha256": sha256_file(tracked_agents)},
                   "run_manifest": None if not run_manifest or not Path(run_manifest).is_file() else {"path": str(run_manifest), "sha256": sha256_file(run_manifest)}},
        "annotation_frame_range": list(gt.frame_range) if gt.frame_range else None,
        "evaluated_frame_count": {k: v["frames"]["count"] for k, v in results["detection"].items() if isinstance(v, dict) and "frames" in v},
        "iou_thresholds": thrs, "min_confidence": min_confidence, "frame_offset": frame_offset, "primary_frame_policy": frame_policy,
        "class_mapping": CLASS_MAPPING, "evaluated_classes": list(SUPPORTED_CLASSES),
        "occlusion_policy": {"outside=1": "ignored", "partial": "retained", "full": "excluded" if exclude_fully_occluded else "retained (recall stratified by occlusion level)",
                             "levels": "pedestrian 'occlusion' attribute 0/1/2 -> none/partial/full; others use the CVAT occluded flag -> occluded_unspecified"},
        "unsupported_class_policy": "train (and any other vehicle type) is counted and never evaluated or mapped; traffic light/sign/crosswalk are ignored and counted; "
                                    "truck/bus predictions are out of scope",
        "matching_method": {"detection": "class-aware per-frame one-to-one assignment (max pairs, then max IoU), IoU >= threshold inclusive",
                            "solver": "scipy.optimize.linear_sum_assignment" if prefer_scipy and _scipy_available() else "pure-python Hungarian",
                            "tracking": "CLEAR-MOT with continuity of the previous correspondence; IDF1 by global assignment on accumulated matched frames"},
        "software": {"python": sys.version.split()[0], "platform": platform.platform(), "scipy": _scipy_version(), "numpy": _numpy_version()},
        "prediction_run": man_info, "limitations": limitations,
        "no_inference": "the evaluator reads existing JSONL files only; it does not run YOLO, ByteTrack or any model",
    }
    out.mkdir(parents=True, exist_ok=True)
    det_report = {"evaluator_version": EVALUATOR_VERSION, "video_id": video_id, "ground_truth": gt_info, "predictions": dict(pred_info["detections"], image_size=size, coverage_used=coverage), "results": results["detection"], "limitations": limitations}
    trk_report = {"evaluator_version": EVALUATOR_VERSION, "video_id": video_id, "ground_truth_identity_validation": gt.id_report,
                  "predictions": pred_info["tracked_agents"], "results": results["tracking"], "limitations": limitations}
    _write_json(out / "pie_detection_metrics.json", det_report)
    _write_json(out / "pie_tracking_metrics.json", trk_report)
    _write_json(out / "pie_evaluation_manifest.json", manifest)
    (out / "pie_evaluation_summary.md").write_text(render_summary(video_id, gt_info, pred_info, results, thrs, frame_policy, limitations, manifest), encoding="utf-8")
    return {"detection": det_report, "tracking": trk_report, "manifest": manifest, "output_dir": str(out)}


def _scipy_available() -> bool:
    try:
        import scipy.optimize  # noqa: F401
        return True
    except ImportError:
        return False


def _scipy_version() -> Optional[str]:
    try:
        import scipy
        return scipy.__version__
    except ImportError:
        return None


def _numpy_version() -> Optional[str]:
    try:
        import numpy
        return numpy.__version__
    except ImportError:
        return None


def render_summary(video_id: str, gt_info: Dict[str, Any], pred_info: Dict[str, Any], results: Dict[str, Any], thrs: Sequence[float], policy: str,
                   limitations: Sequence[str], manifest: Dict[str, Any]) -> str:
    L: List[str] = []
    prim_d = results["detection"]["primary"]
    prim_t = results["tracking"]["primary"]
    c = gt_info["counters"]
    L += ["# PIE perception evaluation: %s" % video_id, "",
          "Evaluator `%s`. Predictions are the existing YOLO11n + ByteTrack outputs; nothing was re-run. Pixels only, no world coordinates." % EVALUATOR_VERSION, "",
          "**This is a single-video evaluation. It does not show that the detector or tracker is accurate, robust or production-ready.**", "",
          "## What was evaluated", "",
          "- Annotation scope: official PIE CVAT XML; annotated frame range %s; %d frames carry any annotation, %d carry supported ground truth." % (
              gt_info["annotation_frame_range"], gt_info["annotated_frames_any_label"], gt_info["frames_with_supported_gt"]),
          "- Evaluated frames (policy `%s`): %d, restricted to the predicted frame range %s. Predictions in other frames are **not** counted as false positives." % (
              policy, prim_d["frames"]["count"], pred_info["coverage_used"]),
          "- Class mapping: pedestrian -> pedestrian; vehicle car -> car; vehicle bicycle -> bike; vehicle train -> `unsupported_train` (not evaluated). Traffic light, sign and crosswalk are ignored.",
          "- Identity: the per-box PIE `id` attribute (not the CVAT track id). Occlusion: `outside=1` ignored, partially occluded retained, fully occluded %s." % (
              "excluded" if "excluded" in manifest["occlusion_policy"]["full"] else "retained, with recall stratified by occlusion level"),
          "- GT box counts (all boxes by label): %s; non-outside: %s." % (gt_info["raw_box_counts_by_label"], gt_info["nonoutside_box_counts_by_label"]),
          "- Unsupported GT (counted, not scored): %s. Discarded boxes: malformed %d, zero area %d, outside the image %d; clipped to the image %d." % (
              gt_info["unsupported_gt_counts"], c.get("discarded_malformed", 0), c.get("discarded_zero_or_negative_area", 0), c.get("discarded_outside_image", 0), c.get("clipped_to_image", 0)),
          "- Supported GT boxes: %d (partially occluded %d, fully occluded %d, interpolated keyframe=0 %d)." % (
              c.get("supported_gt_boxes", 0), c.get("supported_partially_occluded_retained", 0), c.get("supported_fully_occluded", 0), c.get("supported_interpolated_boxes(keyframe=0)", 0)),
          "- GT identity validation: %s" % ("all consistent" if gt_info["identity_validation"]["ok"] else "ISSUES FOUND, see `pie_tracking_metrics.json` (`ground_truth_identity_validation`)"), "",
          "## Detection results (class-aware, one-to-one, fixed operating point, min confidence %s)" % manifest["min_confidence"], "",
          "| IoU | class | TP | FP | FN | precision | recall | F1 | mean IoU |", "|---|---|---|---|---|---|---|---|---|"]
    for t in thrs:
        d = prim_d["by_iou_threshold"]["%.2f" % t]
        for name, m in [("overall", d["overall_supported_classes"])] + [(k, v) for k, v in d["per_class"].items()]:
            L.append("| %.2f | %s | %d | %d | %d | %s | %s | %s | %s |" % (t, name, m["tp"], m["fp"], m["fn"], _fmt(m["precision"]), _fmt(m["recall"]), _fmt(m["f1"]), _fmt(m["mean_iou_matched"])))
    d0 = prim_d["by_iou_threshold"]["%.2f" % thrs[0]]
    L += ["", "Counts at IoU %.2f: %s." % (thrs[0], d0["counts"]),
          "Recall by GT occlusion level: %s." % d0["recall_by_gt_occlusion_level"],
          "Class-mismatch pairs among leftovers: %s. Unmatched predictions overlapping unsupported train GT: %s." % (
              d0["diagnostics"]["class_mismatch_pairs_iou_ge_threshold(leftovers)"], d0["diagnostics"]["unmatched_predictions_overlapping_unsupported_train_gt"]),
          "Frame-alignment sanity check (F1 at IoU %.2f by prediction-frame offset): %s." % (
              thrs[0], {k: _fmt(v["f1"]) for k, v in results["detection"]["frame_alignment_sensitivity_at_lowest_iou_threshold"]["by_offset"].items()}),
          "No mAP is reported: only a single confidence operating point is evaluated.", "",
          "## Tracking results (primary rows: `tracking_validity` true)", "",
          "| IoU | IDTP | IDFP | IDFN | IDP | IDR | IDF1 | MOTA | MOTP (mean IoU) | FP | FN | ID switches | GT ids | pred ids | matched ids |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in thrs:
        m = prim_t["variants"]["primary_tracking_validity_true"]["%.2f" % t]
        L.append("| %.2f | %d | %d | %d | %s | %s | %s | %s | %s | %d | %d | %d | %d | %d | %d |" % (
            t, m["idtp"], m["idfp"], m["idfn"], _fmt(m["id_precision"]), _fmt(m["id_recall"]), _fmt(m["idf1"]), _fmt(m["mota"]), _fmt(m["motp_mean_iou"]),
            m["false_positives"], m["false_negatives"], m["id_switches"], m["gt_identities"], m["predicted_identities"], m["matched_identities_in_global_assignment"]))
    tr = pred_info["tracked_agents"]
    L += ["", "Tracked rows: %d loaded; `is_predicted` true %d; `tracking_validity` true %d / false %d. Diagnostic variants (detection-backed only, all rows) are in `pie_tracking_metrics.json`." % (
        tr["rows_loaded"], tr["rows_is_predicted_true"], tr["rows_tracking_validity_true"], tr["rows_tracking_validity_false"]),
        "ID switches follow each GT identity's last matched predicted identity and persist across unmatched frames and annotation gaps. IDF1 uses one global identity assignment.", "",
        "## Limitations", ""] + ["- %s" % x for x in limitations]
    L += ["", "## Task 4 exit criteria (as verifiable by this evaluator)", "",
          "1. Official PIE XML parsed: yes (%d valid non-outside boxes)." % c.get("valid_nonoutside_boxes", 0),
          "2. Detection metrics generated from the actual `detections.jsonl`: yes.",
          "3. Tracking metrics generated from the actual `tracked_agents.jsonl`: yes.",
          "4. Reports and manifest saved: yes.",
          "5. Focused tests: run separately with pytest (not verifiable inside this script).",
          "6. Policy and limitations documented: yes (above and in `pie_evaluation_manifest.json`).", ""]
    return "\n".join(L)
