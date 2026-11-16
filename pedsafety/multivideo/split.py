"""Official PIE split discovery and the frozen RECORDING-LEVEL split manifest.

Unit of splitting = the PIE recording set (``set01`` ... ``set06``). The clips of one set are consecutive chunks of one recording,
so every clip of a set goes to the same partition; frames, windows and clips are never split randomly.

The official split is read from the PIE toolkit source (``pie_data.py``, shipped with the PIE annotations repository) by parsing its
Python syntax tree with ``ast.literal_eval`` on the dictionary literal whose keys are ``train`` / ``val`` / ``test``; the file is never
executed. If it cannot be found, the stage is BLOCKED unless the user explicitly accepts the reference copy below, which is then
labelled ``REFERENCE_NOT_VERIFIED_AGAINST_SOURCE`` in every manifest.

The official split has no calibration partition. Two schemes are offered (both recording-level):
* ``PIE_OFFICIAL``: train / validation / test exactly as published; calibration stays empty, so calibration remains blocked;
* ``PIE_OFFICIAL_CALIBRATION_FROM_VAL``: a documented DEVIATION that moves whole sets from the official validation sets to a calibration
  partition (default: the last official validation set). This does not create verified conflict labels; Phase 7 calibration stays blocked.

The manifest uses the Phase 9 ``phase9-group-split-1.0.0`` layout so that ``pedsafety.phase9.evaluation`` can assess it unchanged. Its
annotation / identity-mapping / timestamp-alignment / licence fields are filled honestly (``verified: false``), so the Phase 9 gate keeps
independent evaluation BLOCKED until those are actually verified.
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .common import StageError, canonical_sha256, read_json, sha256_file, utc_now, write_json
from . import TARGET_SOURCE

SPLIT_VERSION = "phase9-group-split-1.0.0"
SCHEMES = ("PIE_OFFICIAL", "PIE_OFFICIAL_CALIBRATION_FROM_VAL")
PIE_TOOLKIT_REFERENCE_SPLIT = {"train": ["set01", "set02", "set04"], "val": ["set05", "set06"], "test": ["set03"]}
REFERENCE_SOURCE = "aras62/PIE utilities/pie_data.py (image_set_nums); transcribed reference copy, NOT verified against a local source file"
_SET = re.compile(r"^set\d{2}$")
_CORE_KEYS = ("unit", "scheme", "partitions", "groups_videos", "official_split", "purge_gap_frames", "target_source")


def find_pie_data_py(search_roots: Iterable) -> List[Path]:
    out = []
    for r in search_roots:
        if r and Path(r).is_dir():
            out += sorted(Path(r).rglob("pie_data.py"))
        elif r and Path(r).is_file() and Path(r).name == "pie_data.py":
            out.append(Path(r))
    return out


def parse_official_split(path) -> Dict[str, List[str]]:
    """Return the ``train``/``val``/``test`` set lists found as a dictionary literal in ``pie_data.py`` (AST only; nothing is executed)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if not {"train", "val", "test"} <= set(keys):
            continue
        try:
            d = ast.literal_eval(node)
        except ValueError:
            continue
        if all(isinstance(d.get(k), list) and d[k] and all(isinstance(s, str) and _SET.match(s) for s in d[k]) for k in ("train", "val", "test")):
            hits.append({k: list(d[k]) for k in ("train", "val", "test")})
    uniq = {canonical_sha256(h): h for h in hits}
    if not uniq:
        raise StageError("OFFICIAL_SPLIT_NOT_FOUND", "no train/val/test set-list dictionary literal found in %s" % path)
    if len(uniq) > 1:
        raise StageError("OFFICIAL_SPLIT_AMBIGUOUS", "several different train/val/test dictionaries in %s: %s" % (path, list(uniq.values())))
    return next(iter(uniq.values()))


def resolve_official_split(search_roots: Iterable, *, allow_unverified_reference: bool = False) -> Dict[str, Any]:
    files = find_pie_data_py(search_roots)
    parsed = []
    for f in files:
        try:
            parsed.append({"path": str(f), "sha256": sha256_file(f), "split": parse_official_split(f)})
        except StageError as exc:
            parsed.append({"path": str(f), "sha256": sha256_file(f), "error": str(exc)})
    good = [p for p in parsed if "split" in p]
    if good:
        splits = {canonical_sha256(p["split"]) for p in good}
        if len(splits) > 1:
            raise StageError("OFFICIAL_SPLIT_CONFLICT", "different official splits in %s" % [p["path"] for p in good])
        return {"status": "VERIFIED_FROM_SOURCE", "split": good[0]["split"], "source_files": parsed, "method": "ast.literal_eval of the dictionary literal; file not executed",
                "matches_transcribed_reference": good[0]["split"] == PIE_TOOLKIT_REFERENCE_SPLIT}
    if allow_unverified_reference:
        return {"status": "REFERENCE_NOT_VERIFIED_AGAINST_SOURCE", "split": PIE_TOOLKIT_REFERENCE_SPLIT, "source_files": parsed, "method": REFERENCE_SOURCE,
                "warning": "pie_data.py was not found; the transcribed reference was accepted explicitly by the user and is unverified"}
    raise StageError("OFFICIAL_SPLIT_UNVERIFIED", "pie_data.py (PIE toolkit) was not found under %s; copy the PIE repository's utilities/pie_data.py into the "
                     "raw PIE folder, or set ALLOW_UNVERIFIED_SPLIT_REFERENCE=True to use the transcribed reference (recorded as unverified)" % [str(r) for r in search_roots])


def build_split_manifest(inventory: Dict[str, Any], official: Dict[str, Any], *, scheme: str = "PIE_OFFICIAL",
                         calibration_sets: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    if scheme not in SCHEMES:
        raise StageError("SPLIT_SCHEME", "scheme must be one of %s" % (SCHEMES,))
    off = official["split"]
    parts = {"train": list(off["train"]), "calibration": [], "validation": list(off["val"]), "test": list(off["test"])}
    deviations = []
    if scheme == "PIE_OFFICIAL_CALIBRATION_FROM_VAL":
        cal = list(calibration_sets) if calibration_sets else [off["val"][-1]]
        bad = [s for s in cal if s not in off["val"]]
        if bad or len(cal) >= len(off["val"]):
            raise StageError("CALIBRATION_SETS", "calibration sets must be a strict subset of the official validation sets %s, got %s" % (off["val"], cal))
        parts["calibration"] = cal
        parts["validation"] = [s for s in off["val"] if s not in cal]
        deviations.append("calibration partition %s carved from the official validation sets (recording-level); official validation reduced to %s" % (cal, parts["validation"]))
    present: Dict[str, List[str]] = {}
    for v in inventory["videos"]:
        present.setdefault(v["set"], []).append(v["key"])
    missing_sets = sorted(s for p in parts.values() for s in p if s not in present)
    for p in parts:
        parts[p] = [s for s in parts[p] if s in present]
    groups = {}
    for p, sets in parts.items():
        for s in sets:
            groups[s] = {"partition": p, "videos": sorted(present[s]), "recording_unit": "PIE recording set %s (all clips of the set are chunks of one recording)" % s,
                         "annotation": {"verified": False, "name": "PIE annotations", "type": "official_dataset_annotation", "version": None, "sha256": None,
                                        "protocol_reference": None, "note": "annotation XML is inventoried but not used as trajectory ground truth (ANNOTATED_TARGET is not implemented)"},
                         "identity_mapping": {"verified": False, "reason": "no verified mapping between PIE pedestrian ids and perceived track ids"},
                         "timestamp_alignment": {"verified": False, "reason": "frame alignment between annotations and perception outputs is not verified"},
                         "license": {"name": None, "archive_sha256": None, "acquisition_record": None}}
    core = {"unit": "recording_set", "scheme": scheme, "partitions": parts, "groups_videos": {s: groups[s]["videos"] for s in sorted(groups)},
            "official_split": {"status": official["status"], "split": off}, "purge_gap_frames": 0, "target_source": TARGET_SOURCE}
    content_sha = canonical_sha256({k: core[k] for k in _CORE_KEYS})
    m = {"manifest_version": SPLIT_VERSION, "split_id": "pie_recsplit_%s" % content_sha[:12], "content_sha256": content_sha, "frozen": False, "frozen_utc": None,
         "unit": core["unit"], "scheme": scheme, "partitions": parts, "groups": groups, "purge_gap_frames": 0,
         "purge_rationale": "partitions are whole recordings; no recording is divided, so no chronological purge gap is needed between partitions",
         "official_split": {k: official[k] for k in official if k != "source_files"}, "official_split_source_files": official.get("source_files", []),
         "deviations_from_official": deviations, "sets_in_official_split_without_videos": missing_sets, "target_source": TARGET_SOURCE,
         "methodology": ["the split is created and frozen before any window is generated or any statistic is fitted",
                         "scaler statistics are fitted on train-partition windows only",
                         "no frame, window or clip is assigned randomly; a recording set is never divided"]}
    return m


def check_split_integrity(manifest: Dict[str, Any], inventory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    probs: List[str] = []
    parts, groups = manifest["partitions"], manifest["groups"]
    seen: Dict[str, str] = {}
    for p, sets in parts.items():
        for s in sets:
            if s in seen:
                probs.append("set %s is in both %s and %s" % (s, seen[s], p))
            seen[s] = p
            if groups.get(s, {}).get("partition") != p:
                probs.append("group record of %s disagrees with partition %s" % (s, p))
    owner: Dict[str, str] = {}
    for s, g in groups.items():
        for v in g["videos"]:
            if not v.startswith(s + "_"):
                probs.append("video %s listed under set %s (cross-recording leak)" % (v, s))
            if v in owner:
                probs.append("video %s in groups %s and %s" % (v, owner[v], s))
            owner[v] = s
    for p in ("train", "validation", "test"):
        if not parts.get(p):
            probs.append("partition %s has no recording set with videos" % p)
    if inventory is not None:
        inv = {v["key"] for v in inventory["videos"]}
        unassigned = sorted(inv - set(owner))
        if unassigned:
            probs.append("inventoried videos not assigned to any partition: %s" % unassigned[:20])
        ghost = sorted(set(owner) - inv)
        if ghost:
            probs.append("split lists videos that are not in the inventory: %s" % ghost[:20])
    try:
        from ..phase9.evaluation import validate_split_manifest
        readiness = validate_split_manifest(dict(manifest, frozen=True, frozen_utc=manifest.get("frozen_utc") or "pending"))
    except Exception as exc:                                                                         # noqa: BLE001
        readiness = ["phase9 validator unavailable: %s" % exc]
    return {"ok": not probs, "problems": probs, "videos_per_partition": {p: sum(len(groups[s]["videos"]) for s in sets) for p, sets in parts.items()},
            "phase9_independent_evaluation_problems": readiness,
            "phase9_independent_evaluation_ready": not readiness}


def freeze_split(manifest: Dict[str, Any], splits_dir) -> Dict[str, Any]:
    """Write the manifest once, as ``<splits_dir>/<split_id>.json``. Re-freezing identical content returns the existing frozen file;
    different content under the same id is impossible (id = content hash) and an existing file is never overwritten."""
    integ = check_split_integrity(manifest)
    if not integ["ok"]:
        raise StageError("SPLIT_INTEGRITY", "; ".join(integ["problems"]))
    path = Path(splits_dir) / ("%s.json" % manifest["split_id"])
    if path.is_file():
        old = read_json(path)
        if old.get("content_sha256") != manifest["content_sha256"] or not old.get("frozen"):
            raise StageError("SPLIT_CONFLICT", "%s exists with different content; refusing to overwrite" % path)
        return {"path": path, "manifest": old, "reused": True, "file_sha256": sha256_file(path)}
    m = dict(manifest, frozen=True, frozen_utc=utc_now())
    m["phase9_independent_evaluation_problems"] = integ["phase9_independent_evaluation_problems"]
    write_json(path, m)
    return {"path": path, "manifest": m, "reused": False, "file_sha256": sha256_file(path)}


def load_frozen_split(path) -> Dict[str, Any]:
    m = read_json(path)
    if not m.get("frozen") or m.get("manifest_version") != SPLIT_VERSION:
        raise StageError("SPLIT_NOT_FROZEN", "%s is not a frozen %s manifest" % (path, SPLIT_VERSION))
    core = {"unit": m["unit"], "scheme": m["scheme"], "partitions": m["partitions"], "groups_videos": {s: m["groups"][s]["videos"] for s in sorted(m["groups"])},
            "official_split": {"status": m["official_split"]["status"], "split": m["official_split"]["split"]}, "purge_gap_frames": m["purge_gap_frames"],
            "target_source": m["target_source"]}
    if canonical_sha256(core) != m["content_sha256"]:
        raise StageError("SPLIT_TAMPERED", "%s content hash does not match its recorded content_sha256" % path)
    return m


def video_partitions(manifest: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """``{video_key: {"set": set_id, "partition": partition}}``."""
    return {v: {"set": s, "partition": g["partition"]} for s, g in manifest["groups"].items() for v in g["videos"]}


__all__ = ["resolve_official_split", "parse_official_split", "find_pie_data_py", "build_split_manifest", "check_split_integrity", "freeze_split", "load_frozen_split",
           "video_partitions", "PIE_TOOLKIT_REFERENCE_SPLIT", "SCHEMES", "SPLIT_VERSION"]
