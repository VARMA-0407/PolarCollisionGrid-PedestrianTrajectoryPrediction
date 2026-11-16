"""PIE dataset inventory: videos per recording set, sizes, SHA-256, duplicates, annotation archives and extracted annotation XML.

Nothing is assumed: the six set directories, the number of clips and the annotation files are checked on disk. The reference clip
counts below come from the PIE documentation and are used only for an informational comparison, never as a substitute for a check.
SHA-256 hashing of large videos is cached (keyed by relative path, size and modification time) and the cache is written after
every file, so an interrupted inventory resumes without re-hashing completed videos.
"""

import re
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .common import StageError, read_json, sha256_file, utc_now, video_key, write_json

INVENTORY_VERSION = "multivideo-pie-inventory-1.0.0"
PIE_SETS = ("set01", "set02", "set03", "set04", "set05", "set06")
REFERENCE_CLIP_COUNTS = {"set01": 4, "set02": 3, "set03": 19, "set04": 16, "set05": 2, "set06": 9}   # informational (PIE documentation); not verified here
_VIDEO = re.compile(r"^(video_\d{4})")
_SET = re.compile(r"^set\d{2}$")
ANN_KINDS = {"annt": "_annt.xml", "attributes": "_attributes.xml", "obd": "_obd.xml"}


def _hash_cached(p: Path, rel: str, cache: Dict[str, Any], cache_path: Optional[Path]) -> str:
    st = p.stat()
    c = cache.get(rel)
    if c and c.get("bytes") == st.st_size and c.get("mtime_ns") == st.st_mtime_ns and c.get("sha256"):
        return c["sha256"]
    h = sha256_file(p)
    cache[rel] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h}
    if cache_path is not None:
        write_json(cache_path, cache)
    return h


def scan_annotations(*roots) -> Dict[str, Any]:
    """Find PIE annotation XML (``setNN/video_NNNN_{annt,attributes,obd}.xml``) under the given roots (recursive)."""
    found: Dict[str, Dict[str, str]] = {}
    unparsed: List[str] = []
    for root in roots:
        if root is None or not Path(root).is_dir():
            continue
        for p in sorted(Path(root).rglob("*.xml")):
            kind = next((k for k, suf in ANN_KINDS.items() if p.name.endswith(suf)), None)
            m = _VIDEO.match(p.name)
            if kind is None or m is None or not _SET.match(p.parent.name):
                unparsed.append(str(p))
                continue
            key = video_key(p.parent.name, m.group(1))
            found.setdefault(key, {})
            if kind in found[key] and found[key][kind] != str(p):
                found[key].setdefault("duplicates", []).append(str(p))
            else:
                found[key][kind] = str(p)
    return {"by_video": found, "unparsed_xml": unparsed[:50], "unparsed_xml_count": len(unparsed)}


def extract_annotation_archives(archives: List[Path], dest_root) -> Dict[str, Any]:
    """Extract annotation ZIP archives into NEW sub-directories of ``dest_root`` (existing directories are reused, never overwritten).
    Member paths are checked (no absolute paths, no ``..``)."""
    out = {}
    for a in archives:
        dest = Path(dest_root) / a.stem
        if dest.exists() and any(dest.iterdir()):
            out[str(a)] = {"dest": str(dest), "extracted": False, "reason": "already extracted"}
            continue
        with zipfile.ZipFile(a) as zf:
            for n in zf.namelist():
                if n.startswith("/") or ".." in Path(n).parts:
                    raise StageError("UNSAFE_ARCHIVE", "unsafe member %r in %s" % (n, a))
            dest.mkdir(parents=True, exist_ok=True)
            zf.extractall(dest)
            out[str(a)] = {"dest": str(dest), "extracted": True, "members": len(zf.namelist())}
    return out


def inventory_pie(pie_root, *, clips_dirname: str = "PIE_clips", annotations_dirname: str = "annotations", extra_annotation_roots=(),
                  hash_videos: bool = True, hash_cache_path=None, require_annotations: bool = True,
                  progress: Callable[[str], None] = print) -> Dict[str, Any]:
    root = Path(pie_root)
    clips, ann_root = root / clips_dirname, root / annotations_dirname
    problems: List[str] = []
    warnings: List[str] = []
    if not clips.is_dir():
        raise StageError("PIE_CLIPS_MISSING", "%s does not exist: place the PIE videos in %s/setNN/video_NNNN.mp4" % (clips, clips))
    cache_path = Path(hash_cache_path) if hash_cache_path else None
    cache = read_json(cache_path) if cache_path and cache_path.is_file() else {}
    sets: Dict[str, Any] = {}
    videos: List[Dict[str, Any]] = []
    by_key: Dict[str, List[str]] = {}
    non_mp4 = []
    for s in PIE_SETS:
        sd = clips / s
        if not sd.is_dir():
            sets[s] = {"exists": False, "n_videos": 0, "bytes": 0}
            problems.append("set directory missing: %s" % sd)
            continue
        mp4s = sorted(p for p in sd.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4")
        non_mp4 += [str(p) for p in sd.rglob("*") if p.is_file() and p.suffix.lower() != ".mp4"]
        total = 0
        for i, p in enumerate(mp4s, 1):
            rel = p.relative_to(root).as_posix()
            m = _VIDEO.match(p.stem)
            stem = m.group(1) if m else p.stem
            if not m:
                warnings.append("unexpected video file name (expected video_NNNN.mp4): %s" % rel)
            if p.parent != sd:
                warnings.append("video not directly inside its set directory: %s" % rel)
            size = p.stat().st_size
            total += size
            if size == 0:
                problems.append("empty video file: %s" % rel)
            key = video_key(s, stem)
            by_key.setdefault(key, []).append(rel)
            h = None
            if hash_videos:
                progress("  hashing %s (%d/%d in %s, %.2f GB)" % (rel, i, len(mp4s), s, size / 1e9))
                h = _hash_cached(p, rel, cache, cache_path)
            videos.append({"set": s, "video": stem, "key": key, "relpath": rel, "path": str(p), "bytes": size, "sha256": h})
        sets[s] = {"exists": True, "n_videos": len(mp4s), "bytes": total, "reference_count_pie_documentation": REFERENCE_CLIP_COUNTS[s]}
        if len(mp4s) == 0:
            problems.append("set directory contains no MP4 file: %s" % sd)
        elif len(mp4s) != REFERENCE_CLIP_COUNTS[s]:
            warnings.append("%s has %d videos; the PIE documentation lists %d (informational: the actual files are used)" % (s, len(mp4s), REFERENCE_CLIP_COUNTS[s]))
    dup_keys = {k: v for k, v in by_key.items() if len(v) > 1}
    if dup_keys:
        problems.append("duplicate video identities (same set and video number): %s" % sorted(dup_keys))
    by_hash: Dict[str, List[str]] = {}
    for v in videos:
        if v["sha256"]:
            by_hash.setdefault(v["sha256"], []).append(v["relpath"])
    dup_hash = {h: v for h, v in by_hash.items() if len(v) > 1}
    if dup_hash:
        problems.append("byte-identical duplicate videos: %s" % list(dup_hash.values()))
    # ---- annotations
    archives = sorted(p for p in ann_root.rglob("*.zip")) if ann_root.is_dir() else []
    arch = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in archives]
    ann = scan_annotations(ann_root, *extra_annotation_roots)
    keys = {v["key"] for v in videos}
    without = sorted(k for k in keys if "annt" not in ann["by_video"].get(k, {}))
    orphans = sorted(k for k in ann["by_video"] if k not in keys)
    if not ann_root.is_dir():
        (problems if require_annotations else warnings).append("annotation directory missing: %s" % ann_root)
    if require_annotations and videos and len(without) == len(keys):
        problems.append("no *_annt.xml annotation file was found for any video (extract annotations.zip into %s)" % ann_root)
    if without:
        warnings.append("%d video(s) without a *_annt.xml annotation file: %s" % (len(without), without[:20]))
    for v in videos:
        v["annotations"] = {k: ann["by_video"].get(v["key"], {}).get(k) for k in ANN_KINDS}
    total_bytes = sum(v["bytes"] for v in videos)
    status = "MISSING_REQUIRED_DATA" if problems else ("COMPLETE_WITH_WARNINGS" if warnings else "COMPLETE")
    return {"inventory_version": INVENTORY_VERSION, "created_utc": utc_now(), "pie_root": str(root), "clips_dir": str(clips), "annotations_dir": str(ann_root),
            "status": status, "problems": problems, "warnings": warnings, "sets": sets, "n_videos": len(videos), "total_bytes": total_bytes,
            "total_gb": round(total_bytes / 1e9, 3), "videos": videos, "videos_hashed": bool(hash_videos), "duplicate_keys": dup_keys,
            "duplicate_content": dup_hash, "non_mp4_files": non_mp4[:50], "non_mp4_file_count": len(non_mp4),
            "annotations": {"archives": arch, "xml_by_video": ann["by_video"], "unparsed_xml_count": ann["unparsed_xml_count"], "videos_without_annt_xml": without,
                            "annotations_without_video": orphans, "n_videos_with_annt_xml": len(keys) - len(without)},
            "reference_clip_counts_note": "PIE documentation lists 53 clips (set01 4, set02 3, set03 19, set04 16, set05 2, set06 9); informational only"}


__all__ = ["inventory_pie", "scan_annotations", "extract_annotation_archives", "PIE_SETS", "REFERENCE_CLIP_COUNTS", "INVENTORY_VERSION"]
