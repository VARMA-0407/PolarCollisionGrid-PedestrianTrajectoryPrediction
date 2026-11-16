"""Shared helpers: hashing, atomic JSON, per-stage completion markers and verified publication of a locally computed directory.

Resume model: every per-video stage writes its outputs into a local scratch directory, copies them to the persistent location
(Google Drive) through a ``.partial-<stamp>`` directory, verifies every copied file by SHA-256, renames the directory into place
and only then writes ``_STAGE_COMPLETE.json``. A directory without a valid marker is never treated as complete; it is moved
aside to a quarantine folder (never deleted) before the stage is recomputed. A marker whose configuration hash differs from the
current configuration is reported as STALE and is never silently reused or overwritten.
"""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

MARKER = "_STAGE_COMPLETE.json"
MARKER_VERSION = "multivideo-stage-marker-1.0.0"


class StageError(RuntimeError):
    """Stable-code error raised by the multi-video orchestration (``code`` is machine readable)."""

    def __init__(self, code: str, message: str):
        super().__init__("%s: %s" % (code, message))
        self.code = code


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def write_json(path, obj: Any) -> Path:
    """Atomic JSON write (temporary file + ``os.replace``): a disconnect never leaves a half-written JSON file behind."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str, allow_nan=False), encoding="utf-8")
    os.replace(tmp, p)
    return p


def read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def video_key(set_id: str, video_stem: str) -> str:
    """Unique video identity across recording sets (PIE reuses ``video_0001`` in several sets)."""
    return "%s_%s" % (set_id, video_stem)


def file_records(root, *, hash_files: bool = True, skip: Iterable[str] = (MARKER,)) -> Dict[str, Dict[str, Any]]:
    root = Path(root)
    skip = set(skip)
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in skip and not p.name.endswith(".tmp"):
            rel = p.relative_to(root).as_posix()
            out[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p) if hash_files else None}
    return out


def write_marker(stage_dir, *, stage: str, key: str, config_sha256: str, upstream: Optional[Mapping[str, Any]] = None,
                 detail: Optional[Mapping[str, Any]] = None, files: Optional[Mapping[str, Any]] = None, git_commit: Optional[str] = None) -> Dict[str, Any]:
    d = Path(stage_dir)
    rec = {"marker_version": MARKER_VERSION, "stage": stage, "key": key, "config_sha256": config_sha256, "upstream": dict(upstream or {}),
           "detail": dict(detail or {}), "files": dict(files) if files is not None else file_records(d), "git_commit": git_commit, "completed_utc": utc_now()}
    write_json(d / MARKER, rec)
    return rec


def check_marker(stage_dir, *, stage: str, config_sha256: str, verify: str = "size") -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return (state, marker). state is one of COMPLETE, ABSENT, INCOMPLETE (directory without valid marker), STALE (configuration
    differs), CORRUPT (a recorded file is missing or differs). ``verify`` is ``size`` (fast, used for resume), ``sha256`` (full) or ``none``."""
    d = Path(stage_dir)
    if not d.exists():
        return "ABSENT", None
    mp = d / MARKER
    if not mp.is_file():
        return "INCOMPLETE", None
    try:
        m = read_json(mp)
    except (OSError, ValueError):
        return "INCOMPLETE", None
    if m.get("marker_version") != MARKER_VERSION or m.get("stage") != stage:
        return "INCOMPLETE", m
    if m.get("config_sha256") != config_sha256:
        return "STALE", m
    if verify != "none":
        for rel, r in (m.get("files") or {}).items():
            p = d / rel
            if not p.is_file() or p.stat().st_size != r.get("bytes"):
                return "CORRUPT", m
            if verify == "sha256" and r.get("sha256") and sha256_file(p) != r["sha256"]:
                return "CORRUPT", m
    return "COMPLETE", m


def move_aside(path, quarantine_root) -> Optional[Path]:
    """Move an incomplete / corrupt directory into the quarantine folder (never deletes anything)."""
    p = Path(path)
    if not p.exists():
        return None
    q = Path(quarantine_root)
    q.mkdir(parents=True, exist_ok=True)
    target = q / ("%s__%s__%s" % (p.parent.name, p.name, utc_stamp()))
    i = 0
    while target.exists():
        i += 1
        target = q / ("%s__%s__%s_%d" % (p.parent.name, p.name, utc_stamp(), i))
    shutil.move(str(p), str(target))
    return target


def publish_dir(local_dir, final_dir, quarantine_root) -> Dict[str, Dict[str, Any]]:
    """Copy ``local_dir`` to ``final_dir`` with SHA-256 verification of every file; returns the verified file records.

    An existing ``final_dir`` is moved to quarantine first (callers only publish when no COMPLETE marker exists)."""
    src, dst = Path(local_dir), Path(final_dir)
    recs = file_records(src)
    if dst.exists():
        move_aside(dst, quarantine_root)
    partial = dst.parent / ("%s.partial-%s" % (dst.name, utc_stamp()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, partial)
    bad = [rel for rel, r in recs.items() if sha256_file(partial / rel) != r["sha256"]]
    if bad:
        move_aside(partial, quarantine_root)
        raise StageError("PUBLISH_HASH_MISMATCH", "copy of %s differs from the local result for %s" % (src, bad[:5]))
    os.replace(partial, dst)
    return recs


def copy_dir_verified(src, dst) -> None:
    """Restore a published directory into local scratch (used when a later sub-stage must read an earlier one at its scratch path)."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for rel, r in file_records(src).items():
        if sha256_file(dst / rel) != r["sha256"]:
            raise StageError("RESTORE_HASH_MISMATCH", "restored copy of %s differs" % (src / rel))


def git_state(repo) -> Dict[str, Any]:
    def run(*args):
        try:
            r = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=60)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    status = run("status", "--porcelain")
    return {"branch": run("rev-parse", "--abbrev-ref", "HEAD"), "commit": run("rev-parse", "HEAD"), "describe": run("describe", "--always", "--dirty"),
            "working_tree_clean": (status == "") if status is not None else None, "status_porcelain": (status or "")[:4000] if status is not None else None,
            "remote_origin": run("config", "--get", "remote.origin.url")}


__all__ = ["MARKER", "StageError", "utc_now", "utc_stamp", "sha256_file", "canonical_sha256", "write_json", "read_json", "video_key", "file_records",
           "write_marker", "check_marker", "move_aside", "publish_dir", "copy_dir_verified", "git_state"]
