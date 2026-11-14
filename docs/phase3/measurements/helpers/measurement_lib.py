"""Shared, dependency-light helpers for the PCG measurement notebook (stdlib only; numpy optional).

Nothing here trains a model or touches an experiment artifact. Everything that writes uses exclusive creation
(`make_fresh_dir`, `write_new_json`) so an existing file or directory is never overwritten.
"""

import csv
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LIB_VERSION = "pcg-measurement-lib-1.0"


# ------------------------------------------------------------------ hashing / file records
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_record(path: str, label: str = "", expected_sha256: Optional[str] = None, expected_source: Optional[str] = None) -> Dict[str, Any]:
    """Read-only description of one file. ``matches_previous_hash`` is None when no previous hash is known."""
    rec: Dict[str, Any] = {"label": label, "path": path, "exists": os.path.isfile(path)}
    try:
        rec["resolved_path"] = os.path.realpath(path)
    except OSError:
        rec["resolved_path"] = None
    if rec["exists"]:
        st = os.stat(path)
        rec["size_bytes"] = st.st_size
        rec["modified_utc"] = datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rec["sha256"] = sha256_file(path)
    else:
        rec["size_bytes"] = rec["modified_utc"] = rec["sha256"] = None
    rec["previous_sha256"] = expected_sha256
    rec["previous_hash_source"] = expected_source if expected_sha256 else None
    rec["matches_previous_hash"] = None if (expected_sha256 is None or not rec["exists"]) else (rec["sha256"].lower() == expected_sha256.lower())
    return rec


# ----------------------------------------------------------------------- safe writing
def make_fresh_dir(path: str) -> str:
    """Create ``path`` (and parents). Refuses an existing non-empty directory; an existing EMPTY one is accepted."""
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise FileExistsError("output path exists and is not a directory: %s" % path)
        if os.listdir(path):
            raise FileExistsError("output directory is not empty (nothing is overwritten or deleted): %s" % path)
    os.makedirs(path, exist_ok=True)
    return path


def write_new_json(path: str, obj: Any) -> str:
    """Exclusive create ('x'): raises FileExistsError instead of replacing a file. Returns the SHA-256."""
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_new_csv(path: str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with open(path, "x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


# ------------------------------------------------------------------------- statistics
def percentile_nearest_rank(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    s = sorted(values)
    rank = max(1, int(math.ceil(q / 100.0 * len(s))))
    return s[min(rank, len(s)) - 1]


def describe(values: Sequence[float]) -> Dict[str, Any]:
    """count, mean, std (population), min, p50, p90, p95, p99, max. Percentiles are NEAREST-RANK (no interpolation)."""
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not v:
        return {"count": 0}
    n = len(v)
    mean = sum(v) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / n)
    return {"count": n, "mean": mean, "std": std, "min": min(v), "p50": percentile_nearest_rank(v, 50), "p90": percentile_nearest_rank(v, 90),
            "p95": percentile_nearest_rank(v, 95), "p99": percentile_nearest_rank(v, 99), "max": max(v), "percentile_method": "nearest_rank"}


# --------------------------------------------------------------------------- log parsing
_TRAIN_RE = re.compile(r"^\s*(\d+)/(\d+)\s+\(epoch\s+(\d+)\),\s*train_loss\s*=\s*(-?[\d.]+),\s*time/batch\s*=\s*([\d.]+)\s*$")
_GRID_RE = re.compile(r"grid calculation time for all the data:\s*([\d.]+)\s*seconds")
_TOTAL_RE = re.compile(r"The whole trainig time for\s+(\d+)\s+iteraction was\s+([\d.]+)\s+seconds")
_TEST_RE = re.compile(r"Processed trajectory number\s*:\s*(\d+)\s+out of\s+(\d+)\s+trajectories in time\s+([\d.]+)")


def parse_train_log(text: str) -> Dict[str, Any]:
    """Parse train.py stdout. Only values that are literally printed are returned; nothing is inferred."""
    batches = []
    for line in text.splitlines():
        m = _TRAIN_RE.match(line)
        if m:
            batches.append({"global_batch": int(m.group(1)), "total_batches": int(m.group(2)), "epoch": int(m.group(3)),
                            "train_loss": float(m.group(4)), "time_per_batch_s": float(m.group(5))})
    g, t = _GRID_RE.search(text), _TOTAL_RE.search(text)
    return {"batches": batches, "grid_calculation_s": float(g.group(1)) if g else None,
            "total_training_s": float(t.group(2)) if t else None, "epochs_reported": int(t.group(1)) if t else None}


def parse_test_log(text: str) -> Dict[str, Any]:
    rows = [{"trajectory": int(m.group(1)), "of": int(m.group(2)), "time_s": float(m.group(3))} for m in _TEST_RE.finditer(text)]
    return {"trajectories": rows}


def per_epoch_from_batches(batches: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[int, List[Mapping[str, Any]]] = {}
    for b in batches:
        by.setdefault(int(b["epoch"]), []).append(b)
    out = []
    for e in sorted(by):
        bs = by[e]
        t = [x["time_per_batch_s"] for x in bs]
        l = [x["train_loss"] for x in bs]
        out.append({"epoch": e, "n_batches": len(bs), "mean_loss": sum(l) / len(l), "sum_batch_time_s": sum(t), "batch_time": describe(t)})
    return out
