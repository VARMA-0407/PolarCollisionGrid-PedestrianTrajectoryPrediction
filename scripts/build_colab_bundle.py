"""Build the Colab source ZIP for Phase 3 Task 1 plus a SHA-256 manifest (Option A of the sync decision).

  python scripts/build_colab_bundle.py --out-dir outputs/colab_bundle

Writes into a NEW directory:
  pedsafety_task1_bundle.zip            the source (only files needed to run Task 1)
  pedsafety_task1_bundle.manifest.json  SHA-256 of every file inside the ZIP and of the ZIP itself (upload both)

Only an explicit allow-list is bundled. Never included: datasets, model weights, checkpoints, experiment artifacts, outputs,
caches, secrets, Git metadata, the original PCG training/testing code. Nothing is committed or pushed.
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]

ALLOW_DIRS = ("pedsafety",)                                   # the whole package (adapter and schema code depend on it)
ALLOW_FILES = (
    "docs/phase2/schemas/agent_state.schema.json",            # the approved schema, read by pedsafety.schema at run time
    "configs/phase3/perception_default.json",
    "scripts/run_perception.py", "scripts/benchmark_perception.py", "scripts/make_synthetic_video.py",
    "requirements-perception.txt", "notebooks/phase3_task1_perception_colab.ipynb",
    "docs/phase3/PHASE3_TASK1_PERCEPTION_FRONTEND.md", "docs/phase3/PHASE3_TASK1_EXECUTION_GUIDE.md",
    "scripts/evaluate_pie_perception.py", "docs/phase3/PHASE3_TASK4_PIE_EVALUATION.md",     # Task 4 (evaluator itself lives in pedsafety/perception/)
    "tests/phase3/conftest.py", "tests/phase3/task1_helpers.py",
)
ALLOW_GLOBS = ("tests/phase3/test_task1_*.py", "tests/phase3/test_task4_*.py")
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".onnx", ".engine", ".ckpt", ".pkl", ".npy", ".npz", ".mp4", ".avi", ".mov", ".csv", ".env", ".key", ".pem")
FORBIDDEN_PARTS = ("__pycache__", ".git", "Data", "Store_Results", "experiments", "outputs", "weights", ".pytest_cache")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> List[Path]:
    files: List[Path] = []
    for d in ALLOW_DIRS:
        files += [p for p in (REPO / d).rglob("*") if p.is_file()]
    for f in ALLOW_FILES:
        p = REPO / f
        if not p.is_file():
            raise FileNotFoundError("required bundle file is missing: %s" % f)
        files.append(p)
    for g in ALLOW_GLOBS:
        files += sorted(REPO.glob(g))
    out = []
    for p in sorted(set(files)):
        rel = p.relative_to(REPO)
        if any(part in FORBIDDEN_PARTS for part in rel.parts) or rel.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="a NEW directory; refuses to reuse an existing one")
    ap.add_argument("--name", default="pedsafety_task1_bundle", help="bundle base name (use a different one per task so uploads never overwrite each other)")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    if out.exists():
        print("ERROR: %s already exists; choose a new directory (nothing is overwritten)" % out, file=sys.stderr)
        return 2
    files = collect()
    out.mkdir(parents=True)
    zpath = out / (a.name + ".zip")
    entries = {}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            rel = p.relative_to(REPO).as_posix()
            zf.write(p, rel)
            entries[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    manifest = {"bundle": zpath.name, "bundle_sha256": sha256_file(zpath), "bundle_bytes": zpath.stat().st_size, "file_count": len(entries),
                "files": entries, "excluded": "datasets, weights, checkpoints, experiments, outputs, caches, secrets, git metadata, PCG training/test code",
                "note": "verify with the Colab notebook before running anything; weights are downloaded in Colab only"}
    (out / (a.name + ".manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("bundle   : %s (%d files, %d bytes)" % (zpath, len(entries), manifest["bundle_bytes"]))
    print("manifest : %s" % (out / (a.name + ".manifest.json")))
    print("zip sha256: %s" % manifest["bundle_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
