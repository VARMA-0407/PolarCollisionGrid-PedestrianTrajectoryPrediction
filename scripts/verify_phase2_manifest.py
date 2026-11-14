import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify(manifest_path):
    root = Path(__file__).resolve().parents[1]
    with manifest_path.open("r", encoding="utf-8-sig") as stream:
        manifest = json.load(stream)

    errors = []
    checked = 0
    for artifact in manifest.get("artifacts", []):
        relative_path = Path(artifact["path"])
        if relative_path.is_absolute():
            errors.append("absolute artifact path: {}".format(relative_path))
            continue
        path = root / relative_path
        if not path.exists():
            errors.append("missing input: {}".format(relative_path))
            continue
        checked += 1
        actual_size = path.stat().st_size
        if actual_size != artifact["size_bytes"]:
            errors.append(
                "size mismatch: {} (expected {}, got {})".format(
                    relative_path, artifact["size_bytes"], actual_size
                )
            )
        actual_hash = sha256(path)
        if actual_hash != artifact["sha256"].upper():
            errors.append(
                "hash mismatch: {} (expected {}, got {})".format(
                    relative_path, artifact["sha256"], actual_hash
                )
            )

    declared = {Path(item["path"]) for item in manifest.get("artifacts", [])}
    if Path("manifest_v2.json") in declared:
        errors.append("manifest_v2.json must not be self-hashed")

    result = {
        "manifest": str(manifest_path),
        "declared_artifacts": len(declared),
        "checked_artifacts": checked,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/PCG_T4_5_Epoch_Run/manifest_v2.json"),
    )
    args = parser.parse_args()
    try:
        return verify(args.manifest)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print("Manifest verification error: {}".format(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
