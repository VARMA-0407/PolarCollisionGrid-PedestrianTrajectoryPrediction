"""Stage 6.1: dependency declarations match real imports, and the package version is current."""

import ast
import importlib.util
import json
import re
from pathlib import Path

import pedsafety
from pedsafety import pipeline

REPO = Path(__file__).resolve().parents[2]
PEP440_DEV = re.compile(r"^\d+\.\d+\.\d+\.dev\d+$")


def declared(path: Path, seen=None):
    """Distribution names declared in a requirements file, following -r includes."""
    seen = seen if seen is not None else set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            inc = (path.parent / line[3:].strip()).resolve()
            if inc not in seen:
                seen.add(inc)
                names |= declared(inc, seen)
            continue
        names.add(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


# Import name -> distribution name where they differ (Phase 3 Task 1 added OpenCV).
MODULE_TO_DISTRIBUTION = {"cv2": "opencv-python-headless", "yaml": "pyyaml"}


def third_party_imports(root: Path, skip=()):
    out = set()
    for p in root.rglob("*.py"):
        if any(part in skip for part in p.relative_to(root).parts):
            continue
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            mods = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else (
                [n.module.split(".")[0]] if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module else [])
            for m in mods:
                spec = importlib.util.find_spec(m) if m != "pedsafety" else None
                if spec is not None and "site-packages" in (spec.origin or ""):
                    out.add(MODULE_TO_DISTRIBUTION.get(m, m.lower().replace("_", "-")))
    return out


def test_version_is_a_pep440_dev_version_for_stage_6_1():
    assert PEP440_DEV.match(pedsafety.__version__), pedsafety.__version__
    assert pedsafety.__version__ == "0.6.1.dev0" and "stage1" not in pedsafety.__version__


def test_version_is_reported_in_run_manifests():
    assert pipeline.tool_versions()["pedsafety"] == pedsafety.__version__


def test_every_third_party_import_of_the_package_is_declared():
    """Core modules may only use requirements-pedsafety.txt. The perception frontend (pedsafety/perception) may additionally
    use the separate, optional requirements-perception.txt (Phase 3 Task 1); the core must never depend on it."""
    runtime = declared(REPO / "requirements-pedsafety.txt")
    missing = third_party_imports(REPO / "pedsafety", skip=("perception",)) - runtime
    assert not missing, "undeclared runtime dependencies: %s" % sorted(missing)
    perception = declared(REPO / "requirements-perception.txt")
    missing = third_party_imports(REPO / "pedsafety" / "perception") - runtime - perception
    assert not missing, "undeclared perception dependencies: %s" % sorted(missing)


def test_test_dependencies_are_declared_in_the_dev_file_only():
    dev = declared(REPO / "requirements-dev.txt")
    runtime = declared(REPO / "requirements-pedsafety.txt")
    assert "pytest" in dev and "pytest" not in runtime
    assert not (third_party_imports(REPO / "tests" / "task7") - dev)


def test_jsonschema_is_a_bounded_runtime_dependency():
    text = (REPO / "requirements-pedsafety.txt").read_text(encoding="utf-8")
    assert re.search(r"^jsonschema>=4\.0,<5\s*$", text, re.M)


def test_the_original_pcg_requirements_file_is_left_alone_and_not_duplicated():
    names = declared(REPO / "requirements.txt")
    assert "jsonschema" not in names and "pytest" not in names
    runtime_lines = [l for l in (REPO / "requirements-pedsafety.txt").read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
    assert runtime_lines == ["-r requirements.txt", "jsonschema>=4.0,<5"]          # no unnecessary packages or repeated pins
