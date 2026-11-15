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


# the one package directory (relative to pedsafety/) that may use requirements-phase5.txt in addition to the core file
PHASE5_MODELS_REL = (Path("phase5") / "models",)

# Import name -> distribution name where they differ (Phase 3 Task 1 added OpenCV).
MODULE_TO_DISTRIBUTION = {"cv2": "opencv-python-headless", "yaml": "pyyaml"}


def third_party_imports(root: Path, skip=(), skip_paths=()):
    """``skip``: directory names skipped anywhere (existing behaviour). ``skip_paths``: paths relative to ``root`` whose whole subtree is
    skipped (Phase 5 P5.5): a precise exemption of one package directory, never a name that could match elsewhere."""
    out = set()
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if any(part in skip for part in rel.parts):
            continue
        if any(rel.parts[:len(sp.parts)] == sp.parts for sp in skip_paths):
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
    missing = third_party_imports(REPO / "pedsafety", skip=("perception",), skip_paths=PHASE5_MODELS_REL) - runtime
    assert not missing, "undeclared runtime dependencies: %s" % sorted(missing)
    perception = declared(REPO / "requirements-perception.txt")
    missing = third_party_imports(REPO / "pedsafety" / "perception") - runtime - perception
    assert not missing, "undeclared perception dependencies: %s" % sorted(missing)
    # Phase 5 P5.5: ONLY pedsafety/phase5/models may additionally use requirements-phase5.txt (plain PyTorch)
    phase5 = declared(REPO / "requirements-phase5.txt")
    missing = third_party_imports(REPO / "pedsafety" / "phase5" / "models") - runtime - phase5
    assert not missing, "undeclared Phase 5 model dependencies: %s" % sorted(missing)


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


# ---------------------------------------------------------------------------------------------------------------------------
# Phase 5 P5.5: the Phase 5 dependency file is narrow, and the exemption cannot be used to hide undeclared imports
# ---------------------------------------------------------------------------------------------------------------------------
def _runtime_lines(path: Path):
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]


def test_phase5_requirements_declare_only_torch_on_top_of_the_core_file():
    assert _runtime_lines(REPO / "requirements-phase5.txt") == ["-r requirements-pedsafety.txt", "torch>=2.2"]
    own = declared(REPO / "requirements-phase5.txt") - declared(REPO / "requirements-pedsafety.txt")
    assert own == {"torch"}
    assert "torch" not in declared(REPO / "requirements-pedsafety.txt")                   # the Phase 2 core stays torch-free
    assert "torch" not in declared(REPO / "requirements.txt")
    assert re.search(r"^torch>=2\.2\s*$", (REPO / "requirements-phase5.txt").read_text(encoding="utf-8"), re.M)   # same bound as the perception file
    assert re.search(r"^torch>=2\.2\s*$", (REPO / "requirements-perception.txt").read_text(encoding="utf-8"), re.M)


def test_the_core_package_and_other_phase5_modules_still_need_core_declarations():
    """The exemption is exactly pedsafety/phase5/models: everything else in pedsafety/ (also the rest of pedsafety/phase5) is held to the core file."""
    runtime = declared(REPO / "requirements-pedsafety.txt")
    core = third_party_imports(REPO / "pedsafety", skip=("perception",), skip_paths=PHASE5_MODELS_REL)
    assert core <= runtime and "torch" not in core
    models = third_party_imports(REPO / "pedsafety" / "phase5" / "models")
    assert "torch" in models and models - runtime == {"torch"}                              # models really do import torch, and nothing else extra
    # without the exemption the whole package would fail: the exemption is what the P5.5 change adds
    assert "torch" in third_party_imports(REPO / "pedsafety", skip=("perception",))


def test_skip_paths_exempts_one_subtree_only(tmp_path):
    (tmp_path / "phase5" / "models").mkdir(parents=True)
    (tmp_path / "phase5" / "models" / "enc.py").write_text("import pytest\n", encoding="utf-8")
    (tmp_path / "phase5" / "other.py").write_text("import pytest\n", encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.py").write_text("import pytest\n", encoding="utf-8")        # a different 'models' directory: not exempt
    assert third_party_imports(tmp_path) == {"pytest"}
    only_models = tmp_path / "isolated"
    (only_models / "phase5" / "models").mkdir(parents=True)
    (only_models / "phase5" / "models" / "enc.py").write_text("import pytest\n", encoding="utf-8")
    assert third_party_imports(only_models, skip_paths=PHASE5_MODELS_REL) == set()
    assert third_party_imports(tmp_path, skip_paths=PHASE5_MODELS_REL) == {"pytest"}       # other.py and models/x.py are still scanned


def test_undeclared_imports_in_phase5_models_still_fail_the_guard(tmp_path):
    """The Phase 5 declaration is a whitelist: a package imported by pedsafety/phase5/models that is in no accepted file is reported."""
    (tmp_path / "enc.py").write_text("import torch\nimport pytest\n", encoding="utf-8")
    imported = third_party_imports(tmp_path)
    runtime, phase5 = declared(REPO / "requirements-pedsafety.txt"), declared(REPO / "requirements-phase5.txt")
    assert imported - runtime - phase5 == {"pytest"}                                        # torch is accepted, pytest (undeclared there) is not
    assert "torch" not in (imported - runtime - phase5)


def test_phase5_file_does_not_loosen_the_perception_or_dev_policies():
    assert "torch" not in declared(REPO / "requirements-dev.txt")                          # the dev file keeps including only the core file
    assert "-r requirements-phase5.txt" not in (REPO / "requirements-dev.txt").read_text(encoding="utf-8")
    assert not (third_party_imports(REPO / "pedsafety" / "perception") - declared(REPO / "requirements-pedsafety.txt") - declared(REPO / "requirements-perception.txt"))
