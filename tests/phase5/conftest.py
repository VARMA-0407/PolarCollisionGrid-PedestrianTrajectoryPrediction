import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tests" / "phase4", REPO_ROOT / "tests" / "phase4" / "task3", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest  # noqa: E402

from phase5_helpers import build_scene_task3_run  # noqa: E402


@pytest.fixture(scope="session")
def scene_run_dir(tmp_path_factory):
    """A real Task 3 run directory built by the real Task 2 and Task 3 runners from the synthetic scene."""
    return build_scene_task3_run(tmp_path_factory.mktemp("p5scene"))


# ---------------------------------------------------------------------------------------------------------------------------
# P5.7 fixtures: synthetic-scene window run and a tiny trained model (CPU, single thread, development scale)
# ---------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def p57_windows_run(scene_run_dir, tmp_path_factory):
    from pedsafety.phase5.runner import generate_windows
    from phase5_helpers import CONFIG as WINDOW_CONFIG
    from phase5_helpers import wcfg
    out = tmp_path_factory.mktemp("p57win") / "windows"
    generate_windows(scene_run_dir, wcfg(), out, config_path=str(WINDOW_CONFIG))
    return out


@pytest.fixture(scope="session")
def p57_train_run(p57_windows_run, tmp_path_factory):
    from pedsafety.phase5.models.training_harness import run_training
    from p57_helpers import write_tiny_training_config
    root = tmp_path_factory.mktemp("p57train")
    cfg = write_tiny_training_config(root / "cfg.json")
    out = root / "run"
    manifest = run_training(p57_windows_run, cfg, out, command="test")
    return out, manifest, cfg
