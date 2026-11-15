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
