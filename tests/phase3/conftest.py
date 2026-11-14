import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK7_TESTS = REPO_ROOT / "tests" / "task7"
for path in (REPO_ROOT, TASK7_TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

