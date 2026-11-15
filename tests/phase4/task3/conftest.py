import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for path in (REPO_ROOT, HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
