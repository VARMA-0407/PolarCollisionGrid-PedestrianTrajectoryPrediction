import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (REPO_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
