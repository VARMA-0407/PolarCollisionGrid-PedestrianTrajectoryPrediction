"""Stage 6.1: output-directory policy shared by every command that writes files.

Policy: a run writes only into an output directory that does not exist yet or is EMPTY.
A non-empty directory (any entry at all, hidden files included) or a path that is a file is
refused BEFORE anything is read or written. Nothing is ever deleted or overwritten.

A directory is a complete, valid result of a run only when its completion file is present
(``run_manifest.json`` for ingest/features/build, ``plot_manifest.json`` for plot,
``hbs_smoke_report.json`` for hbs-smoke) AND ``error.json`` is absent. The completion file is
always written last and atomically; ``error.json`` is written for every failed run that owns
its directory. A directory refused by this policy is never touched, not even by ``error.json``.
"""

from pathlib import Path
from typing import Union

from .errors import PedsafetyError

COMPLETION_FILES = {"ingest": "run_manifest.json", "features": "run_manifest.json", "build": "run_manifest.json",
                    "plot": "plot_manifest.json", "hbs-smoke": "hbs_smoke_report.json"}


class OutputDirectoryError(PedsafetyError):
    """The requested output directory violates the policy. ``code`` is stable."""

    def __init__(self, code: str, message: str, path: str):
        self.code, self.path = code, path
        super().__init__("[%s] %s" % (code, message))


def output_dir_state(path: Union[str, Path]) -> str:
    """'absent' | 'empty' | 'non_empty' | 'not_a_directory'."""
    p = Path(path)
    if not p.exists():
        return "absent"
    if not p.is_dir():
        return "not_a_directory"
    return "non_empty" if any(True for _ in p.iterdir()) else "empty"


def check_output_dir(path: Union[str, Path]) -> Path:
    """Raise :class:`OutputDirectoryError` unless the directory is absent or empty. Creates nothing."""
    p = Path(path)
    state = output_dir_state(p)
    if state == "not_a_directory":
        raise OutputDirectoryError("OUTPUT_PATH_NOT_DIRECTORY", "output path exists and is not a directory: %s" % p, str(p))
    if state == "non_empty":
        raise OutputDirectoryError("OUTPUT_DIR_NOT_EMPTY",
                                   "output directory %s already contains files; pedsafety never overwrites or deletes. "
                                   "Choose a new or empty directory for this run" % p, str(p))
    return p


def is_complete_run(path: Union[str, Path], command: str) -> bool:
    """True only if ``command``'s completion file exists and ``error.json`` does not.
    This is the documented test for 'this directory holds the valid result of a finished run'."""
    p = Path(path)
    if command not in COMPLETION_FILES:
        raise ValueError("unknown command %r; expected one of %s" % (command, sorted(COMPLETION_FILES)))
    return (p / COMPLETION_FILES[command]).is_file() and not (p / "error.json").exists()
