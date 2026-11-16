"""Experiment identity, configuration lock and stage status tracking.

* An experiment lives in ``<drive_root>/experiments/<experiment_id>/`` and is never overwritten: a new id is generated unless the
  user explicitly resumes an existing one.
* ``experiment_lock.json`` freezes every configuration value that changes scientific outputs (the "locked" configuration). Resuming
  with a different locked value is refused with a list of the differences, so stale outputs of another configuration are never
  mixed into this experiment.
* ``experiment_status.json`` records every stage (status, timestamps, runtime, detail, error). Status values: COMPLETED,
  COMPLETED_WITH_FAILURES, FAILED, BLOCKED (scientific/data prerequisite missing), BLOCKED_UPSTREAM (a required stage did not
  complete), SKIPPED_DISABLED, REUSED_VERIFIED. A stage that raises is recorded FAILED and never counted as success.
"""

import re
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .common import StageError, canonical_sha256, read_json, utc_now, utc_stamp, write_json

LOCK_VERSION = "multivideo-experiment-lock-1.0.0"
STATUS_VERSION = "multivideo-experiment-status-1.0.0"
OK_STATES = ("COMPLETED", "REUSED_VERIFIED")
_ID = re.compile(r"^[A-Za-z0-9_.-]{3,120}$")


def new_experiment_id(prefix: str = "pie_full") -> str:
    return "%s_%s" % (prefix, utc_stamp())


def create_or_resume_experiment(experiments_root, experiment_id: Optional[str], locked_config: Mapping[str, Any], git: Mapping[str, Any], *,
                                allow_commit_change: bool = False, prefix: str = "pie_full") -> Dict[str, Any]:
    """Create a new experiment directory (``experiment_id`` empty) or resume an existing one after checking the lock."""
    root = Path(experiments_root)
    resume = bool(experiment_id)
    eid = experiment_id or new_experiment_id(prefix)
    if not _ID.match(eid):
        raise StageError("EXPERIMENT_ID", "experiment id %r must match %s" % (eid, _ID.pattern))
    d = root / eid
    lock_path = d / "experiment_lock.json"
    locked = {k: locked_config[k] for k in sorted(locked_config)}
    if resume:
        if not lock_path.is_file():
            raise StageError("RESUME_NOT_FOUND", "cannot resume %s: %s does not exist (leave EXPERIMENT_ID empty to start a new experiment)" % (eid, lock_path))
        lock = read_json(lock_path)
        old = lock["locked_config"]
        diff = sorted(k for k in set(old) | set(locked) if old.get(k) != locked.get(k))
        if diff:
            raise StageError("LOCKED_CONFIG_CHANGED", "resuming %s with different locked configuration values %s; start a new experiment instead "
                             "(old: %s; new: %s)" % (eid, diff, {k: old.get(k) for k in diff}, {k: locked.get(k) for k in diff}))
        if lock.get("git_commit") != git.get("commit"):
            msg = "experiment %s was created at commit %s, the current code is %s" % (eid, lock.get("git_commit"), git.get("commit"))
            if not allow_commit_change:
                raise StageError("CODE_COMMIT_CHANGED", msg + " (set ALLOW_RESUME_ACROSS_COMMITS=True only if the change cannot affect outputs; it is recorded)")
            lock.setdefault("commit_changes_on_resume", []).append({"utc": utc_now(), "from": lock.get("git_commit"), "to": git.get("commit")})
        lock.setdefault("resumes", []).append({"utc": utc_now(), "git": dict(git)})
        write_json(lock_path, lock)
        return {"experiment_id": eid, "dir": d, "resumed": True, "lock": lock}
    if d.exists():
        raise StageError("EXPERIMENT_EXISTS", "%s already exists; experiments are never overwritten" % d)
    d.mkdir(parents=True)
    lock = {"lock_version": LOCK_VERSION, "experiment_id": eid, "created_utc": utc_now(), "locked_config": locked, "locked_config_sha256": canonical_sha256(locked),
            "git_commit": git.get("commit"), "git": dict(git), "resumes": []}
    write_json(lock_path, lock)
    return {"experiment_id": eid, "dir": d, "resumed": False, "lock": lock}


class StageTracker:
    """Persistent stage status register. ``with tracker.stage("S5"):`` records COMPLETED, or FAILED with the error, and never re-raises
    (so the notebook can still produce its final summary); dependants call ``tracker.require(...)``."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = read_json(self.path) if self.path.is_file() else {"status_version": STATUS_VERSION, "stages": {}, "history": []}

    def save(self):
        write_json(self.path, self.data)

    def get(self, name: str) -> Dict[str, Any]:
        return self.data["stages"].get(name, {})

    def status(self, name: str) -> Optional[str]:
        return self.get(name).get("status")

    def set(self, name: str, status: str, **detail):
        rec = {"status": status, "updated_utc": utc_now(), **detail}
        self.data["stages"][name] = rec
        self.data["history"].append({"stage": name, "status": status, "utc": rec["updated_utc"]})
        self.save()
        return rec

    def require(self, name: str, needs: Iterable[str]) -> bool:
        """True if every needed stage is COMPLETED / REUSED_VERIFIED (or COMPLETED_WITH_FAILURES when explicitly accepted by the caller via the
        stage detail ``accepted_partial``); otherwise records BLOCKED_UPSTREAM for ``name`` and returns False."""
        missing = []
        for n in needs:
            r = self.get(n)
            if r.get("status") in OK_STATES or (r.get("status") == "COMPLETED_WITH_FAILURES" and r.get("accepted_partial")):
                continue
            missing.append("%s=%s" % (n, r.get("status", "NOT_RUN")))
        if missing:
            self.set(name, "BLOCKED_UPSTREAM", reason="required stages not completed: %s" % ", ".join(missing))
            print("[BLOCKED_UPSTREAM] %s: required stages not completed: %s" % (name, ", ".join(missing)))
            return False
        return True

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        started = utc_now()
        box: Dict[str, Any] = {"detail": {}, "status": None}
        self.set(name, "RUNNING", started_utc=started)
        print("=" * 100 + "\n[%s] %s started" % (started, name))
        try:
            yield box
        except Exception as exc:                                                                      # noqa: BLE001 - recorded, reported, never silent
            tb = traceback.format_exc()
            self.set(name, "FAILED", started_utc=started, runtime_s=time.perf_counter() - t0, error_code=getattr(exc, "code", type(exc).__name__),
                     error=str(exc)[:2000], traceback_tail=tb.splitlines()[-12:], detail=box["detail"])
            print(tb)
            print("[FAILED] %s: %s" % (name, exc))
            return
        status = box["status"] or "COMPLETED"
        self.set(name, status, started_utc=started, runtime_s=time.perf_counter() - t0, detail=box["detail"], **box.get("extra", {}))
        print("[%s] %s (%.1f s)" % (status, name, time.perf_counter() - t0))

    def summary(self) -> Dict[str, Any]:
        by: Dict[str, list] = {}
        for n, r in self.data["stages"].items():
            by.setdefault(r.get("status", "UNKNOWN"), []).append(n)
        return by


__all__ = ["create_or_resume_experiment", "new_experiment_id", "StageTracker", "OK_STATES"]
