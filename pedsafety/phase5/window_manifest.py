"""Run manifest, git provenance, claims and limitations for a Phase 5 window dataset."""

import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = "phase5-window-run-manifest-1.0.0"
PILOT_LABEL = "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT"

CLAIMS = {
    "coordinate_space": "image (captured raster pixels)",
    "metric_world_coordinates": False, "world_position_valid": False, "physical_features": False,
    "ttc_pet_or_min_separation": False, "collision_or_conflict_probability": False, "ego_motion_compensated": False,
    "independent_ground_truth_targets": False, "forecasting_accuracy": False, "independent_generalization": False,
    "independent_train_validation_test_split": False, "model_trained": False, "real_time_claim": False,
    "note": "Future targets are perceived-track pseudo-targets (PILOT_SMOKE) for pipeline validation only."}

LIMITATIONS = [
    "Pilot only: one PIE video (set01/video_0004) of one scene group; windows are strongly correlated (same scene, camera, drive and "
    "overlapping tracks). No independent train/validation/test split exists; nothing here supports a generalization claim.",
    "PILOT_SMOKE targets are perceived YOLO11n + ByteTrack positions of the same tracks; they inherit detector/tracker errors and are "
    "not independent ground truth. Phase 3 detection/tracking quality on PIE has not been measured.",
    "Image space only (captured_1920x1080 pixels, px/s, s); image velocities mix agent motion with moving-camera ego-motion and perspective.",
    "Node velocities are Task 2 one-frame (1/30 s) causal backward differences with ~48 px/s per-axis jitter; no smoothing.",
    "Bicyclist nodes come from the COCO 'bicycle' detection class (Phase 3 label 'bike' mapped to 'bicyclist' by assumption). A bicycle "
    "detection is not evidence of a rider (parked bicycles are possible). They are stored, flagged semantically_uncertain, and excluded "
    "from the default context.",
    "Strict target policy: windows need the target at all 6 history and 15 future samples; short and fragmented tracks are under-represented, "
    "which biases the pilot towards long, continuously tracked pedestrians.",
    "Track fragments and possible identity switches are not repaired or merged; Phase 3 class-flip holes remain.",
    "Overlapping windows of the same track share frames (stride 0.1 s, window span 2.0 s); the effective sample size is closer to the "
    "number of distinct tracks than to the number of windows.",
]


def git_provenance() -> Dict[str, Any]:
    def run(*a) -> Optional[str]:
        try:
            o = subprocess.run(["git"] + list(a), cwd=str(REPO), capture_output=True, text=True, timeout=20)
            return o.stdout if o.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    lines = [l for l in (status or "").splitlines() if l.strip()]
    tracked = [l[3:] for l in lines if not l.startswith("??")]
    untracked = [l[3:] for l in lines if l.startswith("??")]
    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    return {"commit": commit.strip() if commit else None, "branch": branch.strip() if branch else None,
            "working_tree_dirty": bool(lines) if status is not None else None,
            "tracked_changes": tracked, "untracked_files": untracked,
            "note": "Phase 5 P5.3 code is uncommitted while under review; the dirty flag and file lists record exactly what differed from HEAD."}


def versions() -> Dict[str, Any]:
    try:
        from importlib.metadata import version as _v
        jsv = _v("jsonschema")
    except Exception:                                                  # noqa: BLE001
        jsv = None
    from . import PHASE5_WINDOWS_VERSION
    return {"phase5_windows": PHASE5_WINDOWS_VERSION, "python": platform.python_version(), "numpy": np.__version__, "jsonschema": jsv,
            "platform": platform.platform()}


__all__ = ["MANIFEST_VERSION", "PILOT_LABEL", "CLAIMS", "LIMITATIONS", "git_provenance", "versions"]
