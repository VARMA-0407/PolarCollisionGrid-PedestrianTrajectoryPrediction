"""Target sources for future trajectories.

PILOT_SMOKE (implemented)
    Future pseudo-targets are the perceived Task 3 node positions (footpoint u, v in pixels) of the *same* track at the
    future sampled frames. They inherit every detector/tracker error, are not independent ground truth, and exist only
    to validate data preparation. No forecasting accuracy may be claimed from them.

ANNOTATED_TARGET (future extension, refused)
    Would read independent PIE annotation targets. It needs the annotation XML (not on this machine), a verified
    pedestrian identity mapping between annotations and perceived tracks, frame/timestamp alignment and annotation
    validity checks. Until those exist it raises :class:`TargetSourceUnavailable`; it never falls back to perceived
    positions.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .errors import TargetSourceUnavailable, WindowContractError
from .task3_source import Task3Run
from .window_config import IMPLEMENTED_TARGET_SOURCE_MODES, TARGET_SOURCE_MODES

PILOT_SMOKE_TARGET_SOURCE = "perceived_phase3_bytetrack_tracks_via_phase4_task3_nodes (pseudo-targets, not ground truth)"
ANNOTATED_TARGET_REQUIREMENTS = (
    "PIE annotation XML for each video (video_0004_annt.xml is not on this machine)",
    "verified identity mapping between annotated pedestrians and perceived track ids (no invented mapping)",
    "frame/timestamp alignment between annotations and the 30 fps source frames",
    "annotation validity checks (occlusion, out-of-frame, missing boxes)",
    "independent future targets kept separate from perceived inputs")


@dataclass(frozen=True)
class TargetSequence:
    node_index: np.ndarray      # int64 (F,) Task 3 node index, -1 if absent
    uv_px: np.ndarray           # float32 (F, 2) NaN if absent
    box_h_px: np.ndarray        # float32 (F,) NaN if absent
    border: np.ndarray          # bool (F,)
    valid: np.ndarray           # bool (F,)

    @property
    def complete(self) -> bool:
        return bool(self.valid.all())


def check_target_source(mode: str) -> None:
    if mode not in TARGET_SOURCE_MODES:
        raise WindowContractError("UNKNOWN_TARGET_SOURCE", "unknown target source mode %r" % mode)
    if mode not in IMPLEMENTED_TARGET_SOURCE_MODES:
        raise TargetSourceUnavailable("TARGET_SOURCE_NOT_IMPLEMENTED",
                                      "%s is a documented future extension and is not an active data path. Requirements: %s. "
                                      "Perceived positions are never substituted." % (mode, "; ".join(ANNOTATED_TARGET_REQUIREMENTS)))


def pilot_smoke_targets(run: Task3Run, track_index: int, future_frames: Sequence[int]) -> TargetSequence:
    f = len(future_frames)
    idx = np.full(f, -1, np.int64)
    uv = np.full((f, 2), np.nan, np.float32)
    bh = np.full(f, np.nan, np.float32)
    border = np.zeros(f, bool)
    for m, fr in enumerate(future_frames):
        i: Optional[int] = run.node_at(track_index, int(fr))
        if i is None or not run.masks[i, 0]:
            continue                                  # absent stays absent: no interpolation, no carry-forward
        idx[m] = i
        uv[m] = run.features[i, 0:2]
        bh[m] = run.features[i, 7]
        border[m] = run.border[i]
    return TargetSequence(idx, uv, bh, border, idx >= 0)


def extract_targets(run: Task3Run, track_index: int, future_frames: Sequence[int], mode: str) -> TargetSequence:
    check_target_source(mode)
    return pilot_smoke_targets(run, track_index, future_frames)


__all__ = ["TargetSequence", "check_target_source", "extract_targets", "pilot_smoke_targets", "PILOT_SMOKE_TARGET_SOURCE",
           "ANNOTATED_TARGET_REQUIREMENTS"]
