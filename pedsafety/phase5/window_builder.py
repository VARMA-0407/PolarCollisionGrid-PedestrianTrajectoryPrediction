"""Candidate enumeration and the strict target policy (TARGET_STRICT_CONTEXT_MASKED).

Candidates: every (pedestrian track, anchor frame t0) with t0 on the global anchor grid and
first_frame(track) <= t0 <= last_frame(track). Each candidate is accepted or rejected with exactly one reason, checked
in this fixed precedence order:

    history_before_video_start   h_0 < first video frame
    future_after_video_end       f_{F-1} > last video frame
    target_absent_at_anchor      the target has no node at t0 (track gap at the anchor)
    history_before_track_start   h_0 < first frame of the track
    target_history_gap           a history sample frame has no node of the target (track gap)
    future_after_track_end       f_{F-1} > last frame of the track
    target_future_gap            a future sample frame has no node of the target (track gap)

Nothing is interpolated, forward-filled, assumed stationary, or merged across track ids. Context agents never cause a
rejection: their absence is represented by masks in the graph adapter.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .task3_source import NODE_TYPE_NAMES, Task3Run
from .trajectory_targets import check_target_source, extract_targets
from .window_config import WindowConfig

REJECTION_REASONS = ("history_before_video_start", "future_after_video_end", "target_absent_at_anchor", "history_before_track_start",
                     "target_history_gap", "future_after_track_end", "target_future_gap")


@dataclass
class WindowSet:
    """Accepted windows (W rows) plus rejected candidates (R rows). Arrays only; strings in ``target_track_id``."""
    window_id: np.ndarray                 # int64 (W,)
    target_track_index: np.ndarray        # int32 (W,)
    target_track_id: np.ndarray           # <U (W,)
    anchor_frame: np.ndarray              # int64 (W,)
    history_frames: np.ndarray            # int64 (W, H)
    history_timestamps: np.ndarray        # float64 (W, H)
    future_frames: np.ndarray             # int64 (W, F)
    future_timestamps: np.ndarray         # float64 (W, F)
    history_target_task3_node: np.ndarray # int64 (W, H)
    history_target_source_line: np.ndarray# int64 (W, H)
    history_target_uv_px: np.ndarray      # float32 (W, H, 2)
    target_future_task3_node: np.ndarray  # int64 (W, F)
    target_future_source_line: np.ndarray # int64 (W, F)
    target_future_uv_px: np.ndarray       # float32 (W, F, 2)
    target_future_box_h_px: np.ndarray    # float32 (W, F)
    target_future_border: np.ndarray      # bool (W, F)
    target_future_valid: np.ndarray       # bool (W, F)
    target_border_at_anchor: np.ndarray   # bool (W,)
    rejected_track_index: np.ndarray      # int32 (R,)
    rejected_anchor_frame: np.ndarray     # int64 (R,)
    rejected_reason_code: np.ndarray      # int8 (R,) index into REJECTION_REASONS
    video_first_frame: int
    video_last_frame: int

    @property
    def n_windows(self) -> int:
        return int(self.window_id.shape[0])

    @property
    def n_rejected(self) -> int:
        return int(self.rejected_track_index.shape[0])

    def rejection_counts(self) -> Dict[str, int]:
        c = np.bincount(self.rejected_reason_code.astype(np.int64), minlength=len(REJECTION_REASONS))
        return {r: int(c[i]) for i, r in enumerate(REJECTION_REASONS)}


def _track_spans(run: Task3Run) -> Dict[int, tuple]:
    spans: Dict[int, tuple] = {}
    for t in run.tracks:
        spans[t["track_index"]] = (int(t["first_frame"]), int(t["last_frame"]))
    return spans


def build_windows(run: Task3Run, cfg: WindowConfig, video_first_frame: int = 0, video_last_frame: Optional[int] = None) -> WindowSet:
    check_target_source(cfg.target_source_mode)
    if video_last_frame is None:
        video_last_frame = int(run.frame.max())
    hoff, foff = cfg.history_offsets, cfg.future_offsets
    H, F = hoff.shape[0], foff.shape[0]
    target_codes = [NODE_TYPE_NAMES.index(t) for t in cfg.target_node_types]
    spans = _track_spans(run)
    acc: Dict[str, List] = {k: [] for k in ("tix", "t0", "hn", "tgt")}
    rej_t: List[int] = []
    rej_a: List[int] = []
    rej_r: List[int] = []
    for t in run.tracks:                                   # tracks.json order = canonical track order
        ti = int(t["track_index"])
        if NODE_TYPE_NAMES.index(t["node_type"]) not in target_codes:
            continue
        first, last = spans[ti]
        start = first + ((cfg.anchor_phase_frames - first) % cfg.anchor_stride_frames)
        for t0 in range(start, last + 1, cfg.anchor_stride_frames):
            hist = t0 + hoff
            fut = t0 + foff
            reason = None
            if hist[0] < video_first_frame:
                reason = 0
            elif fut[-1] > video_last_frame:
                reason = 1
            elif run.node_at(ti, t0) is None:
                reason = 2
            elif hist[0] < first:
                reason = 3
            else:
                hn = [run.node_at(ti, int(f)) for f in hist]
                if any(x is None for x in hn):
                    reason = 4
                elif fut[-1] > last:
                    reason = 5
                else:
                    tgt = extract_targets(run, ti, fut.tolist(), cfg.target_source_mode)
                    if not tgt.complete:
                        reason = 6
            if reason is not None:
                rej_t.append(ti)
                rej_a.append(t0)
                rej_r.append(reason)
                continue
            acc["tix"].append(ti)
            acc["t0"].append(t0)
            acc["hn"].append(hn)
            acc["tgt"].append(tgt)
    W = len(acc["t0"])
    t0a = np.array(acc["t0"], np.int64).reshape(W)
    hf = t0a[:, None] + hoff[None, :]
    ff = t0a[:, None] + foff[None, :]
    hn = np.array(acc["hn"], np.int64).reshape(W, H)
    tn = np.array([x.node_index for x in acc["tgt"]], np.int64).reshape(W, F)
    tix = np.array(acc["tix"], np.int32).reshape(W)
    anchor_node = hn[:, -1] if W else np.zeros(0, np.int64)
    ids = np.array([run.track_id(int(i)) for i in tix], dtype=str) if W else np.zeros(0, dtype="<U1")
    return WindowSet(
        window_id=np.arange(W, dtype=np.int64), target_track_index=tix, target_track_id=ids, anchor_frame=t0a,
        history_frames=hf, history_timestamps=run.timestamp[hn] if W else np.zeros((0, H)),
        future_frames=ff, future_timestamps=run.timestamp[tn] if W else np.zeros((0, F)),
        history_target_task3_node=hn, history_target_source_line=run.source_line[hn] if W else np.zeros((0, H), np.int64),
        history_target_uv_px=run.features[hn][:, :, 0:2] if W else np.zeros((0, H, 2), np.float32),
        target_future_task3_node=tn, target_future_source_line=run.source_line[tn] if W else np.zeros((0, F), np.int64),
        target_future_uv_px=np.array([x.uv_px for x in acc["tgt"]], np.float32).reshape(W, F, 2),
        target_future_box_h_px=np.array([x.box_h_px for x in acc["tgt"]], np.float32).reshape(W, F),
        target_future_border=np.array([x.border for x in acc["tgt"]], bool).reshape(W, F),
        target_future_valid=np.array([x.valid for x in acc["tgt"]], bool).reshape(W, F),
        target_border_at_anchor=run.border[anchor_node] if W else np.zeros(0, bool),
        rejected_track_index=np.array(rej_t, np.int32), rejected_anchor_frame=np.array(rej_a, np.int64),
        rejected_reason_code=np.array(rej_r, np.int8), video_first_frame=int(video_first_frame), video_last_frame=int(video_last_frame))


__all__ = ["WindowSet", "build_windows", "REJECTION_REASONS"]
