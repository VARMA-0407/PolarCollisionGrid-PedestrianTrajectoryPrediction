"""Hand-built history inputs for the P5.4 baseline/evaluator tests (no PIE data)."""

import numpy as np

from pedsafety.phase5.baselines import HistoryInput

H, F = 6, 15
TAU = np.array([(m + 1) * 0.1 for m in range(F)])
HIST_T = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0])


def make_hist(uv, *, t=None, vel=None, vvalid=None, pvalid=None, anchor=10.0, cs="captured_1920x1080", unit="pixel", tau=None):
    uv = np.asarray(uv, np.float64)
    n = uv.shape[0]
    t = np.broadcast_to(anchor + (HIST_T if t is None else np.asarray(t, np.float64)), (n, H)).copy()
    vel = np.zeros((n, H, 2)) if vel is None else np.asarray(vel, np.float64)
    vvalid = np.zeros((n, H), bool) if vvalid is None else np.asarray(vvalid, bool)
    pvalid = np.ones((n, H), bool) if pvalid is None else np.asarray(pvalid, bool)
    return HistoryInput(window_id=np.arange(n) + 100, target_track_index=np.arange(n), target_track_id=["pie:video_0004:%d" % i for i in range(n)],
                        anchor_frame=np.full(n, 300), anchor_timestamp=t[:, -1], history_timestamps=t, history_uv_px=uv,
                        history_position_valid=pvalid, history_velocity_px_s=vel, history_velocity_valid=vvalid,
                        horizon_offsets_s=TAU if tau is None else tau, coordinate_system=cs, position_unit=unit, velocity_source="test")


def linear_history(p0, v, t=None):
    """Positions on p(t) = p0 + v * t at the history times (relative to the anchor)."""
    t = HIST_T if t is None else np.asarray(t, np.float64)
    return (np.asarray(p0, np.float64)[None, :] + np.asarray(v, np.float64)[None, :] * t[:, None])[None]
