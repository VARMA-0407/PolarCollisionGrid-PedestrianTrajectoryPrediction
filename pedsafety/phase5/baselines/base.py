"""Common predictor interface: history-only input, single-trajectory output.

``HistoryInput`` is the ONLY object a predictor receives. It is built from the P5.3 window dataset by
:func:`build_history_input`, which copies history fields only; it has no future positions, future timestamps, future
validity or target node references (``__slots__`` forbids adding any). The prediction query times are the configured
horizon offsets ``tau_m = (m + 1) * stride / fps`` relative to the anchor timestamp, derived from the window
configuration, never read from target arrays.

Prediction layout: ``positions`` float64 ``[N, F, 2]`` (u, v in pixels of ``captured_1920x1080``) for the anchor-relative
times ``horizon_offsets_s``; ``valid`` bool ``[N, F]``; ``window_valid`` bool ``[N]``; ``invalid_reason`` int8 ``[N]``
(index into the predictor's ``INVALID_REASONS``, -1 when valid). Invalid entries are NaN, never zero.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from ..errors import WindowContractError

AUTHORIZED_COORDINATE_SYSTEM = "captured_1920x1080"
POSITION_UNIT = "pixel"


class HistoryInput:
    """History-only view of N target windows (H history samples each). Immutable after construction."""

    __slots__ = ("window_id", "target_track_index", "target_track_id", "anchor_frame", "anchor_timestamp", "history_timestamps",
                 "history_uv_px", "history_position_valid", "history_velocity_px_s", "history_velocity_valid", "horizon_offsets_s",
                 "coordinate_system", "position_unit", "velocity_source", "_frozen")

    def __init__(self, *, window_id, target_track_index, target_track_id, anchor_frame, anchor_timestamp, history_timestamps,
                 history_uv_px, history_position_valid, history_velocity_px_s, history_velocity_valid, horizon_offsets_s,
                 coordinate_system, position_unit, velocity_source):
        object.__setattr__(self, "_frozen", False)
        # np.array copies: freezing must never touch the caller's arrays, and later caller edits cannot leak in
        self.window_id = np.array(window_id, np.int64)
        self.target_track_index = np.array(target_track_index, np.int32)
        self.target_track_id = np.array(target_track_id, dtype=str)
        self.anchor_frame = np.array(anchor_frame, np.int64)
        self.anchor_timestamp = np.array(anchor_timestamp, np.float64)
        self.history_timestamps = np.array(history_timestamps, np.float64)
        self.history_uv_px = np.array(history_uv_px, np.float64)
        self.history_position_valid = np.array(history_position_valid, bool)
        self.history_velocity_px_s = np.array(history_velocity_px_s, np.float64)
        self.history_velocity_valid = np.array(history_velocity_valid, bool)
        self.horizon_offsets_s = np.array(horizon_offsets_s, np.float64)
        self.coordinate_system = str(coordinate_system)
        self.position_unit = str(position_unit)
        self.velocity_source = str(velocity_source)
        self._validate()
        for name in self.__slots__:
            v = getattr(self, name)
            if isinstance(v, np.ndarray):
                v.setflags(write=False)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("HistoryInput is immutable")
        object.__setattr__(self, name, value)

    @property
    def n(self) -> int:
        return int(self.window_id.shape[0])

    @property
    def h(self) -> int:
        return int(self.history_timestamps.shape[1])

    @property
    def f(self) -> int:
        return int(self.horizon_offsets_s.shape[0])

    def _validate(self):
        n = self.window_id.shape[0]
        if self.coordinate_system != AUTHORIZED_COORDINATE_SYSTEM or self.position_unit != POSITION_UNIT:
            raise WindowContractError("COORDINATE_SYSTEM_MISMATCH", "history input must be %s / %s, got %s / %s"
                                      % (AUTHORIZED_COORDINATE_SYSTEM, POSITION_UNIT, self.coordinate_system, self.position_unit))
        if self.history_timestamps.ndim != 2 or self.history_timestamps.shape[0] != n:
            raise WindowContractError("HISTORY_SHAPE", "history_timestamps must be [N, H]")
        h = self.history_timestamps.shape[1]
        for name, shape in (("target_track_index", (n,)), ("target_track_id", (n,)), ("anchor_frame", (n,)), ("anchor_timestamp", (n,)),
                            ("history_uv_px", (n, h, 2)), ("history_position_valid", (n, h)), ("history_velocity_px_s", (n, h, 2)),
                            ("history_velocity_valid", (n, h))):
            if getattr(self, name).shape != shape:
                raise WindowContractError("HISTORY_SHAPE", "%s has shape %s, expected %s" % (name, getattr(self, name).shape, shape))
        if self.horizon_offsets_s.ndim != 1 or not np.all(self.horizon_offsets_s > 0) or not np.all(np.diff(self.horizon_offsets_s) > 0):
            raise WindowContractError("HORIZON_OFFSETS", "horizon offsets must be positive and strictly increasing")
        if np.unique(self.window_id).shape[0] != n:
            raise WindowContractError("DUPLICATE_WINDOW_ID", "window ids must be unique")
        if n and not np.allclose(self.history_timestamps[:, -1], self.anchor_timestamp, atol=1e-9, rtol=0):
            raise WindowContractError("ANCHOR_TIMESTAMP", "the last history timestamp must equal the anchor timestamp")
        if n and not np.all(np.diff(self.history_timestamps, axis=1) > 0):
            raise WindowContractError("HISTORY_TIME_ORDER", "history timestamps must be strictly increasing")
        # values flagged valid must be finite; invalid values are ignored by every predictor
        if not np.all(np.isfinite(self.history_uv_px[self.history_position_valid])):
            raise WindowContractError("NONFINITE_VALID_VALUE", "a history position flagged valid is not finite")
        if not np.all(np.isfinite(self.history_velocity_px_s[self.history_velocity_valid])):
            raise WindowContractError("NONFINITE_VALID_VALUE", "a history velocity flagged valid is not finite")
        if not np.all(np.isfinite(self.history_timestamps)):
            raise WindowContractError("NONFINITE_VALID_VALUE", "non-finite history timestamp")


@dataclass
class Prediction:
    baseline_id: str
    positions: np.ndarray          # float64 [N, F, 2]
    valid: np.ndarray              # bool [N, F]
    window_valid: np.ndarray       # bool [N]
    invalid_reason: np.ndarray     # int8 [N], -1 when valid
    invalid_reason_names: Tuple[str, ...]
    window_id: np.ndarray          # int64 [N] (copied from the input for traceability)
    horizon_offsets_s: np.ndarray  # float64 [F]
    config: Dict[str, Any]
    aux: Dict[str, np.ndarray] = field(default_factory=dict)   # per-window diagnostics (e.g. velocity source index)
    coordinate_system: str = AUTHORIZED_COORDINATE_SYSTEM
    position_unit: str = POSITION_UNIT

    def invalid_counts(self) -> Dict[str, int]:
        c = {r: 0 for r in self.invalid_reason_names}
        for code in self.invalid_reason[~self.window_valid].tolist():
            c[self.invalid_reason_names[code]] += 1
        return c


class Predictor:
    """Base class. Subclasses set ``BASELINE_ID``, ``INVALID_REASONS`` and implement ``_predict``."""

    BASELINE_ID = "abstract"
    INVALID_REASONS: Tuple[str, ...] = ()

    def config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def predict(self, hist: HistoryInput) -> Prediction:
        if not isinstance(hist, HistoryInput):
            raise WindowContractError("PREDICTOR_INPUT_TYPE", "predictors accept only a HistoryInput (history-only); got %s" % type(hist).__name__)
        pos, wvalid, reason, aux = self._predict(hist)
        n, f = hist.n, hist.f
        if pos.shape != (n, f, 2):
            raise WindowContractError("PREDICTION_SHAPE", "%s produced %s, expected %s" % (self.BASELINE_ID, pos.shape, (n, f, 2)))
        pos = np.where(wvalid[:, None, None], pos, np.nan)
        valid = np.repeat(wvalid[:, None], f, axis=1)
        if not np.all(np.isfinite(pos[valid])):
            raise WindowContractError("NONFINITE_PREDICTION", "%s produced a non-finite value flagged valid" % self.BASELINE_ID)
        return Prediction(self.BASELINE_ID, pos.astype(np.float64), valid, wvalid.copy(), reason.astype(np.int8), self.INVALID_REASONS,
                          hist.window_id.copy(), hist.horizon_offsets_s.copy(), self.config(), aux)

    def _predict(self, hist: HistoryInput):
        raise NotImplementedError


__all__ = ["HistoryInput", "Prediction", "Predictor", "AUTHORIZED_COORDINATE_SYSTEM", "POSITION_UNIT"]
