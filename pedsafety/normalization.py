"""Stage 5: TRAIN-ONLY normalisation.

``NormalizationState.fit`` accepts windows from the training partition only and
refuses anything else. There is no refit / partial-fit method: once fitted, the
state is immutable and is simply applied to every partition. Statistics are kept
per ``SplitWindow.norm_key()`` (dataset, coordinate mode, units, frame,
calibration) so image, projective and metric data, or different datasets, never
share statistics; a window whose key was never seen in training is an error,
not a fallback.

Fitted per channel from HISTORY values that are available and finite (targets and
identifiers never contribute): count, mean, std (population), scale, a ``constant``
flag. Channels default to the linear ones; heading is excluded because the mean /
std of an angle is meaningless. Missing and non-finite values are neither fitted
nor filled: they stay NaN with their availability mask. Timestamps, frame indices,
ids and class labels are never normalised.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .errors import ConfigError, PedsafetyError
from .geometry import EPS
from .windows import HISTORY_CHANNELS, SplitWindow

NORMALIZATION_VERSION = "pedsafety-normalization-v1"
LINEAR_CHANNELS = ("xs", "ys", "vx", "vy", "speed", "ax", "ay")
STATUS_OK, STATUS_CONSTANT, STATUS_INSUFFICIENT = "ok", "constant", "insufficient"


class NormalizationError(PedsafetyError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__("[%s] %s" % (code, message))


@dataclass(frozen=True)
class NormalizationConfig:
    """constant_policy: 'center_only' (scale 1.0, flagged) or 'error'.
    constant_rel_tol: a channel is constant when std <= tol * max(|mean|, 1);
    default sqrt(eps) (floating-point noise only)."""

    channels: Tuple[str, ...] = LINEAR_CHANNELS
    min_count: int = 2
    constant_policy: str = "center_only"
    constant_rel_tol: float = math.sqrt(EPS)

    def __post_init__(self):
        ch = tuple(self.channels)
        if not ch:
            raise ConfigError("at least one channel must be normalised")
        bad = [c for c in ch if c not in LINEAR_CHANNELS]
        if bad:
            raise ConfigError("channels %s cannot be z-score normalised (allowed: %s; heading is circular and ids/timestamps are not features)"
                              % (bad, LINEAR_CHANNELS))
        if len(set(ch)) != len(ch):
            raise ConfigError("duplicate channels")
        object.__setattr__(self, "channels", ch)
        if isinstance(self.min_count, bool) or not isinstance(self.min_count, int) or self.min_count < 2:
            raise ConfigError("min_count must be an integer >= 2")
        if self.constant_policy not in ("center_only", "error"):
            raise ConfigError("constant_policy must be 'center_only' or 'error'")
        if not (isinstance(self.constant_rel_tol, (int, float)) and math.isfinite(self.constant_rel_tol) and self.constant_rel_tol > 0):
            raise ConfigError("constant_rel_tol must be a positive finite number")

    def to_dict(self) -> Dict[str, Any]:
        return {"channels": list(self.channels), "min_count": self.min_count, "constant_policy": self.constant_policy,
                "constant_rel_tol": self.constant_rel_tol}


@dataclass(frozen=True)
class ChannelStats:
    count: int
    mean: Optional[float]
    std: Optional[float]
    scale: Optional[float]
    status: str
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "std": self.std, "scale": self.scale, "status": self.status, "unit": self.unit}


@dataclass(frozen=True)
class NormalizedWindow:
    """Normalised copies; the source window is unchanged. Unavailable values stay NaN."""

    window: SplitWindow
    history: Mapping[str, np.ndarray]            # normalised channels
    history_passthrough: Mapping[str, np.ndarray]    # un-normalised channels (e.g. heading)
    target_xy: np.ndarray
    target_normalized: bool
    state_sha256: str


@dataclass(frozen=True)
class NormalizationState:
    config: NormalizationConfig
    fitted_on: Mapping[str, Any]
    stats: Mapping[str, Mapping[str, ChannelStats]]    # norm_key -> channel -> stats
    version: str = NORMALIZATION_VERSION

    # ---------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, train_windows: Sequence[SplitWindow], config: Optional[NormalizationConfig] = None,
            fit_info: Optional[Mapping[str, Any]] = None) -> "NormalizationState":
        cfg = config or NormalizationConfig()
        wins = list(train_windows)
        if not wins:
            raise NormalizationError("NO_TRAINING_WINDOWS", "cannot fit normalisation without training windows")
        parts = sorted({w.partition for w in wins})
        if parts != ["train"]:
            raise NormalizationError("LEAKAGE_GUARD", "normalisation may be fitted on the training partition only; got windows from %s" % parts)
        by_key: Dict[str, list] = {}
        for w in wins:
            by_key.setdefault(w.norm_key(), []).append(w)
        stats: Dict[str, Dict[str, ChannelStats]] = {}
        for key in sorted(by_key):
            ws = by_key[key]
            chans: Dict[str, ChannelStats] = {}
            for ch in cfg.channels:
                # Overlapping windows repeat observations: count each (track, frame) once.
                seen = set()
                collected = []
                for w in ws:
                    for i in range(w.obs_len):
                        if w.history_available[ch][i]:
                            ident = (w.track_id, int(w.history_frame_index[i]))
                            if ident not in seen:
                                seen.add(ident)
                                collected.append(float(w.history[ch][i]))
                vals = np.array(collected, dtype=float)
                vals = vals[np.isfinite(vals)]
                unit = ws[0].channel_units()[ch]
                n = int(vals.size)
                if n < cfg.min_count:
                    chans[ch] = ChannelStats(n, None, None, None, STATUS_INSUFFICIENT, unit)
                    continue
                mean, std = float(np.mean(vals)), float(np.std(vals))
                if std <= cfg.constant_rel_tol * max(abs(mean), 1.0):
                    if cfg.constant_policy == "error":
                        raise NormalizationError("CONSTANT_CHANNEL", "channel %r is constant in training data for key %s" % (ch, key))
                    chans[ch] = ChannelStats(n, mean, std, 1.0, STATUS_CONSTANT, unit)
                else:
                    chans[ch] = ChannelStats(n, mean, std, std, STATUS_OK, unit)
            stats[key] = chans
        info = dict(fit_info or {})
        info.update({"partition": "train", "n_windows": len(wins),
                     "train_groups": sorted({w.group_id for w in wins}),
                     "train_window_ids_sha256": hashlib.sha256("\n".join(sorted(w.window_id for w in wins)).encode("utf-8")).hexdigest(),
                     "views": sorted({w.view for w in wins})})
        return cls(cfg, info, stats)

    # -------------------------------------------------------------- applying
    def _key_stats(self, w: SplitWindow) -> Mapping[str, ChannelStats]:
        k = w.norm_key()
        if k not in self.stats:
            raise NormalizationError("UNSEEN_KEY", "no training statistics for %s; windows from a coordinate space, unit or dataset that "
                                     "training never saw are not normalised by fallback" % k)
        return self.stats[k]

    def transform(self, w: SplitWindow) -> NormalizedWindow:
        st = self._key_stats(w)
        hist: Dict[str, np.ndarray] = {}
        for ch in self.config.channels:
            s = st[ch]
            avail = w.history_available[ch]
            if s.status == STATUS_INSUFFICIENT:
                if avail.any():
                    raise NormalizationError("INSUFFICIENT_STATS", "channel %r has too few training values (%d) but the window has data" % (ch, s.count))
                hist[ch] = np.full(w.obs_len, np.nan)
                continue
            out = (w.history[ch] - s.mean) / s.scale
            out[~avail] = np.nan
            out.flags.writeable = False
            hist[ch] = out
        passthrough = {c: w.history[c] for c in HISTORY_CHANNELS if c not in self.config.channels}
        norm_xy = "xs" in self.config.channels and "ys" in self.config.channels and st["xs"].status != STATUS_INSUFFICIENT \
            and st["ys"].status != STATUS_INSUFFICIENT
        if norm_xy:
            t = (w.target_xy - np.array([st["xs"].mean, st["ys"].mean])) / np.array([st["xs"].scale, st["ys"].scale])
        else:
            t = w.target_xy.copy()
        t.flags.writeable = False
        return NormalizedWindow(w, hist, passthrough, t, bool(norm_xy), self.sha256())

    def denormalize_xy(self, w: SplitWindow, xy: Any) -> np.ndarray:
        """Invert the position normalisation (e.g. for model predictions)."""
        st = self._key_stats(w)
        if not ("xs" in self.config.channels and "ys" in self.config.channels):
            raise NormalizationError("POSITIONS_NOT_NORMALISED", "xs/ys are not normalised channels")
        return np.asarray(xy, dtype=float) * np.array([st["xs"].scale, st["ys"].scale]) + np.array([st["xs"].mean, st["ys"].mean])

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "config": self.config.to_dict(), "fitted_on": dict(self.fitted_on),
                "stats": {k: {c: s.to_dict() for c, s in chans.items()} for k, chans in sorted(self.stats.items())}}

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NormalizationState":
        if d.get("version") != NORMALIZATION_VERSION:
            raise NormalizationError("BAD_VERSION", "unsupported normalisation state version %r" % (d.get("version"),))
        c = d["config"]
        cfg = NormalizationConfig(tuple(c["channels"]), c["min_count"], c["constant_policy"], c["constant_rel_tol"])
        stats = {k: {ch: ChannelStats(**s) for ch, s in chans.items()} for k, chans in d["stats"].items()}
        return cls(cfg, dict(d["fitted_on"]), stats)

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NormalizationState":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
