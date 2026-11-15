"""Strict, versioned configuration of the alert state machine (unknown keys rejected; '_' keys are documentation)."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import AlertContractError

CATEGORY_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


def _pos_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


@dataclass(frozen=True)
class AlertConfig:
    config_version: str
    config_status: str
    source_fps: float
    enter_min_category: str
    hold_min_category: str
    escalate_min_category: str
    escalated_hold_min_category: str
    enter_persistence: int
    escalate_persistence: int
    deescalate_persistence: int
    clear_persistence: int
    max_insufficient_observations: int
    max_gap_frames: int
    cooldown_frames: int
    conflicting_indicators_policy: str

    def __post_init__(self):
        if self.config_status != "ENGINEERING_CONFIGURATION_NOT_VALIDATED":
            raise AlertContractError("CONFIG", "config_status must be ENGINEERING_CONFIGURATION_NOT_VALIDATED (no validated evidence exists for any threshold)")
        if not (isinstance(self.config_version, str) and self.config_version):
            raise AlertContractError("CONFIG", "config_version required")
        if not (isinstance(self.source_fps, (int, float)) and not isinstance(self.source_fps, bool) and math.isfinite(self.source_fps) and self.source_fps > 0):
            raise AlertContractError("CONFIG", "source_fps must be a positive number")
        for k in ("enter_min_category", "hold_min_category", "escalate_min_category", "escalated_hold_min_category"):
            if getattr(self, k) not in CATEGORY_RANK:
                raise AlertContractError("CONFIG", "%s must be one of %s" % (k, sorted(CATEGORY_RANK)))
        r = self.ranks()
        if not (r["hold"] <= r["enter"] <= r["escalate"] and r["hold"] <= r["escalated_hold"] <= r["escalate"] and r["enter"] <= r["escalate"]):
            raise AlertContractError("CONFIG", "category ranks must satisfy hold <= enter <= escalate and hold <= escalated_hold <= escalate (hysteresis)")
        if r["hold"] == r["enter"]:
            pass  # allowed but gives no entry/exit hysteresis; persistence still applies
        if not _pos_int(self.enter_persistence) or self.enter_persistence < 2:
            raise AlertContractError("CONFIG", "enter_persistence must be an integer >= 2 (a warning needs at least two consecutive elevated observations)")
        for k in ("escalate_persistence", "deescalate_persistence", "clear_persistence", "max_insufficient_observations", "max_gap_frames"):
            if not _pos_int(getattr(self, k)):
                raise AlertContractError("CONFIG", "%s must be a positive integer" % k)
        if not (isinstance(self.cooldown_frames, int) and not isinstance(self.cooldown_frames, bool) and self.cooldown_frames >= 0):
            raise AlertContractError("CONFIG", "cooldown_frames must be a non-negative integer")
        if self.conflicting_indicators_policy not in ("flag_only", "treat_as_insufficient"):
            raise AlertContractError("CONFIG", "conflicting_indicators_policy must be flag_only or treat_as_insufficient")

    def ranks(self):
        return {"enter": CATEGORY_RANK[self.enter_min_category], "hold": CATEGORY_RANK[self.hold_min_category],
                "escalate": CATEGORY_RANK[self.escalate_min_category], "escalated_hold": CATEGORY_RANK[self.escalated_hold_min_category]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AlertConfig":
        if not isinstance(d, Mapping):
            raise AlertContractError("CONFIG", "configuration must be a JSON object")
        data = {k: v for k, v in d.items() if not k.startswith("_")}
        unk, mis = set(data) - set(cls.__dataclass_fields__), set(cls.__dataclass_fields__) - set(data)
        if unk or mis:
            raise AlertContractError("CONFIG", "unknown keys %s, missing keys %s" % (sorted(unk), sorted(mis)))
        return cls(**data)

    @classmethod
    def load(cls, path) -> "AlertConfig":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise AlertContractError("CONFIG", "cannot read %s: %s" % (path, exc))

    def to_dict(self):
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = ["AlertConfig", "CATEGORY_RANK"]
