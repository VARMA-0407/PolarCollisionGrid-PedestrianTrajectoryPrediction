"""Task 3 configuration (strict: unknown keys rejected; keys starting with '_' are documentation)."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from ..errors import ConfigError, PedsafetyError

NODE_TYPES = ("pedestrian", "vehicle", "bicyclist")              # canonical agent_class values (Q3)
DEFAULT_RELATIONS = (("pedestrian", "pedestrian"), ("vehicle", "pedestrian"), ("pedestrian", "vehicle"),
                     ("bicyclist", "pedestrian"), ("pedestrian", "bicyclist"), ("bicyclist", "vehicle"),
                     ("vehicle", "bicyclist"))
OPTIONAL_VV = ("vehicle", "vehicle")


class InteractionContractError(PedsafetyError):
    """An upstream input contract is broken. ``code`` is stable; nothing is repaired."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__("[%s] %s" % (code, message))


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True)
class InteractionConfig:
    config_version: str
    expected_coordinate_system: str
    enable_vehicle_vehicle: bool = False
    radius_thresholds_sep_over_h: Tuple[float, ...] = (2.0, 5.0, 10.0)
    knn_k: Tuple[int, ...] = (4, 8)
    noise_reference_per_agent_velocity_px_s: float = 48.0

    def __post_init__(self):
        for n in ("config_version", "expected_coordinate_system"):
            v = getattr(self, n)
            if not (isinstance(v, str) and v.strip()):
                raise ConfigError("%s must be a non-empty string" % n)
        if not isinstance(self.enable_vehicle_vehicle, bool):
            raise ConfigError("enable_vehicle_vehicle must be true or false")
        r = tuple(self.radius_thresholds_sep_over_h)
        if not r or not all(_num(x) and x > 0 for x in r) or list(r) != sorted(set(r)):
            raise ConfigError("radius_thresholds_sep_over_h must be a non-empty, strictly increasing list of positive numbers")
        k = tuple(self.knn_k)
        if not k or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in k) or list(k) != sorted(set(k)):
            raise ConfigError("knn_k must be a non-empty, strictly increasing list of integers >= 1")
        if not (_num(self.noise_reference_per_agent_velocity_px_s) and self.noise_reference_per_agent_velocity_px_s > 0):
            raise ConfigError("noise_reference_per_agent_velocity_px_s must be a positive finite number")
        object.__setattr__(self, "radius_thresholds_sep_over_h", tuple(float(x) for x in r))
        object.__setattr__(self, "knn_k", k)

    @property
    def relations(self) -> Tuple[Tuple[str, str], ...]:
        return DEFAULT_RELATIONS + ((OPTIONAL_VV,) if self.enable_vehicle_vehicle else ())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "InteractionConfig":
        if not isinstance(d, Mapping):
            raise ConfigError("configuration must be a JSON object")
        data = {k: v for k, v in d.items() if not k.startswith("_")}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError("unknown configuration keys: %s" % sorted(unknown))
        for k in ("radius_thresholds_sep_over_h", "knn_k"):
            if k in data and not isinstance(data[k], list):
                raise ConfigError("%s must be a list" % k)
        try:
            return cls(**data)
        except TypeError as exc:
            raise ConfigError("configuration is missing required keys: %s" % exc)

    @classmethod
    def load(cls, path) -> "InteractionConfig":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError("cannot read configuration %s: %s" % (path, exc))
        return cls.from_dict(d)

    def to_dict(self):
        d = asdict(self)
        d["radius_thresholds_sep_over_h"] = list(d["radius_thresholds_sep_over_h"])
        d["knn_k"] = list(d["knn_k"])
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
