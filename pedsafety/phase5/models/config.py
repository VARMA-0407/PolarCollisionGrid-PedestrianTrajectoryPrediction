"""Strict configuration of the P5.5 spatial-temporal encoder (unknown keys rejected; keys starting with '_' are documentation).

The history length is NOT stored here: it comes from the accepted window configuration (``WindowConfig.history_length``)
or from the input tensors. The model only declares a temporal receptive field and (optionally) requires it to cover the
history (``require_full_history_coverage``).
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from ...errors import ConfigError

KNOWN_NODE_TYPES = ("pedestrian", "vehicle", "bicyclist")
# approved Task 3 relation set (docs/phase4 decision record Q3/V-V); bicyclist->bicyclist is NOT represented by Task 3
APPROVED_RELATIONS = ("pedestrian__pedestrian", "vehicle__pedestrian", "pedestrian__vehicle", "bicyclist__pedestrian",
                      "pedestrian__bicyclist", "bicyclist__vehicle", "vehicle__bicyclist")
OPTIONAL_VV_RELATION = "vehicle__vehicle"
DEFAULT_RELATIONS = ("pedestrian__pedestrian", "vehicle__pedestrian", "pedestrian__vehicle")
POSITION_MODES = ("target_centred",)
SCALING_SOURCES = ("NOMINAL_DEVELOPMENT_DEFAULT_NOT_FITTED", "TRAIN_PARTITION_STATISTICS")

# raw Task 3 node feature order (verified against the feature dictionary in P5.3): see task3_source.NODE_FEATURES
NODE_INPUT_FEATURES = ("rel_u", "rel_v", "vel_u", "vel_v", "speed", "heading_sin", "heading_cos", "box_h", "frames_since_prev_obs")


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class EncoderConfig:
    config_version: str
    hidden_dim: int
    num_heads: int
    num_spatial_layers: int
    leaky_relu_slope: float
    node_types: Tuple[str, ...]
    relations: Tuple[str, ...]
    allow_vehicle_vehicle: bool
    bicyclist_context_enabled: bool
    use_edge_flags: bool
    position_mode: str
    tcn_kernel_size: int
    tcn_dilations: Tuple[int, ...]
    require_full_history_coverage: bool
    dropout: float
    attention_dropout: float
    feature_scaling: Dict[str, Any]
    init_seed: int

    def __post_init__(self):
        if not (isinstance(self.config_version, str) and self.config_version.strip()):
            raise ConfigError("config_version must be a non-empty string")
        for n in ("hidden_dim", "num_heads", "num_spatial_layers", "tcn_kernel_size", "init_seed"):
            if not _int(getattr(self, n)) or (getattr(self, n) < 1 and n != "init_seed") or getattr(self, n) < 0:
                raise ConfigError("%s must be a positive integer" % n)
        if self.hidden_dim % self.num_heads:
            raise ConfigError("hidden_dim must be divisible by num_heads")
        if not (isinstance(self.leaky_relu_slope, (int, float)) and 0 <= self.leaky_relu_slope < 1):
            raise ConfigError("leaky_relu_slope must be in [0, 1)")
        for n in ("dropout", "attention_dropout"):
            v = getattr(self, n)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v < 1):
                raise ConfigError("%s must be in [0, 1)" % n)
        nt = tuple(self.node_types)
        if not nt or len(set(nt)) != len(nt) or not all(t in KNOWN_NODE_TYPES for t in nt) or "pedestrian" not in nt:
            raise ConfigError("node_types must be distinct known types and include 'pedestrian'")
        if "bicyclist" in nt and not self.bicyclist_context_enabled:
            raise ConfigError("'bicyclist' nodes are semantically uncertain (COCO bicycle detections); set bicyclist_context_enabled=true explicitly")
        if self.bicyclist_context_enabled and "bicyclist" not in nt:
            raise ConfigError("bicyclist_context_enabled requires 'bicyclist' in node_types")
        rel = tuple(self.relations)
        if not rel or len(set(rel)) != len(rel):
            raise ConfigError("relations must be a non-empty list of distinct relation names")
        allowed = set(APPROVED_RELATIONS) | ({OPTIONAL_VV_RELATION} if self.allow_vehicle_vehicle else set())
        for r in rel:
            if r not in allowed:
                raise ConfigError("relation %r is not permitted (approved: %s; vehicle__vehicle needs allow_vehicle_vehicle; "
                                  "bicyclist__bicyclist does not exist in Task 3)" % (r, sorted(allowed)))
            src, dst = r.split("__")
            if src not in nt or dst not in nt:
                raise ConfigError("relation %r uses a node type outside node_types %s" % (r, nt))
        if not any(r.endswith("__pedestrian") for r in rel):
            raise ConfigError("at least one relation must target pedestrians (the forecast targets)")
        if self.position_mode not in POSITION_MODES:
            raise ConfigError("position_mode must be one of %s" % (POSITION_MODES,))
        dil = tuple(self.tcn_dilations)
        if not dil or not all(_int(d) and d >= 1 for d in dil):
            raise ConfigError("tcn_dilations must be a non-empty list of integers >= 1")
        fs = self.feature_scaling
        if not isinstance(fs, Mapping) or fs.get("source") not in SCALING_SOURCES:
            raise ConfigError("feature_scaling.source must be one of %s" % (SCALING_SOURCES,))
        if fs["source"] == "NOMINAL_DEVELOPMENT_DEFAULT_NOT_FITTED":
            for k in ("node", "edge"):
                if not isinstance(fs.get(k), Mapping):
                    raise ConfigError("feature_scaling.%s must be a mapping of nominal scales" % k)
        object.__setattr__(self, "node_types", nt)
        object.__setattr__(self, "relations", rel)
        object.__setattr__(self, "tcn_dilations", dil)

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def receptive_field(self) -> int:
        """Temporal receptive field in samples: 1 + (kernel - 1) * sum(dilations) (one causal conv per dilation level)."""
        return 1 + (self.tcn_kernel_size - 1) * sum(self.tcn_dilations)

    def check_history(self, history_length: int) -> None:
        if not _int(history_length) or history_length < 1:
            raise ConfigError("history_length must be a positive integer")
        if self.require_full_history_coverage and self.receptive_field < history_length:
            raise ConfigError("temporal receptive field %d does not cover history_length %d (kernel %d, dilations %s); "
                              "increase dilations or set require_full_history_coverage=false"
                              % (self.receptive_field, history_length, self.tcn_kernel_size, self.tcn_dilations))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EncoderConfig":
        if not isinstance(d, Mapping):
            raise ConfigError("configuration must be a JSON object")
        data = {k: v for k, v in d.items() if not k.startswith("_")}
        unknown = set(data) - set(cls.__dataclass_fields__)
        missing = set(cls.__dataclass_fields__) - set(data)
        if unknown or missing:
            raise ConfigError("unknown keys %s, missing keys %s" % (sorted(unknown), sorted(missing)))
        for k in ("node_types", "relations", "tcn_dilations"):
            if not isinstance(data[k], (list, tuple)):
                raise ConfigError("%s must be a list" % k)
        return cls(**data)

    @classmethod
    def load(cls, path) -> "EncoderConfig":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError("cannot read configuration %s: %s" % (path, exc))
        return cls.from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("node_types", "relations", "tcn_dilations"):
            d[k] = list(d[k])
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


__all__ = ["EncoderConfig", "APPROVED_RELATIONS", "DEFAULT_RELATIONS", "OPTIONAL_VV_RELATION", "NODE_INPUT_FEATURES", "KNOWN_NODE_TYPES"]
