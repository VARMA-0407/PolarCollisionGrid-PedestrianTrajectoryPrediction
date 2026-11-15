"""Strict configuration of the Phase 6 image-space conflict representation (unknown keys rejected; '_' keys are documentation).

Every threshold here is an EXPLORATORY, unvalidated image-space parameter. None of them is a physical or safety boundary.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from .errors import ConflictContractError

COORDINATE_SYSTEM = "captured_1920x1080"
COORDINATE_UNIT = "pixel"
CONTEXT_TYPES = ("vehicle", "pedestrian", "bicyclist")
CONTEXT_MOTION_MODELS = ("B0_zero_velocity", "B1_last_valid_causal_velocity", "B2_history_linear_regression")
PAIR_TYPE_CODES = {"pedestrian_vehicle": 0, "pedestrian_pedestrian": 1, "pedestrian_bicyclist": 2}
BOX_ASSUMPTION = "CONSTANT_ANCHOR_BOX_SIZE_AT_PREDICTED_FOOTPOINT"


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True)
class ConflictConfig:
    config_version: str
    label: str
    coordinate_system: str
    coordinate_unit: str
    context_agent_types: Tuple[str, ...]
    context_motion_model: str
    baseline_evaluation_config_path: str
    box_assumption: str
    min_valid_steps: int
    score_kappa_sep_over_h: float
    sensitivity_sep_over_h: Tuple[float, ...]
    norm_epsilon: float
    candidate_label_policy: str

    def __post_init__(self):
        if not (isinstance(self.config_version, str) and self.config_version.strip()):
            raise ConflictContractError("CONFIG", "config_version must be a non-empty string")
        if self.label != "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT":
            raise ConflictContractError("CONFIG", "label must be PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT")
        if self.coordinate_system != COORDINATE_SYSTEM or self.coordinate_unit != COORDINATE_UNIT:
            raise ConflictContractError("COORDINATE_SYSTEM", "only %s / %s is authorised (no metric geometry exists)" % (COORDINATE_SYSTEM, COORDINATE_UNIT))
        types = tuple(self.context_agent_types)
        if not types or len(set(types)) != len(types) or any(t not in CONTEXT_TYPES for t in types):
            raise ConflictContractError("CONFIG", "context_agent_types must be distinct values of %s" % (CONTEXT_TYPES,))
        object.__setattr__(self, "context_agent_types", types)
        if self.context_motion_model not in CONTEXT_MOTION_MODELS:
            raise ConflictContractError("CONFIG", "context_motion_model must be one of %s (the only supported agent-motion predictors)" % (CONTEXT_MOTION_MODELS,))
        if not (isinstance(self.baseline_evaluation_config_path, str) and self.baseline_evaluation_config_path):
            raise ConflictContractError("CONFIG", "baseline_evaluation_config_path is required (predictor parameters come from it)")
        if self.box_assumption != BOX_ASSUMPTION:
            raise ConflictContractError("CONFIG", "box_assumption must be %s" % BOX_ASSUMPTION)
        if not (isinstance(self.min_valid_steps, int) and not isinstance(self.min_valid_steps, bool) and self.min_valid_steps >= 1):
            raise ConflictContractError("CONFIG", "min_valid_steps must be a positive integer")
        if not (_num(self.score_kappa_sep_over_h) and self.score_kappa_sep_over_h > 0):
            raise ConflictContractError("CONFIG", "score_kappa_sep_over_h must be positive")
        th = tuple(float(x) for x in self.sensitivity_sep_over_h)
        if not th or any(not _num(x) or x <= 0 for x in th) or list(th) != sorted(set(th)):
            raise ConflictContractError("CONFIG", "sensitivity_sep_over_h must be a non-empty strictly increasing list of positive numbers")
        object.__setattr__(self, "sensitivity_sep_over_h", th)
        if not (_num(self.norm_epsilon) and self.norm_epsilon > 0):
            raise ConflictContractError("CONFIG", "norm_epsilon must be positive")
        if self.candidate_label_policy != "CANDIDATE_ONLY_NEVER_SUPERVISION":
            raise ConflictContractError("CONFIG", "candidate_label_policy must be CANDIDATE_ONLY_NEVER_SUPERVISION")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ConflictConfig":
        if not isinstance(d, Mapping):
            raise ConflictContractError("CONFIG", "configuration must be a JSON object")
        data = {k: v for k, v in d.items() if not k.startswith("_")}
        unknown, missing = set(data) - set(cls.__dataclass_fields__), set(cls.__dataclass_fields__) - set(data)
        if unknown or missing:
            raise ConflictContractError("CONFIG", "unknown keys %s, missing keys %s" % (sorted(unknown), sorted(missing)))
        for k in ("context_agent_types", "sensitivity_sep_over_h"):
            if not isinstance(data[k], (list, tuple)):
                raise ConflictContractError("CONFIG", "%s must be a list" % k)
        return cls(**data)

    @classmethod
    def load(cls, path) -> "ConflictConfig":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConflictContractError("CONFIG", "cannot read %s: %s" % (path, exc))
        return cls.from_dict(d)

    def to_dict(self):
        d = asdict(self)
        d["context_agent_types"] = list(d["context_agent_types"])
        d["sensitivity_sep_over_h"] = list(d["sensitivity_sep_over_h"])
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


__all__ = ["ConflictConfig", "COORDINATE_SYSTEM", "COORDINATE_UNIT", "PAIR_TYPE_CODES", "BOX_ASSUMPTION", "CONTEXT_MOTION_MODELS"]
