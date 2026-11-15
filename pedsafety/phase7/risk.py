"""Risk scoring and categorisation on top of Phase 6 conflict outputs (numpy only, deterministic, no labels, no future data).

Five different things, never interchangeable:
  1. heuristic indicators   Phase 6 ``features`` (image-plane geometry, pixels)
  2. raw conflict score     Phase 6 ``ConflictScores`` (declares its kind; never a probability)
  3. calibrated probability only when a verified ``CalibrationArtifact`` matching the score's provenance is supplied; otherwise NaN + status NOT_CALIBRATED
  4. risk score             ``risk = base * (w + (1 - w) * urgency)``  [unitless, 0..1, an engineering ranking value, NOT a probability]
       base     = calibrated probability if available else the raw score          (``risk_basis`` records which)
       urgency  = 1 - tbar / T,  tbar = mode-weighted time of closest predicted image-plane approach (s), T = last horizon time (s);
                  unitless in [0,1]; NOT a time-to-collision. w = ``urgency_weight_floor`` keeps a non-urgent conflict from vanishing.
       severity: NOT SUPPORTED (no metric geometry / speed / mass / impact annotation) -> NaN with status SEVERITY_NOT_SUPPORTED.
  5. risk category          provisional engineering bins of the risk score from the versioned, persisted config. They are NOT validated safety levels
       unless ``threshold_evidence`` documents an empirical derivation (none exists). Invalid / out-of-distribution rows get INSUFFICIENT_EVIDENCE.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..phase6.pairs import ConflictPairBatch
from ..phase6.scoring import ConflictScores
from .calibration import CalibrationArtifact
from .errors import RiskContractError

CATEGORIES = ("LOW", "MODERATE", "HIGH", "CRITICAL")
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
OUTPUT_SCHEMA = "phase7-risk-output-1.0.0"


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True)
class RiskConfig:
    config_version: str
    category_status: str
    threshold_evidence: str
    moderate_min: float
    high_min: float
    critical_min: float
    urgency_weight_floor: float
    max_abs_log_scale_ratio: float
    max_indicator_disagreement: float
    min_valid_mode_mass: float

    def __post_init__(self):
        if not (isinstance(self.config_version, str) and self.config_version):
            raise RiskContractError("CONFIG", "config_version required")
        if self.category_status not in ("PROVISIONAL_ENGINEERING", "EMPIRICALLY_DERIVED"):
            raise RiskContractError("CONFIG", "category_status must be PROVISIONAL_ENGINEERING or EMPIRICALLY_DERIVED")
        if self.category_status == "EMPIRICALLY_DERIVED" and not self.threshold_evidence.startswith("artifact:"):
            raise RiskContractError("CONFIG", "EMPIRICALLY_DERIVED categories must cite their evidence as 'artifact:<path or hash>'")
        if not all(_num(v) for v in (self.moderate_min, self.high_min, self.critical_min, self.urgency_weight_floor, self.max_abs_log_scale_ratio,
                                     self.max_indicator_disagreement, self.min_valid_mode_mass)):
            raise RiskContractError("CONFIG", "numeric fields must be finite numbers")
        if not (0 < self.moderate_min < self.high_min < self.critical_min <= 1):
            raise RiskContractError("CONFIG", "thresholds must satisfy 0 < moderate < high < critical <= 1")
        if not (0 < self.urgency_weight_floor <= 1) or self.max_abs_log_scale_ratio <= 0 or not (0 < self.max_indicator_disagreement <= 1) or not (0 < self.min_valid_mode_mass <= 1):
            raise RiskContractError("CONFIG", "urgency floor in (0,1], scale limit > 0, disagreement and mode mass in (0,1]")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RiskConfig":
        data = {k: v for k, v in d.items() if not k.startswith("_")}
        unk, mis = set(data) - set(cls.__dataclass_fields__), set(cls.__dataclass_fields__) - set(data)
        if unk or mis:
            raise RiskContractError("CONFIG", "unknown keys %s, missing keys %s" % (sorted(unk), sorted(mis)))
        return cls(**data)

    @classmethod
    def load(cls, path) -> "RiskConfig":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise RiskContractError("CONFIG", "cannot read %s: %s" % (path, exc))

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class RiskOutput:
    schema: str
    pair: Dict[str, np.ndarray]          # identities / frame
    conflict_score: np.ndarray
    conflict_score_kind: str
    calibrated_probability: np.ndarray   # NaN unless calibration_status == CALIBRATED
    calibration_status: str
    temporal_urgency: np.ndarray
    severity: np.ndarray
    severity_status: str
    risk_score: np.ndarray
    risk_basis: str
    risk_category: np.ndarray            # str
    category_status: np.ndarray          # str
    reasons: np.ndarray                  # str ('' when none)
    masks: Dict[str, np.ndarray]
    provenance: Dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.conflict_score.shape[0])

    def to_records(self) -> List[Dict[str, Any]]:
        def f(x):
            return None if not np.isfinite(x) else float(x)
        out = []
        for i in range(self.n):
            out.append({"pedestrian_id": str(self.pair["ped_track_id"][i]), "vehicle_id": str(self.pair["other_track_id"][i]), "anchor_frame": int(self.pair["anchor_frame"][i]),
                        "window_id": int(self.pair["window_id"][i]), "conflict_score": f(self.conflict_score[i]), "conflict_score_kind": self.conflict_score_kind,
                        "calibrated_probability": f(self.calibrated_probability[i]), "calibration_status": self.calibration_status,
                        "temporal_urgency": f(self.temporal_urgency[i]), "severity": None, "severity_status": self.severity_status, "risk_score": f(self.risk_score[i]),
                        "risk_basis": self.risk_basis, "risk_category": str(self.risk_category[i]), "category_status": str(self.category_status[i]),
                        "reason": str(self.reasons[i]) or None, "masks": {k: bool(v[i]) for k, v in self.masks.items()}})
        return out


def _weighted_tmin(batch: ConflictPairBatch):
    f, m = batch.features, batch.masks
    w = np.where(m["mode_valid"], batch.mode_probabilities, 0.0)
    mass = w.sum(axis=1)
    t = np.where(m["mode_valid"], np.nan_to_num(f["t_min_sep_s"]), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mass > 0, (w * t).sum(axis=1) / np.where(mass > 0, mass, 1.0), np.nan), mass


def assess_risk(batch: ConflictPairBatch, scores: ConflictScores, cfg: RiskConfig, calibration: Optional[CalibrationArtifact] = None,
                provenance: Optional[Mapping[str, Any]] = None, checkpoint_file_sha256: Optional[str] = None, phase6_config_sha256: Optional[str] = None) -> RiskOutput:
    P = batch.n_pairs
    if scores.scores.shape != (P,) or not (np.array_equal(scores.window_id, batch.window_id) and np.array_equal(scores.other_track_index, batch.other_track_index)):
        raise RiskContractError("IDENTITY", "scores do not refer to the pairs of the batch")
    if batch.coordinate_system != "captured_1920x1080" or batch.coordinate_unit != "pixel":
        raise RiskContractError("COORDINATE_SYSTEM", "Phase 7 consumes image-space pixel conflict outputs of captured_1920x1080 only")
    s_valid = scores.valid & np.isfinite(scores.scores) & batch.masks["pair_valid"] & batch.context_prediction_valid
    raw = np.where(s_valid, scores.scores, np.nan)
    if np.any(raw[s_valid] < 0) or np.any(raw[s_valid] > 1):
        raise RiskContractError("SCORE_RANGE", "raw conflict scores must lie in [0,1]")
    if calibration is not None:
        if checkpoint_file_sha256 is None or phase6_config_sha256 is None:
            raise RiskContractError("PROVENANCE", "applying a calibration artifact requires the source checkpoint hash and Phase 6 config hash of the scores")
        calibration.verify()
        calibration.check_provenance(score_kind=scores.kind, checkpoint_file_sha256=checkpoint_file_sha256, phase6_config_sha256=phase6_config_sha256,
                                     coordinate_system=batch.coordinate_system)
        prob, cal_status, basis = calibration.apply(raw, s_valid), "CALIBRATED", "calibrated_probability"
        base = prob
    else:
        prob, cal_status, basis = np.full(P, np.nan), "NOT_CALIBRATED", "raw_conflict_score_uncalibrated"
        base = raw
    tbar, mass = _weighted_tmin(batch)
    T = float(batch.horizon_s[-1])
    urgency = np.where(np.isfinite(tbar), np.clip(1.0 - tbar / T, 0.0, 1.0), np.nan)
    w = cfg.urgency_weight_floor
    risk = np.where(s_valid & np.isfinite(urgency), base * (w + (1 - w) * urgency), np.nan)
    f = batch.features
    lsr = f["log_scale_ratio"]
    scale_ok = batch.masks["scale_ok"] & np.isfinite(lsr)
    ood = ~scale_ok | (np.abs(np.nan_to_num(lsr)) > cfg.max_abs_log_scale_ratio)
    low_mass = mass < cfg.min_valid_mode_mass
    disagree = batch.masks["mp_valid"] & s_valid & (np.abs(np.nan_to_num(f["mode_weighted_score"]) - np.nan_to_num(f["mp_score"])) > cfg.max_indicator_disagreement)
    cat = np.full(P, INSUFFICIENT, dtype=object)
    status = np.full(P, INSUFFICIENT, dtype=object)
    reason = np.full(P, "", dtype=object)
    reason[~s_valid] = "INVALID_OR_MISSING_CONFLICT_PREDICTION"
    reason[s_valid & ~np.isfinite(risk)] = "RISK_NOT_COMPUTABLE"
    reason[s_valid & np.isfinite(risk) & ood] = "OUT_OF_DISTRIBUTION_DEPTH_MISMATCH_OR_SCALE_INVALID"
    reason[s_valid & np.isfinite(risk) & ~ood & low_mass] = "INSUFFICIENT_VALID_MODE_MASS"
    ok = s_valid & np.isfinite(risk) & ~ood & ~low_mass
    bins = np.where(risk >= cfg.critical_min, 3, np.where(risk >= cfg.high_min, 2, np.where(risk >= cfg.moderate_min, 1, 0)))
    for i in np.flatnonzero(ok):
        cat[i] = CATEGORIES[bins[i]]
        status[i] = cfg.category_status if not disagree[i] else cfg.category_status + "_CONFLICTING_INDICATORS"
        if disagree[i]:
            reason[i] = "MODE_WEIGHTED_AND_MOST_PROBABLE_MODE_SCORES_DISAGREE"
    prov = dict(provenance or {})
    prov.update({"risk_config_sha256": cfg.sha256(), "risk_config_version": cfg.config_version, "category_status": cfg.category_status, "threshold_evidence": cfg.threshold_evidence,
                 "phase6_provenance": batch.provenance, "phase6_config_sha256": batch.config_sha256, "calibration_artifact_sha256": calibration.sha256 if calibration else None,
                 "source_checkpoint_file_sha256": checkpoint_file_sha256, "coordinate_system": batch.coordinate_system, "coordinate_unit": batch.coordinate_unit,
                 "physical_quantities": False, "future_observations_used": False, "labels_used": False})
    pair = {"window_id": batch.window_id, "ped_track_id": batch.ped_track_id, "other_track_id": batch.other_track_id, "anchor_frame": batch.anchor_frame}
    return RiskOutput(OUTPUT_SCHEMA, pair, raw, scores.kind, prob, cal_status, urgency, np.full(P, np.nan), "SEVERITY_NOT_SUPPORTED_NO_METRIC_GEOMETRY", risk, basis,
                      cat.astype(str), status.astype(str), reason.astype(str), {"conflict_valid": s_valid, "category_assigned": ok, "out_of_distribution": ood}, prov)


def summarize_risk(out: RiskOutput) -> Dict[str, Any]:
    """Descriptive counts only: no accuracy, precision or calibration metric can be derived without verified labels."""
    cats, cnt = np.unique(out.risk_category, return_counts=True)
    rs = out.risk_score[np.isfinite(out.risk_score)]
    return {"n_pairs": out.n, "category_counts": {str(c): int(n) for c, n in zip(cats, cnt)}, "reason_counts": {str(r): int((out.reasons == r).sum()) for r in np.unique(out.reasons) if r},
            "calibration_status": out.calibration_status, "risk_basis": out.risk_basis,
            "risk_score": {"mean": float(rs.mean()), "median": float(np.median(rs)), "p90": float(np.quantile(rs, 0.9)), "max": float(rs.max())} if rs.size else None,
            "evaluation": "DESCRIPTIVE_ONLY_NO_VERIFIED_LABELS"}


__all__ = ["RiskConfig", "RiskOutput", "assess_risk", "summarize_risk", "CATEGORIES", "INSUFFICIENT", "OUTPUT_SCHEMA"]
