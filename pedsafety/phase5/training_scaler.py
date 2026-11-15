"""Train-only feature statistics for the P5.5 encoder.

Statistics are fitted from the exact derived node and edge feature spaces used
by ``SpatioTemporalEncoder``.  Raw anchors and future supervision are never
part of the fitted or transformed values.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .errors import WindowContractError
from .models.config import NODE_INPUT_FEATURES
from .models.scaling import FeatureScaler
from .task3_source import EDGE_FEATURES


STATISTICS_SCHEMA_VERSION = "phase5-train-feature-statistics-1.0.0"
CONSTANT_SCALE = 1.0


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _hash_body(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _as_tuple(values: Sequence[Any], dtype) -> tuple:
    return tuple(np.asarray(values, dtype=dtype).tolist())


def _derived_node_values(batch: Mapping[str, Any]):
    try:
        features = np.asarray(batch["node_features"], dtype=np.float64)
        masks = np.asarray(batch["node_masks"], dtype=bool)
        present = np.asarray(batch["present"], dtype=bool)
    except (KeyError, TypeError, ValueError) as exc:
        raise WindowContractError("SCALER_INPUT_MISSING", "missing node scaling inputs: %s" % exc)
    if features.ndim != 4 or features.shape[-1] != 10:
        raise WindowContractError("SCALER_INPUT_SHAPE", "node_features must be [B,H,A,10]")
    if masks.shape != features.shape[:3] + (4,) or present.shape != features.shape[:3]:
        raise WindowContractError("SCALER_INPUT_SHAPE", "node masks/presence do not match node_features")
    reference = features[:, -1, 0, 0:2]
    reference_valid = present[:, -1, 0] & masks[:, -1, 0, 0]
    if not reference_valid.all() or not np.isfinite(reference[reference_valid]).all():
        raise WindowContractError("SCALER_INVALID_ANCHOR", "valid target anchors must be finite")
    relative = features[..., 0:2] - reference[:, None, None, :]
    raw = np.concatenate((relative, features[..., 2:9]), axis=-1)
    valid = np.concatenate(
        (
            present[..., None],
            present[..., None],
            masks[..., 1:2],
            masks[..., 1:2],
            masks[..., 1:2],
            masks[..., 2:3],
            masks[..., 2:3],
            present[..., None],
            masks[..., 3:4],
        ),
        axis=-1,
    ) & present[..., None]
    return raw, valid


def _masked_values(values: np.ndarray, valid: np.ndarray, label: str) -> np.ndarray:
    if valid.shape != values.shape:
        raise WindowContractError("SCALER_INPUT_SHAPE", "%s mask does not match values" % label)
    selected = values[valid]
    if selected.size and not np.isfinite(selected).all():
        raise WindowContractError("SCALER_NONFINITE_VALID", "%s contains non-finite valid values" % label)
    return selected


class TrainOnlyFeatureStatistics:
    """Immutable train-partition statistics compatible with ``FeatureScaler``."""

    def __init__(
        self,
        *,
        fit_partition: str,
        split_identity: str,
        source_identity: Mapping[str, Any],
        configuration: Mapping[str, Any],
        node_mean: Sequence[float],
        node_scale: Sequence[float],
        node_valid_count: Sequence[int],
        edge_mean: Sequence[float],
        edge_scale: Sequence[float],
        edge_valid_count: Sequence[int],
        statistics_sha256: Optional[str] = None,
    ):
        if fit_partition != "train":
            raise WindowContractError("SCALER_NOT_TRAIN_ONLY", "fit_partition must be exactly 'train'")
        if not isinstance(split_identity, str) or not split_identity.strip():
            raise WindowContractError("SCALER_SPLIT_IDENTITY", "split_identity must be non-empty")
        if not isinstance(source_identity, Mapping):
            raise WindowContractError("SCALER_SOURCE_IDENTITY", "source_identity must be a mapping")
        self.fit_partition = fit_partition
        self.split_identity = split_identity
        self.source_identity = deepcopy(dict(source_identity))
        self.configuration = deepcopy(dict(configuration))
        self.node_mean = _as_tuple(node_mean, np.float64)
        self.node_scale = _as_tuple(node_scale, np.float64)
        self.node_valid_count = _as_tuple(node_valid_count, np.int64)
        self.edge_mean = _as_tuple(edge_mean, np.float64)
        self.edge_scale = _as_tuple(edge_scale, np.float64)
        self.edge_valid_count = _as_tuple(edge_valid_count, np.int64)
        self.statistics_sha256 = statistics_sha256
        self._validate()
        if statistics_sha256 is not None and statistics_sha256 != self.compute_hash():
            raise WindowContractError("SCALER_HASH_MISMATCH", "statistics_sha256 does not match persisted statistics")

    def _validate(self) -> None:
        if not isinstance(self.configuration, Mapping):
            raise WindowContractError("SCALER_CONFIGURATION", "configuration must be a mapping")
        if tuple(self.configuration.get("node_feature_names", ())) != tuple(NODE_INPUT_FEATURES):
            raise WindowContractError("SCALER_FEATURE_ORDER", "node feature ordering is incompatible")
        if tuple(self.configuration.get("edge_feature_names", ())) != tuple(EDGE_FEATURES):
            raise WindowContractError("SCALER_FEATURE_ORDER", "edge feature ordering is incompatible")
        if self.configuration.get("constant_scale") != CONSTANT_SCALE:
            raise WindowContractError("SCALER_CONFIGURATION", "constant_scale is incompatible")
        for label, values, names in (
            ("node_mean", self.node_mean, NODE_INPUT_FEATURES),
            ("node_scale", self.node_scale, NODE_INPUT_FEATURES),
            ("node_valid_count", self.node_valid_count, NODE_INPUT_FEATURES),
            ("edge_mean", self.edge_mean, EDGE_FEATURES),
            ("edge_scale", self.edge_scale, EDGE_FEATURES),
            ("edge_valid_count", self.edge_valid_count, EDGE_FEATURES),
        ):
            if len(values) != len(names):
                raise WindowContractError("SCALER_DIMENSION", "%s has incompatible feature dimension" % label)
            if label.endswith("valid_count"):
                if any(int(v) <= 0 for v in values):
                    raise WindowContractError("SCALER_EMPTY_FEATURE", "%s contains an empty feature count" % label)
            elif not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                raise WindowContractError("SCALER_NONFINITE", "%s contains non-finite values" % label)
        if (np.asarray(self.node_scale) <= 0).any() or (np.asarray(self.edge_scale) <= 0).any():
            raise WindowContractError("SCALER_SCALE", "all scales must be positive")

    @classmethod
    def fit(
        cls,
        batches: Iterable[Mapping[str, Any]],
        *,
        fit_partition: str,
        split_identity: str,
        source_identity: Mapping[str, Any],
        configuration: Optional[Mapping[str, Any]] = None,
    ) -> "TrainOnlyFeatureStatistics":
        if fit_partition != "train":
            raise WindowContractError("SCALER_NOT_TRAIN_ONLY", "statistics may only be fitted with fit_partition='train'")
        node_sum = np.zeros(len(NODE_INPUT_FEATURES), np.float64)
        node_sum_sq = np.zeros_like(node_sum)
        node_count = np.zeros_like(node_sum, dtype=np.int64)
        edge_sum = np.zeros(len(EDGE_FEATURES), np.float64)
        edge_sum_sq = np.zeros_like(edge_sum)
        edge_count = np.zeros_like(edge_sum, dtype=np.int64)
        batch_count = 0
        for batch in batches:
            batch_count += 1
            node_values, node_valid = _derived_node_values(batch)
            for feature in range(node_values.shape[-1]):
                values = _masked_values(node_values[..., feature], node_valid[..., feature], "node feature %s" % NODE_INPUT_FEATURES[feature])
                node_sum[feature] += values.sum(dtype=np.float64)
                node_sum_sq[feature] += np.square(values, dtype=np.float64).sum(dtype=np.float64)
                node_count[feature] += values.size
            edges = batch.get("edges")
            if not isinstance(edges, Mapping):
                raise WindowContractError("SCALER_INPUT_MISSING", "batch lacks relation-specific edges")
            for relation, edge in edges.items():
                try:
                    attrs = np.asarray(edge.get("edge_attr", edge.get("attr")), dtype=np.float64)
                    masks = np.asarray(edge.get("edge_mask", edge.get("mask")), dtype=bool)
                except (KeyError, TypeError, ValueError) as exc:
                    raise WindowContractError("SCALER_INPUT_MISSING", "invalid edge inputs for %s: %s" % (relation, exc))
                if attrs.ndim != 2 or attrs.shape[1] != len(EDGE_FEATURES):
                    raise WindowContractError("SCALER_INPUT_SHAPE", "edge_attr for %s must be [E,14]" % relation)
                if masks.shape != attrs.shape:
                    raise WindowContractError("SCALER_INPUT_SHAPE", "edge_mask for %s does not match edge_attr" % relation)
                for feature in range(attrs.shape[1]):
                    values = _masked_values(attrs[:, feature], masks[:, feature], "edge feature %s" % EDGE_FEATURES[feature])
                    edge_sum[feature] += values.sum(dtype=np.float64)
                    edge_sum_sq[feature] += np.square(values, dtype=np.float64).sum(dtype=np.float64)
                    edge_count[feature] += values.size
        if batch_count == 0:
            raise WindowContractError("SCALER_EMPTY_INPUT", "no training batches were supplied")
        if (node_count == 0).any() or (edge_count == 0).any():
            raise WindowContractError("SCALER_EMPTY_FEATURE", "every fitted feature needs at least one valid value")
        node_mean = node_sum / node_count
        edge_mean = edge_sum / edge_count
        node_var = np.maximum(node_sum_sq / node_count - np.square(node_mean), 0.0)
        edge_var = np.maximum(edge_sum_sq / edge_count - np.square(edge_mean), 0.0)
        node_scale = np.where(node_var > 0, np.sqrt(node_var), CONSTANT_SCALE)
        edge_scale = np.where(edge_var > 0, np.sqrt(edge_var), CONSTANT_SCALE)
        cfg = {
            "node_feature_names": list(NODE_INPUT_FEATURES),
            "edge_feature_names": list(EDGE_FEATURES),
            "center": "mean",
            "scale": "population_standard_deviation",
            "constant_scale": CONSTANT_SCALE,
        }
        if configuration is not None:
            cfg.update(deepcopy(dict(configuration)))
        return cls(
            fit_partition=fit_partition,
            split_identity=split_identity,
            source_identity=source_identity,
            configuration=cfg,
            node_mean=node_mean,
            node_scale=node_scale,
            node_valid_count=node_count,
            edge_mean=edge_mean,
            edge_scale=edge_scale,
            edge_valid_count=edge_count,
        )

    def _body(self) -> dict:
        return {
            "schema_version": STATISTICS_SCHEMA_VERSION,
            "fit_partition": self.fit_partition,
            "split_identity": self.split_identity,
            "source_identity": deepcopy(self.source_identity),
            "configuration": deepcopy(self.configuration),
            "node": {
                "feature_names": list(NODE_INPUT_FEATURES),
                "mean": list(self.node_mean),
                "scale": list(self.node_scale),
                "valid_count": list(self.node_valid_count),
            },
            "edge": {
                "feature_names": list(EDGE_FEATURES),
                "mean": list(self.edge_mean),
                "scale": list(self.edge_scale),
                "valid_count": list(self.edge_valid_count),
            },
        }

    def compute_hash(self) -> str:
        return _hash_body(self._body())

    def to_dict(self) -> dict:
        body = self._body()
        body["statistics_sha256"] = self.compute_hash()
        return body

    def save(self, path) -> str:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.compute_hash()

    @classmethod
    def load(
        cls,
        path,
        *,
        expected_source_identity: Optional[Mapping[str, Any]] = None,
        expected_split_identity: Optional[str] = None,
    ) -> "TrainOnlyFeatureStatistics":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise WindowContractError("SCALER_INVALID_FILE", "cannot read statistics: %s" % exc)
        if payload.get("schema_version") != STATISTICS_SCHEMA_VERSION:
            raise WindowContractError("SCALER_SCHEMA_VERSION", "unsupported statistics schema")
        if expected_source_identity is not None and payload.get("source_identity") != dict(expected_source_identity):
            raise WindowContractError("SCALER_PROVENANCE", "statistics source identity does not match")
        if expected_split_identity is not None and payload.get("split_identity") != expected_split_identity:
            raise WindowContractError("SCALER_PROVENANCE", "statistics split identity does not match")
        try:
            node, edge = payload["node"], payload["edge"]
            if not isinstance(node, Mapping) or not isinstance(edge, Mapping):
                raise WindowContractError("SCALER_INVALID_FILE", "node and edge statistics must be mappings")
            if tuple(node["feature_names"]) != tuple(NODE_INPUT_FEATURES) or tuple(edge["feature_names"]) != tuple(EDGE_FEATURES):
                raise WindowContractError("SCALER_FEATURE_ORDER", "persisted feature ordering is incompatible")
            return cls(
                fit_partition=payload["fit_partition"],
                split_identity=payload["split_identity"],
                source_identity=payload["source_identity"],
                configuration=payload["configuration"],
                node_mean=node["mean"],
                node_scale=node["scale"],
                node_valid_count=node["valid_count"],
                edge_mean=edge["mean"],
                edge_scale=edge["scale"],
                edge_valid_count=edge["valid_count"],
                statistics_sha256=payload["statistics_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WindowContractError("SCALER_INVALID_FILE", "statistics field is missing: %s" % exc)

    def feature_scaler(self) -> FeatureScaler:
        """Return the unchanged fitted scaler consumed by ``SpatioTemporalEncoder``."""
        return FeatureScaler.from_arrays(
            self.node_mean,
            self.node_scale,
            self.edge_mean,
            self.edge_scale,
            "TRAIN_PARTITION_STATISTICS",
            {
                "fitted": True,
                "fit_partition": self.fit_partition,
                "split_identity": self.split_identity,
                "source_identity": deepcopy(self.source_identity),
                "statistics_sha256": self.compute_hash(),
            },
        )


def fit_feature_statistics(
    batches: Iterable[Mapping[str, Any]],
    *,
    fit_partition: str,
    split_identity: str,
    source_identity: Mapping[str, Any],
    configuration: Optional[Mapping[str, Any]] = None,
) -> TrainOnlyFeatureStatistics:
    """Explicit public fitting entry point for train-only P5.7 statistics."""
    return TrainOnlyFeatureStatistics.fit(
        batches,
        fit_partition=fit_partition,
        split_identity=split_identity,
        source_identity=source_identity,
        configuration=configuration,
    )


__all__ = [
    "CONSTANT_SCALE",
    "STATISTICS_SCHEMA_VERSION",
    "TrainOnlyFeatureStatistics",
    "fit_feature_statistics",
]
