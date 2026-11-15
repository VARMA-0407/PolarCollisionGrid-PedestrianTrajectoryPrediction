"""P5.7.1 deterministic dataset and batch adapter.

This module adapts the persisted P5.3 window and graph-bank artifacts to the
P5.5/P5.6 interfaces.  Future arrays are retained as supervision fields but
are never passed to :class:`EncoderInput`.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .errors import WindowContractError
from .graph_window_adapter import collate_windows
from .models.encoder import EncoderInput
from .window_config import WindowConfig
from .window_storage import read_dataset
from .window_validation import validate_window_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingWindowSample:
    """One deterministic window record returned by ``TrainingWindowDataset``."""

    index: int
    window_id: int
    video_id: str
    group_id: str
    target_track_id: str
    target_track_index: int
    anchor_frame: int
    history_frames: np.ndarray
    future_frames: np.ndarray
    target_future_uv_px: np.ndarray
    target_future_valid: np.ndarray
    anchor_uv_px: np.ndarray
    target_valid: bool


class TrainingWindowDataset:
    """Validated, indexable view over an existing P5.3 window run.

    Graph construction remains owned by P5.3.  This class only selects windows,
    delegates graph collation to ``collate_windows``, and adds training metadata.
    """

    def __init__(
        self,
        run_dir,
        config: Optional[WindowConfig] = None,
        *,
        include_optional: bool = False,
        max_agents: Optional[int] = None,
        verify_hashes: bool = True,
    ):
        self.run_dir = Path(run_dir)
        manifest_path = self.run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise WindowContractError("DATASET_MISSING_MANIFEST", "missing %s" % manifest_path)
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise WindowContractError("DATASET_INVALID_MANIFEST", "cannot read %s: %s" % (manifest_path, exc))
        if not isinstance(self.manifest, Mapping):
            raise WindowContractError("DATASET_INVALID_MANIFEST", "run manifest must be a JSON object")

        self.hashes_verified = bool(verify_hashes)
        self._verify_manifest_outputs()
        try:
            self.ws, self.bank, self.extra = read_dataset(self.run_dir)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise WindowContractError("DATASET_INVALID_ARTIFACT", "cannot load P5.3 artifacts: %s" % exc)

        manifest_config = self.manifest.get("config", {}).get("content")
        if not isinstance(manifest_config, Mapping):
            raise WindowContractError("MANIFEST_CONFIG_MISSING", "run manifest lacks config.content")
        try:
            manifest_cfg = WindowConfig.from_dict(manifest_config)
        except Exception as exc:
            raise WindowContractError("MANIFEST_CONFIG_INVALID", str(exc))
        if config is not None and config.sha256() != manifest_cfg.sha256():
            raise WindowContractError("CONFIG_MISMATCH", "supplied WindowConfig does not match the run manifest")
        self.config = config if config is not None else manifest_cfg
        self.provenance = {
            "sources": deepcopy(self.manifest.get("sources", {})),
            "outputs": deepcopy(self.manifest.get("outputs", {})),
            "git": deepcopy(self.manifest.get("git", {})),
            "label": self.manifest.get("label"),
            "run_manifest_sha256": _sha256(manifest_path),
            "artifact_hashes_verified": self.hashes_verified,
        }
        self.include_optional = bool(include_optional)
        self.max_agents = max_agents
        self._validate_dataset()

    def _verify_manifest_outputs(self) -> None:
        outputs = self.manifest.get("outputs")
        if not isinstance(outputs, Mapping) or not outputs:
            raise WindowContractError("MANIFEST_OUTPUTS_MISSING", "run manifest lacks outputs")
        for name, record in outputs.items():
            if not isinstance(name, str) or Path(name).name != name:
                raise WindowContractError("MANIFEST_OUTPUT_INVALID", "invalid output name %r" % (name,))
            path = self.run_dir / name
            if not path.is_file() or not isinstance(record, Mapping):
                raise WindowContractError("DATASET_MISSING_FILE", "missing manifest output %s" % name)
            if self.hashes_verified and record.get("sha256") != _sha256(path):
                raise WindowContractError("MANIFEST_HASH_MISMATCH", "hash mismatch for %s" % path)

    def _validate_dataset(self) -> None:
        target_bank = self.extra.get("history_target_bank_node")
        if target_bank is None:
            raise WindowContractError("DATASET_REQUIRED_ARRAY_MISSING", "history_target_bank_node is missing")
        try:
            report = validate_window_dataset(self.ws, self.bank, target_bank, self.config)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise WindowContractError("DATASET_VALIDATION_FAILED", "malformed P5.3 artifact: %s" % exc)
        if not report["ok"]:
            failed = [name for name, result in report["checks"].items() if not result["pass"]]
            raise WindowContractError("DATASET_VALIDATION_FAILED", "failed checks: %s" % ", ".join(failed))
        n = self.ws.n_windows
        for name in ("video_id", "group_id"):
            values = self.extra.get(name)
            if values is None or values.shape != (n,):
                raise WindowContractError("DATASET_REQUIRED_ARRAY_MISSING", "%s must have shape [%d]" % (name, n))
        if self.manifest.get("counts", {}).get("accepted_windows") != n:
            raise WindowContractError("MANIFEST_COUNT_MISMATCH", "manifest accepted_windows does not match windows.npz")

    def __len__(self) -> int:
        return self.ws.n_windows

    def __getitem__(self, index: int) -> TrainingWindowSample:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        target_history = self.ws.history_target_uv_px[index, -1]
        if target_history.shape != (2,) or not np.isfinite(target_history).all():
            raise WindowContractError("INVALID_ANCHOR", "window %d has a non-finite final history anchor" % index)
        return TrainingWindowSample(
            index=index,
            window_id=int(self.ws.window_id[index]),
            video_id=str(self.extra["video_id"][index]),
            group_id=str(self.extra["group_id"][index]),
            target_track_id=str(self.ws.target_track_id[index]),
            target_track_index=int(self.ws.target_track_index[index]),
            anchor_frame=int(self.ws.anchor_frame[index]),
            history_frames=self.ws.history_frames[index].copy(),
            future_frames=self.ws.future_frames[index].copy(),
            target_future_uv_px=self.ws.target_future_uv_px[index].copy(),
            target_future_valid=self.ws.target_future_valid[index].copy(),
            anchor_uv_px=target_history.copy(),
            target_valid=True,
        )

    def collate(self, samples: Sequence[TrainingWindowSample]) -> Dict[str, Any]:
        """Create a deterministic batch compatible with ``EncoderInput``."""
        if not samples:
            raise WindowContractError("EMPTY_BATCH", "cannot collate an empty batch")
        indices = [int(sample.index) for sample in samples]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(self) for i in indices):
            raise WindowContractError("INVALID_BATCH_INDEX", "batch contains duplicate or out-of-range indices")
        expected = [self[i] for i in indices]
        if any(a.window_id != b.window_id for a, b in zip(samples, expected)):
            raise WindowContractError("SAMPLE_DATASET_MISMATCH", "sample does not belong to this dataset")

        batch = collate_windows(
            self.bank,
            self.ws,
            indices,
            self.config,
            include_optional=self.include_optional,
            max_agents=self.max_agents,
        )
        anchor = np.asarray(batch["node_features"][:, -1, 0, :2], dtype=np.float32)
        target_valid = np.asarray(batch["present"][:, -1, 0] & batch["slot_valid"][:, 0], dtype=bool)
        if not target_valid.all() or not np.isfinite(anchor).all():
            raise WindowContractError("INVALID_ANCHOR", "every batch target must have a finite final history anchor")
        if not np.array_equal(anchor, np.asarray([s.anchor_uv_px for s in expected], dtype=np.float32)):
            raise WindowContractError("ANCHOR_MISMATCH", "collated target anchor differs from stored history target")

        batch["anchor_uv_px"] = anchor
        batch["target_valid"] = target_valid
        batch["metadata"] = {
            "window_id": np.asarray([s.window_id for s in expected], dtype=np.int64),
            "video_id": np.asarray([s.video_id for s in expected], dtype=str),
            "group_id": np.asarray([s.group_id for s in expected], dtype=str),
            "target_track_id": np.asarray([s.target_track_id for s in expected], dtype=str),
            "target_track_index": np.asarray([s.target_track_index for s in expected], dtype=np.int32),
            "anchor_frame": np.asarray([s.anchor_frame for s in expected], dtype=np.int64),
            "history_frames": np.asarray([s.history_frames for s in expected], dtype=np.int64),
            "future_frames": np.asarray([s.future_frames for s in expected], dtype=np.int64),
            "source_artifact": str(self.run_dir.resolve()),
            "run_manifest_sha256": _sha256(self.run_dir / "run_manifest.json"),
            "dataset_label": self.manifest.get("label"),
            "provenance": deepcopy(self.provenance),
        }
        return batch

    @staticmethod
    def encoder_input(batch: Mapping[str, Any]) -> EncoderInput:
        """Build P5.5 input; future supervision and metadata are excluded."""
        forbidden = {"target_future_uv_px", "target_future_valid", "future_frames", "metadata", "anchor_uv_px", "target_valid"}
        history_batch = {key: value for key, value in batch.items() if key not in forbidden}
        return EncoderInput.from_collated_batch(history_batch)

    def fit_scaler_statistics(
        self,
        indices: Sequence[int],
        *,
        fit_partition: str,
        split_identity: str,
        batch_size: int = 32,
        configuration: Optional[Mapping[str, Any]] = None,
    ):
        """Fit P5.7.2 statistics from explicitly selected training windows only."""
        if fit_partition != "train":
            raise WindowContractError("SCALER_NOT_TRAIN_ONLY", "fit_partition must be exactly 'train'")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise WindowContractError("SCALER_BATCH_SIZE", "batch_size must be a positive integer")
        selected = [int(i) for i in indices]
        if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= len(self) for i in selected):
            raise WindowContractError("SCALER_INDICES", "indices must be a non-empty unique dataset index sequence")
        from .training_scaler import fit_feature_statistics

        batches = (
            self.collate([self[i] for i in selected[start : start + batch_size]])
            for start in range(0, len(selected), batch_size)
        )
        source_identity = {
            "run_manifest_sha256": self.provenance["run_manifest_sha256"],
            "sources": deepcopy(self.provenance["sources"]),
            "label": self.provenance["label"],
        }
        return fit_feature_statistics(
            batches,
            fit_partition=fit_partition,
            split_identity=split_identity,
            source_identity=source_identity,
            configuration=configuration,
        )


__all__ = ["TrainingWindowDataset", "TrainingWindowSample"]
