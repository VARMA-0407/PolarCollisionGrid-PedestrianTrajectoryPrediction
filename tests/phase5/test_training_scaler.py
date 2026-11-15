import json

import numpy as np
import pytest
import torch

from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.training_scaler import fit_feature_statistics


def _batch(seed=0, constant=False):
    rng = np.random.default_rng(seed)
    node = rng.normal(size=(1, 6, 2, 10)).astype(np.float32)
    node[..., 0:2] += 100.0
    node[..., 8] = 3.0
    masks = np.ones((1, 6, 2, 4), dtype=bool)
    present = np.ones((1, 6, 2), dtype=bool)
    attrs = np.full((2, 14), 2.0 if constant else 0.0, dtype=np.float32)
    if not constant:
        attrs[:] = rng.normal(size=attrs.shape)
    return {
        "node_features": node,
        "node_masks": masks,
        "present": present,
        "edges": {
            "pedestrian__pedestrian": {
                "edge_attr": attrs,
                "edge_mask": np.ones(attrs.shape, dtype=bool),
            }
        },
        "target_future_uv_px": rng.normal(size=(1, 15, 2)).astype(np.float32),
        "target_future_valid": np.ones((1, 15), dtype=bool),
        "anchor_uv_px": node[:, -1, 0, :2].copy(),
    }


def _fit(batch, **kwargs):
    defaults = {
        "fit_partition": "train",
        "split_identity": "pilot-train",
        "source_identity": {"dataset": "synthetic", "sha256": "abc"},
    }
    defaults.update(kwargs)
    return fit_feature_statistics([batch], **defaults)


def test_train_only_masked_fit_and_constant_scale():
    stats = _fit(_batch(constant=True))
    assert stats.fit_partition == "train"
    assert all(value == 1.0 for value in stats.edge_scale)
    assert all(value > 0 for value in stats.node_valid_count)
    with pytest.raises(WindowContractError, match="SCALER_NOT_TRAIN_ONLY"):
        _fit(_batch(), fit_partition="validation")


def test_masked_values_are_excluded_but_valid_nonfinite_values_rejected():
    batch = _batch()
    batch["node_features"][0, 0, 1, 2] = np.nan
    batch["node_masks"][0, 0, 1, 1] = False
    batch["edges"]["pedestrian__pedestrian"]["edge_attr"][0, 0] = np.inf
    batch["edges"]["pedestrian__pedestrian"]["edge_mask"][0, 0] = False
    assert _fit(batch).node_valid_count[2] == 11
    batch["node_masks"][0, 0, 1, 1] = True
    with pytest.raises(WindowContractError, match="SCALER_NONFINITE_VALID"):
        _fit(batch)


def test_future_targets_do_not_influence_statistics():
    first = _batch(3)
    second = _batch(3)
    second["target_future_uv_px"][:] = 1e9
    second["target_future_valid"][:] = False
    a = _fit(first)
    b = _fit(second)
    assert a.to_dict() == b.to_dict()


def test_save_load_hash_and_provenance_validation(tmp_path):
    stats = _fit(_batch())
    path = tmp_path / "statistics.json"
    digest = stats.save(path)
    loaded = type(stats).load(
        path,
        expected_source_identity={"dataset": "synthetic", "sha256": "abc"},
        expected_split_identity="pilot-train",
    )
    assert loaded.compute_hash() == digest
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["node"]["mean"][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WindowContractError, match="SCALER_HASH_MISMATCH"):
        type(stats).load(path)
    payload = stats.to_dict()
    payload["node"]["feature_names"][0] = "wrong_order"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WindowContractError, match="SCALER_FEATURE_ORDER"):
        type(stats).load(path)


def test_pilot_adapter_fits_only_explicit_training_indices():
    from pathlib import Path

    from pedsafety.phase5.training_data import TrainingWindowDataset

    run = Path(__file__).resolve().parents[2] / "experiments" / "phase5_task3_windows_pie_video_0004" / "20261115T035825Z_pie_video_0004" / "primary"
    if not run.is_dir():
        pytest.skip("committed pilot artifact is unavailable")
    dataset = TrainingWindowDataset(run)
    stats = dataset.fit_scaler_statistics([0, 1], fit_partition="train", split_identity="pilot-train")
    assert stats.source_identity["run_manifest_sha256"] == dataset.provenance["run_manifest_sha256"]
    assert len(stats.node_mean) == 9
    assert len(stats.edge_mean) == 14
    with pytest.raises(WindowContractError, match="SCALER_NOT_TRAIN_ONLY"):
        dataset.fit_scaler_statistics([0], fit_partition="validation", split_identity="pilot-validation")


def test_feature_order_and_raw_supervision_are_unchanged():
    batch = _batch()
    anchor = batch["anchor_uv_px"].copy()
    future = batch["target_future_uv_px"].copy()
    scaler = _fit(batch).feature_scaler()
    assert scaler.source == "TRAIN_PARTITION_STATISTICS"
    raw = torch.ones((1, 9))
    valid = torch.ones((1, 9), dtype=torch.bool)
    scaled = scaler.scale_node(raw, valid)
    assert torch.isfinite(scaled).all()
    np.testing.assert_array_equal(batch["anchor_uv_px"], anchor)
    np.testing.assert_array_equal(batch["target_future_uv_px"], future)
