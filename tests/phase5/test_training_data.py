import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.models.decoder import DecoderConfig, MultimodalTrajectoryDecoder
from pedsafety.phase5.models.encoder import SpatioTemporalEncoder
from pedsafety.phase5.models.config import EncoderConfig
from pedsafety.phase5.training_data import TrainingWindowDataset


REPO = Path(__file__).resolve().parents[2]
PILOT_RUN = REPO / "experiments" / "phase5_task3_windows_pie_video_0004" / "20261115T035825Z_pie_video_0004" / "primary"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_run(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    copied = tmp_path / "run"
    shutil.copytree(PILOT_RUN, copied)
    return copied


def _update_manifest_hashes(run_dir):
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, record in manifest["outputs"].items():
        if (run_dir / name).is_file():
            record["sha256"] = _sha256(run_dir / name)
            record["bytes"] = (run_dir / name).stat().st_size
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _rewrite_npz(path, mutate):
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    mutate(arrays)
    np.savez(path, **arrays)


@pytest.fixture(scope="module")
def dataset():
    if not PILOT_RUN.is_dir():
        pytest.skip("committed pilot artifact is unavailable")
    return TrainingWindowDataset(PILOT_RUN)


def test_dataset_length_and_deterministic_indexing(dataset):
    assert len(dataset) == 2327
    first = dataset[0]
    again = dataset[0]
    assert first.window_id == again.window_id
    assert first.video_id == "pie:video_0004"
    assert first.group_id == "pie:video_0004"
    np.testing.assert_array_equal(first.history_frames, again.history_frames)
    np.testing.assert_array_equal(first.future_frames, again.future_frames)


def test_sample_metadata_anchor_and_targets(dataset):
    sample = dataset[0]
    assert sample.target_track_id
    assert sample.target_track_index >= 0
    assert sample.anchor_frame == sample.history_frames[-1]
    assert sample.anchor_uv_px.shape == (2,)
    assert sample.target_future_uv_px.shape == (15, 2)
    assert sample.target_future_valid.shape == (15,)
    assert sample.target_future_valid.all()


def test_batch_contract_and_metadata(dataset):
    batch = dataset.collate([dataset[0], dataset[1]])
    assert batch["node_features"].dtype == np.float32
    assert batch["node_features"].shape[:2] == (2, 6)
    assert batch["node_features"].shape[-1] == 10
    assert batch["node_masks"].shape == batch["node_features"].shape[:3] + (4,)
    assert batch["node_masks"].dtype == bool
    assert batch["present"].dtype == bool
    assert batch["slot_valid"].dtype == bool
    assert batch["target_future_uv_px"].shape == (2, 15, 2)
    assert batch["target_future_uv_px"].dtype == np.float32
    assert batch["target_future_valid"].shape == (2, 15)
    assert batch["anchor_uv_px"].shape == (2, 2)
    assert batch["anchor_uv_px"].dtype == np.float32
    assert batch["target_valid"].shape == (2,)
    assert batch["target_valid"].all()
    assert batch["target_slot"] == 0
    assert set(batch["edges"]) == set(batch["active_relations"])
    metadata = batch["metadata"]
    assert metadata["video_id"].tolist() == ["pie:video_0004", "pie:video_0004"]
    assert metadata["group_id"].tolist() == ["pie:video_0004", "pie:video_0004"]
    assert metadata["target_track_id"].shape == (2,)
    assert metadata["history_frames"].shape == (2, 6)
    assert metadata["future_frames"].shape == (2, 15)
    assert metadata["source_artifact"] == str(PILOT_RUN.resolve())
    assert metadata["provenance"]["sources"] == dataset.manifest["sources"]
    assert metadata["provenance"]["outputs"] == dataset.manifest["outputs"]
    assert metadata["provenance"]["artifact_hashes_verified"] is True


def test_anchor_is_final_raw_history_position(dataset):
    sample = dataset[11]
    batch = dataset.collate([sample])
    np.testing.assert_array_equal(batch["anchor_uv_px"][0], dataset.ws.history_target_uv_px[11, -1])
    np.testing.assert_array_equal(batch["anchor_uv_px"][0], batch["node_features"][0, -1, 0, :2])


def test_encoder_input_excludes_future_and_metadata(dataset):
    batch = dataset.collate([dataset[0], dataset[1]])
    encoder_input = dataset.encoder_input(batch)
    assert encoder_input.present.shape == (2, 6, batch["node_features"].shape[2])
    assert encoder_input.node_features.shape[-1] == 10
    assert not hasattr(encoder_input, "target_future_uv_px")
    assert not hasattr(encoder_input, "future_frames")

    changed = dict(batch)
    changed["target_future_uv_px"] = batch["target_future_uv_px"] + 100000.0
    changed["target_future_valid"] = np.zeros_like(batch["target_future_valid"])
    original = dataset.encoder_input(batch)
    altered = dataset.encoder_input(changed)
    np.testing.assert_array_equal(original.node_features.numpy(), altered.node_features.numpy())
    np.testing.assert_array_equal(original.history_frames.numpy(), altered.history_frames.numpy())


def test_decoder_compatibility_uses_no_future_targets(dataset):
    batch = dataset.collate([dataset[0]])
    cfg = DecoderConfig(
        config_version="test",
        num_modes=5,
        history_steps=6,
        future_steps=15,
        sampling_hz=10.0,
        coordinate_space="image",
        coordinate_unit="pixels",
        output_representation="raw_pixel_offsets",
        input_embedding_dim=64,
        decoder_hidden_dim=16,
        dropout=0.0,
        regression_loss="smooth_l1",
        classification_loss="cross_entropy",
        classification_weight=0.1,
        winner_selection="minimum_masked_ADE",
        autoregressive=False,
    )
    decoder = MultimodalTrajectoryDecoder(cfg)
    output = decoder(
        torch.zeros(1, 64),
        torch.as_tensor(batch["anchor_uv_px"]),
        torch.as_tensor(batch["target_valid"]),
    )
    assert tuple(output.trajectories_uv_px.shape) == (1, 5, 15, 2)
    assert tuple(output.prediction_valid.shape) == (1, 5, 15)


def test_actual_encoder_to_decoder_cpu_smoke(dataset):
    batch = dataset.collate([dataset[0]])
    encoder_input = dataset.encoder_input(batch)
    encoder = SpatioTemporalEncoder(
        EncoderConfig.load(REPO / "configs" / "phase5" / "encoder_default.json"),
        history_length=6,
    ).eval()
    decoder = MultimodalTrajectoryDecoder(
        DecoderConfig.load(REPO / "configs" / "phase5" / "decoder_default.json")
    ).eval()
    with torch.no_grad():
        encoded = encoder(encoder_input)
        decoded = decoder(
            encoded.target_summary,
            torch.as_tensor(batch["anchor_uv_px"]),
            torch.as_tensor(batch["target_valid"]),
        )
    assert tuple(encoded.target_summary.shape) == (1, 64)
    assert tuple(decoded.trajectories_uv_px.shape) == (1, 5, 15, 2)
    assert torch.isfinite(decoded.trajectories_uv_px).all()
    assert torch.isfinite(decoded.mode_probabilities).all()
    torch.testing.assert_close(decoded.mode_probabilities.sum(dim=1), torch.ones(1))


def test_edges_preserve_relations_batch_indices_and_variable_agents(dataset):
    counts = [int(dataset.collate([dataset[i]])["slot_valid"].sum()) for i in range(len(dataset))]
    low, high = min(range(len(counts)), key=counts.__getitem__), max(range(len(counts)), key=counts.__getitem__)
    batch = dataset.collate([dataset[low], dataset[high]])
    assert len(batch["active_relations"]) >= 2
    max_agents = batch["node_features"].shape[2]
    single_batches = [dataset.collate([dataset[low]]), dataset.collate([dataset[high]])]
    for relation, edge in batch["edges"].items():
        assert edge["bhsd"].shape[1] == 4
        assert edge["edge_attr"].shape[1] == 14
        assert edge["edge_mask"].shape == edge["edge_attr"].shape
        assert edge["flags"].shape[1] == 6
        if edge["bhsd"].size:
            assert edge["bhsd"][:, 0].min() >= 0
            assert edge["bhsd"][:, 0].max() < 2
            assert edge["bhsd"][:, 1].min() >= 0
            assert edge["bhsd"][:, 1].max() < 6
            assert edge["bhsd"][:, 2:].min() >= 0
            assert edge["bhsd"][:, 2:].max() < max_agents
        expected = []
        for batch_index, single in enumerate(single_batches):
            part = single["edges"][relation]
            bhsd = part["bhsd"].copy()
            bhsd[:, 0] = batch_index
            expected.append((bhsd, part["edge_attr"], part["edge_mask"], part["flags"]))
        for actual, expected_parts in zip(
            (edge["bhsd"], edge["edge_attr"], edge["edge_mask"], edge["flags"]),
            (np.concatenate([part[0] for part in expected]),
             np.concatenate([part[1] for part in expected]),
             np.concatenate([part[2] for part in expected]),
             np.concatenate([part[3] for part in expected])),
        ):
            np.testing.assert_array_equal(actual, expected_parts)


def test_repeated_collation_is_deterministic(dataset):
    first = dataset.collate([dataset[3], dataset[19]])
    second = dataset.collate([dataset[3], dataset[19]])
    for key in ("node_features", "node_masks", "present", "track_index", "target_future_uv_px", "anchor_uv_px"):
        np.testing.assert_equal(first[key], second[key])
    for relation in first["edges"]:
        for key in ("bhsd", "edge_attr", "edge_mask", "flags"):
            np.testing.assert_equal(first["edges"][relation][key], second["edges"][relation][key])


def test_manifest_missing_and_tampered_outputs_rejected(tmp_path):
    with pytest.raises(WindowContractError, match="DATASET_MISSING_MANIFEST"):
        TrainingWindowDataset(tmp_path)
    copied = _copy_run(tmp_path)
    (copied / "smoke_manifest.json").unlink()
    with pytest.raises(WindowContractError, match="DATASET_MISSING_FILE"):
        TrainingWindowDataset(copied)
    copied = _copy_run(tmp_path / "tampered")
    (copied / "smoke_manifest.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(WindowContractError, match="MANIFEST_HASH_MISMATCH"):
        TrainingWindowDataset(copied)
    bypassed = TrainingWindowDataset(copied, verify_hashes=False)
    assert bypassed.hashes_verified is False
    metadata = bypassed.collate([bypassed[0]])["metadata"]
    assert metadata["provenance"]["artifact_hashes_verified"] is False


def test_malformed_shapes_nonfinite_values_and_invalid_target_rejected(tmp_path):
    copied = _copy_run(tmp_path / "shape")
    _rewrite_npz(copied / "windows.npz", lambda arrays: arrays.__setitem__("history_frames", arrays["history_frames"][:, :5]))
    _update_manifest_hashes(copied)
    with pytest.raises(WindowContractError, match="DATASET_VALIDATION_FAILED"):
        TrainingWindowDataset(copied)

    copied = _copy_run(tmp_path / "node")
    def corrupt_node(arrays):
        arrays["n__features"][0, 0] = np.inf
    _rewrite_npz(copied / "graph_bank.npz", corrupt_node)
    _update_manifest_hashes(copied)
    with pytest.raises(WindowContractError, match="DATASET_VALIDATION_FAILED"):
        TrainingWindowDataset(copied)

    copied = _copy_run(tmp_path / "edge")
    def corrupt_edge(arrays):
        for key in arrays:
            if key.startswith("e__") and key.endswith("__edge_attr"):
                mask_key = key[:-len("edge_attr")] + "edge_mask"
                if arrays[key].size and arrays[mask_key].any():
                    position = np.argwhere(arrays[mask_key])[0]
                    arrays[key][tuple(position)] = np.inf
                    return
        raise AssertionError("pilot graph bank has no valid edge attribute")
    _rewrite_npz(copied / "graph_bank.npz", corrupt_edge)
    _update_manifest_hashes(copied)
    with pytest.raises(WindowContractError, match="DATASET_VALIDATION_FAILED"):
        TrainingWindowDataset(copied)

    copied = _copy_run(tmp_path / "target")
    def corrupt_target(arrays):
        with np.load(copied / "graph_bank.npz", allow_pickle=False) as bank:
            tracks = bank["n__track_index"]
        target = arrays["target_track_index"][0]
        candidate = int(np.flatnonzero(tracks != target)[0])
        arrays["history_target_bank_node"][0, 0] = candidate
    _rewrite_npz(copied / "windows.npz", corrupt_target)
    _update_manifest_hashes(copied)
    with pytest.raises(WindowContractError, match="DATASET_VALIDATION_FAILED"):
        TrainingWindowDataset(copied)


def test_missing_observations_remain_masked_and_unfilled(dataset):
    for index in range(100):
        batch = dataset.collate([dataset[index]])
        missing = (~batch["present"]) & batch["slot_valid"][:, None]
        if missing.any():
            assert not batch["node_masks"][missing].any()
            assert np.isnan(batch["node_features"][missing]).all()
            return
    pytest.fail("pilot sample did not expose a missing context observation")


def test_malformed_encoder_batch_rejected(dataset):
    batch = dataset.collate([dataset[0]])
    malformed = dict(batch)
    del malformed["node_features"]
    with pytest.raises(WindowContractError, match="INPUT_MISSING"):
        dataset.encoder_input(malformed)
    with pytest.raises(WindowContractError, match="INVALID_BATCH_INDEX"):
        dataset.collate([dataset[0], dataset[0]])


def test_empty_batch_rejected(dataset):
    with pytest.raises(WindowContractError, match="EMPTY_BATCH"):
        dataset.collate([])
