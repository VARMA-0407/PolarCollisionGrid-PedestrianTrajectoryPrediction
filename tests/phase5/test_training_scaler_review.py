"""P5.7.2 review: train-only fitting, derived feature order == encoder's, mask-aware statistics, constant features, deterministic
serialization / hash, provenance verification and unchanged raw anchors / targets."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.models import EncoderConfig, EncoderInput, SpatioTemporalEncoder
from pedsafety.phase5.models.scaling import FeatureScaler
from pedsafety.phase5.training_data import TrainingWindowDataset
from pedsafety.phase5.training_scaler import TrainOnlyFeatureStatistics, fit_feature_statistics
from enc_helpers import CONFIG as ENC_CONFIG
from enc_helpers import make_input

SRC = {"dataset": "synthetic", "sha256": "abc"}


def batch_from(inp: EncoderInput):
    """A collated-batch-like dict (numpy) from a synthetic EncoderInput, plus raw supervision arrays."""
    B = inp.present.shape[0]
    return {"node_features": inp.node_features.numpy().copy(), "node_masks": inp.node_masks.numpy().copy(), "present": inp.present.numpy().copy(),
            "edges": {r: {"bhsd": e.bhsd.numpy().copy(), "edge_attr": e.attr.numpy().copy(), "edge_mask": e.mask.numpy().copy()} for r, e in inp.edges.items()},
            "anchor_uv_px": inp.node_features[:, -1, 0, :2].numpy().copy(), "target_future_uv_px": np.random.default_rng(0).normal(size=(B, 15, 2)).astype(np.float32),
            "target_future_valid": np.ones((B, 15), bool)}


def fit(batches, **kw):
    d = dict(fit_partition="train", split_identity="pilot-train", source_identity=SRC)
    d.update(kw)
    return fit_feature_statistics(batches, **d)


def test_derived_feature_space_and_order_equal_the_encoders():
    """The statistics are fitted in exactly the space the encoder scales: encoder node inputs == (derived value - mean) / scale where valid, 0 elsewhere."""
    inp = make_input(B=3, H=6, A=6, types=(0, 0, 1, 1, 0), absent=[(0, 1, 2), (1, 3, 4), (2, 0, 1)], seed=21)
    b = batch_from(inp)
    stats = fit([b])
    enc = SpatioTemporalEncoder(EncoderConfig.load(ENC_CONFIG), 6, scaler=stats.feature_scaler())
    got = enc._node_inputs(inp)[..., :9].numpy().astype(np.float64)                    # scaled, masked -> 0
    f, m, p = b["node_features"].astype(np.float64), b["node_masks"], b["present"]
    ref = f[:, -1, 0, 0:2]
    raw = np.concatenate([f[..., 0:2] - ref[:, None, None, :], f[..., 2:9]], axis=-1)
    valid = np.stack([p, p, m[..., 1] & p, m[..., 1] & p, m[..., 1] & p, m[..., 2] & p, m[..., 2] & p, p, m[..., 3] & p], axis=-1)
    expect = np.where(valid, (raw - np.array(stats.node_mean)) / np.array(stats.node_scale), 0.0)
    np.testing.assert_allclose(got, expect, rtol=1e-4, atol=1e-5)
    assert (got[~valid] == 0).all()
    for j in range(9):                                                                 # fitted data is standardised per feature
        v = got[..., j][valid[..., j]]
        assert abs(v.mean()) < 1e-3 and (abs(v.std() - 1) < 1e-3 or stats.node_scale[j] == 1.0), j


def test_statistics_are_validity_mask_aware():
    inp = make_input(B=2, H=6, A=5, types=(0, 0, 1, 1), absent=[(0, 2, 1)], seed=22)
    base = batch_from(inp)
    s0 = fit([base])
    dirty = copy.deepcopy(base)
    dirty["node_features"][..., 2][~(dirty["node_masks"][..., 1] & dirty["present"])] = 9e9          # garbage under False masks
    dirty["node_features"][..., 5][~(dirty["node_masks"][..., 2] & dirty["present"])] = 9e9
    for e in dirty["edges"].values():
        e["edge_attr"][~e["edge_mask"]] = 9e9
    assert fit([dirty]).compute_hash() == s0.compute_hash()
    bad = copy.deepcopy(base)
    idx = np.argwhere(bad["present"] & bad["node_masks"][..., 1])[0]
    bad["node_features"][idx[0], idx[1], idx[2], 3] = np.nan                                         # NaN under a TRUE mask is a contract failure
    with pytest.raises(WindowContractError, match="SCALER_NONFINITE_VALID"):
        fit([bad])


def test_constant_feature_scale_is_one_and_scaled_value_is_zero():
    b = batch_from(make_input(B=2, H=6, A=5, types=(0, 0, 1, 1), seed=23))
    b["node_features"][..., 7] = np.where(b["present"], 100.0, np.nan)                               # constant box height
    for e in b["edges"].values():
        e["edge_attr"][:, 13] = 0.5
        e["edge_mask"][:, 13] = True
    s = fit([b])
    assert s.node_scale[7] == 1.0 and abs(s.node_mean[7] - 100.0) < 1e-9 and s.edge_scale[13] == 1.0 and abs(s.edge_mean[13] - 0.5) < 1e-9
    sc = s.feature_scaler()
    raw = torch.zeros(1, 9)
    raw[0, 7] = 100.0
    out = sc.scale_node(raw, torch.ones(1, 9, dtype=torch.bool))
    assert float(out[0, 7]) == 0.0 and torch.isfinite(out).all()


def test_train_partition_only_and_explicit_batches_only():
    b1, b2 = batch_from(make_input(B=2, seed=24)), batch_from(make_input(B=2, seed=25))
    with pytest.raises(WindowContractError, match="SCALER_NOT_TRAIN_ONLY"):
        fit([b1], fit_partition="validation")
    with pytest.raises(WindowContractError, match="SCALER_NOT_TRAIN_ONLY"):
        fit([b1], fit_partition="test")
    with pytest.raises(WindowContractError, match="SCALER_EMPTY_INPUT"):
        fit([])
    assert fit([b1]).compute_hash() != fit([b1, b2]).compute_hash()                                  # exactly the supplied batches determine the statistics
    assert fit([b2]).compute_hash() == fit([copy.deepcopy(b2)]).compute_hash()


def test_future_targets_and_anchors_never_enter_or_change(p57_windows_run):
    ds = TrainingWindowDataset(p57_windows_run)
    batch = ds.collate([ds[i] for i in range(len(ds))])
    snap = {k: np.array(batch[k]) for k in ("anchor_uv_px", "target_future_uv_px", "target_future_valid", "target_valid")}
    s_a = fit([batch])
    mutated = dict(batch, target_future_uv_px=np.full_like(batch["target_future_uv_px"], 1e9), target_future_valid=np.zeros_like(batch["target_future_valid"]),
                   anchor_uv_px=np.full_like(batch["anchor_uv_px"], -7.0))
    assert fit([mutated]).compute_hash() == s_a.compute_hash()
    enc = SpatioTemporalEncoder(EncoderConfig.load(ENC_CONFIG), 6, scaler=s_a.feature_scaler())
    with torch.no_grad():
        enc(TrainingWindowDataset.encoder_input(batch))
    for k, v in snap.items():                                                                        # raw supervision untouched by fitting and by the scaled encoder pass
        np.testing.assert_array_equal(batch[k], v)


def test_dataset_helper_fits_exactly_the_selected_windows(p57_windows_run):
    ds = TrainingWindowDataset(p57_windows_run)
    sel = [0, 3, 5, 9]
    a = ds.fit_scaler_statistics(sel, fit_partition="train", split_identity="s", batch_size=3)
    b = ds.fit_scaler_statistics(sel, fit_partition="train", split_identity="s", batch_size=4)
    assert a.compute_hash() == b.compute_hash() or np.allclose(a.node_mean, b.node_mean, rtol=1e-12)     # batch size does not change the fitted values
    assert a.compute_hash() != ds.fit_scaler_statistics(list(range(len(ds))), fit_partition="train", split_identity="s").compute_hash()
    manual = fit([ds.collate([ds[i] for i in sel])], split_identity="s", source_identity=a.source_identity)
    np.testing.assert_allclose(manual.node_mean, a.node_mean, rtol=1e-12)
    np.testing.assert_allclose(manual.edge_scale, a.edge_scale, rtol=1e-12)
    for bad in ([], [0, 0], [len(ds)]):
        with pytest.raises(WindowContractError, match="SCALER_INDICES"):
            ds.fit_scaler_statistics(bad, fit_partition="train", split_identity="s")


def test_deterministic_serialization_hash_and_roundtrip(tmp_path):
    b = batch_from(make_input(B=3, seed=26))
    s = fit([b])
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    h1, h2 = s.save(p1), fit([copy.deepcopy(b)]).save(p2)
    assert h1 == h2 == s.compute_hash() and p1.read_bytes() == p2.read_bytes()
    loaded = TrainOnlyFeatureStatistics.load(p1, expected_source_identity=SRC, expected_split_identity="pilot-train")
    p3 = tmp_path / "c.json"
    assert loaded.save(p3) == h1 and p3.read_bytes() == p1.read_bytes()                              # load -> save is byte-identical
    body = json.loads(p1.read_text(encoding="utf-8"))
    assert body["statistics_sha256"] == h1 and body["fit_partition"] == "train"
    p4 = tmp_path / "d.json"
    p4.write_text(json.dumps({k: body[k] for k in reversed(list(body))}), encoding="utf-8")        # key order does not change the canonical hash
    assert TrainOnlyFeatureStatistics.load(p4).compute_hash() == h1


def test_provenance_and_tamper_verification(tmp_path):
    s = fit([batch_from(make_input(B=2, seed=27))])
    p = tmp_path / "s.json"
    s.save(p)
    with pytest.raises(WindowContractError, match="SCALER_PROVENANCE"):
        TrainOnlyFeatureStatistics.load(p, expected_source_identity={"dataset": "other", "sha256": "abc"})
    with pytest.raises(WindowContractError, match="SCALER_PROVENANCE"):
        TrainOnlyFeatureStatistics.load(p, expected_split_identity="other-split")
    body = json.loads(p.read_text(encoding="utf-8"))
    body["node"]["mean"][0] += 1e-6
    t = tmp_path / "t.json"
    t.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(WindowContractError, match="SCALER_HASH_MISMATCH"):
        TrainOnlyFeatureStatistics.load(t)
    body = json.loads(p.read_text(encoding="utf-8"))
    body["fit_partition"] = "validation"
    v = tmp_path / "v.json"
    v.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(WindowContractError, match="SCALER_NOT_TRAIN_ONLY"):
        TrainOnlyFeatureStatistics.load(v)
    body = json.loads(p.read_text(encoding="utf-8"))
    body["node"]["feature_names"] = list(reversed(body["node"]["feature_names"]))
    o = tmp_path / "o.json"
    o.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(WindowContractError, match="SCALER_FEATURE_ORDER"):
        TrainOnlyFeatureStatistics.load(o)


def test_feature_scaler_contract_and_no_torch_import_outside_models():
    s = fit([batch_from(make_input(B=2, seed=28))])
    sc = s.feature_scaler()
    assert isinstance(sc, FeatureScaler) and sc.source == "TRAIN_PARTITION_STATISTICS" and sc.provenance["fit_partition"] == "train"
    assert sc.provenance["statistics_sha256"] == s.compute_hash()
    src = (Path(__file__).resolve().parents[2] / "pedsafety" / "phase5" / "training_scaler.py").read_text(encoding="utf-8")
    assert "import torch" not in src                                                                 # dependency policy: torch only under pedsafety/phase5/models
