"""P5.5 integrated encoder: history-only contract, identity association, masks, centring, leakage sensitivity, adapter integration."""

import dataclasses

import numpy as np
import pytest
import torch

from pedsafety.errors import ConfigError
from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.graph_window_adapter import collate_windows
from pedsafety.phase5.models import EdgeBatch, EncoderInput, FeatureScaler, SpatioTemporalEncoder
from pedsafety.phase5.models.encoder import HISTORY_KEYS
from pedsafety.phase5.runner import generate_windows
from pedsafety.phase5.window_storage import read_dataset
from phase5_helpers import CONFIG as WINDOW_CONFIG
from phase5_helpers import wcfg
from enc_helpers import CONFIG as ENC_CONFIG
from enc_helpers import cfg, cfg_bike, make_input, pad_slots, permute_slots, replace


def enc(c=None, H=6):
    m = SpatioTemporalEncoder(c or cfg(), history_length=H)
    m.eval()
    return m


ABSENT = [(0, 1, 3), (0, 2, 3), (1, 4, 2), (1, 0, 4)]


def test_forward_shapes_masks_and_target_context_roles():
    inp = make_input(B=3, H=6, A=8, types=(0, 0, 1, 1, 0), absent=ABSENT[:2] + [(2, 5, 4)])
    o = enc()(inp)
    assert o.sequence_embeddings.shape == (3, 8, 6, 64) and o.spatial_embeddings.shape == (3, 6, 8, 64)
    assert o.target_summary.shape == (3, 64) and o.last_step_embeddings.shape == (3, 8, 64)
    assert torch.equal(o.target_summary, o.sequence_embeddings[:, 0, -1])
    assert o.is_target[:, 0].all() and not o.is_target[:, 1:].any() and (o.node_type[:, 0] == 0).all()
    assert o.receptive_field == 7
    assert (o.sequence_embeddings[~o.presence] == 0).all() and (o.spatial_embeddings.permute(0, 2, 1, 3)[~o.presence] == 0).all()
    assert (o.sequence_embeddings[~o.slot_valid] == 0).all()                       # padded slots
    assert not o.last_step_mask[2, 4] and o.last_step_mask[:, 0].all()               # absent at the last step -> masked, not filled
    assert torch.isfinite(o.sequence_embeddings).all()
    assert torch.equal(o.window_id, inp.window_id) and torch.equal(o.track_index, inp.track_index)


def test_variable_agent_counts_across_frames_and_windows():
    a = make_input(B=2, H=6, A=7, types=(0, 0, 1), absent=[(0, 2, 1), (0, 3, 1), (1, 0, 2)])
    b = make_input(B=2, H=6, A=7, types=(0, 1, 1, 0, 0, 1), absent=[(0, 5, 4), (1, 2, 3)], seed=1)
    for inp in (a, b):
        o = enc()(inp)
        assert torch.isfinite(o.sequence_embeddings).all() and (o.presence.sum() > 0)


def test_padding_amount_does_not_change_results():
    small = make_input(B=2, H=6, A=5, types=(0, 0, 1, 1, 0), absent=ABSENT[:2], seed=4)
    big = pad_slots(small, 4)
    m = enc()
    o1, o2 = m(small), m(big)
    assert o2.sequence_embeddings.shape[1] == 9
    assert torch.allclose(o1.target_summary, o2.target_summary, atol=1e-6)
    assert torch.allclose(o1.sequence_embeddings, o2.sequence_embeddings[:, :5], atol=1e-6)
    assert (o2.sequence_embeddings[:, 5:] == 0).all()


def test_track_identity_association_slot_permutation_equivariance():
    inp = make_input(B=2, H=6, A=6, types=(0, 0, 1, 1, 0, 1), absent=ABSENT, seed=2)
    m = enc()
    o = m(inp)
    p = np.array([0, 4, 2, 5, 1, 3])
    o_p = m(permute_slots(inp, p))
    assert torch.allclose(o_p.target_summary, o.target_summary, atol=1e-5)
    assert torch.allclose(o_p.sequence_embeddings, o.sequence_embeddings[:, p], atol=1e-5)
    assert torch.equal(o_p.track_index, o.track_index[:, p])                           # track identities travel with their embeddings


def test_same_track_keeps_same_slot_across_frames_is_required_by_construction():
    inp = make_input(B=1, H=6, A=5, types=(0, 0, 1, 1))
    with pytest.raises(WindowContractError, match="DUPLICATE_TRACK_SLOT"):          # validated when the input is constructed
        replace(inp, track_index=torch.tensor([[100, 101, 101, 103, -1]]))


def test_future_target_arrays_cannot_reach_the_encoder():
    inp_fields = {f.name for f in dataclasses.fields(EncoderInput)}
    assert not any("future" in n or "target" in n for n in inp_fields)
    assert not any("future" in k or "target_" in k for k in HISTORY_KEYS)
    m = enc()
    with pytest.raises(WindowContractError, match="ENCODER_INPUT_TYPE"):
        m({"node_features": torch.zeros(1)})
    with pytest.raises(TypeError):
        EncoderInput(**{**{f.name: getattr(make_input(), f.name) for f in dataclasses.fields(EncoderInput)}, "target_future_uv_px": torch.zeros(2, 15, 2)})


def test_collated_batch_future_arrays_are_ignored_and_mutating_them_changes_nothing():
    inp = make_input(B=2, H=6, A=6, types=(0, 0, 1, 1, 0), absent=ABSENT[:2])
    batch = {"window_id": inp.window_id.numpy(), "anchor_frame": inp.anchor_frame.numpy(), "history_frames": inp.history_frames.numpy(),
             "track_index": inp.track_index.numpy(), "node_type": inp.node_type.numpy(), "slot_valid": inp.slot_valid.numpy(),
             "present": inp.present.numpy(), "node_features": inp.node_features.numpy(), "node_masks": inp.node_masks.numpy(),
             "edges": {r: {"bhsd": e.bhsd.numpy(), "edge_attr": e.attr.numpy(), "edge_mask": e.mask.numpy(), "flags": e.flags.numpy()} for r, e in inp.edges.items()},
             "target_future_uv_px": np.zeros((2, 15, 2), np.float32), "target_future_valid": np.ones((2, 15), bool),
             "future_frames": np.arange(15)[None].repeat(2, 0)}
    m = enc()
    out1 = m(EncoderInput.from_collated_batch(batch)).sequence_embeddings
    batch["target_future_uv_px"][:] = 1e9
    batch["future_frames"][:] = -7
    batch["target_future_valid"][:] = False
    out2 = m(EncoderInput.from_collated_batch(batch)).sequence_embeddings
    assert torch.equal(out1, out2)
    broken = {k: v for k, v in batch.items() if k != "present"}
    with pytest.raises(WindowContractError, match="INPUT_MISSING"):
        EncoderInput.from_collated_batch(broken)


def test_leakage_test_is_sensitive_injecting_future_information_into_history_changes_outputs():
    """Meta-test: if future information WERE put into a history field, the outputs would change, so the invariance tests above can fail."""
    inp = make_input(B=2, H=6, A=6, types=(0, 0, 1, 1, 0), seed=3)
    m = enc()
    base = m(inp).target_summary
    f = inp.node_features.clone()
    f[:, -1, 0, 0:2] += 120.0                                       # pretend a future position was written into the last history sample
    leaked = m(replace(inp, node_features=f)).target_summary
    assert not torch.allclose(base, leaked)
    f2 = inp.node_features.clone()
    f2[:, 2, 1, 0:2] += 80.0                                        # a context agent's history position is also used
    assert not torch.allclose(base, m(replace(inp, node_features=f2)).target_summary)


def test_absolute_position_is_not_an_input_translation_invariance():
    inp = make_input(B=2, H=6, A=6, types=(0, 0, 1, 1, 0), absent=ABSENT[:2], seed=5)
    f = inp.node_features.clone()
    f[..., 0] += 321.0
    f[..., 1] += -123.0                                             # NaNs at absent nodes stay NaN
    m = enc()
    assert torch.allclose(m(replace(inp, node_features=f)).sequence_embeddings, m(inp).sequence_embeddings, atol=1e-3)


def test_masked_values_have_no_influence():
    inp = make_input(B=2, H=6, A=6, types=(0, 0, 1, 1, 0), absent=ABSENT, seed=6)
    m = enc()
    base = m(inp)
    # (a) masked edge attributes: NaN / huge values
    edges = {}
    for r, e in inp.edges.items():
        a = torch.where(e.mask, e.attr, torch.full_like(e.attr, float("nan")))
        edges[r] = EdgeBatch(e.bhsd, a, e.mask, e.flags)
    out_nan = m(replace(inp, edges=edges))
    edges_big = {r: EdgeBatch(e.bhsd, torch.where(e.mask, e.attr, torch.full_like(e.attr, 1e30)), e.mask, e.flags) for r, e in inp.edges.items()}
    out_big = m(replace(inp, edges=edges_big))
    assert torch.equal(out_nan.sequence_embeddings, base.sequence_embeddings) and torch.equal(out_big.sequence_embeddings, base.sequence_embeddings)
    # (b) masked node velocity / heading / gap values
    f = inp.node_features.clone()
    vm = inp.node_masks[..., 1] & inp.present
    f[..., 2][~vm] = 5e8
    f[..., 5][~(inp.node_masks[..., 2] & inp.present)] = 7e8
    f[..., 8][~(inp.node_masks[..., 3] & inp.present)] = 9e8
    assert torch.equal(m(replace(inp, node_features=f)).sequence_embeddings, base.sequence_embeddings)
    # (c) a valid value does matter (including the gap feature)
    f2 = inp.node_features.clone()
    ok = inp.node_masks[:, -1, 0, 3]
    f2[:, -1, 0, 8] = torch.where(ok, f2[:, -1, 0, 8] + 7.0, f2[:, -1, 0, 8])
    assert ok.any() and not torch.equal(m(replace(inp, node_features=f2)).target_summary, base.target_summary)
    # (d) invalid node features are never read
    f3 = inp.node_features.clone()
    f3[~inp.present] = 1e12
    assert torch.equal(m(replace(inp, node_features=f3)).sequence_embeddings, base.sequence_embeddings)


def test_nonfinite_valid_values_are_rejected():
    inp = make_input(B=1, H=6, A=5, types=(0, 0, 1, 1))
    f = inp.node_features.clone()
    f[0, 2, 1, 0] = float("nan")
    with pytest.raises(WindowContractError, match="NONFINITE_VALID_VALUE"):
        replace(inp, node_features=f)
    e = dict(inp.edges)
    r = "pedestrian__pedestrian"
    a = e[r].attr.clone()
    a[0, 0] = float("inf")
    e[r] = EdgeBatch(e[r].bhsd, a, e[r].mask, e[r].flags)
    with pytest.raises(WindowContractError, match="NONFINITE_VALID_VALUE"):
        replace(inp, edges=e)


def test_temporal_causality_of_the_integrated_encoder():
    inp = make_input(B=2, H=6, A=6, types=(0, 0, 1, 1, 0), absent=[(0, 1, 3)], seed=7)
    m = enc()
    base = m(inp).sequence_embeddings
    f = inp.node_features.clone()
    # keep the centring reference (the target's LAST sample) fixed so that only the changed sample can matter
    f[:, 4, 1:, 7] *= 1.5                                           # change only a non-last, non-target sample (box height of context agents)
    chg = m(replace(inp, node_features=f)).sequence_embeddings
    assert torch.equal(chg[:, :, :4], base[:, :, :4]) and not torch.equal(chg[:, :, 4:], base[:, :, 4:])


def test_target_centring_uses_only_the_last_history_sample():
    inp = make_input(B=1, H=6, A=5, types=(0, 0, 1, 1), seed=8)
    m = enc()
    base = m(inp).target_summary
    f = inp.node_features.clone()
    f[:, -1, 0, 0:2] += 55.0                                        # moving the reference changes every relative position
    assert not torch.allclose(m(replace(inp, node_features=f)).target_summary, base)


def test_node_type_distinctions_and_unsupported_types():
    inp = make_input(B=1, H=6, A=5, types=(0, 0, 1, 1))
    bike = replace(inp, node_type=torch.tensor([[0, 0, 2, 1, -1]]))
    with pytest.raises(WindowContractError, match="UNSUPPORTED_NODE_TYPE"):
        enc()(bike)
    m = enc()
    swapped = replace(inp, node_type=torch.tensor([[0, 1, 0, 1, -1]]))
    assert not torch.allclose(m(inp).sequence_embeddings[:, 1], m(swapped).sequence_embeddings[:, 1])
    with pytest.raises(WindowContractError, match="TARGET_NOT_PEDESTRIAN"):
        replace(inp, node_type=torch.tensor([[1, 0, 1, 1, -1]]))


def test_default_excludes_bicycle_context_and_opt_in_works():
    inp = make_input(B=2, H=6, A=7, types=(0, 0, 1, 2, 2), relations=("pedestrian__pedestrian", "vehicle__pedestrian", "pedestrian__vehicle", "bicyclist__pedestrian",
                                                                      "pedestrian__bicyclist", "bicyclist__vehicle", "vehicle__bicyclist"), seed=9)
    with pytest.raises(WindowContractError, match="UNSUPPORTED_(RELATION|NODE_TYPE)"):
        enc()(inp)
    o = enc(cfg_bike())(inp)
    assert torch.isfinite(o.sequence_embeddings).all() and (o.node_type == 2).any()
    assert cfg().bicyclist_context_enabled is False and "bicyclist" not in cfg().node_types


def test_missing_edge_flags_when_enabled():
    inp = make_input(B=1, H=6, A=5, types=(0, 0, 1, 1))
    m = enc(cfg(use_edge_flags=True))
    assert torch.isfinite(m(inp).sequence_embeddings).all()
    no_flags = {r: EdgeBatch(e.bhsd, e.attr, e.mask, None) for r, e in inp.edges.items()}
    with pytest.raises(WindowContractError, match="INPUT_MISSING"):
        m(replace(inp, edges=no_flags))


def test_history_length_configurable_and_receptive_field_guard():
    for H in (2, 4, 6, 7):
        o = enc(H=H)(make_input(B=2, H=H, A=5, types=(0, 0, 1, 1), seed=H))
        assert o.sequence_embeddings.shape[2] == H
    with pytest.raises(ConfigError, match="does not cover"):
        SpatioTemporalEncoder(cfg(), history_length=10)
    long_ok = SpatioTemporalEncoder(cfg(require_full_history_coverage=False), history_length=10)
    assert long_ok(make_input(B=1, H=10, A=5, types=(0, 0, 1, 1))).sequence_embeddings.shape[2] == 10
    with pytest.raises(ConfigError, match="does not cover"):
        enc()(make_input(B=1, H=10, A=5, types=(0, 0, 1, 1)))                         # forward guards the actual input length too


def test_input_validation_errors():
    with pytest.raises(WindowContractError, match="TARGET_MISSING"):
        make_input(B=2, H=6, A=5, types=(0, 0, 1, 1), absent=[(0, 5, 0)])
    ok = make_input(B=2, H=6, A=5, types=(0, 0, 1, 1))
    with pytest.raises(WindowContractError, match="HISTORY_ORDER"):
        replace(ok, history_frames=ok.history_frames.flip(1))
    with pytest.raises(WindowContractError, match="INPUT_SHAPE"):
        replace(ok, node_masks=ok.node_masks[..., :3])
    e = dict(ok.edges)
    e["pedestrian__pedestrian"] = EdgeBatch(torch.tensor([[0, 0, 0, 99]]), torch.zeros(1, 14), torch.ones(1, 14, dtype=torch.bool), None)
    with pytest.raises(WindowContractError, match="EDGE_ENDPOINT"):
        replace(ok, edges=e)
    with pytest.raises(WindowContractError, match="PRESENT_WITHOUT_SLOT"):
        replace(ok, slot_valid=torch.tensor([[True, True, True, False, False]] * 2))


def test_scaler_interface_train_only_statistics():
    c = cfg()
    s = FeatureScaler.from_config(c.feature_scaling)
    assert s.source == "NOMINAL_DEVELOPMENT_DEFAULT_NOT_FITTED" and s.provenance["fitted"] is False
    fs = {"source": "TRAIN_PARTITION_STATISTICS", "fit_partition": "val", "statistics_sha256": "x" * 64,
          "node_offset": {}, "node_scale": {}, "edge_offset": {}, "edge_scale": {}}
    with pytest.raises(WindowContractError, match="SCALER_NOT_TRAIN_ONLY"):
        FeatureScaler.from_config(fs)
    with pytest.raises(WindowContractError, match="SCALER_NOT_TRAIN_ONLY"):
        FeatureScaler.from_config(dict(fs, fit_partition="train", statistics_sha256=""))
    ok = FeatureScaler.from_config(dict(fs, fit_partition="train"))
    assert ok.provenance["fit_partition"] == "train"
    inp = make_input(B=1, H=6, A=5, types=(0, 0, 1, 1))
    m = SpatioTemporalEncoder(c, 6, scaler=ok)                                     # statistics can be supplied later
    assert torch.isfinite(m(inp).sequence_embeddings).all()
    with pytest.raises(ConfigError):
        FeatureScaler.from_config(dict(c.feature_scaling, node={**c.feature_scaling["node"], "box_h": 0.0}))
    with pytest.raises(ConfigError):
        cfg(feature_scaling={"source": "FITTED_ON_PILOT"})


def test_forward_backward_smoke_all_parameters_receive_finite_gradients():
    inp = make_input(B=3, H=6, A=7, types=(0, 0, 1, 1, 0, 1), absent=ABSENT, seed=10)
    m = SpatioTemporalEncoder(cfg(), 6)
    m.train()
    o = m(inp)
    (o.target_summary.pow(2).mean() + o.sequence_embeddings.pow(2).mean()).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    assert sum(p.grad.abs().sum() for p in m.parameters()) > 0
    assert m.parameter_count() == sum(p.numel() for p in m.parameters()) > 0


def test_deterministic_and_seed_isolated():
    inp = make_input(B=2, H=6, A=6, types=(0, 0, 1, 1, 0), absent=ABSENT[:2], seed=11)
    torch.manual_seed(1234)
    before = torch.rand(1)
    torch.manual_seed(1234)
    a = enc()(inp).sequence_embeddings
    after = torch.rand(1)
    assert torch.equal(before, after)                                                # model construction did not consume global RNG state
    assert torch.equal(a, enc()(inp).sequence_embeddings)
    assert not torch.equal(a, enc(cfg(init_seed=3))(inp).sequence_embeddings)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device detected; CUDA is not exercised in this environment")
def test_optional_cuda_forward_backward_smoke_matches_cpu():
    inp = make_input(B=2, H=6, A=6, types=(0, 0, 1, 1, 0), absent=ABSENT[:2], seed=12)
    cpu = enc()
    gpu = SpatioTemporalEncoder(cfg(), 6).cuda()
    gpu.eval()
    ref = cpu(inp).sequence_embeddings
    out = gpu(inp.to("cuda")).sequence_embeddings
    assert torch.allclose(out.cpu(), ref, atol=1e-4)
    gpu(inp.to("cuda")).target_summary.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in gpu.parameters() if p.grad is not None)


def test_input_to_cpu_roundtrip_preserves_content():
    inp = make_input(B=2, H=6, A=5, types=(0, 0, 1, 1), absent=ABSENT[:1], seed=13)
    again = inp.to("cpu")
    assert torch.equal(enc()(inp).sequence_embeddings, enc()(again).sequence_embeddings)


# ---------------- real P5.3 adapter path (synthetic scene run through the actual Task 2 -> Task 3 -> P5.3 pipeline) ----------------
@pytest.fixture(scope="module")
def p53(scene_run_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("p55win") / "windows"
    generate_windows(scene_run_dir, wcfg(), out, config_path=str(WINDOW_CONFIG))
    ws, bank, extra = read_dataset(out)
    return wcfg(), ws, bank


def test_encoder_on_real_p53_collated_batches(p53):
    wc, ws, bank = p53
    m = SpatioTemporalEncoder(cfg(), history_length=wc.history_length)
    m.eval()
    batch = collate_windows(bank, ws, range(ws.n_windows), wc)
    inp = EncoderInput.from_collated_batch(batch)
    o = m(inp)
    B, H, A = batch["present"].shape
    assert o.sequence_embeddings.shape == (B, A, wc.history_length, 64) and torch.isfinite(o.sequence_embeddings).all()
    assert torch.equal(o.track_index[:, 0], torch.as_tensor(ws.target_track_index.astype(np.int64)))      # slot 0 = the window's target track
    assert o.presence[:, 0].all() and (o.node_type[:, 0] == 0).all()
    assert set(batch["active_relations"]) == set(cfg().relations)
    # P5.3 default context contains no bicyclist node: the default encoder accepts it as is
    assert not (o.node_type == 2).any()
    # a future-only agent (vehicle 9 appears at frame 50) never appears as an encoder slot
    assert (o.presence.sum(dim=(1, 2)) > 0).all()


def test_encoder_with_optional_bicycle_context_on_real_batches(p53):
    wc, ws, bank = p53
    m = SpatioTemporalEncoder(cfg_bike(), history_length=wc.history_length)
    m.eval()
    batch = collate_windows(bank, ws, range(ws.n_windows), wc, include_optional=True)
    o = m(EncoderInput.from_collated_batch(batch))
    assert (o.node_type == 2).any() and torch.isfinite(o.sequence_embeddings).all()
    with pytest.raises(WindowContractError, match="UNSUPPORTED_(RELATION|NODE_TYPE)"):
        enc()(EncoderInput.from_collated_batch(batch))                                # default encoder refuses bicycle-derived input


def test_real_batch_future_arrays_do_not_matter(p53):
    wc, ws, bank = p53
    m = SpatioTemporalEncoder(cfg(), history_length=wc.history_length)
    m.eval()
    batch = collate_windows(bank, ws, range(0, ws.n_windows, 2), wc)
    a = m(EncoderInput.from_collated_batch(batch)).sequence_embeddings
    batch["target_future_uv_px"][:] = np.nan
    batch["future_frames"][:] = 0
    assert torch.equal(a, m(EncoderInput.from_collated_batch(batch)).sequence_embeddings)
