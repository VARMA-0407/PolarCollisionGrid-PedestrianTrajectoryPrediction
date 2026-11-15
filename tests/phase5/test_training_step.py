"""P5.7.3: integrated training step (batch -> scaler -> encoder -> decoder -> masked loss -> gradients), history-only isolation, masks, numerics."""

import copy
import dataclasses
import inspect

import numpy as np
import pytest
import torch

from pedsafety.errors import PedsafetyError
from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.models import EncoderConfig
from pedsafety.phase5.models.decoder import DecoderConfig
from pedsafety.phase5.models.forecaster import ForecastingModel, check_compatibility
from pedsafety.phase5.models.training_step import TrainBatch, TrainingNumericsError, eval_step, forward_loss, make_train_batch, train_step
from pedsafety.phase5.training_data import TrainingWindowDataset
from enc_helpers import CONFIG as ENC_CONFIG
from enc_helpers import cfg as enc_cfg
from phase5_helpers import wcfg

REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
DEC_CONFIG = REPO / "configs" / "phase5" / "decoder_default.json"
CW = 0.1


@pytest.fixture(scope="module")
def ds(p57_windows_run):
    return TrainingWindowDataset(p57_windows_run)


@pytest.fixture(scope="module")
def stats(ds):
    return ds.fit_scaler_statistics(list(range(len(ds))), fit_partition="train", split_identity="synthetic-train", batch_size=8)


def make_model(ds, stats, seed=1):
    return ForecastingModel(EncoderConfig.load(ENC_CONFIG), DecoderConfig.load(DEC_CONFIG), ds.config, scaler=stats.feature_scaler(), decoder_seed=seed)


def tbatch(ds, idx=range(8), device="cpu"):
    return make_train_batch(ds.collate([ds[i] for i in idx]), device)


def with_threads(n):
    class Ctx:
        def __enter__(self):
            self.old = torch.get_num_threads()
            torch.set_num_threads(n)

        def __exit__(self, *a):
            torch.set_num_threads(self.old)
    return Ctx()


def test_integration_shapes_and_finite_loss(ds, stats):
    m = make_model(ds, stats)
    tb = tbatch(ds)
    out, loss = forward_loss(m, tb, CW)
    B = 8
    assert tuple(out.encoded.target_summary.shape) == (B, 64)
    assert tuple(out.decoded.trajectories_uv_px.shape) == (B, 5, 15, 2) and tuple(out.decoded.mode_probabilities.shape) == (B, 5)
    assert torch.isfinite(loss.total_loss) and loss.valid_window_count == B
    torch.testing.assert_close(out.decoded.mode_probabilities.sum(dim=1), torch.ones(B))
    # raw pixel anchor preserved: trajectories == anchor + raw offsets exactly
    assert torch.equal(out.decoded.trajectories_uv_px, tb.anchor_uv_px[:, None, None, :] + out.decoded.trajectory_offsets_px)


def test_gradients_reach_encoder_and_decoder_and_parameters_change(ds, stats):
    m = make_model(ds, stats)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    r = train_step(m, opt, tbatch(ds), classification_weight=CW, grad_clip_norm=10.0)
    assert r.grad_norm_encoder > 0 and r.grad_norm_decoder > 0 and np.isfinite(r.grad_norm_total)
    assert r.parameters_without_grad == []                                                              # every parameter of the default configuration got a gradient
    for part in ("encoder.spatial.", "encoder.temporal.", "decoder.trajectory_head", "decoder.mode_head", "decoder.input_projection"):
        names = [n for n in before if n.startswith(part) or part in n]
        assert names and any(not torch.equal(before[n], dict(m.named_parameters())[n]) for n in names), part
    assert not any(torch.equal(before[n], dict(m.named_parameters())[n]) for n in before if "scaler" in n)   # (scaler buffers are not parameters)
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_scaler_buffers_are_not_trained(ds, stats):
    m = make_model(ds, stats)
    before = {k: v.clone() for k, v in m.encoder.scaler.state_dict().items()}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(2):
        train_step(m, opt, tbatch(ds), classification_weight=CW)
    assert all(torch.equal(before[k], v) for k, v in m.encoder.scaler.state_dict().items())
    assert not any("scaler" in n for n, _ in m.named_parameters())


def test_model_forward_signature_has_no_future_argument_and_input_has_no_future_fields(ds, stats):
    params = list(inspect.signature(ForecastingModel.forward).parameters)
    assert params == ["self", "encoder_input", "anchor_uv_px", "target_valid"]
    names = [f.name for f in dataclasses.fields(type(tbatch(ds).encoder_input))]
    assert not any("future" in n or "target" in n for n in names)
    assert [f.name for f in dataclasses.fields(TrainBatch)][-3:] == ["target_future_uv_px", "target_future_valid", "window_id"]


def test_future_targets_used_only_for_loss_not_prediction(ds, stats):
    m = make_model(ds, stats)
    tb = tbatch(ds)
    out1, loss1 = forward_loss(m, tb, CW)
    tb2 = dataclasses.replace(tb, target_future_uv_px=tb.target_future_uv_px + 123.0)
    out2, loss2 = forward_loss(m, tb2, CW)
    assert torch.equal(out1.decoded.trajectories_uv_px, out2.decoded.trajectories_uv_px) and torch.equal(out1.decoded.mode_logits, out2.decoded.mode_logits)
    assert not torch.equal(loss1.total_loss, loss2.total_loss)                                          # ... while the loss does depend on the supervision
    # a model call recorded by a spy never receives a tensor equal to any future array
    seen = []
    orig = m.forward
    m.forward = lambda *a, **k: (seen.append((a, k)), orig(*a, **k))[1]
    forward_loss(m, tb, CW)
    flat = [x for x in seen[0][0]] + list(seen[0][1].values())
    fut = [tb.target_future_uv_px, tb.target_future_valid]
    for x in flat:
        assert not (isinstance(x, torch.Tensor) and any(x.shape == f.shape and torch.equal(x, f) for f in fut))
    assert len(seen[0][0]) == 3 and not seen[0][1]


def test_future_leak_into_history_input_is_detected_by_the_isolation_test(ds, stats):
    """Meta-test: if future positions WERE written into the history input, predictions change, so the invariance test above would fail."""
    m = make_model(ds, stats)
    tb = tbatch(ds)
    base = forward_loss(m, tb, CW)[0].decoded.trajectories_uv_px
    enc = tb.encoder_input
    f = enc.node_features.clone()
    f[:, -1, 0, :2] += 80.0
    leaked = dataclasses.replace(tb, encoder_input=type(enc)(**{**{k: getattr(enc, k) for k in enc.__dataclass_fields__}, "node_features": f}))
    assert not torch.equal(forward_loss(m, leaked, CW)[0].decoded.trajectories_uv_px, base)


def test_masked_supervision_invalid_targets_do_not_contaminate_loss_or_gradients(ds, stats):
    m = make_model(ds, stats)
    tb = tbatch(ds)
    fv = tb.target_future_valid.clone()
    fv[0, 7:] = False
    fv[1, :3] = False
    t = tb.target_future_uv_px.clone()
    base = dataclasses.replace(tb, target_future_valid=fv, target_future_uv_px=torch.where(fv[..., None], t, torch.full_like(t, float("nan"))))
    other = dataclasses.replace(tb, target_future_valid=fv, target_future_uv_px=torch.where(fv[..., None], t, torch.full_like(t, 1e9)))
    l1, l2 = forward_loss(m, base, CW)[1], forward_loss(m, other, CW)[1]
    assert torch.isfinite(l1.total_loss) and torch.equal(l1.total_loss, l2.total_loss)                  # NaN or 1e9 under a False mask: no effect
    m.zero_grad()
    l1.total_loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    # a window with no valid future step is excluded and counted, not zero-filled
    fv0 = fv.clone()
    fv0[2] = False
    r = forward_loss(m, dataclasses.replace(tb, target_future_valid=fv0), CW)[1]
    assert r.valid_window_count == 7 and not bool(r.valid_window_mask[2]) and r.diagnostics["excluded_windows"] == 1
    # target_valid False excludes the window too
    tv = tb.target_valid.clone()
    tv[3] = False
    assert forward_loss(m, dataclasses.replace(tb, target_valid=tv), CW)[1].valid_window_count == 7
    with pytest.raises(WindowContractError, match="NO_VALID_TARGETS"):
        forward_loss(m, dataclasses.replace(tb, target_future_valid=torch.zeros_like(fv)), CW)


def test_nonfinite_valid_values_rejected(ds, stats):
    m = make_model(ds, stats)
    tb = tbatch(ds)
    t = tb.target_future_uv_px.clone()
    t[0, 0, 0] = float("nan")
    with pytest.raises(WindowContractError, match="NONFINITE_VALID_VALUE"):
        forward_loss(m, dataclasses.replace(tb, target_future_uv_px=t), CW)
    a = tb.anchor_uv_px.clone()
    a[0, 0] = float("inf")
    with pytest.raises(WindowContractError, match="NONFINITE_VALID_VALUE"):
        forward_loss(m, dataclasses.replace(tb, anchor_uv_px=a), CW)


def test_nonfinite_loss_and_gradient_are_fatal_and_leave_parameters_unchanged(ds, stats):
    m = make_model(ds, stats)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    h = next(iter(m.decoder.parameters())).register_hook(lambda g: g * float("inf"))
    with pytest.raises(TrainingNumericsError, match="NONFINITE_GRADIENT"):
        train_step(m, opt, tbatch(ds), classification_weight=CW)
    h.remove()
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters())                              # the optimizer step was not taken
    with torch.no_grad():
        m.decoder.mode_head.bias.fill_(float("nan"))
    with pytest.raises((TrainingNumericsError, WindowContractError)):
        train_step(m, opt, tbatch(ds), classification_weight=CW)


def test_gradient_clipping_and_scheduler_and_optimizer_step(ds, stats):
    m = make_model(ds, stats)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
    r = train_step(m, opt, tbatch(ds), classification_weight=CW, grad_clip_norm=1e-3, scheduler=sched)
    assert r.grad_clipped is True and r.grad_norm_total > 1e-3
    post = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in m.parameters() if p.grad is not None))
    assert float(post) <= 1e-3 * (1 + 1e-4)                                                              # the gradients that were applied are clipped
    assert opt.param_groups[0]["lr"] < 1e-3 and sched.last_epoch == 1 and r.lr == 1e-3
    r2 = train_step(m, opt, tbatch(ds), classification_weight=CW, grad_clip_norm=None)
    assert r2.grad_clipped is False


def test_training_step_is_deterministic_single_thread(ds, stats):
    def run():
        with with_threads(1):
            m = make_model(ds, stats)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            return [train_step(m, opt, tbatch(ds, range(i, i + 8)), classification_weight=CW, grad_clip_norm=10.0).loss for i in (0, 4, 8)], \
                {n: p.detach().clone() for n, p in m.named_parameters()}
    (l1, p1), (l2, p2) = run(), run()
    assert l1 == l2 and all(torch.equal(p1[n], p2[n]) for n in p1)


def test_loss_decreases_when_overfitting_a_tiny_batch(ds, stats):
    with with_threads(1):
        m = make_model(ds, stats)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)          # raw-pixel offsets (P5.6 contract) need a moderately high rate to fit quickly
        tb = tbatch(ds, range(6))
        losses = [train_step(m, opt, tb, classification_weight=CW, grad_clip_norm=10.0).loss for _ in range(40)]
    assert losses[-1] < 0.5 * losses[0]                                                                 # measured in development: 23.9 -> 6.6


def test_eval_step_has_no_side_effects(ds, stats):
    m = make_model(ds, stats)
    m.train()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    r = eval_step(m, tbatch(ds), classification_weight=CW)
    assert m.training is True and all(p.grad is None for p in m.parameters())
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters()) and np.isfinite(r.loss) and r.grad_norm_total is None


def test_mode_and_winner_diagnostics(ds, stats):
    m = make_model(ds, stats)
    r = eval_step(m, tbatch(ds), classification_weight=CW)
    assert sum(r.winner_counts) == r.usable_windows == 8 and len(r.winner_counts) == 5
    assert r.oracle_mode_ade_px <= r.selected_mode_ade_px + 1e-6 and np.isfinite(r.mean_mode_entropy) and 0 < r.max_mode_probability_mean <= 1


def test_compatibility_checks():
    ec, dc, wc = EncoderConfig.load(ENC_CONFIG), DecoderConfig.load(DEC_CONFIG), wcfg()
    check_compatibility(ec, dc, wc)
    d = dc.to_dict()
    d["input_embedding_dim"] = 32
    with pytest.raises(PedsafetyError):
        check_compatibility(ec, DecoderConfig.from_dict(d), wc)
    d = dc.to_dict()
    d["future_steps"] = 10
    with pytest.raises(PedsafetyError):
        check_compatibility(ec, DecoderConfig.from_dict(d), wc)
    d = dc.to_dict()
    d["sampling_hz"] = 5.0
    with pytest.raises(PedsafetyError):
        check_compatibility(ec, DecoderConfig.from_dict(d), wc)


def test_make_train_batch_requires_supervision_keys(ds):
    batch = ds.collate([ds[0], ds[1]])
    for key in ("anchor_uv_px", "target_future_valid", "metadata"):
        broken = {k: v for k, v in batch.items() if k != key}
        with pytest.raises(WindowContractError, match="BATCH_MISSING_KEY"):
            make_train_batch(broken)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device detected; CUDA is not exercised in this environment")
def test_cuda_training_step_matches_cpu_semantics(ds, stats):
    m = make_model(ds, stats).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    r = train_step(m, opt, tbatch(ds, device="cuda"), classification_weight=CW, grad_clip_norm=10.0)
    assert np.isfinite(r.loss) and r.grad_norm_encoder > 0 and r.grad_norm_decoder > 0


def test_cpu_device_batch_tensors_live_on_cpu(ds):
    tb = tbatch(ds)
    assert tb.anchor_uv_px.device.type == "cpu" and tb.encoder_input.node_features.device.type == "cpu" and tb.target_future_uv_px.dtype == torch.float32
