"""P5.5 temporal encoder: causality (prefix invariance, no right-padding), receptive field, masks, agent independence."""

import pytest
import torch

from pedsafety.errors import ConfigError
from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.models import CausalDilatedTCN
from enc_helpers import cfg

D = 16


def tcn(k=3, dil=(1, 2), seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m = CausalDilatedTCN(D, D, k, dil)
    m.eval()
    return m


def rand(M=5, T=6, seed=0, p_absent=0.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, T, D, generator=g)
    pres = torch.rand(M, T, generator=g) >= p_absent
    return x, pres


def test_shapes_for_h6_and_configurable_history_lengths():
    m = tcn()
    for T in (1, 2, 4, 6, 10, 16):
        x, p = rand(3, T)
        y = m(x, p)
        assert y.shape == (3, T, D) and torch.isfinite(y).all()


def test_causal_prefix_invariance_future_inputs_cannot_change_earlier_outputs():
    m = tcn()
    x, p = rand(6, 6, seed=1, p_absent=0.2)
    y = m(x, p)
    for t0 in range(1, 6):
        x2, p2 = x.clone(), p.clone()
        x2[:, t0:] = torch.randn_like(x2[:, t0:]) * 10
        p2[:, t0:] = ~p2[:, t0:]                                  # even presence of later samples may change
        y2 = m(x2, p2)
        assert torch.equal(y2[:, :t0], y[:, :t0]), t0
        assert not torch.equal(y2[:, t0:], y[:, t0:])             # the test itself is sensitive: later outputs do change


def test_no_right_padding_leakage_truncation_equals_prefix():
    m = tcn()
    x, p = rand(4, 9, seed=2)
    y = m(x, p)
    for T in range(1, 9):                                          # different tensor lengths may use different float32 summation order
        assert torch.allclose(m(x[:, :T], p[:, :T]), y[:, :T], atol=1e-6, rtol=0), T
    # sensitivity: a genuine right-side dependency would be many orders of magnitude larger than float rounding
    x2 = x.clone()
    x2[:, 5:] += 3.0
    assert torch.equal(m(x2, p)[:, :5], y[:, :5])


def test_left_only_padding_in_every_block():
    m = tcn(k=3, dil=(1, 2, 4))
    for blk, d in zip(m.blocks, (1, 2, 4)):
        assert blk.left_pad == 2 * d and blk.conv.padding == (0,)


def test_receptive_field_formula_and_empirical_dependency():
    for k, dil in ((3, (1, 2)), (2, (1, 2, 4)), (3, (1,)), (5, (1, 2, 4))):
        m = tcn(k, dil)
        rf = 1 + (k - 1) * sum(dil)
        assert m.receptive_field == rf
        T = rf + 4
        x, p = rand(2, T, seed=3)
        y = m(x, p)
        for j in range(T):
            x2 = x.clone()
            x2[:, j] = x2[:, j] + 3.0
            changed = not torch.equal(m(x2, p)[:, T - 1], y[:, T - 1])
            assert changed == (j >= T - rf), (k, dil, j)           # output at the last step depends on exactly the last RF samples


def test_config_receptive_field_covers_history_h6_and_is_checked():
    c = cfg()
    assert c.receptive_field == 7 >= 6
    c.check_history(6)
    with pytest.raises(ConfigError, match="does not cover"):
        c.check_history(10)
    cfg(require_full_history_coverage=False).check_history(10)     # explicit opt-out
    assert cfg(tcn_dilations=[1, 2, 4]).receptive_field == 15


def test_absent_steps_are_exactly_zero_and_garbage_there_is_ignored():
    m = tcn()
    x, p = rand(6, 6, seed=4, p_absent=0.4)
    y = m(x, p)
    assert (y[~p] == 0).all()
    x2 = x.clone()
    x2[~p] = float("nan")                                          # NaN / garbage at absent samples must not matter
    y2 = m(x2, p)
    assert torch.equal(y2, y) and torch.isfinite(y2).all()
    x3 = x.clone()
    x3[~p] = 1e9
    assert torch.equal(m(x3, p), y)


def test_missing_observation_is_not_filled_by_neighbours():
    m = tcn()
    x, _ = rand(1, 6, seed=5)
    full = torch.ones(1, 6, dtype=torch.bool)
    gap = full.clone()
    gap[0, 2] = False
    y_full, y_gap = m(x, full), m(x, gap)
    assert (y_gap[0, 2] == 0).all() and (y_full[0, 2] != 0).any()
    assert not torch.equal(y_gap[0, 3], y_full[0, 3])              # the absence is visible downstream (mask), not hidden by copying
    assert torch.equal(y_gap[0, :2], y_full[0, :2])                # earlier steps unaffected


def test_presence_bit_distinguishes_absent_from_zero_embedding():
    m = tcn()
    x = torch.zeros(2, 6, D)
    p = torch.ones(2, 6, dtype=torch.bool)
    p[1, 3] = False
    y = m(x, p)
    assert not torch.equal(y[0, 4], y[1, 4])


def test_agents_are_independent_no_cross_agent_normalisation():
    m = tcn()
    x, p = rand(5, 6, seed=6)
    y = m(x, p)
    x2 = x.clone()
    x2[2] = torch.randn(6, D) * 50
    y2 = m(x2, p)
    keep = torch.tensor([0, 1, 3, 4])
    assert torch.equal(y2[keep], y[keep]) and not torch.equal(y2[2], y[2])


def test_finite_outputs_and_gradients():
    m = tcn()
    m.train()
    x, p = rand(8, 6, seed=7, p_absent=0.3)
    x.requires_grad_(True)
    m(x, p).pow(2).sum().backward()
    assert torch.isfinite(x.grad).all() and all(torch.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None)
    assert (x.grad[~p] == 0).all()                                 # absent inputs receive no gradient


def test_shape_and_dtype_validation():
    m = tcn()
    x, p = rand(2, 6)
    with pytest.raises(WindowContractError, match="TEMPORAL_SHAPE"):
        m(x, p[:, :5])
    with pytest.raises(WindowContractError, match="TEMPORAL_SHAPE"):
        m(x, p.float())
    with pytest.raises(WindowContractError, match="TEMPORAL_SHAPE"):
        m(x[0], p[0])


def test_deterministic():
    x, p = rand(4, 6, seed=8)
    assert torch.equal(tcn()(x, p), tcn()(x, p)) and not torch.equal(tcn()(x, p), tcn(seed=1)(x, p))
