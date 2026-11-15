"""P5.6 masked WTA and classification-loss tests."""

import pytest
import torch

from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.models.losses import masked_wta_loss


def make_case(batch=2, modes=5, future=3):
    gt = torch.zeros(batch, future, 2)
    gt[0, :, 0] = torch.arange(future, dtype=torch.float32)
    gt[1, :, 1] = torch.arange(future, dtype=torch.float32)
    pred = torch.zeros(batch, modes, future, 2)
    pred[0, min(1, modes - 1)] = gt[0]
    pred[1, min(3, modes - 1)] = gt[1]
    valid = torch.ones(batch, future, dtype=torch.bool)
    agents = torch.ones(batch, dtype=torch.bool)
    logits = torch.zeros(batch, modes)
    return pred, logits, gt, valid, agents


def test_masked_ade_selects_winner_and_returns_diagnostics():
    pred, logits, gt, valid, agents = make_case()
    out = masked_wta_loss(pred, logits, gt, valid, agents)
    assert out.winner_index.tolist() == [1, 3]
    assert out.valid_window_count == 2
    assert torch.allclose(out.mode_ade_px[0, 1], torch.tensor(0.0))
    assert torch.allclose(out.mode_ade_px[1, 3], torch.tensor(0.0))


def test_masked_values_and_nan_targets_do_not_contaminate_loss():
    pred, logits, gt, valid, agents = make_case()
    valid[:, -1] = False
    gt[:, -1] = float("nan")
    out = masked_wta_loss(pred, logits, gt, valid, agents)
    assert torch.isfinite(out.total_loss)
    pred[:, :, -1] = float("nan")
    out2 = masked_wta_loss(pred, logits, gt, valid, agents)
    assert torch.isfinite(out2.total_loss)


def test_valid_nonfinite_values_and_all_invalid_batch_are_rejected():
    pred, logits, gt, valid, agents = make_case()
    gt[0, 0, 0] = float("nan")
    with pytest.raises(WindowContractError, match="NONFINITE_VALID_VALUE"):
        masked_wta_loss(pred, logits, gt, valid, agents)
    valid[:] = False
    with pytest.raises(WindowContractError, match="NO_VALID_TARGETS"):
        masked_wta_loss(pred, logits, gt, valid, agents)


def test_invalid_windows_are_excluded_and_classification_uses_logits():
    pred, logits, gt, valid, agents = make_case()
    agents[1] = False
    logits[0, 1] = 4.0
    out = masked_wta_loss(pred, logits, gt, valid, agents, classification_weight=0.5)
    assert out.valid_window_count == 1 and not out.valid_window_mask[1]
    assert torch.isfinite(out.classification_loss)
    assert torch.allclose(out.total_loss, out.regression_loss + 0.5 * out.classification_loss)


def test_k1_has_active_regression_and_zero_classification_loss():
    pred, logits, gt, valid, agents = make_case(modes=1)
    pred.requires_grad_()
    logits.requires_grad_()
    with torch.no_grad():
        pred[:, 0, :, 0] += 1.0
    out = masked_wta_loss(pred, logits, gt, valid, agents)
    assert torch.equal(out.winner_index, torch.zeros(2, dtype=torch.long))
    assert out.classification_loss.item() == 0.0
    assert out.regression_loss.item() >= 0.0
    assert out.total_loss.item() > 0.0
    out.total_loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert logits.grad is None


def test_smooth_l1_gradients_and_classification_weight_effect():
    pred, logits, gt, valid, agents = make_case()
    pred.requires_grad_()
    logits.requires_grad_()
    out = masked_wta_loss(pred, logits, gt, valid, agents, classification_weight=0.25)
    out.total_loss.backward()
    assert torch.isfinite(pred.grad).all() and torch.isfinite(logits.grad).all()
    assert out.classification_loss.item() > 0


def test_deterministic_two_mode_toy_selects_distinct_paths_without_training():
    gt = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])
    pred = torch.tensor([
        [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 0.0]]],
        [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 0.0]]],
    ])
    logits = torch.zeros(2, 2)
    valid = torch.ones(2, 2, dtype=torch.bool)
    out = masked_wta_loss(pred, logits, gt, valid, torch.ones(2, dtype=torch.bool))
    assert out.winner_index.tolist() == [1, 0]
