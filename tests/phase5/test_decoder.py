"""P5.6 direct decoder contract and evaluator compatibility tests."""

import numpy as np
import pytest
import torch

from pedsafety.errors import ConfigError
from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.models.decoder import DecoderConfig, MultimodalTrajectoryDecoder, evaluator_arrays
from pedsafety.phase5.models.losses import masked_wta_loss


def cfg(**updates):
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "configs" / "phase5" / "decoder_default.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    return DecoderConfig.from_dict(data)


def test_shapes_probabilities_and_anchor_reconstruction():
    torch.manual_seed(2)
    model = MultimodalTrajectoryDecoder(cfg())
    summary = torch.randn(3, 64)
    anchor = torch.tensor([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]])
    out = model(summary, anchor, torch.tensor([True, True, True]))
    assert out.trajectory_offsets_px.shape == (3, 5, 15, 2)
    assert out.trajectories_uv_px.shape == (3, 5, 15, 2)
    assert out.mode_logits.shape == (3, 5)
    assert out.mode_probabilities.shape == (3, 5)
    assert out.prediction_valid.shape == (3, 5, 15)
    assert torch.allclose(out.trajectories_uv_px, out.trajectory_offsets_px + anchor[:, None, None])
    assert torch.isfinite(out.mode_probabilities).all()
    assert (out.mode_probabilities >= 0).all()
    assert torch.allclose(out.mode_probabilities.sum(dim=1), torch.ones(3))


def test_invalid_target_has_explicit_invalid_prediction_mask():
    model = MultimodalTrajectoryDecoder(cfg())
    out = model(torch.zeros(2, 64), torch.zeros(2, 2), torch.tensor([True, False]))
    assert out.prediction_valid[0].all() and not out.prediction_valid[1].any()
    assert torch.isfinite(out.trajectories_uv_px).all()


def test_input_validation_and_no_future_interface():
    model = MultimodalTrajectoryDecoder(cfg())
    with pytest.raises(WindowContractError, match="DECODER_INPUT_SHAPE"):
        model(torch.zeros(2, 63), torch.zeros(2, 2), torch.ones(2, dtype=torch.bool))
    with pytest.raises(WindowContractError, match="DECODER_INPUT_MASK"):
        model(torch.zeros(2, 64), torch.zeros(2, 2), torch.ones(2))
    with pytest.raises(WindowContractError, match="NONFINITE_VALID_VALUE"):
        model(torch.full((2, 64), float("nan")), torch.zeros(2, 2), torch.ones(2, dtype=torch.bool))


def test_determinism_gradients_and_evaluator_arrays():
    model = MultimodalTrajectoryDecoder(cfg())
    model.eval()
    summary = torch.randn(2, 64)
    anchor = torch.randn(2, 2)
    valid = torch.ones(2, dtype=torch.bool)
    a, b = model(summary, anchor, valid), model(summary, anchor, valid)
    assert torch.equal(a.trajectories_uv_px, b.trajectories_uv_px)
    model.train()
    trajectory_output = model(summary, anchor, valid)
    trajectory_output.trajectories_uv_px.sum().backward()
    trajectory_parameters = (
        model.input_projection, model.norm, model.trajectory_head,
    )
    for module in trajectory_parameters:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters())
    assert all(p.grad is None for p in model.mode_head.parameters())

    model.zero_grad()
    classification_output = model(summary, anchor, valid)
    torch.nn.functional.cross_entropy(
        classification_output.mode_logits, torch.zeros(2, dtype=torch.long)
    ).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.mode_head.parameters())

    model.zero_grad()
    combined_output = model(summary, anchor, valid)
    target = torch.zeros(2, 15, 2)
    combined_loss = masked_wta_loss(
        combined_output.trajectories_uv_px,
        combined_output.mode_logits,
        target,
        torch.ones(2, 15, dtype=torch.bool),
        valid,
    )
    combined_loss.total_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.parameters())
    arrays = evaluator_arrays(a)
    assert arrays["positions_uv_px"].shape == (2, 5, 15, 2)
    assert arrays["valid"].shape == (2, 5, 15)
    assert arrays["mode_probabilities"].shape == (2, 5)
    assert np.isfinite(arrays["positions_uv_px"]).all()


def test_config_rejects_scaling_and_architecture_changes():
    with pytest.raises(ConfigError):
        cfg(output_representation="standardized_offsets")
    with pytest.raises(ConfigError):
        cfg(autoregressive=True)
    with pytest.raises(ConfigError):
        cfg(coordinate_unit="pixel")
