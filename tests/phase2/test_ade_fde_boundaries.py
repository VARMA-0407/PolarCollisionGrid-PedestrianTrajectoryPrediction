import math

import torch

from utils.helper import get_final_error, get_mean_error


def test_ade_ignores_agents_that_are_not_observed():
    predicted = torch.tensor(
        [
            [[1.0, 0.0], [100.0, 100.0]],
            [[3.0, 0.0], [100.0, 100.0]],
        ]
    )
    truth = torch.zeros_like(predicted)

    result = get_mean_error(
        predicted,
        truth,
        assumedNodesPresent=[[7, 8], [7, 8]],
        ObsNodesPresent=[7],
        using_cuda=False,
        look_up={7: 0, 8: 1},
    )

    assert math.isclose(result.item(), 2.0)


def test_ade_includes_empty_timesteps_as_zero():
    predicted = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    truth = torch.zeros_like(predicted)

    result = get_mean_error(
        predicted,
        truth,
        assumedNodesPresent=[[7], []],
        ObsNodesPresent=[7],
        using_cuda=False,
        look_up={7: 0},
    )

    assert math.isclose(result.item(), 0.5)


def test_fde_uses_only_the_final_prediction_timestep():
    predicted = torch.tensor(
        [
            [[100.0, 0.0]],
            [[3.0, 4.0]],
        ]
    )
    truth = torch.zeros_like(predicted)

    result = get_final_error(
        predicted,
        truth,
        assumedNodesPresent=[[7], [7]],
        ObsNodesPresent=[7],
        look_up={7: 0},
    )

    assert math.isclose(result.item(), 5.0)


def test_fde_returns_zero_when_no_observed_agent_is_present():
    predicted = torch.ones((1, 1, 2))
    truth = torch.zeros_like(predicted)

    result = get_final_error(
        predicted,
        truth,
        assumedNodesPresent=[[8]],
        ObsNodesPresent=[7],
        look_up={8: 0},
    )

    assert result == 0
