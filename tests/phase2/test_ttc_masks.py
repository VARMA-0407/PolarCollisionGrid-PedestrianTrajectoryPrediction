import numpy as np
import torch

from utils.Interaction import getInteractionGridMask


def test_ttc_mask_marks_first_matching_time_and_angle_sector():
    pedestrians = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    vehicles = torch.tensor([[10.0, 0.0, 0.0, 0.0]])

    mask, ttc_mask = getInteractionGridMask(
        pedestrians,
        vehicles,
        TTC_min=[10.0, 20.0],
        d_min=1.0,
        num_sector=4,
        is_heterogeneous=True,
    )

    assert mask.shape == (1, 1, 8)
    assert ttc_mask.shape == (1, 1, 8)
    assert mask[0, 0, 0] == 1
    assert np.count_nonzero(mask) == 1
    assert np.isclose(ttc_mask[0, 0, 0], 1.0)


def test_ttc_mask_omits_non_conflicting_pairs_and_self_pairs():
    frame = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
        ]
    )

    mask, ttc_mask = getInteractionGridMask(
        frame,
        frame,
        TTC_min=[5.0],
        d_min=1.0,
        num_sector=4,
        is_heterogeneous=False,
    )

    assert mask.shape == (2, 2, 4)
    assert ttc_mask.shape == (2, 2, 4)
    assert mask[0, 1, 0] == 0
    assert mask[0, 0].sum() == 0
    assert mask.sum() == 0
    assert ttc_mask.sum() == 0


def test_ttc_occupancy_mask_collapses_agent_dimension():
    pedestrians = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    vehicles = torch.tensor([[10.0, 0.0, 0.0, 0.0]])

    mask, ttc_mask = getInteractionGridMask(
        pedestrians,
        vehicles,
        TTC_min=[10.0],
        d_min=1.0,
        num_sector=4,
        is_heterogeneous=True,
        is_occupancy=True,
    )

    assert mask.shape == (1, 4)
    assert ttc_mask.shape == (1, 1, 4)
    assert mask[0, 0] == 1
    assert np.count_nonzero(mask) == 1
