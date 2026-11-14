import math

import numpy as np

from utils.Interaction import (
    Time2Conflict,
    approach_angle_sector,
    approach_dir_sector,
)


def test_time2conflict_returns_negative_one_for_stationary_agents():
    result = Time2Conflict(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([0.0, 0.0]),
        1.0,
    )

    assert result == -1


def test_time2conflict_returns_negative_two_when_already_inside_threshold():
    result = Time2Conflict(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.5, 0.0]),
        np.array([0.0, 0.0]),
        1.0,
    )

    assert result == -2


def test_time2conflict_returns_first_positive_boundary_crossing():
    result = Time2Conflict(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([10.0, 0.0]),
        np.array([0.0, 0.0]),
        1.0,
    )

    assert math.isclose(result, 9.0)


def test_time2conflict_returns_negative_one_for_past_conflict():
    result = Time2Conflict(
        np.array([0.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([10.0, 0.0]),
        np.array([0.0, 0.0]),
        1.0,
    )

    assert result == -1


def test_approach_sectors_use_counterclockwise_boundaries():
    current_position = np.array([0.0, 0.0])
    current_velocity = np.array([1.0, 0.0])

    assert approach_dir_sector(
        current_position, current_velocity, np.array([1.0, 0.0]), 8
    ) == 0
    assert approach_dir_sector(
        current_position, current_velocity, np.array([0.0, 1.0]), 8
    ) == 2
    assert approach_angle_sector(
        current_velocity, np.array([0.0, 1.0]), 8
    ) == 2
