import dataclasses

import numpy as np
import pytest

from pedsafety.chronological import (
    ChronologicalSplitError,
    RawSequence,
    SplitBoundaries,
    split_raw_sequences,
)
from pedsafety.errors import ConfigError
from pedsafety.evaluation import (
    PredictionCandidate,
    evaluate_predictions,
    select_prediction,
)
from pedsafety.normalization import NormalizationError, NormalizationState
from pedsafety.trajectory import SmoothingPolicy, TrajectoryConfig
from pedsafety.windows import WindowConfig
from stage5_fixtures import dataset


def sequence(n=20, scene="hbs"):
    return RawSequence("track-1", tuple(float(i) for i in range(n)), tuple(range(n)), scene)


def test_chronological_order_and_split_boundary_integrity():
    result = split_raw_sequences(
        [sequence()],
        SplitBoundaries(train_end=7.0, validation_end=14.0),
        WindowConfig(obs_len=3, pred_len=2, window_stride=1),
        feature_lookback_frames=0,
    )
    assert result.windows["train"]
    assert result.windows["validation"]
    assert result.windows["test"]
    for partition, windows in result.windows.items():
        for w in windows:
            assert tuple(sorted(w.history_indices + w.target_indices)) == w.history_indices + w.target_indices
            if partition == "train":
                assert w.target_end <= 7.0
            elif partition == "validation":
                assert 7.0 <= w.observation_start and w.target_end <= 14.0
            else:
                assert w.observation_start >= 14.0


def test_crossing_windows_and_dependency_overlap_are_recorded_not_reassigned():
    result = split_raw_sequences(
        [sequence(16)],
        SplitBoundaries(train_end=7.0, validation_end=12.0),
        WindowConfig(obs_len=3, pred_len=2),
        feature_lookback_frames=1,
    )
    reasons = {e.reason for e in result.excluded}
    assert "purged_dependency_overlap" in reasons
    assert result.purge_lookback_frames == 1
    assert result.purge_lookback_seconds_by_sequence["track-1"] == pytest.approx(1.0)


def test_purge_boundary_is_inclusive_for_dependency_start():
    result = split_raw_sequences(
        [sequence(12)],
        SplitBoundaries(train_end=5.0, validation_end=9.0),
        WindowConfig(obs_len=2, pred_len=2),
        feature_lookback_frames=1,
    )
    assert not any(w.partition == "validation" and w.dependency_start < 5.0 for w in result.all_windows)


def test_invalid_timestamps_and_short_sequences_fail_explicitly():
    with pytest.raises(ChronologicalSplitError, match="INVALID_TIMESTAMP"):
        RawSequence("bad", (0.0, float("nan")), (0, 1))
    with pytest.raises(ChronologicalSplitError, match="INSUFFICIENT_SEQUENCE_LENGTH"):
        split_raw_sequences([sequence(3)], SplitBoundaries(2.0, 4.0), WindowConfig(obs_len=2, pred_len=2))
    with pytest.raises(ConfigError):
        SplitBoundaries(4.0, 3.0)


def test_hbs_single_scene_is_explicitly_labeled_temporal_only():
    result = split_raw_sequences([sequence(scene="one-scene")], SplitBoundaries(7.0, 14.0), WindowConfig(3, 2))
    assert "within one scene" in result.single_scene_warning
    assert "scene-independent" in result.single_scene_warning


def test_causal_configuration_rejects_future_dependent_smoothing():
    cfg = TrajectoryConfig(smoothing=SmoothingPolicy("offline_centered_moving_average", window=3))
    with pytest.raises(ConfigError):
        from pedsafety.causal_trajectory import assert_causal_config
        assert_causal_config(cfg)


def test_future_label_mutation_does_not_change_observed_input():
    base = dataset()
    original = base.windows["train"][5]
    recs, prov = __import__("stage5_fixtures").corpus()
    cut = int(original.target_frame_index[0])
    for row in recs:
        if row["track_id"] == original.track_id and row["frame_index"] >= cut:
            row["image_x"] += 999.0
            row["image_y"] -= 999.0
    changed = dataset(recs, prov)
    mutated = {w.window_id: w for p in changed.windows.values() for w in p}[original.window_id]
    for channel in original.history:
        assert np.array_equal(original.history[channel], mutated.history[channel], equal_nan=True)
    assert not np.array_equal(original.target_xy, mutated.target_xy)


def test_train_only_normalization_and_non_train_mutation_invariance():
    ds = dataset()
    state = ds.fit_normalization()
    with pytest.raises(NormalizationError, match="training partition only"):
        NormalizationState.fit(ds.windows["train"] + ds.windows["validation"][:1])
    assert state.fitted_on["partition"] == "train"


def test_evaluator_uses_fixed_non_oracle_selection_and_records_invalid_cases():
    truth = {"w": np.array([[0.0, 0.0], [1.0, 0.0]])}
    candidates = {"w": [PredictionCandidate(np.array([[100.0, 0.0], [100.0, 0.0]]), 0.2),
                         PredictionCandidate(np.array([[0.0, 0.0], [1.0, 0.0]]), 0.8)]}
    result = evaluate_predictions(candidates, truth, selection_rule="model_score", coordinate_space="image")
    assert result.valid_cases[0].ade > 0.0
    with pytest.raises(ConfigError, match="oracle"):
        select_prediction(candidates["w"], "best_of_k")
    invalid = evaluate_predictions({}, truth)
    assert invalid.cases[0].status == "invalid" and invalid.cases[0].reason == "missing_prediction"


def test_test_labels_cannot_change_prediction_selection():
    predictions = {"w": [PredictionCandidate(np.zeros((2, 2)), 0.1),
                         PredictionCandidate(np.ones((2, 2)), 0.2)]}
    a = evaluate_predictions(predictions, {"w": np.zeros((2, 2))}, selection_rule="model_score")
    b = evaluate_predictions(predictions, {"w": np.full((2, 2), 500.0)}, selection_rule="model_score")
    assert np.array_equal(a.cases[0].prediction, b.cases[0].prediction)


def test_evaluator_does_not_label_image_space_as_metres():
    with pytest.raises(ConfigError):
        evaluate_predictions({"w": [PredictionCandidate(np.zeros((2, 2)))]},
                             {"w": np.zeros((2, 2))}, coordinate_space="image", physical_units=True)

