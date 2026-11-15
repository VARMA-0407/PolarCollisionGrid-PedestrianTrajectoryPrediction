"""P5.3 window configuration: H = 6, F = 15, 10 Hz from 30 fps, dependency interval, strict validation."""

import json

import numpy as np
import pytest

from pedsafety.errors import ConfigError
from pedsafety.phase5.errors import TargetSourceUnavailable
from pedsafety.phase5.trajectory_targets import check_target_source
from pedsafety.phase5.window_config import WindowConfig
from phase5_helpers import CONFIG, wcfg


def test_default_config_values_are_the_approved_ones():
    c = WindowConfig.load(CONFIG)
    assert (c.source_fps, c.target_sampling_hz, c.sampling_stride_frames) == (30.0, 10.0, 3)
    assert (c.history_length, c.prediction_length) == (6, 15)
    assert (c.history_span_s, c.future_horizon_s) == (0.5, 1.5)
    assert c.coordinate_system == "captured_1920x1080" and c.coordinate_unit == "pixel"
    assert c.target_source_mode == "PILOT_SMOKE"
    assert c.target_node_types == ("pedestrian",)
    assert c.default_context_node_types == ("pedestrian", "vehicle")
    assert c.optional_context_node_types == ("bicyclist",) and c.semantically_uncertain_node_types == ("bicyclist",)
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key in ("config_version", "missing_observation_policy", "track_continuity_policy"):
        assert raw[key]


def test_six_observation_history_indexing():
    c = wcfg()
    assert c.history_offsets.tolist() == [-15, -12, -9, -6, -3, 0]
    assert c.history_offsets.shape == (6,)


def test_fifteen_observation_future_indexing():
    c = wcfg()
    assert c.future_offsets.tolist() == [3 * (m + 1) for m in range(15)]
    assert c.future_offsets[-1] == 45


def test_exact_timestamp_offsets():
    d = wcfg().derived()
    assert d["history_offsets_s"] == [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
    assert d["future_offsets_s"] == [round(0.1 * (m + 1), 9) for m in range(15)]
    assert d["sampling_interval_s"] == pytest.approx(0.1, abs=1e-12)


def test_30fps_to_10hz_stride_conversion_enforced():
    assert wcfg().sampling_interval_s == pytest.approx(0.1)
    with pytest.raises(ConfigError, match="target_sampling_hz"):
        wcfg(sampling_stride_frames=2)
    with pytest.raises(ConfigError, match="target_sampling_hz"):
        wcfg(source_fps=25.0)
    with pytest.raises(ConfigError, match="history_span_s"):
        wcfg(history_length=5)              # stale H = 5 with span 0.5 s is inconsistent
    with pytest.raises(ConfigError, match="future_horizon_s"):
        wcfg(prediction_length=10)


def test_dependency_interval_and_purge_for_h6_f15():
    c = wcfg()
    assert c.dependency_offsets == (-16, 45)
    assert c.dependency_length_frames == 62
    assert c.minimum_purge_gap_frames == 62


def test_config_rejects_unknown_missing_and_bad_semantics():
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ConfigError, match="unknown"):
        WindowConfig.from_dict(dict(raw, surprise=1))
    del raw["history_length"]
    with pytest.raises(ConfigError, match="missing"):
        WindowConfig.from_dict(raw)
    with pytest.raises(ConfigError, match="pedestrian prediction targets only"):
        wcfg(target_node_types=["pedestrian", "vehicle"])
    with pytest.raises(ConfigError, match="semantically uncertain"):
        wcfg(default_context_node_types=["pedestrian", "vehicle", "bicyclist"], optional_context_node_types=[])
    with pytest.raises(ConfigError, match="coordinate_unit"):
        wcfg(coordinate_unit="metre")
    with pytest.raises(ConfigError, match="target_source_mode"):
        wcfg(target_source_mode="ORACLE")


def test_annotated_target_mode_is_refused_not_substituted():
    c = wcfg(target_source_mode="ANNOTATED_TARGET")          # a valid documented mode ...
    with pytest.raises(TargetSourceUnavailable, match="never substituted"):
        check_target_source(c.target_source_mode)             # ... but not an active data path


def test_config_hash_is_stable_and_sensitive():
    assert wcfg().sha256() == wcfg().sha256()
    assert wcfg().sha256() != wcfg(anchor_phase_frames=1).sha256()
    assert np.array_equal(wcfg().history_offsets, WindowConfig.load(CONFIG).history_offsets)
