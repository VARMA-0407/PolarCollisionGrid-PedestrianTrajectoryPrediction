import copy
import dataclasses
import json

import numpy as np
import pytest

from pedsafety.calibration import VerificationClaim
from pedsafety.errors import ConfigError
from pedsafety.normalization import (
    LINEAR_CHANNELS, NORMALIZATION_VERSION, NormalizationConfig, NormalizationError, NormalizationState,
)
from pedsafety.agent_state import build_agent_states
from pedsafety.projection import TransformConfig, transform_agent_states
from pedsafety.splitting import PARTITIONS, GroupingPolicy, SplitConfig, compute_groups, make_plan
from pedsafety.trajectory import CoordinateMode, TrajectoryConfig
from pedsafety.windows import WindowConfig
from stage3_fixtures import RASTER, make_manifest
from stage4_fixtures import make_track
from stage5_fixtures import GROUP, SPLIT, WIN, corpus, corpus_observations, dataset


def unique_values(windows, ch):
    seen, vals = set(), []
    for w in windows:
        for i in range(w.obs_len):
            k = (w.track_id, int(w.history_frame_index[i]))
            if w.history_available[ch][i] and k not in seen:
                seen.add(k)
                vals.append(float(w.history[ch][i]))
    return np.array(vals)


def err(fn, *a, **k):
    with pytest.raises(NormalizationError) as ei:
        fn(*a, **k)
    return ei.value


# ===================================================== 6-7 train-only fitting
def test_statistics_are_the_training_statistics_of_unique_observations():
    ds = dataset()
    st = ds.fit_normalization()
    key = ds.windows["train"][0].norm_key()
    for ch in LINEAR_CHANNELS:
        v = unique_values(ds.windows["train"], ch)
        s = st.stats[key][ch]
        assert s.count == v.size and s.mean == pytest.approx(v.mean(), rel=1e-12) and s.std == pytest.approx(v.std(), rel=1e-12)
        assert s.status == "ok" and s.scale == s.std
    assert st.stats[key]["xs"].unit == "pixel" and st.stats[key]["vx"].unit == "pixel/s" and st.stats[key]["ax"].unit == "pixel/s^2"


def test_fit_refuses_anything_but_training_windows():
    ds = dataset()
    for bad in (ds.windows["validation"], ds.windows["test"], ds.windows["train"] + ds.windows["validation"][:1]):
        assert err(NormalizationState.fit, bad).code == "LEAKAGE_GUARD"
    assert err(NormalizationState.fit, []).code == "NO_TRAINING_WINDOWS"
    assert not any(hasattr(NormalizationState, n) for n in ("refit", "partial_fit", "update", "fit_transform"))
    st = ds.fit_normalization()
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.stats = {}


def test_changing_validation_and_test_data_does_not_change_training_statistics():
    base = dataset()
    recs, prov = corpus()
    keys, _ = compute_groups(recs, prov, GROUP)
    non_train = {g for g in base.plan.assignments if base.plan.assignments[g] != "train"}
    rng = np.random.default_rng(0)
    for r, k in zip(recs, keys):
        if k in non_train:
            r["image_x"] += float(rng.normal(0, 800))
            r["image_y"] += float(rng.normal(0, 800))
    pert = dataset(recs, prov)
    assert pert.plan.sha256() == base.plan.sha256()                                  # grouping is value-independent
    a, b = base.fit_normalization(), pert.fit_normalization()
    assert a.to_dict() == b.to_dict() and a.sha256() == b.sha256()
    vb, vp = base.windows["validation"][0], {w.window_id: w for w in pert.windows["validation"]}[base.windows["validation"][0].window_id]
    assert not np.allclose(a.transform(vb).history["xs"], b.transform(vp).history["xs"], equal_nan=True)   # the data really changed


def test_changing_the_training_data_does_change_the_statistics():
    base = dataset()
    recs, prov = corpus()
    keys, _ = compute_groups(recs, prov, GROUP)
    for r, k in zip(recs, keys):
        if base.plan.assignments[k] == "train":
            r["image_x"] += 100.0
    assert dataset(recs, prov).fit_normalization().sha256() != base.fit_normalization().sha256()


def test_the_same_fitted_transform_is_applied_to_every_partition():
    ds = dataset()
    st = ds.fit_normalization()
    key = ds.windows["train"][0].norm_key()
    s = st.stats[key]["xs"]
    for p in PARTITIONS:
        w = ds.windows[p][0]
        n = st.transform(w)
        assert np.allclose(n.history["xs"], (w.history["xs"] - s.mean) / s.scale, equal_nan=True)
        assert n.state_sha256 == st.sha256() and n.target_normalized
        assert np.allclose(n.target_xy, (w.target_xy - [s.mean, st.stats[key]["ys"].mean]) / [s.scale, st.stats[key]["ys"].scale])
    tr = np.concatenate([st.transform(w).history["xs"][w.history_available["xs"]] for w in ds.windows["train"][:1]])
    assert np.isfinite(tr).all()


def test_fit_info_records_the_training_split_identity():
    ds = dataset()
    st = ds.fit_normalization()
    f = st.fitted_on
    assert f["partition"] == "train" and f["split_plan_sha256"] == ds.plan.sha256() and f["split_seed"] == 7 and f["grouping_unit"] == "scene"
    assert f["train_groups"] == ds.plan.groups_in("train") and f["n_windows"] == len(ds.windows["train"]) and f["window_config"]["obs_len"] == 4
    assert f["view"] == "causal" and f["views"] == ["CAUSAL"] and len(f["train_window_ids_sha256"]) == 64


# ======================================== explicit handling of odd features
def test_constant_channels_are_flagged_and_not_divided_by_zero():
    obs = []
    for s in range(4):
        obs += make_track("t0", frames=range(14), seq="s%d" % s, scene="sc%d" % s, vx=0.0, vy=0.0, x0=300.0, y0=400.0)
    res = build_agent_states(obs)
    ds = dataset(res.records, res.provenance)
    st = ds.fit_normalization()
    s = st.stats[ds.windows["train"][0].norm_key()]
    assert s["xs"].status == "constant" and s["xs"].scale == 1.0 and s["vx"].status == "constant" and s["speed"].status == "constant"
    n = st.transform(ds.windows["train"][0])
    assert np.nanmax(np.abs(n.history["xs"])) == pytest.approx(0.0, abs=1e-9) and np.isfinite(n.history["xs"]).all()
    assert err(NormalizationState.fit, ds.windows["train"], NormalizationConfig(constant_policy="error")).code == "CONSTANT_CHANNEL"


def test_missing_values_stay_missing_after_normalisation_never_zero():
    ds = dataset()
    st = ds.fit_normalization()
    w = ds.windows["train"][0]
    n = st.transform(w)
    assert np.isnan(n.history["vx"][0]) and not w.history_available["vx"][0]
    assert np.isnan(n.history["ax"][:2]).all()
    assert np.isfinite(n.history["xs"]).all() and w.history["xs"].tolist() != n.history["xs"].tolist()


def test_insufficient_statistics_are_explicit():
    ds = dataset()
    st = NormalizationState.fit(ds.windows["train"], NormalizationConfig(min_count=10 ** 6))
    s = st.stats[ds.windows["train"][0].norm_key()]["xs"]
    assert s.status == "insufficient" and s.mean is None and s.scale is None
    assert err(st.transform, ds.windows["train"][0]).code == "INSUFFICIENT_STATS"


def test_heading_ids_and_timestamps_are_not_normalised_as_features():
    for bad in (("heading",), ("frame_index",), ("timestamp",), ("xs", "heading"), ()):
        with pytest.raises(ConfigError):
            NormalizationConfig(channels=bad)
    ds = dataset()
    w = ds.windows["train"][2]
    n = ds.fit_normalization().transform(w)
    assert "heading" not in n.history and "heading" in n.history_passthrough
    assert np.array_equal(n.history_passthrough["heading"], w.history["heading"], equal_nan=True)
    for name in ("history_frame_index", "history_timestamp", "target_frame_index", "target_timestamp", "window_id", "track_id", "agent_class"):
        assert getattr(n.window, name) is getattr(w, name) or np.array_equal(getattr(n.window, name), getattr(w, name), equal_nan=True) or getattr(n.window, name) == getattr(w, name)


def test_subset_of_channels_and_positions_left_raw_when_not_selected():
    ds = dataset()
    st = NormalizationState.fit(ds.windows["train"], NormalizationConfig(channels=("vx", "vy", "speed")))
    n = st.transform(ds.windows["train"][0])
    assert not n.target_normalized and np.array_equal(n.target_xy, ds.windows["train"][0].target_xy)
    assert "xs" in n.history_passthrough
    with pytest.raises(NormalizationError, match="POSITIONS_NOT_NORMALISED"):
        st.denormalize_xy(ds.windows["train"][0], [[0.0, 0.0]])


def test_denormalisation_inverts_position_normalisation():
    ds = dataset()
    st = ds.fit_normalization()
    w = ds.windows["validation"][0]
    assert np.allclose(st.denormalize_xy(w, st.transform(w).target_xy), w.target_xy, rtol=1e-12)


def test_config_validation():
    for kw in (dict(min_count=1), dict(constant_policy="zero"), dict(constant_rel_tol=0), dict(channels=("xs", "xs"))):
        with pytest.raises(ConfigError):
            NormalizationConfig(**kw)


# =============================================== unseen coordinate spaces / datasets
def test_windows_from_an_unseen_dataset_are_not_normalised_by_fallback():
    obs = corpus_observations(n_scenes=3, dataset="AAA") + corpus_observations(n_scenes=3, dataset="BBB")
    res = build_agent_states(obs)
    _, infos = compute_groups(res.records, res.provenance, GROUP)
    groups = sorted(infos)
    assign = [(g, "train") for g in groups if ":aaa:" in g and not g.endswith("sc2")] + [(g, "validation") for g in groups if g == "scene:aaa:sc2"] \
        + [(g, "test") for g in groups if ":bbb:" in g]
    ds = dataset(res.records, res.provenance, split=SplitConfig(seed=1, explicit_assignments=assign))
    st = ds.fit_normalization()
    assert all(k.startswith("AAA|") for k in st.stats)
    assert st.transform(ds.windows["validation"][0]).state_sha256
    e = err(st.transform, ds.windows["test"][0])
    assert e.code == "UNSEEN_KEY" and "BBB" in str(e)


def test_image_and_metric_spaces_get_separate_statistics_and_keep_units():
    recs, prov = corpus()
    s3 = transform_agent_states(recs, prov, make_manifest(), RASTER, TransformConfig(promote_eligible_to_canonical=True))
    assert s3.counters.get("projected", 0) == len(recs) and s3.counters["promoted"] == len(recs)
    metric = dataset(s3.records, prov, s3.projections, trajectory=TrajectoryConfig(coordinate_mode=CoordinateMode.METRIC_MANIFEST_ELIGIBLE))
    image = dataset(s3.records, prov, s3.projections)
    wm, wi = metric.windows["train"][0], image.windows["train"][0]
    assert (wm.position_unit, wm.velocity_unit, wm.physical, wm.calibration_id) == ("m", "m/s", True, "synthetic-cal")
    assert (wi.position_unit, wi.physical) == ("pixel", False) and wm.norm_key() != wi.norm_key() and "synthetic_plane" in wm.norm_key()
    sm = metric.fit_normalization().stats[wm.norm_key()]["xs"]
    assert sm.unit == "m" and metric.fit_normalization().fitted_on["trajectory_config"]["coordinate_mode"] == "metric_manifest_eligible"
    # one fitted state can never normalise a window of the other space
    st_img = image.fit_normalization()
    assert err(st_img.transform, wm).code == "UNSEEN_KEY"


# ================================================================= serialisation
def test_state_round_trips_exactly_through_json(tmp_path):
    ds = dataset()
    st = ds.fit_normalization()
    path = st.save(tmp_path / "norm.json")
    loaded = NormalizationState.load(path)
    assert loaded.to_dict() == st.to_dict() and loaded.sha256() == st.sha256() and loaded.config == st.config
    w = ds.windows["test"][1]
    a, b = st.transform(w), loaded.transform(w)
    for ch in a.history:
        assert np.array_equal(a.history[ch], b.history[ch], equal_nan=True)
    assert np.array_equal(a.target_xy, b.target_xy)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == NORMALIZATION_VERSION and raw["fitted_on"]["split_plan_sha256"] == ds.plan.sha256()


def test_loading_rejects_unknown_versions(tmp_path):
    st = dataset().fit_normalization()
    d = st.to_dict()
    d["version"] = "other"
    assert err(NormalizationState.from_dict, d).code == "BAD_VERSION"


def test_normalised_windows_do_not_modify_the_source_windows():
    ds = dataset()
    w = ds.windows["train"][0]
    before = {k: v.copy() for k, v in w.history.items()}
    ds.fit_normalization().transform(w)
    assert all(np.array_equal(before[k], w.history[k], equal_nan=True) for k in before)
