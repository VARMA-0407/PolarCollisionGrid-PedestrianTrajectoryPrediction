import json

import pytest

from pedsafety.errors import ConfigError
from pedsafety.pipeline_config import load_config
from pedsafety.splitting import SplitError
from stage6_fixtures import load_cfg, save_cfg, synthetic


@pytest.fixture()
def syn(tmp_path):
    return synthetic(tmp_path)


def test_valid_config_loads_and_resolves_relative_csv(syn):
    csv_path, cfg_path = syn
    cfg = load_config(cfg_path)
    assert cfg.input_csv == str(csv_path.resolve())
    assert cfg.grouping.unit == "scene" and cfg.windows.obs_len == 5 and cfg.view == "causal"
    assert len(cfg.sha256()) == 64


def test_config_hash_ignores_key_order_but_tracks_content(syn, tmp_path):
    _, p = syn
    d = load_cfg(p)
    a = load_config(p).sha256()
    q = save_cfg(tmp_path / "c2.json", dict(reversed(list(d.items()))))
    assert load_config(q).sha256() == a
    d["split"]["seed"] = 8
    assert load_config(save_cfg(tmp_path / "c3.json", d)).sha256() != a


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.update(bogus=1), "unknown key"),
    (lambda d: d["input"].update(bogus=1), "unknown key"),
    (lambda d: d["windows"].update(obs_len=0), "obs_len"),
    (lambda d: d.update(config_version=2), "config_version"),
    (lambda d: d.update(view="both"), "view"),
    (lambda d: d["input"].pop("columns"), "missing required"),
    (lambda d: d["input"].update(position_unit="m"), "position"),
    (lambda d: d["split"].update(seed=True), "seed"),
    (lambda d: d["trajectory"].update(coordinate_mode="metric_guess"), "coordinate_mode"),
    (lambda d: d["agent_state"].update(velocity_contract={"space": "world"}), "world"),
])
def test_invalid_config_is_rejected_with_a_readable_message(syn, tmp_path, mutate, needle):
    _, p = syn
    d = load_cfg(p)
    mutate(d)
    with pytest.raises(ConfigError) as ei:
        load_config(save_cfg(tmp_path / "bad.json", d))
    assert needle in str(ei.value)


def test_missing_and_malformed_config_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.json")
    (tmp_path / "x.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(tmp_path / "x.json")


def test_grouping_has_no_default_and_build_sections_are_required(syn, tmp_path):
    _, p = syn
    d = load_cfg(p)
    d.pop("grouping")
    cfg = load_config(save_cfg(tmp_path / "ng.json", d))
    assert cfg.grouping is None
    with pytest.raises(ConfigError, match="no default grouping"):
        cfg.require_build_sections()


def test_recording_grouping_needs_an_explicit_map_and_rejects_duplicates(syn, tmp_path):
    _, p = syn
    d = load_cfg(p)
    d["grouping"] = {"unit": "recording", "independence_basis": "stated"}
    with pytest.raises(ConfigError, match="recording_map"):
        load_config(save_cfg(tmp_path / "r1.json", d))
    d["grouping"]["recording_map"] = [["SYNTHETIC", "s0", "recA"], ["SYNTHETIC", "s0", "recB"]]
    with pytest.raises(SplitError) as ei:
        load_config(save_cfg(tmp_path / "r2.json", d))
    assert ei.value.code == "CONFLICTING_RECORDING_MAP"
    d["grouping"]["recording_map"] = [["SYNTHETIC", "s0"]]
    with pytest.raises(ConfigError, match="triples"):
        load_config(save_cfg(tmp_path / "r3.json", d))


def test_missing_calibration_manifest_file_is_a_config_error(syn, tmp_path):
    _, p = syn
    d = load_cfg(p)
    d["calibration"] = {"manifest": "absent.json", "observed_raster": {}}
    with pytest.raises(ConfigError, match="calibration manifest"):
        load_config(save_cfg(tmp_path / "cm.json", d))


def test_input_override_replaces_csv_without_touching_config_hash_semantics(syn, tmp_path):
    _, p = syn
    other = tmp_path / "other.csv"
    other.write_text("x\n", encoding="utf-8")
    cfg = load_config(p, input_csv_override=str(other))
    assert cfg.input_csv == str(other)
    assert json.loads(json.dumps(cfg.resolved))["input"]["csv"] == str(other)
