"""Read-only HBS smoke test and visualisation labelling/rendering tests."""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from pedsafety import pipeline, viz
from pedsafety.cli import main
from pedsafety.causal_trajectory import process_causal
from pedsafety.pipeline_config import load_config
from pedsafety.trajectory import CoordinateMode, TrajectoryConfig
from stage4_fixtures import make_track, stage2, stage3
from stage6_fixtures import calibrated_setup, load_cfg, save_cfg, synthetic

REPO = Path(__file__).resolve().parents[2]
HBS = REPO / "Data" / "HBS" / "hbs.csv"
HBS_CFG = REPO / "configs" / "phase2" / "pedsafety_hbs_smoke.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# --------------------------------------------------------------------- HBS
@pytest.mark.skipif(not HBS.is_file(), reason="local HBS file not present")
def test_hbs_smoke_is_read_only_and_makes_no_metric_claim(tmp_path):
    before = sha(HBS)
    rc = main(["hbs-smoke", str(HBS_CFG), "--out", str(tmp_path), "--max-rows", "3000"])
    assert rc == 0 and sha(HBS) == before
    r = jl(tmp_path / "hbs_smoke_report.json")
    assert r["status"] == "pass" and all(r["checks"].values())
    assert r["source"]["sha256"] == before and r["rows_limit"] == 3000
    assert r["split_attempt"]["refused"] and r["split_attempt"]["code"] == "INSUFFICIENT_GROUPS"
    assert r["counts"]["scenes"] == ["hbs:0"]
    ident = next(iter(r["features"]["identities"].values()))
    assert ident["physical"] is False and ident["position_unit"] == "unverified_planar_units"
    assert "metric coordinates" in r["not_claimed"] and "a train/validation/test partition" in r["not_claimed"]
    assert (tmp_path / "hbs_sample.csv").read_text(encoding="utf-8").count("\n") == 3001      # a copy: header + 3000 rows
    assert not list(tmp_path.glob("windows_*"))


@pytest.mark.skipif(not HBS.is_file(), reason="local HBS file not present")
def test_hbs_build_is_refused_not_faked(tmp_path):
    """Even with a grouping block present, one scene cannot be split. The tool must refuse."""
    d = load_cfg(HBS_CFG)
    d["input"]["csv"] = str(HBS)
    d["grouping"] = {"unit": "scene", "independence_basis": "attempt: HBS has one scene"}
    d["split"] = {"seed": 1}
    d["windows"] = {"obs_len": 4, "pred_len": 4}
    sample = tmp_path / "s.csv"
    sample.write_text("".join(HBS.open(encoding="utf-8").readlines()[:2001]), encoding="utf-8")
    d["input"]["csv"] = str(sample)
    cfg = save_cfg(tmp_path / "c.json", d)
    assert main(["build", str(cfg), "--out", str(tmp_path / "o")]) == 4
    assert jl(tmp_path / "o" / "error.json")["code"] == "INSUFFICIENT_GROUPS"
    assert not list((tmp_path / "o").glob("windows_*"))


def test_smoke_check_fails_when_positions_are_declared_metric(tmp_path):
    """Negative control: the smoke checks must be able to fail."""
    csv_path, _ = synthetic(tmp_path, n_scenes=1)
    cfg_dict = {"config_version": 1,
                "input": {"csv": str(csv_path), "source_dataset": "X", "columns": {"track_id": "track", "frame_index": "frame", "timestamp": "t_s",
                                                                                   "label": "cls", "x": "foot_x", "y": "foot_y",
                                                                                   "scene_id": "scene", "sequence_id": "sequence"},
                          "position_semantics": "image_pixel", "position_unit": "pixel", "position_coordinate_system": "img"}}
    cfg = load_config(save_cfg(tmp_path / "c.json", cfg_dict))
    rep = pipeline.run_hbs_smoke(cfg, tmp_path / "o")
    assert rep["status"] == "fail" and rep["checks"]["positions_unverified_planar_only"] is False
    assert main(["hbs-smoke", str(tmp_path / "c.json"), "--out", str(tmp_path / "o2")]) == pipeline.EXIT_CHECK_FAILED


def test_smoke_rejects_bad_max_rows(tmp_path):
    csv_path, cfg = synthetic(tmp_path, n_scenes=1)
    with pytest.raises(Exception, match="max_rows"):
        pipeline.run_hbs_smoke(load_config(cfg), tmp_path / "o", max_rows=0)


# --------------------------------------------------------------------- viz
def traj(mode, **kw):
    recs, prov = stage2(make_track("a", frames=range(6)))
    if mode in ("projective_unverified", "metric_manifest_eligible"):
        res = stage3(recs, prov, promote=(mode == "metric_manifest_eligible"))
        out = process_causal(res.records, prov, res.projections, TrajectoryConfig(coordinate_mode=mode))
    else:
        out = process_causal(recs, prov, None, TrajectoryConfig(coordinate_mode=mode))
    return out.trajectories


def test_axis_labels_follow_the_data_unit_status():
    img = viz.axis_spec(traj("image_pixel")[0])
    assert img["xlabel"] == "x [pixel]" and img["invert_y"] is True and "not metric" in img["banner"]
    proj = viz.axis_spec(traj("projective_unverified")[0])
    assert proj["invert_y"] is False and "not metric" in proj["banner"] and "m]" not in proj["xlabel"]
    met = viz.axis_spec(traj("metric_manifest_eligible")[0])
    assert met["physical"] is True and "x [m]" == met["xlabel"] and "not independently verified" in met["banner"]
    assert "not metric" not in met["banner"] or "metric by software" in met["banner"]


def test_planar_label_never_claims_a_unit():
    recs, prov = stage2(make_track("a", frames=range(4)))
    # the planar mode needs unverified_planar declarations; the label rule is exercised directly through a stub
    class T:
        coordinate_mode, position_unit, velocity_unit, physical = "unverified_planar", "unverified_planar_units", "unverified_planar_units/s", False
    s = viz.axis_spec(T)
    assert "UNVERIFIED" in s["banner"] and s["xlabel"] == "x [unverified_planar_units]" and s["invert_y"] is False


def test_identities_are_never_drawn_on_one_axis():
    mixed = traj("image_pixel") + traj("projective_unverified")
    assert len(viz._groups(mixed)) == 2
    assert len(viz._groups(traj("image_pixel") + traj("image_pixel"))) == 1


def test_figures_render_png_files(tmp_path):
    t = traj("image_pixel") + traj("projective_unverified")
    ok, why = viz.plot_trajectories(tmp_path / "t.png", trajectories=t)
    ok2, _ = viz.plot_kinematics(tmp_path / "k.png", trajectories=t)
    assert ok and ok2 and why == ""
    for n in ("t.png", "k.png"):
        assert (tmp_path / n).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and (tmp_path / n).stat().st_size > 2000


def test_figures_refuse_to_draw_nothing(tmp_path):
    ok, why = viz.plot_trajectories(tmp_path / "t.png", trajectories=[])
    assert ok is False and "no trajectories" in why and not (tmp_path / "t.png").exists()
    assert viz.plot_split(tmp_path / "s.png", dataset=None)[0] is False
    assert viz.plot_windows(tmp_path / "w.png", dataset=None)[0] is False


def test_plot_command_draws_what_is_possible_and_reports_the_rest(tmp_path):
    _, cfg = synthetic(tmp_path)
    assert main(["plot", str(cfg), "--out", str(tmp_path / "p")]) == 0
    m = jl(tmp_path / "p" / "plot_manifest.json")
    assert set(m["written"]) == {"trajectories", "kinematics", "split", "windows"}
    assert "projection" in m["skipped"] and "no calibration" in m["skipped"]["projection"]
    for f in m["written"].values():
        assert (tmp_path / "p" / f).stat().st_size > 2000


def test_plot_command_skips_split_figures_without_grouping(tmp_path):
    _, cfg = synthetic(tmp_path)
    d = load_cfg(cfg)
    d.pop("grouping")
    save_cfg(cfg, d)
    assert main(["plot", str(cfg), "--out", str(tmp_path / "p")]) == 0
    m = jl(tmp_path / "p" / "plot_manifest.json")
    assert set(m["written"]) == {"trajectories", "kinematics"} and set(m["skipped"]) >= {"split", "windows", "projection"}


def test_projection_figure_with_calibration(tmp_path):
    _, cfg = calibrated_setup(tmp_path, mode="projective_unverified", promote=False)
    assert main(["plot", str(cfg), "--which", "projection,trajectories", "--out", str(tmp_path / "p")]) == 0
    assert set(jl(tmp_path / "p" / "plot_manifest.json")["written"]) == {"projection", "trajectories"}


# ------------------------------------------- negative controls for the smoke checks
def planar_multi_scene_cfg(tmp_path, n_scenes):
    csv_path, _ = synthetic(tmp_path, n_scenes=n_scenes)
    cols = {"track_id": "track", "frame_index": "frame", "timestamp": "t_s", "label": "cls", "x": "foot_x", "y": "foot_y",
            "scene_id": "scene", "sequence_id": "sequence"}
    d = {"config_version": 1, "input": {"csv": str(csv_path), "source_dataset": "X", "columns": cols, "position_semantics": "unverified_planar",
                                         "position_unit": None, "position_coordinate_system": "x_local"}}
    return load_config(save_cfg(tmp_path / "pc.json", d))


def test_smoke_split_check_fails_when_data_has_enough_scenes(tmp_path):
    rep = pipeline.run_hbs_smoke(planar_multi_scene_cfg(tmp_path, 6), tmp_path / "o")
    assert rep["split_attempt"]["refused"] is False
    assert rep["checks"]["split_refused_insufficient_groups"] is False and rep["checks"]["single_scene"] is False
    assert rep["status"] == "fail"


def test_smoke_physical_and_unit_checks_fail_on_a_metric_looking_feature(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    stub = SimpleNamespace(physical=True, velocity_unit="m/s", position_unit="m", arrays={"dt": np.array([0.5])})
    monkeypatch.setattr(pipeline, "describe_features", lambda ing, t: ({"n_tracks": 1, "identities": {}}, SimpleNamespace(trajectories=[stub])))
    rep = pipeline.run_hbs_smoke(planar_multi_scene_cfg(tmp_path, 1), tmp_path / "o")
    assert rep["checks"]["no_physical_features"] is False and rep["checks"]["no_metre_unit_anywhere"] is False
    assert rep["checks"]["velocity_units_unverified"] is False and rep["status"] == "fail"


def test_image_panels_invert_y_and_other_panels_do_not(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(viz, "_save", lambda fig, path: seen.append([ax.yaxis_inverted() for ax in fig.axes]))
    viz.plot_trajectories(tmp_path / "a.png", trajectories=traj("image_pixel"))
    viz.plot_trajectories(tmp_path / "b.png", trajectories=traj("projective_unverified"))
    assert seen == [[True], [False]]
