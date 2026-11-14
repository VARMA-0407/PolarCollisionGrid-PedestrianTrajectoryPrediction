import copy
import dataclasses
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from pedsafety.agent_state import build_agent_states
from pedsafety.calibration import VerificationClaim
from pedsafety.causal_trajectory import process_causal
from pedsafety.geometry import apply_homography
from pedsafety.io.csv_adapter import CsvAdapterConfig, CsvAnnotationAdapter
from pedsafety.io.synthetic import generate_scene, synthetic_csv_config, write_scene_csv
from pedsafety.offline_trajectory import process_offline
from pedsafety.schema import collect_violations
from pedsafety.trajectory import CoordinateMode, TrajectoryConfig
from stage3_fixtures import H_TRUE, RASTER, make_manifest, raw
from stage4_fixtures import col, make_track, same, snapshot, stage2, stage3

PROJ = TrajectoryConfig(coordinate_mode=CoordinateMode.PROJECTIVE_UNVERIFIED)
METRIC = TrajectoryConfig(coordinate_mode=CoordinateMode.METRIC_MANIFEST_ELIGIBLE)
DECL = TrajectoryConfig(coordinate_mode=CoordinateMode.DECLARED_WORLD_UNVERIFIED)
PLANAR = TrajectoryConfig(coordinate_mode=CoordinateMode.UNVERIFIED_PLANAR)


def pipeline(manifest=None, promote=False, **kw):
    recs, prov = stage2(make_track(frames=range(10), x0=300.0, y0=400.0, vx=20.0, vy=10.0, **kw))
    s3 = stage3(recs, prov, manifest, promote)
    return s3.records, prov, s3.projections


def expected_plane():
    pts = [(300.0 + 20.0 * f / 10, 400.0 + 10.0 * f / 10) for f in range(10)]
    return apply_homography(H_TRUE, pts).xy


# --------------------------------------------------- 16 projective never metric
@pytest.mark.parametrize("manifest,promote", [
    (make_manifest(plane_unit=None), True),            # no declared scale
    (make_manifest(verification=VerificationClaim()), True),   # unit declared, evidence incomplete -> ineligible
    (make_manifest(), False),                          # eligible, but promotion not enabled
])
def test_unverified_projective_coordinates_never_become_metric(manifest, promote):
    recs, prov, projs = pipeline(manifest, promote)
    t = process_causal(recs, prov, projs, PROJ).trajectories[0]
    exp = expected_plane()
    assert (t.position_unit, t.velocity_unit, t.acceleration_unit) == ("projective_plane_units", "projective_plane_units/s", "projective_plane_units/s^2")
    assert t.physical is False and t.physical_basis is None and t.unit_status == "projective_unverified_scale"
    assert "m" not in {t.position_unit, t.velocity_unit} and t.calibration_id == "synthetic-cal" and len(t.manifest_sha256) == 64
    dt = np.diff(col(t, "timestamp"))
    assert col(t, "vx")[1:] == pytest.approx(np.diff(exp[:, 0]) / dt, rel=1e-9) and col(t, "x") == pytest.approx(exp[:, 0])
    # and the metric mode refuses the very same inputs
    m = process_causal(recs, prov, projs, METRIC).trajectories[0]
    assert np.isnan(col(m, "vx")).all() and set(col(m, "position_state")) == {"world_position_valid_is_false"}
    assert m.physical is False and m.position_unit == "unavailable"


def test_projective_mode_needs_a_projection_that_succeeded():
    recs, prov, projs = pipeline(make_manifest())
    t = process_causal(recs, prov, None, PROJ).trajectories[0]
    assert set(col(t, "position_state")) == {"projection_missing"} and np.isnan(col(t, "vx")).all()
    far = stage2(make_track(frames=range(3), x0=2500.0, y0=400.0, vx=0.0, vy=0.0))
    s3 = stage3(*far)
    t2 = process_causal(far[0], far[1], s3.projections, PROJ).trajectories[0]
    assert set(col(t2, "position_state")) == {"not_projected(outside_image)"} and np.isnan(col(t2, "xs")).all()


# ----------------------------------------------- 17 verified metric with provenance
def test_metric_features_only_with_eligible_promoted_projection_and_explicit_provenance():
    manifest = make_manifest()
    recs, prov, projs = pipeline(manifest, promote=True)
    t = process_causal(recs, prov, projs, METRIC).trajectories[0]
    exp = expected_plane()
    dt = np.diff(col(t, "timestamp"))
    assert (t.position_unit, t.velocity_unit, t.acceleration_unit) == ("m", "m/s", "m/s^2")
    assert t.physical is True and t.physical_basis == "software_manifest_checks_only" and t.unit_status == "metric_by_manifest_checks_only"
    assert t.frame_name == "synthetic_plane" and t.calibration_id == "synthetic-cal" and t.manifest_sha256 == manifest.sha256()
    assert col(t, "vx")[1:] == pytest.approx(np.diff(exp[:, 0]) / dt, rel=1e-9)
    assert col(t, "vy")[1:] == pytest.approx(np.diff(exp[:, 1]) / dt, rel=1e-9)
    assert col(t, "speed")[1:] == pytest.approx(np.hypot(np.diff(exp[:, 0]), np.diff(exp[:, 1])) / dt, rel=1e-9)
    assert all(collect_violations(r) == [] for r in recs)


def test_metric_mode_refuses_missing_or_inconsistent_provenance():
    recs, prov, projs = pipeline(make_manifest(), promote=True)
    none = process_causal(recs, prov, None, METRIC).trajectories[0]
    assert set(col(none, "position_state")) == {"metric_requires_projection_provenance"} and np.isnan(col(none, "vx")).all()
    forged = copy.deepcopy(recs)
    forged[4]["world_x"] += 1.0                                    # record no longer matches its projection
    t = process_causal(forged, prov, projs, METRIC).trajectories[0]
    assert col(t, "position_state")[4] == "record_and_projection_disagree" and np.isnan(col(t, "vx")[4:6]).all()
    weak = [dataclasses.replace(p, manifest_sha256=None) if i == 2 else p for i, p in enumerate(projs)]
    t2 = process_causal(recs, prov, weak, METRIC).trajectories[0]
    assert col(t2, "position_state")[2] == "projection_provenance_incomplete"
    unit_swap = copy.deepcopy(recs)
    unit_swap[3]["world_coordinate_unit"] = "cm"
    assert col(process_causal(unit_swap, prov, projs, METRIC).trajectories[0], "position_state")[3] == "record_and_projection_disagree"
    not_valid = copy.deepcopy(recs)
    not_valid[1]["world_position_valid"] = False
    assert col(process_causal(not_valid, prov, projs, METRIC).trajectories[0], "position_state")[1] == "world_position_valid_is_false"


def test_frame_or_calibration_change_inside_a_track_breaks_differencing():
    recs, prov, projs = pipeline(make_manifest(), promote=True)
    alt = [dataclasses.replace(p, calibration_id="other-cal") if i >= 5 else p for i, p in enumerate(projs)]
    t = process_causal(recs, prov, alt, METRIC).trajectories[0]
    # rows 5+ carry a different calibration id in the projection -> they disagree with the promoted record? no: ids are not
    # compared to the record, so the identity (which includes the calibration id) changes and the chain restarts
    assert col(t, "velocity_state")[4] == "valid" and col(t, "velocity_state")[5] == "frame_identity_changed"
    assert np.isnan(col(t, "vx")[5]) and col(t, "velocity_state")[6] == "valid"


def test_mixed_units_never_difference_across_a_unit_change():
    recs, prov, projs = pipeline(make_manifest(), promote=True)
    alt = copy.deepcopy(recs)
    projs2 = list(projs)
    for i in range(5, 10):
        alt[i]["world_coordinate_unit"] = "cm"
        projs2[i] = dataclasses.replace(projs[i], plane_unit="cm")
    t = process_causal(alt, prov, projs2, METRIC).trajectories[0]
    assert col(t, "velocity_state")[5] == "frame_identity_changed" and t.velocity_unit == "m/s"


# --------------------------------------------------- 18 declared-world provenance
def declared_obs(n=6, unit="m"):
    return [dataclasses.replace(raw("w", f, 0.0, 0.0, sem="world_coordinate", unit=unit, cs="site_map"),
                                pos_x=10.0 + 2.0 * f, pos_y=5.0 + 1.0 * f) for f in range(n)]


def test_declared_world_positions_are_kept_but_labelled_unverified_and_non_physical():
    recs, prov = stage2(declared_obs())
    assert all(r["world_x"] is None and r["world_position_valid"] is False for r in recs)       # Stage 2: not in canonical record
    t = process_causal(recs, prov, None, DECL).trajectories[0]
    assert t.position_unit == "declared:m" and t.velocity_unit == "declared:m/s" and t.physical is False
    assert t.unit_status == "declared_world_not_calibration_verified" and t.frame_name == "site_map" and t.physical_basis is None
    assert col(t, "vx")[1:] == pytest.approx([20.0] * 5, rel=1e-9) and col(t, "vy")[1:] == pytest.approx([10.0] * 5, rel=1e-9)
    assert math.isnan(col(t, "vx")[0])
    # not silently promoted or mixed into other modes
    for cfg in (METRIC, TrajectoryConfig(), PLANAR):
        other = process_causal(recs, prov, None, cfg).trajectories[0]
        assert np.isnan(col(other, "vx")).all() and other.physical is False, cfg.coordinate_mode
    assert set(col(process_causal(recs, prov, None, TrajectoryConfig()).trajectories[0], "position_state")) == {"no_image_position(semantics=world_coordinate)"}


def test_unverified_planar_positions_are_unitless_and_non_physical():
    obs = [dataclasses.replace(raw("u", f, 0.0, 0.0, sem="unverified_planar", unit=None, cs="hbs_local"), pos_x=1.0 * f, pos_y=0.0) for f in range(5)]
    recs, prov = stage2(obs)
    t = process_causal(recs, prov, None, PLANAR).trajectories[0]
    assert t.position_unit == "unverified_planar_units" and t.velocity_unit == "unverified_planar_units/s" and t.physical is False
    assert col(t, "vx")[1:] == pytest.approx([10.0] * 4)
    assert np.isnan(col(process_causal(recs, prov, None, DECL).trajectories[0], "vx")).all()


def test_image_mode_keeps_working_for_image_tracks_next_to_declared_ones():
    obs = make_track("img", frames=range(4)) + declared_obs(4)
    recs, prov = stage2(obs)
    res = process_causal(recs, prov, None, TrajectoryConfig())
    by = {t.raw_track_id: t for t in res.trajectories}
    assert by["img"].velocity_unit == "pixel/s" and np.isnan(col(by["w"], "vx")).all()


# --------------------------------------------------------- 23 regression / end to end
def test_full_pipeline_stage1_to_stage4_on_synthetic_scene(tmp_path):
    scene = generate_scene(seed=3, position_noise_px=0.5)
    parsed = CsvAnnotationAdapter(write_scene_csv(scene, tmp_path / "s.csv"), synthetic_csv_config()).parse()
    built = build_agent_states(parsed.observations)
    assert built.report.ok and all(collect_violations(r) == [] for r in built.records)
    before = snapshot(built.records, built.provenance)
    c = process_causal(built.records, built.provenance, None, TrajectoryConfig())
    o = process_offline(built.records, built.provenance, None, TrajectoryConfig(max_interpolation_gap_frames=4, allow_imputed_support=True))
    assert (built.records, built.provenance) == (before[0], before[1])
    assert c.issues == [] and o.issues == [] and len(c.trajectories) == len(o.trajectories) == 5
    gap = next(t for t in o.trajectories if t.raw_track_id == "ped_gap")
    assert gap.n_imputed == 4 and gap.n_missing == 0 and list(col(gap, "origin")[10:14]) == ["imputed"] * 4
    cg = next(t for t in c.trajectories if t.raw_track_id == "ped_gap")
    assert cg.sampling.missing_frames == (10, 11, 12, 13) and cg.n_imputed == 0
    assert all(collect_violations(r) == [] for r in built.records)


def test_stage3_outputs_feed_stage4_and_remain_schema_valid():
    recs, prov, projs = pipeline(make_manifest(), promote=True)
    assert all(collect_violations(r) == [] for r in recs)
    t = process_offline(recs, prov, projs, METRIC).trajectories[0]
    assert t.VIEW == "OFFLINE_NON_CAUSAL" and t.physical and t.velocity_unit == "m/s"


# ------------------------------------------------------------- HBS read-only smoke
HBS = Path(__file__).resolve().parents[2] / "Data" / "HBS" / "hbs.csv"


@pytest.mark.skipif(not HBS.is_file(), reason="local HBS file not present")
def test_hbs_sample_produces_unverified_planar_features_only(tmp_path):
    before = hashlib.sha256(HBS.read_bytes()).hexdigest()
    sample = tmp_path / "hbs.csv"
    sample.write_text("\n".join(HBS.read_text(encoding="utf-8").splitlines()[:1501]) + "\n", encoding="utf-8")
    cfg = CsvAdapterConfig("HBS", {"track_id": "agent_id", "frame_index": "frame_id", "timestamp": "timestamp", "label": "label",
                                   "x": "pos_x", "y": "pos_y", "vel_x": "vel_x", "vel_y": "vel_y", "scene_id": "scene_id"},
                           sequence_id="hbs_seq", position_semantics="unverified_planar", position_unit=None,
                           position_coordinate_system="hbs_unverified_local")
    built = build_agent_states(CsvAnnotationAdapter(sample, cfg).parse().observations)
    assert built.report.ok
    res = process_causal(built.records, built.provenance, None, PLANAR)
    assert len(res.trajectories) >= 2 and res.issues == []
    for t in res.trajectories:
        assert t.physical is False and t.velocity_unit == "unverified_planar_units/s" and "m/s" not in t.velocity_unit
        d = col(t, "dt")
        d = d[np.isfinite(d)]
        assert d.size == 0 or d.max() == pytest.approx(0.5)               # HBS 0.5 s spacing observed in the data, not assumed
    img = process_causal(built.records, built.provenance, None, TrajectoryConfig())
    assert all(np.isnan(col(t, "vx")).all() for t in img.trajectories)
    assert hashlib.sha256(HBS.read_bytes()).hexdigest() == before
