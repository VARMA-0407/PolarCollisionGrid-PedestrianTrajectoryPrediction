"""Pairwise features, masks, relations and sensitivity flags on hand-built node tables (hand-computed oracles)."""

import math

import numpy as np
import pytest

from pedsafety.interaction.config import NODE_TYPES
from pedsafety.interaction.diagnostics import theoretical_edge_counts
from pedsafety.interaction.nodes import NODE_FLOAT_FEATURES, NODE_MASKS, build_node_table, canonical_track_key
from pedsafety.interaction.pairwise import EDGE_FEATURES, build_edges, flag_names
from task3_helpers import cfg3, jr

F = {n: i for i, n in enumerate(EDGE_FEATURES)}


def edge(edges, rel, src_track, dst_track, nt):
    e = edges[rel]
    tid = {t: i for i, t in enumerate(nt.track_ids)}
    s, d = tid["pie:video_0004:%s" % src_track], tid["pie:video_0004:%s" % dst_track]
    hit = [k for k in range(e.frame.size) if nt.track_index[e.edge_index[0, k]] == s and nt.track_index[e.edge_index[1, k]] == d]
    assert len(hit) == 1
    return e.edge_attr[hit[0]].astype(np.float64), e.edge_mask[hit[0]], e.flags[hit[0]]


def two_peds():
    # A (track 1): footpoint (103, 500), h 60, moving right 90 px/s.  B (track 2): (160, 420), h 80, stationary (velocity 0).
    return [jr("1", 1, 103.0, 500.0, h=60.0, vel=(90.0, 0.0)), jr("2", 1, 160.0, 420.0, h=80.0, vel=(0.0, 0.0))]


def test_exact_pairwise_geometry_and_direction():
    nt = build_node_table(two_peds())
    edges, _ = build_edges(nt, cfg3())
    a, m, _ = edge(edges, "pedestrian__pedestrian", "2", "1", nt)          # edge B -> A: source B, destination A
    sep = math.hypot(57.0, -80.0)                                            # 98.2293
    assert a[F["du_px"]] == pytest.approx(57.0) and a[F["dv_px"]] == pytest.approx(-80.0) and a[F["sep_px"]] == pytest.approx(sep, rel=1e-6)
    assert a[F["bearing_sin"]] == pytest.approx(-80.0 / sep, rel=1e-6) and a[F["bearing_cos"]] == pytest.approx(57.0 / sep, rel=1e-6)
    assert a[F["sep_over_h"]] == pytest.approx(sep / 80.0, rel=1e-6) and a[F["log_h_ratio"]] == pytest.approx(math.log(80.0 / 60.0), rel=1e-6)
    assert a[F["dv_row_px"]] == pytest.approx(-80.0)
    assert a[F["dvel_u_px_s"]] == pytest.approx(-90.0) and a[F["dvel_v_px_s"]] == pytest.approx(0.0)
    # A moves right toward B: pixel separation shrinks -> closing rate positive = 57*90/sep
    assert a[F["closing_rate_px_s"]] == pytest.approx(57.0 * 90.0 / sep, rel=1e-6)
    # destination A heads +u (heading 0); relative bearing = bearing of B as seen from A
    assert a[F["rel_bearing_sin"]] == pytest.approx(-80.0 / sep, rel=1e-6) and a[F["rel_bearing_cos"]] == pytest.approx(57.0 / sep, rel=1e-6)
    assert m.all()


def test_antisymmetry_and_symmetry():
    nt = build_node_table(two_peds())
    edges, _ = build_edges(nt, cfg3())
    ab, _, _ = edge(edges, "pedestrian__pedestrian", "1", "2", nt)
    ba, _, _ = edge(edges, "pedestrian__pedestrian", "2", "1", nt)
    for f in ("du_px", "dv_px", "dv_row_px", "log_h_ratio", "dvel_u_px_s", "dvel_v_px_s", "bearing_sin", "bearing_cos"):
        assert ab[F[f]] == pytest.approx(-ba[F[f]], abs=1e-5), f
    for f in ("sep_px", "sep_over_h", "closing_rate_px_s", "overlap_ios"):
        assert ab[F[f]] == pytest.approx(ba[F[f]], rel=1e-6), f


@pytest.mark.parametrize("vel_j,vel_i,expected_sign", [((0.0, 0.0), (90.0, 0.0), 1), ((0.0, 0.0), (-90.0, 0.0), -1), ((0.0, 90.0), (0.0, 90.0), 0)])
def test_closing_rate_signs(vel_j, vel_i, expected_sign):
    # j at (200, 500), i at (100, 500): i moving right approaches, moving left recedes, both moving down = parallel
    nt = build_node_table([jr("1", 1, 100.0, 500.0, vel=vel_i), jr("2", 1, 200.0, 500.0, vel=vel_j)])
    a, m, _ = edge(build_edges(nt, cfg3())[0], "pedestrian__pedestrian", "2", "1", nt)
    assert m[F["closing_rate_px_s"]]
    assert a[F["closing_rate_px_s"]] == pytest.approx(90.0 * expected_sign, abs=1e-6)


def test_relative_bearing_uses_destination_heading_and_seam():
    # destination heading 0 (moving +u); source directly above in the image (dv < 0) -> rel bearing -pi/2
    nt = build_node_table([jr("1", 1, 100.0, 500.0, vel=(10.0, 0.0)), jr("2", 1, 100.0, 400.0, vel=(0.0, 0.0))])
    a, _, _ = edge(build_edges(nt, cfg3())[0], "pedestrian__pedestrian", "2", "1", nt)
    assert a[F["rel_bearing_sin"]] == pytest.approx(-1.0, abs=1e-6) and a[F["rel_bearing_cos"]] == pytest.approx(0.0, abs=1e-6)
    # seam: destination heading pi - 0.01, source at bearing -pi + 0.01 -> rel bearing +0.02
    h = math.pi - 0.01
    b = -math.pi + 0.01
    nt = build_node_table([jr("1", 1, 500.0, 500.0, vel=(math.cos(h) * 50, math.sin(h) * 50)),
                           jr("2", 1, 500.0 + 100 * math.cos(b), 500.0 + 100 * math.sin(b), vel=(0.0, 0.0))])
    a, _, _ = edge(build_edges(nt, cfg3())[0], "pedestrian__pedestrian", "2", "1", nt)
    assert a[F["rel_bearing_sin"]] == pytest.approx(math.sin(0.02), abs=2e-4) and a[F["rel_bearing_cos"]] == pytest.approx(math.cos(0.02), abs=2e-4)


def test_velocity_and_heading_masks():
    # j has no valid velocity; i has velocity but zero speed -> heading invalid
    nt = build_node_table([jr("1", 1, 100.0, 500.0, vel=(0.0, 0.0)), jr("2", 1, 160.0, 420.0, vel=None)])
    a, m, _ = edge(build_edges(nt, cfg3())[0], "pedestrian__pedestrian", "2", "1", nt)
    for f in ("dvel_u_px_s", "dvel_v_px_s", "closing_rate_px_s", "rel_bearing_sin", "rel_bearing_cos"):
        assert not m[F[f]] and math.isnan(a[F[f]]), f
    for f in ("du_px", "dv_px", "sep_px", "bearing_sin", "sep_over_h", "log_h_ratio", "overlap_ios"):
        assert m[F[f]] and math.isfinite(a[F[f]]), f
    fi = {n: i for i, n in enumerate(NODE_FLOAT_FEATURES)}
    mi = {n: i for i, n in enumerate(NODE_MASKS)}
    k = list(nt.track_ids).index("pie:video_0004:2")
    assert not nt.masks[k, mi["velocity_valid"]] and math.isnan(nt.features[k, fi["vel_u_px_s"]]) and math.isnan(nt.features[k, fi["heading_sin"]])


def test_zero_separation_is_masked_not_nan_leaking():
    nt = build_node_table([jr("1", 1, 300.0, 500.0, vel=(10.0, 0.0)), jr("2", 1, 300.0, 500.0, vel=(0.0, 5.0))])
    a, m, _ = edge(build_edges(nt, cfg3())[0], "pedestrian__pedestrian", "2", "1", nt)
    assert a[F["sep_px"]] == 0.0 and a[F["sep_over_h"]] == 0.0 and m[F["sep_px"]]
    for f in ("bearing_sin", "bearing_cos", "rel_bearing_sin", "rel_bearing_cos", "closing_rate_px_s"):
        assert not m[F[f]] and math.isnan(a[F[f]])
    assert a[F["dvel_u_px_s"]] == pytest.approx(-10.0) and m[F["dvel_u_px_s"]]
    assert a[F["overlap_ios"]] == pytest.approx(1.0)


@pytest.mark.parametrize("j_box,expected", [((300.0, 500.0), 0.0), ((105.0, 495.0), (15 * 55) / (20 * 60.0)), ((100.0, 500.0), 1.0)])
def test_overlap_ios(j_box, expected):
    # i: box x [90,110], y [440,500] (area 1200). j same size, footpoint moved.
    nt = build_node_table([jr("1", 1, 100.0, 500.0), jr("2", 1, j_box[0], j_box[1])])
    a, _, _ = edge(build_edges(nt, cfg3())[0], "pedestrian__pedestrian", "2", "1", nt)
    assert a[F["overlap_ios"]] == pytest.approx(expected, rel=1e-6)


def test_nested_box_overlap_is_one():
    nt = build_node_table([jr("1", 1, 100.0, 500.0, h=200.0, w=200.0), jr("2", 1, 100.0, 480.0, h=40.0, w=20.0)])
    a, _, _ = edge(build_edges(nt, cfg3())[0], "pedestrian__pedestrian", "2", "1", nt)
    assert a[F["overlap_ios"]] == pytest.approx(1.0)


def mixed_frame():
    return [jr("1", 1, 100.0, 500.0), jr("2", 1, 200.0, 500.0), jr("3", 1, 400.0, 600.0, cls="vehicle", h=150.0, w=200.0),
            jr("4", 1, 700.0, 600.0, cls="vehicle", h=150.0, w=200.0), jr("5", 1, 900.0, 650.0, cls="bicyclist", h=90.0, w=40.0),
            jr("6", 1, 1000.0, 650.0, cls="bicyclist", h=90.0, w=40.0)]


def test_relation_typing_no_self_edges_and_vv_flag():
    nt = build_node_table(mixed_frame())
    edges, nr = build_edges(nt, cfg3())
    counts = {k: e.frame.size for k, e in edges.items()}
    assert counts == {"pedestrian__pedestrian": 2, "vehicle__pedestrian": 4, "pedestrian__vehicle": 4, "bicyclist__pedestrian": 4,
                      "pedestrian__bicyclist": 4, "bicyclist__vehicle": 4, "vehicle__bicyclist": 4}
    assert nr == {"bicyclist__bicyclist": 2, "vehicle__vehicle": 2}
    for k, e in edges.items():
        s, d = k.split("__")
        assert (e.edge_index[0] != e.edge_index[1]).all()
        assert all(NODE_TYPES[nt.node_type[x]] == s for x in e.edge_index[0]) and all(NODE_TYPES[nt.node_type[x]] == d for x in e.edge_index[1])
    edges_vv, nr_vv = build_edges(nt, cfg3(enable_vehicle_vehicle=True))
    assert edges_vv["vehicle__vehicle"].frame.size == 2 and nr_vv == {"bicyclist__bicyclist": 2}
    assert int((nt.node_type == 2).sum()) == 2                               # bicyclists stay a separate type


def test_vv_disabled_by_default_in_code_and_shipped_config():
    import json
    from pedsafety.interaction.config import InteractionConfig
    from task3_helpers import T3_CONFIG
    d = json.loads(T3_CONFIG.read_text(encoding="utf-8"))
    assert d["enable_vehicle_vehicle"] is False
    d.pop("enable_vehicle_vehicle")
    c = InteractionConfig.from_dict(d)                                        # key omitted -> code default
    assert c.enable_vehicle_vehicle is False and ("vehicle", "vehicle") not in c.relations
    assert len(c.relations) == 7 and ("bicyclist", "bicyclist") not in c.relations


def test_complete_graph_equals_theoretical_and_flags_never_filter():
    nt = build_node_table(mixed_frame())
    edges, _ = build_edges(nt, cfg3(radius_thresholds_sep_over_h=[0.001], knn_k=[1]))
    theo = theoretical_edge_counts(nt)
    for k, e in edges.items():
        assert e.frame.size == theo[k]
    assert sum(e.flags[:, 1].sum() for e in edges.values()) == 0             # radius flag false everywhere, edges still stored


def test_knn_tie_breaking_uses_canonical_track_order():
    # destination track 1 at (500, 500); sources 10, 2, 3 at the same distance (100 px) and same height -> ties
    rows = [jr("1", 1, 500.0, 500.0), jr("10", 1, 600.0, 500.0), jr("2", 1, 400.0, 500.0), jr("3", 1, 500.0, 600.0)]
    nt = build_node_table(rows)
    assert nt.track_ids == sorted(nt.track_ids, key=canonical_track_key) and nt.track_ids[1:] == ["pie:video_0004:2", "pie:video_0004:3", "pie:video_0004:10"]
    edges, _ = build_edges(nt, cfg3(knn_k=[1, 2]))
    fn = list(flag_names(cfg3(knn_k=[1, 2])))
    picks = {}
    for src in ("2", "3", "10"):
        _, _, fl = edge(edges, "pedestrian__pedestrian", src, "1", nt)
        picks[src] = (bool(fl[fn.index("knn_1")]), bool(fl[fn.index("knn_2")]))
    assert picks == {"2": (True, True), "3": (False, True), "10": (False, False)}


def test_knn_ranks_across_relations_and_radius_flags():
    # pedestrian destination 1; pedestrian source 2 at sep_over_h 2.5; vehicle source 3 nearer (sep_over_h 1.0)
    rows = [jr("1", 1, 500.0, 500.0, h=60.0), jr("2", 1, 650.0, 500.0, h=60.0), jr("3", 1, 500.0, 650.0, cls="vehicle", h=150.0, w=50.0)]
    nt = build_node_table(rows)
    c = cfg3(radius_thresholds_sep_over_h=[1.0, 2.0, 5.0], knn_k=[1])
    edges, _ = build_edges(nt, c)
    fn = list(flag_names(c))
    _, _, f_ped = edge(edges, "pedestrian__pedestrian", "2", "1", nt)
    _, _, f_veh = edge(edges, "vehicle__pedestrian", "3", "1", nt)
    assert f_veh[fn.index("knn_1")] and not f_ped[fn.index("knn_1")]           # global ranking across relations (P4)
    assert [bool(f_ped[fn.index("sep_over_h_le_%g" % r)]) for r in (1, 2, 5)] == [False, False, True]   # 150/60 = 2.5
    assert [bool(f_veh[fn.index("sep_over_h_le_%g" % r)]) for r in (1, 2, 5)] == [True, True, True]      # 150/150 = 1.0 (inclusive)


def test_truncation_flag_propagation_keeps_values():
    nt = build_node_table([jr("1", 1, 100.0, 500.0, border=True, h=60.0), jr("2", 1, 200.0, 500.0, h=60.0), jr("3", 1, 300.0, 500.0, h=60.0)])
    edges, _ = build_edges(nt, cfg3())
    e = edges["pedestrian__pedestrian"]
    tid = {t: i for i, t in enumerate(nt.track_ids)}
    b = tid["pie:video_0004:1"]
    for k in range(e.frame.size):
        touches = b in (nt.track_index[e.edge_index[0, k]], nt.track_index[e.edge_index[1, k]])
        assert bool(e.flags[k, 0]) == touches
    a, m, fl = edge(edges, "pedestrian__pedestrian", "2", "1", nt)
    assert fl[0] and a[F["sep_over_h"]] == pytest.approx(100.0 / 60.0, rel=1e-6) and m[F["sep_over_h"]]


def test_missing_frame_track_age_and_frames_since_prev_obs():
    rows = [jr("1", 0, 100.0, 500.0, gap=None), jr("2", 0, 200.0, 500.0, gap=None), jr("2", 1, 203.0, 500.0, gap=0),
            jr("1", 3, 109.0, 500.0, gap=2), jr("2", 3, 209.0, 500.0, gap=1)]
    nt = build_node_table(rows)
    assert 1 in nt.frame_starts and nt.frame_starts[1][1] - nt.frame_starts[1][0] == 1          # frame 1: only track 2
    edges, _ = build_edges(nt, cfg3())
    assert 1 not in set(edges["pedestrian__pedestrian"].frame.tolist())                        # no edge without a partner
    fi = {n: i for i, n in enumerate(NODE_FLOAT_FEATURES)}
    mi = {n: i for i, n in enumerate(NODE_MASKS)}
    rows_out = {(int(nt.frame[k]), nt.track_ids[nt.track_index[k]][-1]): k for k in range(nt.n)}
    k = rows_out[(3, "1")]
    assert nt.features[k, fi["track_age_frames"]] == 3 and nt.features[k, fi["frames_since_prev_obs"]] == 2
    k0 = rows_out[(0, "1")]
    assert not nt.masks[k0, mi["frames_since_prev_obs_valid"]] and np.isnan(nt.features[k0, fi["frames_since_prev_obs"]])
    assert nt.features[k0, fi["track_age_frames"]] == 0


def test_invalid_and_rejected_rows_are_not_nodes():
    rows = [jr("1", 1, 100.0, 500.0), jr("2", 1, 200.0, 500.0, obs="invalid", pos="observation_valid_not_true"),
            jr("3", 1, 300.0, 500.0, obs="rejected", pos="rejected_duplicate_observation", gap=None)]
    nt = build_node_table(rows)
    assert nt.n == 1 and nt.track_ids == ["pie:video_0004:1"]


def test_scalability_max_frame_and_large_input():
    import time
    rows = []
    for f in range(400):
        for t in range(23):                                                   # the PIE maximum of 23 agents in one frame
            cls = ("pedestrian", "vehicle", "bicyclist")[t % 3]
            rows.append(jr(str(t), f, 50.0 + 80 * t, 400.0 + 20 * (t % 5), cls=cls, vel=(1.0 + t, 0.5), gap=0 if f else None))
    nt = build_node_table(rows)
    t0 = time.perf_counter()
    edges, nr = build_edges(nt, cfg3())
    elapsed = time.perf_counter() - t0
    theo = theoretical_edge_counts(nt)
    assert all(e.frame.size == theo[k] for k, e in edges.items())
    assert sum(e.frame.size for e in edges.values()) + sum(nr.values()) == 400 * 23 * 22
    assert elapsed < 60.0                                                     # 9,200 nodes / 202,400 pairs; generous bound
