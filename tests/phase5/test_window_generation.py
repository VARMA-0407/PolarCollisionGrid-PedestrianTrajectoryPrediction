"""P5.3 window generation on a real (synthetic-input) Task 3 run: indexing, gaps, masks, context, relations, leakage."""

import copy
import json

import numpy as np
import pytest

from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.graph_window_adapter import build_graph_bank, collate_windows, window_target_bank_nodes
from pedsafety.phase5.task3_source import load_task3_run
from pedsafety.phase5.trajectory_targets import pilot_smoke_targets
from pedsafety.phase5.upstream import verify_upstream
from pedsafety.phase5.window_builder import REJECTION_REASONS, build_windows
from pedsafety.phase5.window_validation import validate_window_dataset
from phase5_helpers import track_index, wcfg


@pytest.fixture(scope="module")
def built(scene_run_dir):
    cfg = wcfg()
    run = load_task3_run(scene_run_dir, cfg)
    ws = build_windows(run, cfg)
    bank = build_graph_bank(run, ws, cfg)
    return cfg, run, ws, bank, window_target_bank_nodes(bank, ws)


def _windows_of(ws, ti):
    return np.flatnonzero(ws.target_track_index == ti)


def _reasons_of(ws, ti):
    sel = ws.rejected_track_index == ti
    return {int(a): REJECTION_REASONS[int(r)] for a, r in zip(ws.rejected_anchor_frame[sel], ws.rejected_reason_code[sel])}


def test_accepted_windows_and_targets(built):
    cfg, run, ws, bank, _ = built
    t1, t3 = track_index(run, "1"), track_index(run, "3")
    assert sorted(set(ws.target_track_index.tolist())) == sorted([t1, t3])
    assert ws.anchor_frame[_windows_of(ws, t1)].tolist() == list(range(15, 43, 3))
    assert ws.n_windows == 20
    assert ws.n_windows + ws.n_rejected == sum(len(range((t["first_frame"] + 2) // 3 * 3, t["last_frame"] + 1, 3))
                                               for t in run.tracks if t["node_type"] == "pedestrian")


def test_six_history_and_fifteen_future_frames(built):
    cfg, run, ws, *_ = built
    w = _windows_of(ws, track_index(run, "1"))[0]          # anchor 15
    assert ws.history_frames[w].tolist() == [0, 3, 6, 9, 12, 15]
    assert ws.future_frames[w].tolist() == list(range(18, 61, 3))
    assert ws.history_frames.shape[1] == 6 and ws.future_frames.shape[1] == 15


def test_exact_timestamps_and_10hz(built):
    cfg, run, ws, *_ = built
    assert np.allclose(ws.history_timestamps, ws.history_frames / 30.0, atol=1e-12, rtol=0)
    assert np.allclose(ws.future_timestamps, ws.future_frames / 30.0, atol=1e-12, rtol=0)
    allt = np.concatenate([ws.history_timestamps, ws.future_timestamps], axis=1)
    assert np.allclose(np.diff(allt, axis=1), 0.1, atol=1e-9)
    assert np.allclose(ws.history_timestamps[:, -1] - ws.history_timestamps[:, 0], 0.5, atol=1e-9)
    assert np.allclose(ws.future_timestamps[:, -1] - ws.history_timestamps[:, -1], 1.5, atol=1e-9)


def test_history_future_separation(built):
    cfg, run, ws, *_ = built
    assert np.all(ws.future_timestamps[:, 0] > ws.history_timestamps[:, -1])
    for h, f in zip(ws.history_frames, ws.future_frames):
        assert not set(h.tolist()) & set(f.tolist())
    assert np.array_equal(ws.history_frames[:, -1], ws.anchor_frame)


def test_track_gap_at_sampled_frame_rejects_with_reasons(built):
    cfg, run, ws, *_ = built
    t2 = track_index(run, "2")
    assert _windows_of(ws, t2).size == 0
    r = _reasons_of(ws, t2)
    assert r[30] == "target_absent_at_anchor"
    assert all(r[a] == "target_future_gap" for a in range(15, 30, 3))      # frame 30 lies in the future
    assert all(r[a] == "target_history_gap" for a in range(33, 43, 3))     # frame 30 lies in the history
    assert all(r[a] == "history_before_video_start" for a in range(0, 15, 3))
    assert all(r[a] == "future_after_video_end" for a in range(45, 88, 3))


def test_missing_unsampled_frame_keeps_window_and_mask(built):
    cfg, run, ws, bank, tbn = built
    t3 = track_index(run, "3")
    assert ws.anchor_frame[_windows_of(ws, t3)].tolist() == list(range(15, 43, 3))
    n32 = run.node_at(t3, 32)
    assert run.node_at(t3, 31) is None and n32 is not None
    assert not run.masks[n32, 1] and np.isnan(run.features[n32, 2])        # Task 2 velocity invalid after the gap
    w = _windows_of(ws, t3)[0]
    b = collate_windows(bank, ws, [w], cfg)
    assert b["present"][0, :, 0].all()


def test_no_interpolation_pilot_targets_missing_stays_missing(built):
    cfg, run, ws, *_ = built
    t2 = track_index(run, "2")
    tg = pilot_smoke_targets(run, t2, [27, 30, 33])
    assert tg.valid.tolist() == [True, False, True]
    assert np.isnan(tg.uv_px[1]).all() and tg.node_index[1] == -1
    assert not tg.complete
    # every stored target value is a real Task 3 node of the same track and frame
    fn = ws.target_future_task3_node
    assert np.array_equal(run.features[fn][:, :, :2], ws.target_future_uv_px)
    assert np.array_equal(run.frame[fn], ws.future_frames)


def test_no_forward_fill_for_absent_context(built):
    cfg, run, ws, bank, _ = built
    t1, t2 = track_index(run, "1"), track_index(run, "2")
    w = int(np.flatnonzero((ws.target_track_index == t1) & (ws.anchor_frame == 33))[0])   # history 18..33 includes frame 30
    b = collate_windows(bank, ws, [w], cfg)
    s = int(np.flatnonzero(b["track_index"][0] == t2)[0])
    h30 = ws.history_frames[w].tolist().index(30)
    assert not b["present"][0, h30, s]
    assert np.isnan(b["node_features"][0, h30, s]).all() and not b["node_masks"][0, h30, s].any()
    assert b["present"][0, h30 - 1, s] and b["present"][0, h30 + 1, s]
    # no edge touches the absent slot at that step
    for r, e in b["edges"].items():
        at = e["bhsd"][e["bhsd"][:, 1] == h30]
        assert not np.any((at[:, 2] == s) | (at[:, 3] == s))


def test_no_track_id_merging(built):
    cfg, run, ws, *_ = built
    t6, t7 = track_index(run, "6"), track_index(run, "7")
    assert _windows_of(ws, t6).size == 0 and _windows_of(ws, t7).size == 0   # merged 0..89 would give 10 windows
    assert set(_reasons_of(ws, t6).values()) <= {"history_before_video_start", "future_after_track_end"}
    assert set(_reasons_of(ws, t7).values()) <= {"history_before_track_start", "future_after_video_end"}
    for i in range(ws.n_windows):
        ti = ws.target_track_index[i]
        assert np.all(run.track_index[ws.history_target_task3_node[i]] == ti)
        assert np.all(run.track_index[ws.target_future_task3_node[i]] == ti)


def test_pedestrian_only_targets(built):
    cfg, run, ws, bank, tbn = built
    assert np.all(run.node_type[ws.history_target_task3_node] == 0)
    for lab in ("4", "5", "9"):                                             # continuous vehicle never a target
        ti = track_index(run, lab)
        assert ti not in ws.target_track_index and ti not in ws.rejected_track_index


def test_vehicle_context_preserved(built):
    cfg, run, ws, bank, _ = built
    b = collate_windows(bank, ws, range(ws.n_windows), cfg)
    t4 = track_index(run, "4")
    assert np.all(np.any(b["track_index"] == t4, axis=1))
    assert set(b["active_relations"]) == {"pedestrian__pedestrian", "vehicle__pedestrian", "pedestrian__vehicle"}
    assert b["edges"]["vehicle__pedestrian"]["bhsd"].shape[0] > 0


def test_future_only_agents_are_invisible(built):
    cfg, run, ws, bank, _ = built
    t9 = track_index(run, "9")                                              # appears at frame 50: future only
    assert t9 not in bank.nodes["track_index"]
    b = collate_windows(bank, ws, range(ws.n_windows), cfg, include_optional=True)
    assert not np.any(b["track_index"] == t9)


def test_bicycle_semantic_uncertainty(built):
    cfg, run, ws, bank, _ = built
    t5 = track_index(run, "5")
    unc = bank.nodes["semantically_uncertain"]
    assert np.array_equal(unc, bank.nodes["node_type"] == 2) and unc.any()
    w = int(np.flatnonzero(ws.anchor_frame == 15)[0])                       # history 0..15; bike present only at 12, 15
    d = collate_windows(bank, ws, [w], cfg)
    assert not np.any(d["track_index"] == t5)                               # excluded by default
    o = collate_windows(bank, ws, [w], cfg, include_optional=True)
    s = int(np.flatnonzero(o["track_index"][0] == t5)[0])
    assert o["semantically_uncertain"][0, s] and o["node_type"][0, s] == 2
    assert o["present"][0, :, s].tolist() == [False, False, False, False, True, True]
    assert ws.n_windows == int(np.sum(ws.target_track_index >= 0))          # windows never require a bicycle node


def test_approved_relation_filtering(built):
    cfg, run, ws, bank, _ = built
    rels = set(bank.edges)
    assert rels == {"pedestrian__pedestrian", "vehicle__pedestrian", "pedestrian__vehicle", "bicyclist__pedestrian",
                    "pedestrian__bicyclist", "bicyclist__vehicle", "vehicle__bicyclist"}
    assert "bicyclist__bicyclist" not in rels and "vehicle__vehicle" not in rels
    assert set(bank.active_relations(cfg, False)) == {"pedestrian__pedestrian", "vehicle__pedestrian", "pedestrian__vehicle"}
    assert set(bank.active_relations(cfg, True)) == rels


def test_graph_references(built):
    cfg, run, ws, bank, tbn = built
    assert np.all(bank.nodes["track_index"][tbn] == ws.target_track_index[:, None])
    assert np.array_equal(bank.nodes["frame"][tbn], ws.history_frames)
    for name, e in bank.edges.items():
        assert np.array_equal(bank.nodes["frame"][e.edge_index[0]], e.frame)
        assert np.array_equal(bank.nodes["task3_node_index"][e.edge_index], run.edges[name].edge_index[:, e.task3_edge_index])
    assert set(bank.frames.tolist()) == set(ws.history_frames.ravel().tolist())
    assert bank.frames.max() <= ws.anchor_frame.max()


def test_feature_validity_masks_carried(built):
    cfg, run, ws, bank, _ = built
    b = collate_windows(bank, ws, range(ws.n_windows), cfg, include_optional=True)
    f, m, p = b["node_features"], b["node_masks"], b["present"]
    assert np.array_equal(np.isfinite(f[..., 2]), m[..., 1])
    assert np.array_equal(np.isfinite(f[..., 0]), p)
    assert not m[~p].any()
    for e in b["edges"].values():
        assert np.array_equal(np.isfinite(e["edge_attr"]), e["edge_mask"])


def test_validation_passes_and_detects_tampering(built):
    cfg, run, ws, bank, tbn = built
    rep = validate_window_dataset(ws, bank, tbn, cfg, run)
    assert rep["ok"], [k for k, v in rep["checks"].items() if not v["pass"]]
    bad = copy.deepcopy(ws)
    bad.history_timestamps[0, 2], bad.history_timestamps[0, 3] = bad.history_timestamps[0, 3], bad.history_timestamps[0, 2]
    r2 = validate_window_dataset(bad, bank, tbn, cfg, run)
    assert not r2["checks"]["history_timestamps_strictly_increasing"]["pass"]
    bad = copy.deepcopy(ws)
    bad.target_future_uv_px[0, 3] += 0.5                                    # an "interpolated"/edited target
    r3 = validate_window_dataset(bad, bank, tbn, cfg, run)
    assert not r3["checks"]["no_interpolation_future_targets_are_task3_nodes"]["pass"]


def test_future_frame_never_enters_input(built):
    cfg, run, ws, bank, _ = built
    bad = copy.deepcopy(ws)
    bad.history_frames[0, 2] = bad.future_frames[0, 0]
    with pytest.raises(WindowContractError, match="FUTURE_FRAME_IN_INPUT"):
        bank.window_graphs(bad, 0, cfg)


def test_agent_overflow_is_an_error(built):
    cfg, run, ws, bank, _ = built
    with pytest.raises(WindowContractError, match="AGENT_OVERFLOW"):
        collate_windows(bank, ws, [0], cfg, max_agents=1)


def test_invalid_coordinate_metadata_rejected(scene_run_dir):
    with pytest.raises(WindowContractError, match="COORDINATE_SYSTEM_MISMATCH"):
        load_task3_run(scene_run_dir, wcfg(coordinate_system="captured_640x360"))


def test_tampered_task3_output_rejected(scene_run_dir, tmp_path):
    import shutil
    d = tmp_path / "t3copy"
    shutil.copytree(scene_run_dir, d)
    fd = d / "feature_dictionary.json"
    obj = json.loads(fd.read_text(encoding="utf-8"))
    obj["coordinate_system"] = "world_metric"
    fd.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(WindowContractError, match="TASK3_OUTPUT_SIZE|TASK3_OUTPUT_HASH"):
        load_task3_run(d, wcfg())
    # same size, different content -> hash check
    raw = bytearray((scene_run_dir / "feature_dictionary.json").read_bytes())
    raw[raw.index(b"captured_1920x1080")] = ord("C")
    fd.write_bytes(bytes(raw))
    with pytest.raises(WindowContractError, match="TASK3_OUTPUT_HASH"):
        load_task3_run(d, wcfg())


def test_true_claim_rejected(scene_run_dir, tmp_path):
    import shutil
    d = tmp_path / "t3claim"
    shutil.copytree(scene_run_dir, d)
    mp = d / "run_manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["claims"]["metric_world_coordinates"] = True
    mp.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(WindowContractError, match="TASK3_MANIFEST_SCHEMA|TASK3_CLAIM_TRUE"):
        load_task3_run(d, wcfg())


def test_upstream_chain_and_source_line_join(built):
    cfg, run, *_ = built
    rep = verify_upstream(run, cfg)
    assert rep["source_line_join"]["all_match"] and rep["source_line_join"]["nodes_checked"] == run.n_nodes
    assert rep["causal_rows"]["rows_matched_to_nodes"] == run.n_nodes
    assert rep["task2"]["offline_sidecar_consumed"] is False
    assert rep["source_labels_by_node_type"]["bicyclist"] == {"bike": 51}
