"""Runner/CLI end to end: outputs, dtypes, dictionary, manifest integrity, determinism, order invariance, source immutability."""

import json
import random
import subprocess
import sys

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from pedsafety.interaction.nodes import NODE_FLOAT_FEATURES, NODE_MASKS
from pedsafety.interaction.pairwise import EDGE_FEATURES
from pedsafety.interaction.runner import run_interaction_representation
from pedsafety.interaction.storage import load_npz
from pedsafety.output_dir import OutputDirectoryError
from task3_helpers import REPO, T3_CONFIG, cfg3, make_task2_run, read_rows, resync, scene_records, sha, write_rows

SCHEMAS = REPO / "docs" / "phase4" / "schemas"
EXPECTED = {"nodes.npz", "tracks.json", "feature_dictionary.json", "interaction_diagnostics.json", "config_resolved.json",
            "interaction_summary.md", "run_manifest.json"} | {"edges_%s.npz" % r for r in (
                "pedestrian__pedestrian", "vehicle__pedestrian", "pedestrian__vehicle", "bicyclist__pedestrian", "pedestrian__bicyclist",
                "bicyclist__vehicle", "vehicle__bicyclist")}


@pytest.fixture()
def run(tmp_path):
    causal, manifest, p3 = make_task2_run(tmp_path, scene_records())
    before = {p.name: sha(p) for p in (causal, manifest, p3)}
    m = run_interaction_representation(causal, manifest, p3, cfg3(), tmp_path / "out")
    return tmp_path / "out", m, (causal, manifest, p3), before


def test_outputs_dtypes_and_shapes(run):
    out, m, _, _ = run
    assert {p.name for p in out.iterdir()} == EXPECTED
    n = load_npz(out / "nodes.npz")
    N = n["frame"].shape[0]
    assert N == 16 and n["features"].dtype == np.float32 and n["features"].shape == (N, len(NODE_FLOAT_FEATURES))
    assert n["masks"].dtype == bool and n["masks"].shape == (N, len(NODE_MASKS)) and n["frame"].dtype == np.int64
    assert list(n["feature_names"]) == list(NODE_FLOAT_FEATURES)
    order = list(zip(n["frame"].tolist(), n["track_index"].tolist()))
    assert order == sorted(order)                                                          # canonical node order
    e = load_npz(out / "edges_pedestrian__pedestrian.npz")
    E = e["frame"].shape[0]
    assert e["edge_index"].shape == (2, E) and e["edge_index"].dtype == np.int64 and e["edge_attr"].shape == (E, len(EDGE_FEATURES))
    assert e["edge_attr"].dtype == np.float32 and e["edge_mask"].dtype == bool and list(e["relation"]) == ["pedestrian", "pedestrian"]
    assert not (np.isfinite(e["edge_attr"]) & ~e["edge_mask"]).any() and np.isfinite(e["edge_attr"][e["edge_mask"]]).all()


def test_manifest_integrity_and_schema(run):
    out, m, _, _ = run
    v = Draft202012Validator(json.loads((SCHEMAS / "interaction_run_manifest.schema.json").read_text(encoding="utf-8")))
    assert not list(v.iter_errors(m))
    assert "run_manifest.json" not in m["outputs"]                                          # the manifest does not hash itself
    for name, info in m["outputs"].items():
        assert sha(out / name) == info["sha256"] and (out / name).stat().st_size == info["bytes"]
    assert m["inputs"]["unchanged_after_run"] is True and m["claims"]["physical_features"] is False
    assert m["counts"]["edges_by_relation"]["pedestrian__pedestrian"] == 8                 # 2 pedestrians x 4 frames x 2 directions
    assert m["counts"]["candidate_pairs_not_represented"] == {}                            # one vehicle, one bicyclist per frame
    assert m["validation"]["edges_equal_theoretical_bound"] is True


def test_peak_memory_is_measured_on_this_platform(run):
    _, m, _, _ = run
    assert isinstance(m["peak_memory_bytes"], int) and m["peak_memory_bytes"] > 10 * 2 ** 20     # a Python process with numpy


def test_feature_dictionary_complete_and_nonphysical(run):
    out, _, _, _ = run
    d = json.loads((out / "feature_dictionary.json").read_text(encoding="utf-8"))
    v = Draft202012Validator(json.loads((SCHEMAS / "interaction_feature_dictionary.schema.json").read_text(encoding="utf-8")))
    assert not list(v.iter_errors(d))
    assert [f["name"] for f in d["node_features"]] == list(NODE_FLOAT_FEATURES)
    assert [f["name"] for f in d["edge_features"]] == list(EDGE_FEATURES)
    for f in d["node_features"] + d["edge_features"] + d["edge_flags"]:
        assert f["physical"] is False
    for f in d["node_features"] + d["edge_features"]:
        assert f["unit"] and f["formula"] and f["mask"]


def test_deterministic_output_hashes(tmp_path):
    causal, manifest, p3 = make_task2_run(tmp_path, scene_records())
    m1 = run_interaction_representation(causal, manifest, p3, cfg3(), tmp_path / "o1")
    m2 = run_interaction_representation(causal, manifest, p3, cfg3(), tmp_path / "o2")
    assert {k: v["sha256"] for k, v in m1["outputs"].items()} == {k: v["sha256"] for k, v in m2["outputs"].items()}


def test_input_order_invariance(tmp_path):
    causal, manifest, p3 = make_task2_run(tmp_path, scene_records())
    m1 = run_interaction_representation(causal, manifest, p3, cfg3(), tmp_path / "o1")
    rows = read_rows(causal)
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame_index"], []).append(r)
    rnd = random.Random(7)
    shuffled = []
    for f in sorted(by_frame):                                       # shuffle rows of different tracks within each frame
        g = by_frame[f][:]
        rnd.shuffle(g)
        shuffled.extend(g)
    assert shuffled != rows
    write_rows(causal, shuffled)
    resync(manifest, causal, p3)
    m2 = run_interaction_representation(causal, manifest, p3, cfg3(), tmp_path / "o2")
    keys = [k for k in m1["outputs"] if k.endswith(".npz") or k in ("tracks.json", "interaction_diagnostics.json", "feature_dictionary.json")]
    for k in keys:
        if k == "interaction_diagnostics.json":
            continue                                                 # embeds no input order but compared field-wise below
        assert m1["outputs"][k]["sha256"] == m2["outputs"][k]["sha256"], k
    d1 = json.loads((tmp_path / "o1" / "interaction_diagnostics.json").read_text(encoding="utf-8"))
    d2 = json.loads((tmp_path / "o2" / "interaction_diagnostics.json").read_text(encoding="utf-8"))
    assert d1 == d2


def test_vv_enabled_by_configuration(tmp_path):
    recs = scene_records() + [__import__("task2_helpers").rec(track="8", frame=f, fx=1500.0, fy=800.0, h=150.0, w=200.0, cls="vehicle",
                                                              source_label="truck") for f in range(4)]
    causal, manifest, p3 = make_task2_run(tmp_path, recs)
    m0 = run_interaction_representation(causal, manifest, p3, cfg3(), tmp_path / "o0")
    assert "edges_vehicle__vehicle.npz" not in m0["outputs"] and m0["counts"]["candidate_pairs_not_represented"] == {"vehicle__vehicle": 8}
    m1 = run_interaction_representation(causal, manifest, p3, cfg3(enable_vehicle_vehicle=True), tmp_path / "o1")
    assert m1["counts"]["edges_by_relation"]["vehicle__vehicle"] == 8


def test_source_inputs_unchanged_and_output_policy(run, tmp_path):
    out, m, (causal, manifest, p3), before = run
    assert {p.name: sha(p) for p in (causal, manifest, p3)} == before
    with pytest.raises(OutputDirectoryError):
        run_interaction_representation(causal, manifest, p3, cfg3(), out)                    # non-empty directory refused


def test_cli_exit_codes(tmp_path):
    causal, manifest, p3 = make_task2_run(tmp_path, scene_records())
    script = str(REPO / "scripts" / "run_interaction_representation.py")
    args = ["--causal", str(causal), "--task2-manifest", str(manifest), "--phase3", str(p3), "--config", str(T3_CONFIG)]
    ok = subprocess.run([sys.executable, script] + args + ["--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    bad = subprocess.run([sys.executable, script, "--causal", str(causal.parent / "offline_kinematics.jsonl")] + args[2:] + ["--out", str(tmp_path / "o2")],
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "OFFLINE_INPUT_REFUSED" in bad.stderr and not (tmp_path / "o2").exists()
