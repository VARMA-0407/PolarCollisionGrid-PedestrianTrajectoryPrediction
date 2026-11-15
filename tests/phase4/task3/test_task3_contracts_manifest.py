"""Remaining contract codes: Task 2 manifest/config checks, causal-file framing, node-level checks and run-time guards."""

import json
import math

import pytest

from pedsafety.interaction import runner as runner_mod
from pedsafety.interaction.config import InteractionContractError
from pedsafety.interaction.input_adapter import load_inputs
from pedsafety.interaction.runner import run_interaction_representation
from task3_helpers import cfg3, make_task2_run, read_rows, resync, scene_records, sha, write_rows


@pytest.fixture()
def t2(tmp_path):
    return make_task2_run(tmp_path, scene_records())


def expect(code, causal, manifest, p3):
    with pytest.raises(InteractionContractError) as ei:
        load_inputs(causal, manifest, p3, cfg3())
    assert ei.value.code == code, str(ei.value)


def edit_manifest(manifest, fn):
    m = json.loads(manifest.read_text(encoding="utf-8"))
    fn(m)
    manifest.write_text(json.dumps(m), encoding="utf-8")


def edit_config(manifest, fn, raw=None):
    p = manifest.parent / "config_resolved.json"
    if raw is not None:
        p.write_text(raw, encoding="utf-8")
    else:
        d = json.loads(p.read_text(encoding="utf-8"))
        fn(d)
        p.write_text(json.dumps(d), encoding="utf-8")
    edit_manifest(manifest, lambda m: m["outputs"]["config_resolved.json"].__setitem__("sha256", sha(p)))


@pytest.mark.parametrize("fn,code", [
    (lambda m: m.__setitem__("status", "failed"), "TASK2_MANIFEST_IDENTITY"),
    (lambda m: m["outputs"].pop("causal_kinematics.jsonl"), "TASK2_MANIFEST_NO_CAUSAL"),
    (lambda m: m["sidecar_schema"].__setitem__("version", "image-kinematics-sidecar-0.9.0"), "SIDECAR_SCHEMA_MISMATCH"),
    (lambda m: m["modes"].__setitem__("offline_rows_in_causal_output", True), "TASK2_MODES"),
    (lambda m: m["conventions"].__setitem__("coordinate_system", "captured_640x360"), "TASK2_CONVENTIONS"),
    (lambda m: m["outputs"]["causal_kinematics.jsonl"].__setitem__("rows", 15), "CAUSAL_ROW_COUNT"),
])
def test_task2_manifest_checks(t2, fn, code):
    causal, manifest, p3 = t2
    edit_manifest(manifest, fn)
    expect(code, causal, manifest, p3)


def test_unreadable_and_bad_manifest(t2):
    causal, manifest, p3 = t2
    manifest.write_text("{not json", encoding="utf-8")
    expect("UNREADABLE_TASK2_MANIFEST", causal, manifest, p3)
    manifest.write_text("[1, 2]", encoding="utf-8")
    expect("BAD_TASK2_MANIFEST", causal, manifest, p3)


def test_missing_input(t2):
    causal, manifest, p3 = t2
    expect("INPUT_MISSING", causal.parent / "absent.jsonl", manifest, p3)


@pytest.mark.parametrize("fn,raw,code", [
    (lambda d: d.__setitem__("max_derivative_gap_frames", 1), None, "TASK2_GAP_BRIDGING"),
    (lambda d: d.__setitem__("expected_image_coordinate_system", "captured_640x360"), None, "TASK2_RASTER"),
    (lambda d: d.__setitem__("frame_step", 0), None, "TASK2_CONFIG"),
    (None, "{broken", "UNREADABLE_TASK2_CONFIG"),
    (None, "[]", "BAD_TASK2_CONFIG"),
])
def test_task2_config_checks(t2, fn, raw, code):
    causal, manifest, p3 = t2
    edit_config(manifest, fn, raw)
    expect(code, causal, manifest, p3)


def test_config_hash_mismatch(t2):
    causal, manifest, p3 = t2
    p = manifest.parent / "config_resolved.json"
    p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")       # changed without updating the manifest
    expect("TASK2_CONFIG_HASH", causal, manifest, p3)


def test_blank_and_malformed_causal_lines(t2):
    causal, manifest, p3 = t2
    text = causal.read_text(encoding="utf-8").splitlines()
    causal.write_text("\n".join(text[:2] + [""] + text[2:]) + "\n", encoding="utf-8")
    resync_hash_only(manifest, causal, len(text))
    expect("CAUSAL_BLANK_LINE", causal, manifest, p3)
    causal.write_text("\n".join(text[:2] + ["{oops"] + text[3:]) + "\n", encoding="utf-8")
    resync_hash_only(manifest, causal, len(text))
    expect("CAUSAL_MALFORMED_LINE", causal, manifest, p3)


def resync_hash_only(manifest, causal, rows):
    edit_manifest(manifest, lambda m: m["outputs"]["causal_kinematics.jsonl"].update(sha256=sha(causal), rows=rows))


def test_unknown_node_type_and_non_finite_position(t2):
    causal, manifest, p3 = t2
    rows = read_rows(causal)
    rows[0]["agent_class"] = "animal"                                         # sidecar schema allows any string here
    write_rows(causal, rows)
    resync(manifest, causal, p3)
    expect("UNKNOWN_NODE_TYPE", causal, manifest, p3)
    rows = read_rows(causal)
    rows[0]["agent_class"] = "pedestrian"
    rows[0]["position_x"] = float("nan")                                      # JSON NaN is a schema 'number'
    write_rows(causal, rows)
    resync(manifest, causal, p3)
    expect("POSITION_NON_FINITE", causal, manifest, p3)


def test_input_changed_during_run_is_detected(t2, tmp_path, monkeypatch):
    causal, manifest, p3 = t2
    real = runner_mod.build_edges

    def tamper(nt, cfg):
        p3.write_text(p3.read_text(encoding="utf-8") + "\n", encoding="utf-8")   # modify a source mid-run
        return real(nt, cfg)
    monkeypatch.setattr(runner_mod, "build_edges", tamper)
    with pytest.raises(InteractionContractError) as ei:
        run_interaction_representation(causal, manifest, p3, cfg3(), tmp_path / "out")
    assert ei.value.code == "INPUT_CHANGED_DURING_RUN"
    assert not (tmp_path / "out" / "run_manifest.json").exists()             # no completion marker


def test_output_schema_guard_blocks_physical_claims(t2, tmp_path, monkeypatch):
    causal, manifest, p3 = t2
    monkeypatch.setattr(runner_mod, "CLAIMS", dict(runner_mod.CLAIMS, physical_features=True))
    with pytest.raises(InteractionContractError) as ei:
        run_interaction_representation(causal, manifest, p3, cfg3(), tmp_path / "out")
    assert ei.value.code == "OUTPUT_SCHEMA_VIOLATION"
    assert not (tmp_path / "out" / "run_manifest.json").exists()
