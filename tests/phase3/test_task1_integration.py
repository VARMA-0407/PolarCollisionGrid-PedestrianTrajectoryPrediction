"""Task 1 integration: video -> detector -> real ByteTrack -> structured output -> schema -> visualisation -> benchmark.

All normal tests use the locally generated synthetic video with the MOCK replay detector (plumbing only) and one real-wrapper
test with RANDOMLY INITIALISED weights built locally from the model yaml (no download). None of them says anything about
detection quality. The last test is the optional real-weights/real-video test: it is skipped unless both
``PEDSAFETY_WEIGHTS`` and ``PEDSAFETY_VIDEO`` point to existing files.
"""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("ultralytics")

from pedsafety.errors import ConfigError  # noqa: E402
from pedsafety.perception.benchmark import latency_stats, run_benchmark  # noqa: E402
from pedsafety.perception.config import PerceptionConfig, load_config  # noqa: E402
from pedsafety.perception.export import read_jsonl  # noqa: E402
from pedsafety.perception.runner import build_detector, execute_run, make_run_dir  # noqa: E402
from pedsafety.perception.runtime_errors import VideoSourceError  # noqa: E402
from pedsafety.perception.synthetic_video import ReplayDetector, build_ground_truth, generate_synthetic_video  # noqa: E402
from pedsafety.perception.video_source import OpenCVVideoSource  # noqa: E402
from pedsafety.schema import collect_violations  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    path = str(d / "scene.mp4")
    gt = generate_synthetic_video(path, width=320, height=180, fps=10.0, n_frames=60, fourcc="mp4v")
    return path, gt


def _run(tmp_path, synth, **kw):
    path, gt = synth
    out = make_run_dir(tmp_path, "t")
    cfg = kw.pop("cfg", PerceptionConfig())
    return out, execute_run(cfg, path, out, backend="replay", replay_gt=gt, **kw)


def test_synthetic_ground_truth_is_deterministic_and_well_formed():
    a, b = build_ground_truth(), build_ground_truth()
    assert a == b and a["n_frames"] == 120
    for f in a["frames"]:
        for o in f["objects"]:
            x0, y0, x1, y1 = o["bbox_xyxy"]
            assert 0 <= x0 < x1 <= a["width"] and 0 <= y0 < y1 <= a["height"]


def test_end_to_end_outputs_manifest_and_contracts(tmp_path, synth):
    out, m = _run(tmp_path, synth, save_video=True, scene_id="synthetic_scene")
    assert set(m["outputs"]) == {"tracked_agents.jsonl", "detections.jsonl", "annotated.mp4", "agent_states.jsonl", "run_manifest.json"}
    assert m["is_mock"] is True and m["backend"] == "replay" and m["results"]["frames_skipped"] == 0 and m["results"]["issues_by_code"] == {}
    assert m["components"]["frame_source"]["parameters"]["timestamp_source"] == "frame_index_over_fps"
    assert m["components"]["tracker"]["parameters"]["implementation"] == "ultralytics.trackers.byte_tracker.BYTETracker"
    assert json.loads((out / "run_manifest.json").read_text())["manifest_version"] == m["manifest_version"]
    rows = read_jsonl(str(out / "tracked_agents.jsonl"))
    assert rows and all(r["coordinate_space"] == "image_pixels" and "world_x" not in r for r in rows)
    per_frame = {}
    for r in rows:
        per_frame.setdefault(r["frame_index"], []).append(r["track_id"])
    assert all(len(v) == len(set(v)) for v in per_frame.values())                            # no duplicate active ids in a frame
    ts = {r["frame_index"]: r["timestamp_s"] for r in rows}
    assert all(t == pytest.approx(i / 10.0) for i, t in ts.items())


def test_agent_states_are_schema_valid_image_space_and_marked_mock(tmp_path, synth):
    out, m = _run(tmp_path, synth, scene_id="synthetic_scene")
    recs = read_jsonl(str(out / "agent_states.jsonl"))
    assert recs and m["agent_states"]["is_mock"] is True and "mock" in m["agent_states"]["source_dataset"]
    for r in recs[:50]:
        assert collect_violations(r) == []
        assert r["world_position_valid"] is False and r["world_x"] is None and r["velocity_x"] is None
        assert r["image_coordinate_unit"] == "pixel"


def test_identity_persistence_through_crossing_and_dropout_on_the_synthetic_scene(tmp_path):
    path = str(tmp_path / "s.mp4")
    gt = generate_synthetic_video(path, fourcc="mp4v")
    out = make_run_dir(tmp_path, "id")
    execute_run(PerceptionConfig(), path, out, backend="replay", replay_gt=gt)
    rows = read_jsonl(str(out / "tracked_agents.jsonl"))
    by = {f["frame_index"]: f["objects"] for f in gt["frames"]}
    owner = {}
    for r in rows:
        for o in by[r["frame_index"]]:
            if o["bbox_xyxy"] == r["bbox_xyxy"]:
                owner.setdefault(o["gt_id"], set()).add(r["track_id"])
    assert len(owner[1]) == 1 and len(owner[2]) == 1 and owner[1] != owner[2]               # two pedestrians cross: no ID switch
    assert len(owner[3]) == 1                                                                 # 6-frame detector dropout: same id afterwards
    assert len(owner[5]) == 1 and len(owner[6]) == 1


def test_annotated_video_has_one_frame_per_processed_frame(tmp_path, synth):
    out, m = _run(tmp_path, synth, save_video=True)
    with OpenCVVideoSource(str(out / "annotated.mp4")) as s:
        assert len(list(s.frames())) == m["results"]["frames_processed"]


def test_runs_never_reuse_a_directory(tmp_path):
    first = make_run_dir(tmp_path, "same")
    (first / "marker.txt").write_text("keep", encoding="utf-8")
    try:
        second = make_run_dir(tmp_path, "same")                 # same second, same name: must not hand back the existing directory
    except FileExistsError:
        second = None
    assert second is None or second != first
    assert (first / "marker.txt").read_text(encoding="utf-8") == "keep"


def test_error_handling_missing_video_and_bad_backend(tmp_path, synth):
    with pytest.raises(VideoSourceError):
        execute_run(PerceptionConfig(), str(tmp_path / "nope.mp4"), make_run_dir(tmp_path, "x"), backend="replay", replay_gt=synth[1])
    with pytest.raises(ConfigError):
        build_detector(PerceptionConfig(), "replay", None)
    with pytest.raises(ConfigError):
        build_detector(PerceptionConfig(), "magic", None)


def test_source_is_released_even_when_the_detector_cannot_be_built(tmp_path, synth):
    out = make_run_dir(tmp_path, "rel")
    with pytest.raises(Exception):
        execute_run(PerceptionConfig(), synth[0], out, backend="yolo")                       # default weights path does not exist here
    assert not any(p.name == "run_manifest.json" for p in out.iterdir())


def test_benchmark_separates_passes_and_refuses_claims_for_mocks(synth):
    path, gt = synth
    cfg = PerceptionConfig()
    rep = run_benchmark(lambda: OpenCVVideoSource(path), ReplayDetector(gt), cfg, warmup=0)
    assert rep["frames"] == 60 and rep["decode_only"]["frames"] == 60 and rep["tracking_only"]["frames"] == 60
    assert rep["combined"]["per_frame_wall_incl_decode"]["frames"] == 60
    assert rep["detection_only"]["detect_stage"]["frames"] == 60 and rep["combined"]["track_stage"]["frames"] == 60
    for k in ("mean_ms", "p50_ms", "p95_ms"):
        assert rep["combined"]["per_frame_wall_incl_decode"][k] > 0
    assert rep["supports_performance_claims"] is False                                       # mock detector: never a claim
    assert rep["hardware_declared"] and rep["environment"]["python"] and "scope" in rep
    assert rep["gpu_memory_measured"] is False or rep["environment"]["cuda"]["available"]


def test_latency_stats_arithmetic_and_empty_input():
    s = latency_stats([0.01, 0.02, 0.03, 0.04])
    assert s["mean_ms"] == pytest.approx(25.0) and s["p50_ms"] == pytest.approx(20.0) and s["p95_ms"] == pytest.approx(40.0)
    assert s["processing_fps"] == pytest.approx(40.0)
    assert latency_stats([]) is None


def test_real_wrapper_with_randomly_initialised_weights_plumbing_only(tmp_path, synth):
    """Exercises the real Ultralytics load/predict path on CPU. Random weights: the output means nothing; this checks that
    loading, hashing, the device choice, the manifest and the output files all work."""
    from ultralytics import YOLO
    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    names.update({i: "class_%d" % i for i in range(80) if i not in names})
    m = YOLO("yolo11n.yaml")
    m.model.names = names
    w = tmp_path / "random_init.pt"
    m.save(str(w))
    cfg = PerceptionConfig.from_dict({"detector": {"model_path": str(w), "device": "cpu", "conf": 0.001},
                                      "tracker": {"track_low_thresh": 0.001, "track_high_thresh": 0.001, "new_track_thresh": 0.001},
                                      "video": {"max_frames": 5}})
    out = make_run_dir(tmp_path, "rand")
    man = execute_run(cfg, synth[0], out, backend="yolo", save_video=True)
    p = man["components"]["detector"]["parameters"]
    assert man["is_mock"] is False and p["device"] == "cpu" and len(p["weights_sha256"]) == 64 and man["results"]["frames_seen"] == 5
    assert (out / "annotated.mp4").exists() and (out / "tracked_agents.jsonl").exists()


@pytest.mark.skipif(not (os.environ.get("PEDSAFETY_WEIGHTS") and os.environ.get("PEDSAFETY_VIDEO")
                         and Path(os.environ.get("PEDSAFETY_WEIGHTS", "")).is_file() and Path(os.environ.get("PEDSAFETY_VIDEO", "")).is_file()),
                    reason="optional real-model test: set PEDSAFETY_WEIGHTS and PEDSAFETY_VIDEO to existing files (never downloaded here)")
def test_optional_real_weights_real_video(tmp_path):
    cfg = load_config(REPO / "configs" / "phase3" / "perception_default.json")
    cfg = PerceptionConfig.from_dict({**cfg.to_dict(), "detector": {**cfg.to_dict()["detector"], "model_path": os.environ["PEDSAFETY_WEIGHTS"]},
                                      "video": {"max_frames": 100}})
    out = make_run_dir(tmp_path, "real")
    man = execute_run(cfg, os.environ["PEDSAFETY_VIDEO"], out, backend="yolo", save_video=True)
    r = man["results"]
    assert man["is_mock"] is False and r["frames_seen"] > 0 and r["frames_skipped"] == 0
    rows = read_jsonl(str(out / "tracked_agents.jsonl"))
    for fr in {x["frame_index"] for x in rows}:
        ids = [x["track_id"] for x in rows if x["frame_index"] == fr]
        assert len(ids) == len(set(ids))


def _cli(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(REPO / "scripts" / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_exit_codes_and_no_empty_directory_left_behind(tmp_path, synth, capsys):
    cli = _cli("run_perception")
    root = tmp_path / "out"
    assert cli.main(["--video", str(tmp_path / "missing.mp4"), "--output-root", str(root)]) == 2
    assert cli.main(["--video", synth[0], "--model", str(tmp_path / "missing.pt"), "--output-root", str(root)]) == 2
    err = capsys.readouterr().err
    assert "video file not found" in err and "never downloaded automatically" in err
    assert not root.exists() or list(root.iterdir()) == []


def test_cli_replay_run_succeeds_and_writes_outputs(tmp_path, synth):
    cli = _cli("run_perception")
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(synth[1]), encoding="utf-8")
    assert cli.main(["--video", synth[0], "--backend", "replay", "--replay-gt", str(gt), "--output-root", str(tmp_path / "o"), "--save-video"]) == 0
    run = next((tmp_path / "o").iterdir())
    assert {p.name for p in run.iterdir()} >= {"tracked_agents.jsonl", "run_manifest.json", "annotated.mp4"}
