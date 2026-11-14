"""Task 1: configuration validation, the additive ``class_id`` field, and import independence. Stdlib/numpy only."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pedsafety.errors import ConfigError
from pedsafety.perception import PerceptionPipeline, RawDetection, DetectionRecord
from pedsafety.perception.config import PerceptionConfig, load_config
from pedsafety.perception.mocks import MockDetector, MockTracker, ScriptedFrameSource, synthetic_frames
from pedsafety.perception.types import detection_problem

REPO = Path(__file__).resolve().parents[2]
DEFAULT_JSON = REPO / "configs" / "phase3" / "perception_default.json"


def test_default_json_loads_and_matches_dataclass_defaults():
    cfg = load_config(DEFAULT_JSON)
    assert cfg.to_dict() == PerceptionConfig().to_dict()
    assert cfg.warnings() == []
    assert cfg.detector.coco_class_ids == (0, 1, 2, 5, 7)           # motorcycle (3) is not selected
    assert "motorcycle" not in cfg.detector.label_map


def test_none_returns_defaults():
    assert load_config(None) == PerceptionConfig()


@pytest.mark.parametrize("patch", [
    {"detector": {"conf": 0.0}}, {"detector": {"conf": 1.5}}, {"detector": {"iou": 0}}, {"detector": {"device": "tpu"}},
    {"detector": {"imgsz": 10}}, {"detector": {"coco_class_ids": []}}, {"detector": {"coco_class_ids": [-1]}},
    {"detector": {"label_map": {"person": "alien"}}}, {"detector": {"half": "yes"}}, {"detector": {"model_path": ""}},
    {"tracker": {"track_buffer": -1}}, {"tracker": {"match_thresh": 0}}, {"tracker": {"track_low_thresh": 0.9, "track_high_thresh": 0.2}},
    {"tracker": {"box_source": "centre"}}, {"tracker": {"class_policy": "majority"}}, {"pipeline": {"out_of_frame": "wrap"}},
    {"pipeline": {"bbox_tolerance_px": -1}}, {"video": {"max_frames": 0}}, {"video": {"fps_override": 0}}, {"video": {"sequence_id": ""}},
    {"detector": {"unknown_key": 1}}, {"surprise": {}}, {"tracker": {"track_buffer": True}}, {"detector": {"conf": float("nan")}},
])
def test_invalid_configuration_is_rejected(patch):
    with pytest.raises(ConfigError):
        PerceptionConfig.from_dict(patch)


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"config_version": "other"}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(wrong)


def test_consistency_warnings():
    cfg = PerceptionConfig.from_dict({"detector": {"conf": 0.5}})
    assert any("track_low_thresh" in w for w in cfg.warnings())
    assert any("half" in w for w in PerceptionConfig.from_dict({"detector": {"half": True, "device": "cpu"}}).warnings())


def test_pipeline_config_uses_the_configured_label_map():
    cfg = PerceptionConfig.from_dict({"detector": {"label_map": {"person": "pedestrian", "motorcycle": "bike"}}})
    lm = cfg.pipeline_config().label_map
    assert lm.resolve("motorcycle").value == "bike"                  # an explicit, configured choice
    assert load_config(DEFAULT_JSON).pipeline_config().label_map.resolve("motorcycle") is None


# ---------------------------------------------------------------- additive class_id
def test_class_id_is_optional_and_backward_compatible():
    d = RawDetection(0, "person", (1.0, 1.0, 5.0, 9.0), 0.8)          # old four-argument form still works
    assert d.class_id is None and detection_problem(d) is None


@pytest.mark.parametrize("bad", [True, 1.5, "0"])
def test_class_id_type_is_validated(bad):
    assert detection_problem(RawDetection(0, "person", (1.0, 1.0, 5.0, 9.0), 0.8, bad)) is not None


def test_class_id_reaches_the_detection_record():
    frames = list(synthetic_frames(2, width=100, height=100))
    dets = {0: [RawDetection(0, "person", (10.0, 10.0, 30.0, 60.0), 0.9, 0)], 1: [RawDetection(0, "car", (50.0, 50.0, 90.0, 80.0), 0.8, 2)]}
    pipe = PerceptionPipeline(ScriptedFrameSource(frames), MockDetector(dets), MockTracker())
    res = pipe.run()
    got = [(d.label_raw, d.class_id) for f in res.frames for d in f.detections]
    assert got == [("person", 0), ("car", 2)]


def test_detection_record_default_class_id():
    assert DetectionRecord.__dataclass_fields__["class_id"].default is None


# ---------------------------------------------------------------- import independence
def test_config_export_and_errors_import_without_heavy_libraries():
    code = ("import sys\n"
            "import pedsafety.perception, pedsafety.perception.config, pedsafety.perception.export, pedsafety.perception.runtime_errors\n"
            "bad = [m for m in ('cv2', 'ultralytics', 'torch', 'torchvision', 'lap', 'lapx', 'filterpy') if m in sys.modules]\n"
            "print(','.join(bad))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
