"""Task 1B: YOLO wrapper contract, tested with an injected fake model (no weights, no ultralytics needed)."""

import numpy as np
import pytest

from pedsafety.perception import PerceptionPipeline, PipelineConfig
from pedsafety.perception.config import DetectorSettings, PerceptionConfig
from pedsafety.perception.mocks import ScriptedFrameSource, MockTracker
from pedsafety.perception.runtime_errors import DetectorLoadError
from pedsafety.perception.yolo_detector import UltralyticsDetector, resolve_device

from task1_helpers import FakeYoloModel, frame_meta


def _img(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _det(model, **settings):
    return UltralyticsDetector(DetectorSettings(device="cpu", **settings), model=model)


def test_output_structure_original_labels_ids_boxes_confidence():
    m = FakeYoloModel([[10, 20, 50, 120], [200, 100, 300, 160]], [0.91, 0.55], [0, 2])
    out = _det(m).detect(frame_meta(), _img())
    assert [(d.detection_id, d.label, d.class_id) for d in out] == [(0, "person", 0), (1, "car", 2)]
    assert out[0].bbox_xyxy == (10.0, 20.0, 50.0, 120.0) and all(isinstance(v, float) for v in out[0].bbox_xyxy)
    assert out[0].confidence == pytest.approx(0.91, abs=1e-6) and out[1].confidence == pytest.approx(0.55, abs=1e-6)


def test_empty_detections_return_an_empty_sequence():
    assert list(_det(FakeYoloModel()).detect(frame_meta(), _img())) == []


def test_original_class_is_preserved_and_mapping_is_explicit_and_configurable():
    """bicycle and motorcycle are NOT silently equated: the detector keeps both labels; the label map decides."""
    boxes = [[10, 10, 40, 60], [100, 10, 140, 60], [200, 10, 260, 60]]
    model = FakeYoloModel(boxes, [0.9, 0.9, 0.9], [1, 3, 7])
    det = _det(model)
    raw = det.detect(frame_meta(), _img())
    assert [d.label for d in raw] == ["bicycle", "motorcycle", "truck"]
    src = ScriptedFrameSource([(frame_meta(), _img())])
    res = PerceptionPipeline(src, det, MockTracker()).run()
    cats = [(d.label_raw, d.category.value) for d in res.frames[0].detections]
    assert cats == [("bicycle", "bike"), ("truck", "truck")]                      # motorcycle rejected by default
    assert [i.code.value for i in res.issues] == ["unsupported_class"]
    cfg = PerceptionConfig.from_dict({"detector": {"label_map": {"bicycle": "bike", "motorcycle": "bike", "truck": "truck"}}})
    res2 = PerceptionPipeline(ScriptedFrameSource([(frame_meta(), _img())]), det, MockTracker(), cfg.pipeline_config()).run()
    assert [(d.label_raw, d.category.value) for d in res2.frames[0].detections] == [("bicycle", "bike"), ("motorcycle", "bike"), ("truck", "truck")]


def test_thresholds_classes_and_device_are_passed_to_the_model():
    m = FakeYoloModel()
    _det(m, conf=0.33, iou=0.45, imgsz=320, max_det=50).detect(frame_meta(), _img())
    kw = m.calls[0]
    assert (kw["conf"], kw["iou"], kw["imgsz"], kw["max_det"], kw["device"], kw["classes"]) == (0.33, 0.45, 320, 50, "cpu", [0, 1, 2, 5, 7])
    assert kw["verbose"] is False and "half" not in kw                            # half is only sent when requested on a GPU
    m2 = FakeYoloModel()
    _det(m2, coco_class_ids=None).detect(frame_meta(), _img())
    assert "classes" not in m2.calls[0]


def test_half_is_ignored_on_cpu_and_sent_on_cuda():
    m = FakeYoloModel()
    _det(m, half=True).detect(frame_meta(), _img())
    assert "half" not in m.calls[0]
    g = UltralyticsDetector(DetectorSettings(device="cuda", half=True), model=FakeYoloModel(), cuda_available=True)
    g.detect(frame_meta(), _img())
    assert g._model.calls[0]["half"] is True and g._model.calls[0]["device"] == "cuda:0"


@pytest.mark.parametrize("payload", [None, np.zeros((480, 640), np.uint8), np.zeros((480, 640, 4), np.uint8), np.zeros((100, 100, 3), np.uint8)])
def test_bad_payload_raises(payload):
    with pytest.raises(ValueError):
        _det(FakeYoloModel()).detect(frame_meta(), payload)


def test_pipeline_converts_a_detector_exception_into_an_issue():
    src = ScriptedFrameSource([(frame_meta(), None)])
    res = PerceptionPipeline(src, _det(FakeYoloModel()), MockTracker()).run()
    assert [i.code.value for i in res.issues] == ["detector_error"] and not res.frames[0].processed


def test_invalid_model_path_is_a_clear_error_and_nothing_is_downloaded(tmp_path):
    with pytest.raises(DetectorLoadError, match="never downloaded"):
        UltralyticsDetector(DetectorSettings(model_path=str(tmp_path / "missing.pt"), device="cpu"))
    with pytest.raises(DetectorLoadError, match="never downloaded"):
        UltralyticsDetector(DetectorSettings(model_path="yolo11n.pt", device="cpu"))   # a bare name must not trigger a download


def test_model_without_class_names_or_with_unknown_class_ids_is_rejected():
    with pytest.raises(DetectorLoadError):
        UltralyticsDetector(DetectorSettings(device="cpu"), model=FakeYoloModel(names={}))
    with pytest.raises(DetectorLoadError, match="class ids"):
        UltralyticsDetector(DetectorSettings(device="cpu", coco_class_ids=(0, 99)), model=FakeYoloModel())


def test_unreadable_weights_file_is_wrapped(tmp_path):
    pytest.importorskip("ultralytics")
    bad = tmp_path / "broken.pt"
    bad.write_bytes(b"not a model")
    with pytest.raises(DetectorLoadError):
        UltralyticsDetector(DetectorSettings(model_path=str(bad), device="cpu"))


@pytest.mark.parametrize("requested,cuda,expected", [("cpu", True, "cpu"), ("cpu", False, "cpu"), ("auto", False, "cpu"),
                                                     ("auto", True, "cuda:0"), ("cuda", True, "cuda:0")])
def test_device_selection(requested, cuda, expected):
    assert resolve_device(requested, cuda) == expected


def test_cuda_request_without_cuda_fails_loudly_and_unknown_device_is_rejected():
    with pytest.raises(DetectorLoadError, match="not available"):
        resolve_device("cuda", False)
    with pytest.raises(DetectorLoadError):
        resolve_device("tpu", False)


def test_component_info_records_parameters_and_marks_injected_models_as_mock():
    d = _det(FakeYoloModel(), conf=0.2)
    info = d.info
    assert info.kind == "detector" and info.is_mock is True                         # a fake model can never back a claim
    p = dict(info.parameters)
    assert p["conf"] == "0.2" and p["device"] == "cpu" and p["class_ids"] == "0,1,2,5,7" and p["weights_sha256"] == "n/a"


def test_real_weights_file_hash_is_recorded(tmp_path):
    """Hash recording only: a file that Ultralytics cannot load never gets this far, so use an injected loader substitute."""
    from pedsafety.perception.yolo_detector import file_sha256
    f = tmp_path / "w.bin"
    f.write_bytes(b"abc")
    assert file_sha256(str(f)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
