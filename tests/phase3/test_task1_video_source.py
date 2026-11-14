"""Task 1A: OpenCV video ingestion, on tiny locally generated videos (skipped when OpenCV is not installed)."""

import pytest

cv2 = pytest.importorskip("cv2")

from pedsafety.perception.runtime_errors import VideoSourceError  # noqa: E402
from pedsafety.perception.video_source import OpenCVVideoSource  # noqa: E402

from task1_helpers import write_test_video  # noqa: E402


@pytest.fixture()
def video(tmp_path):
    p = tmp_path / "clip.avi"
    write_test_video(str(p), 64, 48, 10.0, 12)
    return str(p)


def test_valid_input_properties(video):
    s = OpenCVVideoSource(video)
    assert (s.width, s.height) == (64, 48) and s.fps == pytest.approx(10.0)
    assert s.frame_count in (None, 12) and s.coordinate_system == "captured_64x48"
    assert s.info.kind == "frame_source" and s.info.is_mock is False
    s.close()


def test_frame_indices_timestamps_and_end_of_video(video):
    with OpenCVVideoSource(video) as s:
        got = [(m.frame_index, m.timestamp_s, m.width, m.height, m.timestamp_source, p.shape) for m, p in s.frames()]
    assert [g[0] for g in got] == list(range(12))                                   # monotonically increasing, from 0, none skipped
    assert [g[1] for g in got] == pytest.approx([i / 10.0 for i in range(12)])      # frame_index / fps
    assert all(g[4] == "frame_index_over_fps" and g[5] == (48, 64, 3) for g in got)
    assert s.frames_delivered == 12 and s.stats["frames_missing_vs_advisory_count"] in (None, 0)
    assert len(s.decode_latencies_s) == 12 and all(x >= 0 for x in s.decode_latencies_s)


def test_resource_is_released_after_exhaustion_early_exit_and_context_exit(video):
    s = OpenCVVideoSource(video)
    list(s.frames())
    assert s.released
    s2 = OpenCVVideoSource(video)
    it = s2.frames()
    next(it)
    assert not s2.released
    it.close()                                                                       # consumer stops early
    assert s2.released
    with OpenCVVideoSource(video) as s3:
        assert not s3.released
    assert s3.released
    s3.close()                                                                       # idempotent


def test_source_is_one_shot(video):
    s = OpenCVVideoSource(video)
    list(s.frames())
    with pytest.raises(VideoSourceError, match="one-shot"):
        list(s.frames())


def test_max_frames_limits_without_changing_timing(video):
    with OpenCVVideoSource(video, max_frames=5) as s:
        metas = [m for m, _ in s.frames()]
    assert [m.frame_index for m in metas] == [0, 1, 2, 3, 4] and metas[-1].timestamp_s == pytest.approx(0.4)


@pytest.mark.parametrize("w,h,fps", [(64, 48, 10.0), (96, 64, 25.0), (32, 32, 5.0), (128, 72, 30.0)])
def test_different_resolutions_and_frame_rates_are_preserved_not_resampled(tmp_path, w, h, fps):
    p = str(tmp_path / "v.avi")
    write_test_video(p, w, h, fps, 6)
    with OpenCVVideoSource(p) as s:
        metas = [m for m, _ in s.frames()]
    assert len(metas) == 6 and all((m.width, m.height) == (w, h) for m in metas)
    assert metas[1].timestamp_s == pytest.approx(1.0 / fps) and metas[0].nominal_fps == pytest.approx(fps)


def test_invalid_paths_and_files(tmp_path):
    with pytest.raises(VideoSourceError, match="not found"):
        OpenCVVideoSource(str(tmp_path / "missing.mp4"))
    with pytest.raises(VideoSourceError, match="not found"):
        OpenCVVideoSource(str(tmp_path))                                              # a directory is not a video
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 10)
    with pytest.raises(VideoSourceError, match="could not open"):
        OpenCVVideoSource(str(junk))


@pytest.mark.parametrize("kw", [{"max_frames": 0}, {"max_frames": True}, {"fps_override": 0}, {"fps_override": float("nan")}, {"fps_override": -5}])
def test_invalid_arguments_are_rejected(video, kw):
    with pytest.raises(VideoSourceError):
        OpenCVVideoSource(video, **kw)


def test_fps_override_is_used_and_labelled(video):
    with OpenCVVideoSource(video, fps_override=20.0) as s:
        m = [m for m, _ in s.frames()]
    assert m[1].timestamp_s == pytest.approx(0.05) and dict(s.info.parameters)["fps_origin"] == "config_override"


def test_truncated_video_ends_cleanly_and_reports_shortfall(tmp_path):
    p = tmp_path / "t.avi"
    write_test_video(str(p), 64, 48, 10.0, 12)
    data = p.read_bytes()
    p.write_bytes(data[: len(data) * 3 // 5])                                         # chop the file mid-stream
    try:
        s = OpenCVVideoSource(str(p))
    except VideoSourceError:
        return                                                                        # OpenCV may refuse a truncated index: also clean
    with s:
        n = len(list(s.frames()))
    assert 0 <= n <= 12 and s.released
