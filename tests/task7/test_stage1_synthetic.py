import dataclasses

import pytest

from pedsafety.io.csv_adapter import CsvAnnotationAdapter
from pedsafety.io.synthetic import (
    AgentSpec, default_specs, generate_scene, synthetic_csv_config, write_scene_csv,
)


def test_deterministic_same_seed_and_noise():
    a = generate_scene(seed=3, position_noise_px=1.5)
    b = generate_scene(seed=3, position_noise_px=1.5)
    c = generate_scene(seed=4, position_noise_px=1.5)
    assert a.observations == b.observations
    assert a.observations != c.observations


def test_zero_noise_is_seed_independent_and_matches_analytic_truth():
    a, b = generate_scene(seed=1), generate_scene(seed=99)
    assert a.observations == b.observations
    t = 7 / a.fps
    o = next(o for o in a.observations if o.track_id == "ped_accel" and o.frame_index == 7)
    assert o.pos_x == pytest.approx(400.0 + 0.5 * 5.0 * t * t)
    assert o.pos_y == pytest.approx(350.0 - 10.0 * t + 0.5 * 2.0 * t * t)
    assert o.vel_x == pytest.approx(5.0 * t) and o.vel_y == pytest.approx(-10.0 + 2.0 * t)
    assert o.timestamp == t


def test_noise_stream_independent_of_dropped_frames():
    with_gap = generate_scene(seed=5, position_noise_px=2.0)
    spec = [dataclasses.replace(s, missing_frames=frozenset()) for s in default_specs()]
    full = generate_scene(seed=5, position_noise_px=2.0, specs=tuple(spec))
    idx = {(o.track_id, o.frame_index): o for o in full.observations}
    for o in with_gap.observations:
        assert idx[(o.track_id, o.frame_index)].pos_x == o.pos_x


def test_gaps_and_lifespans_respected():
    s = generate_scene()
    frames = {}
    for o in s.observations:
        frames.setdefault(o.track_id, []).append(o.frame_index)
    assert not set(range(10, 14)) & set(frames["ped_gap"])
    assert min(frames["ped_late_early"]) == 5 and max(frames["ped_late_early"]) == 24
    assert max(frames["ped_const"]) == 39
    assert all(f == sorted(f) for f in frames.values())
    assert len(s.truth) == len(s.observations)


def test_box_is_footpoint_anchored_and_well_ordered():
    for o in generate_scene(position_noise_px=3.0).observations:
        x0, y0, x1, y1 = o.bbox_xyxy
        assert x0 < x1 and y0 < y1
        assert (x0 + x1) / 2 == pytest.approx(o.pos_x) and y1 == o.pos_y
        assert (o.position_semantics, o.position_unit) == ("image_pixel", "pixel")


def test_csv_round_trip_is_exact(tmp_path):
    s = generate_scene(seed=2, position_noise_px=0.7)
    p = write_scene_csv(s, tmp_path / "s.csv")
    r = CsvAnnotationAdapter(p, synthetic_csv_config()).parse()
    assert r.issues == []
    got = [dataclasses.replace(o, source_line=0, extras=()) for o in r.observations]
    assert got == s.observations


def test_custom_specs_and_input_validation():
    s = generate_scene(n_frames=3, specs=(AgentSpec("a", "pedestrian", (0, 0), (1, 1)),))
    assert len(s.observations) == 3
    for bad in (dict(n_frames=0), dict(fps=0), dict(position_noise_px=-1)):
        with pytest.raises(ValueError):
            generate_scene(**bad)


@pytest.mark.parametrize("kw,match", [
    (dict(fps=float("nan")), "fps"),
    (dict(fps=float("inf")), "fps"),
    (dict(fps=-10.0), "fps"),
    (dict(fps="10"), "fps"),
    (dict(fps=True), "fps"),
    (dict(position_noise_px=float("nan")), "position_noise_px"),
    (dict(position_noise_px=float("inf")), "position_noise_px"),
    (dict(position_noise_px=-0.1), "position_noise_px"),
    (dict(confidence=float("nan")), "confidence"),
    (dict(confidence=float("inf")), "confidence"),
    (dict(confidence=1.5), "confidence"),
    (dict(confidence=-0.1), "confidence"),
    (dict(n_frames=2.5), "n_frames"),
    (dict(n_frames=True), "n_frames"),
    (dict(n_frames=0), "n_frames"),
])
def test_invalid_generator_inputs_name_the_argument(kw, match):
    with pytest.raises(ValueError, match=match):
        generate_scene(**kw)


def test_boundary_confidence_and_int_fps_accepted():
    for conf in (0.0, 1.0):
        assert generate_scene(n_frames=2, confidence=conf).observations[0].detection_confidence == conf
    assert generate_scene(n_frames=2, fps=10).fps == 10.0
    assert generate_scene(n_frames=2, fps=10).observations[0].position_coordinate_system == "synthetic_image"
