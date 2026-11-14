"""Causal leakage, causal/offline separation, offline central differences and smoothing behaviour."""

import ast
import math
import random

import pytest

from pedsafety.errors import ConfigError
from pedsafety.image_kinematics import (
    CausalImageKinematics, NonCausalRowError, TimeAwareEMA, process_causal_records, require_causal_rows, validate_phase3_record,
)
from pedsafety.image_kinematics_offline import process_offline_records
from pedsafety.image_kinematics_run import load_sidecar_validator
from task2_helpers import REPO, cfg, deep, lines, track

SMOOTHINGS = [{"kind": "none", "alpha": None, "tau_s": None}, {"kind": "causal_ema", "alpha": 0.4, "tau_s": None},
              {"kind": "time_aware_ema", "alpha": None, "tau_s": 0.15}]


def wobbly(n=40, seed=3):
    rnd = random.Random(seed)
    return track([(f, f / 30.0 + (0.004 if f % 7 == 3 else 0.0), 300 + 4 * f + rnd.uniform(-2, 2), 600 - 2 * f + rnd.uniform(-2, 2))
                  for f in range(n) if f not in (17, 18)])


# --------------------------------------------------------------- leakage
@pytest.mark.parametrize("sm", SMOOTHINGS)
def test_changing_future_observations_leaves_earlier_causal_rows_unchanged(sm):
    base = wobbly()
    k = 20
    changed = deep(base)
    for r in changed[k:]:
        r["image_x"] += 250.0
        r["footpoint"][0] += 250.0
        r["bbox_xyxy"][0] += 250.0
        r["bbox_xyxy"][2] += 250.0
    c = cfg(smoothing=sm)
    a = process_causal_records(base, lines(base), c).rows
    b = process_causal_records(changed, lines(changed), c).rows
    assert a[:k] == b[:k]                                        # byte-for-byte identical earlier rows
    assert a[k] != b[k]                                          # the change itself is visible


@pytest.mark.parametrize("sm", SMOOTHINGS)
def test_stream_outputs_are_fixed_before_later_records_exist(sm):
    """Each step is computed and frozen before the next record is even handed to the driver; the batch result must equal it."""
    recs = wobbly()
    c = cfg(smoothing=sm)
    drv = CausalImageKinematics(c)
    frozen = []
    for i, r in enumerate(recs):
        st = drv.update(r, validate_phase3_record(r, c), i)
        frozen.append((st.raw.vx, st.raw.ax, None if st.filtered is None else st.filtered.vx))
    batch = process_causal_records(recs, lines(recs), c)
    again = [(batch.steps[i].raw.vx, batch.steps[i].raw.ax, None if batch.steps[i].filtered is None else batch.steps[i].filtered.vx)
             for i in range(len(recs))]
    assert repr(frozen) == repr(again)


def test_time_aware_ema_update_reads_only_the_current_sample():
    ema = TimeAwareEMA(0.2)
    assert ema.update(0, 0.0, (0.0, 0.0), "id") == (0.0, 0.0)
    y = ema.update(1, 0.1, (10.0, 0.0), "id")
    assert y[0] == pytest.approx(10.0 * (1 - math.exp(-0.5)))   # 3.9347
    assert ema.last_alpha == pytest.approx(1 - math.exp(-0.5))


def test_causal_module_does_not_import_offline_module():
    tree = ast.parse((REPO / "pedsafety" / "image_kinematics.py").read_text(encoding="utf-8"))
    names = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
            {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any("offline" in (m or "") for m in names)


# ------------------------------------------------- causal / offline separation
def test_offline_rows_are_labelled_and_refused_by_causal_guard():
    recs = wobbly()
    c = cfg()
    causal = process_causal_records(recs, lines(recs), c)
    off = process_offline_records(recs, lines(recs), causal, c)
    assert off and all(r["view"] == "OFFLINE_NON_CAUSAL" and r["velocity_method"] == "central_difference_actual_dt" for r in off)
    require_causal_rows(causal.rows)
    with pytest.raises(NonCausalRowError):
        require_causal_rows(causal.rows + off[:1])


def test_offline_row_cannot_be_relabelled_causal():
    recs = wobbly()
    c = cfg()
    causal = process_causal_records(recs, lines(recs), c)
    off = process_offline_records(recs, lines(recs), causal, c)
    v = load_sidecar_validator()
    forged = dict(off[5], view="CAUSAL")                         # central-difference method under a CAUSAL label
    assert list(v.iter_errors(forged))
    with pytest.raises(NonCausalRowError):
        require_causal_rows([dict(off[5], view="CAUSAL")])
    assert not list(v.iter_errors(off[5])) and not list(v.iter_errors(causal.rows[5]))


def test_offline_central_difference_exact_for_quadratic_motion():
    a = 60.0
    recs = track([(f, f / 30.0, 100.0 + 0.5 * a * (f / 30.0) ** 2, 500.0) for f in range(6)])
    c = cfg()
    off = process_offline_records(recs, lines(recs), process_causal_records(recs, lines(recs), c), c)
    assert off[0]["velocity_state"] == "insufficient_neighbours" and off[-1]["velocity_state"] == "insufficient_neighbours"
    for k in range(1, 5):
        assert off[k]["velocity_x"] == pytest.approx(a * k / 30.0, rel=1e-9)        # central difference = v(t_k) on equal spacing
        assert off[k]["acceleration_x"] == pytest.approx(a, rel=1e-6)


def test_offline_irregular_spacing():
    a = 60.0
    ts = [0.0, 0.03, 0.08]
    recs = track([(f, t, 100.0 + 0.5 * a * t * t, 500.0) for f, t in enumerate(ts)])
    c = cfg()
    off = process_offline_records(recs, lines(recs), process_causal_records(recs, lines(recs), c), c)
    assert off[1]["velocity_x"] == pytest.approx(0.5 * a * (0.08 + 0.0), rel=1e-9)  # secant slope 2.4 (not v(0.03)=1.8)
    assert off[1]["acceleration_x"] == pytest.approx(a, rel=1e-6)                   # exact on any spacing
    assert off[1]["dt_s"] == pytest.approx(0.08)


def test_offline_never_bridges_a_gap():
    recs = track([(f, f / 30.0, 100.0 + 3 * f, 500.0) for f in (0, 1, 2, 4, 5, 6)])
    c = cfg()
    off = process_offline_records(recs, lines(recs), process_causal_records(recs, lines(recs), c), c)
    assert [r["velocity_state"] for r in off] == ["insufficient_neighbours", "valid", "gap_adjacent", "gap_adjacent", "valid",
                                                  "insufficient_neighbours"]
    assert off[2]["velocity_x"] is None and off[3]["velocity_x"] is None


# ------------------------------------------------------------- smoothing
def test_smoothing_configuration_validation():
    with pytest.raises(ConfigError):
        cfg(smoothing={"kind": "causal_ema", "alpha": 1.5, "tau_s": None})
    with pytest.raises(ConfigError):
        cfg(smoothing={"kind": "time_aware_ema", "alpha": None, "tau_s": 0.0})
    with pytest.raises(ConfigError):
        cfg(smoothing={"kind": "kalman", "alpha": None, "tau_s": None})
    assert cfg().smoothing.kind == "none"                                           # default: disabled


@pytest.mark.parametrize("sm", SMOOTHINGS[1:])
def test_raw_estimates_are_preserved_when_filtering(sm):
    recs = wobbly()
    raw = process_causal_records(recs, lines(recs), cfg()).rows
    flt = process_causal_records(recs, lines(recs), cfg(smoothing=sm)).rows
    keys = ("position_x", "velocity_x", "velocity_y", "speed", "velocity_state", "acceleration_x", "heading")
    assert [{k: r[k] for k in keys} for r in raw] == [{k: r[k] for k in keys} for r in flt]
    assert any(r["filtered_velocity_x"] is not None for r in flt)
    assert all(r["filtered_velocity_x"] is None for r in raw)


def _step_response(sample_times, smoothing):
    """Position steps from 0 to 10 px right after t=0; returns the filtered position at the last sample."""
    recs = track([(f, t, 100.0 + (0.0 if f == 0 else 10.0), 500.0) for f, t in enumerate(sample_times)])
    rows = process_causal_records(recs, lines(recs), cfg(smoothing=smoothing)).rows
    return rows[-1]["filtered_position_x"] - 100.0


def test_time_aware_ema_is_sampling_invariant_but_fixed_alpha_is_not():
    coarse, fine = [0.0, 0.1, 0.2], [0.0, 0.05, 0.1, 0.15, 0.2]
    ta = {"kind": "time_aware_ema", "alpha": None, "tau_s": 0.2}
    fa = {"kind": "causal_ema", "alpha": 0.5, "tau_s": None}
    expected = 10.0 * (1.0 - math.exp(-0.2 / 0.2))                                  # 6.3212 after 0.2 s, any sampling
    assert _step_response(coarse, ta) == pytest.approx(expected, rel=1e-12)
    assert _step_response(fine, ta) == pytest.approx(expected, rel=1e-12)
    # fixed alpha=0.5 per SAMPLE: 2 updates -> 10*(1-0.25)=7.5 ; 4 updates -> 10*(1-0.0625)=9.375
    assert _step_response(coarse, fa) == pytest.approx(7.5) and _step_response(fine, fa) == pytest.approx(9.375)


def test_filter_restarts_after_a_gap():
    recs = track([(0, 0.0, 100.0, 500.0), (1, 1 / 30, 110.0, 500.0), (3, 3 / 30, 200.0, 500.0)])
    rows = process_causal_records(recs, lines(recs), cfg(smoothing={"kind": "time_aware_ema", "alpha": None, "tau_s": 0.2})).rows
    assert rows[2]["filtered_position_x"] == 200.0 and rows[2]["filtered_velocity_state"] == "gap_before"


def test_noisy_observations_filtering_reduces_velocity_error():
    rnd = random.Random(12345)
    sigma, v, n = 1.0, 90.0, 400
    recs = track([(f, f / 30.0, 200.0 + v * f / 30.0 + rnd.gauss(0, sigma), 500.0 + rnd.gauss(0, sigma)) for f in range(n)])
    raw = process_causal_records(recs, lines(recs), cfg()).rows
    flt = process_causal_records(recs, lines(recs), cfg(smoothing={"kind": "time_aware_ema", "alpha": None, "tau_s": 0.2})).rows
    warm = 60                                                                        # skip filter warm-up (2 s >> tau)
    e_raw = math.sqrt(sum((r["velocity_x"] - v) ** 2 for r in raw[warm:]) / (n - warm))
    e_flt = math.sqrt(sum((r["filtered_velocity_x"] - v) ** 2 for r in flt[warm:]) / (n - warm))
    # theory for raw backward differences: sqrt(2) * sigma / dt = 42.4 px/s
    assert 34.0 < e_raw < 51.0
    assert e_flt < 0.5 * e_raw
