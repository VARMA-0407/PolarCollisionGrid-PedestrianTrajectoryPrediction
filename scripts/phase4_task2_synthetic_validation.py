"""Phase 4 Task 2 synthetic validation: smoothing evaluation (raw vs fixed-alpha EMA vs time-aware EMA).

Deterministic (fixed seeds). Synthetic data only: no PIE data, no model, no video. Writes a new/empty output directory:
    smoothing_evaluation.json   metrics per scenario and filter
    run_manifest.json           config, seeds, script hash, output hashes (written last)

Scenarios (all image space, 30 fps nominal, 12 s):
    * constant velocity (90, -30) px/s with Gaussian footpoint noise sigma in {0.5, 1, 2} px
    * the same with irregular sampling (dt drawn uniformly from [0.5, 1.5] x 1/30 s)
    * an abrupt manoeuvre: velocity switches from (90, 0) to (0, 90) px/s at t = 6 s (sigma 1 px), to measure lag

Metrics (after a 2 s warm-up): velocity RMSE [px/s], heading RMSE [rad, wrapped], acceleration RMSE [px/s^2];
for the manoeuvre, the delay until the filtered heading stays within 0.2 rad of the new direction.
These numbers describe the filters on synthetic data only; they are not evidence about PIE footpoint noise.
"""

import argparse
import datetime
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.image_kinematics import ImageKinematicsConfig, process_causal_records  # noqa: E402
from pedsafety.image_kinematics_run import _atomic_write_text, sha256_file  # noqa: E402
from pedsafety.output_dir import check_output_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FILTERS = [{"kind": "none", "alpha": None, "tau_s": None},
           {"kind": "causal_ema", "alpha": 0.3, "tau_s": None}, {"kind": "causal_ema", "alpha": 0.5, "tau_s": None},
           {"kind": "time_aware_ema", "alpha": None, "tau_s": 0.1}, {"kind": "time_aware_ema", "alpha": None, "tau_s": 0.2},
           {"kind": "time_aware_ema", "alpha": None, "tau_s": 0.4}]
WARMUP_S, DURATION_S, FPS = 2.0, 12.0, 30.0


def make_records(times, pos_fn, sigma, seed):
    rnd = random.Random(seed)
    out = []
    for f, t in enumerate(times):
        x, y = pos_fn(t)
        fx, fy = x + rnd.gauss(0, sigma), y + rnd.gauss(0, sigma)
        out.append({"track_id": "synthetic:s:1", "sequence_id": "synthetic:s", "scene_id": "synthetic:s", "agent_class": "pedestrian",
                    "frame_index": f, "timestamp": t, "bbox_xyxy": [fx - 10, fy - 60, fx + 10, fy], "footpoint": [fx, fy], "image_x": fx,
                    "image_y": fy, "image_coordinate_system": "captured_1920x1080", "image_coordinate_unit": "pixel", "world_x": None,
                    "world_y": None, "world_position_valid": False, "observation_valid": True, "interpolation_flag": False})
    return out


def times_uniform():
    return [f / FPS for f in range(int(DURATION_S * FPS))]


def times_irregular(seed):
    rnd, t, out = random.Random(seed), 0.0, []
    while t < DURATION_S:
        out.append(t)
        t += rnd.uniform(0.5, 1.5) / FPS
    return out


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def evaluate(records, cfg_dict, vel_fn, acc_fn):
    cfg = ImageKinematicsConfig.from_dict(dict(cfg_dict))
    rows = process_causal_records(records, list(range(1, len(records) + 1)), cfg).rows
    p = "" if cfg.smoothing.kind == "none" else "filtered_"
    ev, eh, ea = [], [], []
    for r in rows:
        if r["timestamp"] < WARMUP_S:
            continue
        tv = vel_fn(r["timestamp"])
        if r[p + "velocity_state"] == "valid":
            ev.append((r[p + "velocity_x"] - tv[0]) ** 2 + (r[p + "velocity_y"] - tv[1]) ** 2)
        if r[p + "heading_state"] == "valid":
            eh.append(wrap(r[p + "heading"] - math.atan2(tv[1], tv[0])) ** 2)
        if r[p + "acceleration_state"] == "valid":
            ta = acc_fn(r["timestamp"])
            ea.append((r[p + "acceleration_x"] - ta[0]) ** 2 + (r[p + "acceleration_y"] - ta[1]) ** 2)
    rm = lambda e: math.sqrt(sum(e) / len(e)) if e else None  # noqa: E731
    return rows, p, {"n_rows": len(rows), "velocity_rmse_px_per_s": rm(ev), "heading_rmse_rad": rm(eh), "acceleration_rmse_px_per_s2": rm(ea)}


def manoeuvre_lag(rows, p, t_switch, new_heading, tol=0.2):
    after = [r for r in rows if r["timestamp"] >= t_switch and r[p + "heading_state"] == "valid"]
    for k, r in enumerate(after):
        if all(abs(wrap(q[p + "heading"] - new_heading)) <= tol for q in after[k:k + 30]):
            return r["timestamp"] - t_switch
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = check_output_dir(a.out)
    base = json.loads((REPO / "configs" / "phase4" / "image_kinematics_default.json").read_text(encoding="utf-8"))
    base = {k: v for k, v in base.items() if not k.startswith("_")}
    results = {"description": __doc__.strip().splitlines()[0], "warmup_s": WARMUP_S, "duration_s": DURATION_S, "nominal_fps": FPS, "scenarios": []}
    v0 = (90.0, -30.0)
    cv_pos = lambda t: (200 + v0[0] * t, 800 + v0[1] * t)  # noqa: E731
    for sampling in ("uniform", "irregular"):
        for sigma in (0.5, 1.0, 2.0):
            seed = int(1000 * sigma) + (7 if sampling == "irregular" else 0)
            times = times_uniform() if sampling == "uniform" else times_irregular(seed + 1)
            recs = make_records(times, cv_pos, sigma, seed)
            sc = {"scenario": "constant_velocity", "sampling": sampling, "sigma_px": sigma, "seed": seed, "true_velocity_px_per_s": list(v0),
                  "theory_raw_velocity_rmse_px_per_s_uniform": math.sqrt(2) * sigma * FPS * math.sqrt(2) if sampling == "uniform" else None,
                  "filters": []}
            for flt in FILTERS:
                _, _, m = evaluate(recs, dict(base, smoothing=flt), lambda t: v0, lambda t: (0.0, 0.0))
                sc["filters"].append(dict(filter=flt, **m))
            results["scenarios"].append(sc)
    t_sw = 6.0
    man_pos = lambda t: (200 + 90 * min(t, t_sw), 500 + 90 * max(t - t_sw, 0.0))  # noqa: E731
    man_vel = lambda t: (90.0, 0.0) if t < t_sw else (0.0, 90.0)  # noqa: E731
    for sampling in ("uniform", "irregular"):
        times = times_uniform() if sampling == "uniform" else times_irregular(99)
        noisy = make_records(times, man_pos, 1.0, 4242)
        clean = make_records(times, man_pos, 0.0, 4242)        # lag is measured without noise so that noise is not counted as lag
        sc = {"scenario": "abrupt_manoeuvre", "sampling": sampling, "sigma_px_for_rmse": 1.0, "sigma_px_for_lag": 0.0, "seed": 4242,
              "switch_time_s": t_sw, "lag_definition": "time after the switch until the heading stays within 0.2 rad of the new direction",
              "filters": []}
        for flt in FILTERS:
            _, _, m = evaluate(noisy, dict(base, smoothing=flt), man_vel, lambda t: (0.0, 0.0))
            m.pop("acceleration_rmse_px_per_s2")              # acceleration is an impulse at the switch: RMSE is not meaningful
            rows, p, _ = evaluate(clean, dict(base, smoothing=flt), man_vel, lambda t: (0.0, 0.0))
            theory = None if flt["kind"] != "time_aware_ema" else flt["tau_s"] * math.log(1.0 + 1.0 / math.tan(0.2))
            sc["filters"].append(dict(filter=flt, heading_settle_delay_noise_free_s=manoeuvre_lag(rows, p, t_sw, math.pi / 2),
                                      theory_continuous_time_delay_s=theory, **m))
        results["scenarios"].append(sc)
    # theory note: raw backward-difference velocity error per axis = sqrt(2)*sigma/dt, two axes -> x sqrt(2)
    results["notes"] = ["theory_raw_velocity_rmse uses sqrt(2)*sigma/dt per axis and sqrt(2) for the two axes (uniform sampling only)",
                        "fixed-alpha causal_ema smooths per SAMPLE, so its effective time constant changes with the sampling interval; "
                        "time_aware_ema uses alpha_t = 1 - exp(-dt/tau)",
                        "synthetic Gaussian noise only; real footpoint noise is not Gaussian or independent (box jitter, occlusion, truncation)"]
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.mkdir(parents=True, exist_ok=True)
    h = _atomic_write_text(out / "smoothing_evaluation.json", json.dumps(results, indent=2) + "\n")
    man = {"manifest_version": "phase4-task2-synthetic-validation-1.0.0", "created_utc": started, "script": "scripts/phase4_task2_synthetic_validation.py",
           "script_sha256": sha256_file(Path(__file__)), "config_base": "configs/phase4/image_kinematics_default.json",
           "config_base_sha256": sha256_file(REPO / "configs" / "phase4" / "image_kinematics_default.json"), "filters": FILTERS,
           "outputs": {"smoothing_evaluation.json": {"sha256": h}}, "data": "synthetic only"}
    _atomic_write_text(out / "run_manifest.json", json.dumps(man, indent=2) + "\n")
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
