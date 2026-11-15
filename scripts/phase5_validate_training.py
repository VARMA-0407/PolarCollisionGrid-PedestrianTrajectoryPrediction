"""Phase 5 P5.7.4 CLI: independently validate a pilot training run (read-only unless --report names a new file).

Checks: every data artifact hash; the run-specific environment record; labels and claims; configuration hashes; the P5.3 window run
(re-verified); the train-only scaler statistics (hash, provenance, partition) and - with --recompute-scaler - an INDEPENDENT numpy
recomputation of every statistic from the collated history batches; the JSONL log (parses, finite, epoch records equal the manifest);
every checkpoint (file hash, canonical state hash, metadata provenance, optimizer/scheduler state, schedule horizon, scaler buffers equal
the statistics); and - with --compare-with - bitwise equality of all data artifacts with a second run (determinism).
Exit codes: 0 all checks pass, 1 a check failed, 2 contract failure, 3 unexpected failure.
"""

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.image_kinematics_run import sha256_file  # noqa: E402
from pedsafety.phase5.models.checkpoint import load_checkpoint  # noqa: E402
from pedsafety.phase5.models.encoder import HISTORY_KEYS, EncoderInput  # noqa: E402
from pedsafety.phase5.models.training_harness import TrainingConfig, load_run_configs, scaler_source_identity  # noqa: E402
from pedsafety.phase5.training_data import TrainingWindowDataset  # noqa: E402
from pedsafety.phase5.training_scaler import TrainOnlyFeatureStatistics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _independent_scaler(ds, n):
    """Plain-numpy recomputation of the derived node/edge statistics (does not use training_scaler's code)."""
    ns, nq, nc = np.zeros(9), np.zeros(9), np.zeros(9)
    es, eq, ec = np.zeros(14), np.zeros(14), np.zeros(14)
    for s in range(0, n, 64):
        b = ds.collate([ds[i] for i in range(s, min(s + 64, n))])
        f, m, p = b["node_features"].astype(np.float64), b["node_masks"], b["present"]
        ref = f[:, -1, 0, 0:2]
        cols = [f[..., 0] - ref[:, None, None, 0], f[..., 1] - ref[:, None, None, 1], f[..., 2], f[..., 3], f[..., 4], f[..., 5], f[..., 6], f[..., 7], f[..., 8]]
        vals = [p, p, m[..., 1] & p, m[..., 1] & p, m[..., 1] & p, m[..., 2] & p, m[..., 2] & p, p, m[..., 3] & p]
        for j, (c, v) in enumerate(zip(cols, vals)):
            x = c[v]
            ns[j] += x.sum()
            nq[j] += (x * x).sum()
            nc[j] += x.size
        for e in b["edges"].values():
            a, mk = e["edge_attr"].astype(np.float64), e["edge_mask"]
            for j in range(14):
                x = a[:, j][mk[:, j]]
                es[j] += x.sum()
                eq[j] += (x * x).sum()
                ec[j] += x.size
    nm, em = ns / nc, es / ec
    nv, ev = np.maximum(nq / nc - nm ** 2, 0), np.maximum(eq / ec - em ** 2, 0)
    return nm, np.where(nv > 0, np.sqrt(nv), 1.0), nc, em, np.where(ev > 0, np.sqrt(ev), 1.0), ec


def validate_run(run_dir, compare_with=None, recompute_scaler=False) -> dict:
    d = Path(run_dir)
    m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    out = {"run": str(d), "checks": {}}

    def chk(name, ok, detail=None):
        out["checks"][name] = {"pass": bool(ok), "detail": detail}
    bad = [n for n, r in m["data_artifacts"].items() if not (d / n).is_file() or sha256_file(d / n) != r["sha256"] or (d / n).stat().st_size != r["bytes"]]
    chk("data_artifact_hashes_match_manifest", not bad, bad or sorted(m["data_artifacts"]))
    env = m["environment_json"]
    chk("environment_record_hash_matches", (d / "environment.json").is_file() and sha256_file(d / "environment.json") == env["sha256"])
    chk("label_pilot_only_and_claims_false", m["label"] == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" and all(v is False for v in m["claims"].values() if isinstance(v, bool)))
    chk("policy_final_epoch_only_no_selection", m["training"]["checkpoint_selection"] == "FINAL_EPOCH_ONLY" and m["claims"]["best_checkpoint_selected"] is False
        and m["claims"]["independent_validation"] is False)
    inp = m["inputs"]
    tcfg = TrainingConfig.load(REPO / inp["training_config"]["path"])
    enc, dec, _, _ = load_run_configs(tcfg)
    chk("config_hashes_match", tcfg.sha256() == inp["training_config"]["sha256_canonical"] and enc.sha256() == inp["encoder_config"]["sha256_canonical"]
        and dec.sha256() == inp["decoder_config"]["sha256_canonical"])
    resolved = json.loads((d / "config_resolved.json").read_text(encoding="utf-8"))
    chk("resolved_config_equals_loaded_configs", resolved["training_config"] == tcfg.to_dict() and resolved["encoder_config"] == enc.to_dict() and resolved["decoder_config"] == dec.to_dict())
    ds = TrainingWindowDataset(REPO / inp["windows_run"])
    chk("window_run_reverified_and_matches", ds.provenance["run_manifest_sha256"] == inp["windows_run_manifest_sha256"] and ds.config.sha256() == inp["window_config_sha256"]
        and len(ds) == inp["windows"])
    stats = TrainOnlyFeatureStatistics.load(d / "scaler_statistics.json", expected_source_identity=scaler_source_identity(ds), expected_split_identity=inp["scaler"]["split_identity"])
    chk("scaler_statistics_provenance_and_hash", stats.compute_hash() == inp["scaler"]["statistics_sha256"] and stats.fit_partition == "train"
        and min(stats.node_valid_count + stats.edge_valid_count) == inp["scaler"]["valid_counts_min"] > 0)
    if recompute_scaler:
        nm, ns_, nc, em, es_, ec = _independent_scaler(ds, len(ds))
        ok = (np.allclose(nm, stats.node_mean, rtol=1e-9, atol=1e-9) and np.allclose(ns_, stats.node_scale, rtol=1e-8, atol=1e-9) and np.array_equal(nc, stats.node_valid_count)
              and np.allclose(em, stats.edge_mean, rtol=1e-9, atol=1e-9) and np.allclose(es_, stats.edge_scale, rtol=1e-8, atol=1e-9) and np.array_equal(ec, stats.edge_valid_count))
        chk("scaler_statistics_independently_recomputed", ok)
    lines = (d / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(l) for l in lines]
    epochs = [r for r in recs if r.get("type") == "epoch"]
    finite = all(isinstance(r.get("loss", r.get("mean_loss")), float) and math.isfinite(r.get("loss", r.get("mean_loss"))) for r in recs)
    chk("log_parses_and_losses_finite", finite and len(epochs) == m["training"]["epochs_run_in_this_directory"], len(recs))
    chk("log_epoch_records_match_manifest", epochs[-1] == m["training"]["final_epoch_summary"] and epochs[0] == m["training"]["first_epoch_summary"])
    chk("log_has_no_wall_clock_fields", not any(k in r for r in recs for k in ("time", "seconds", "runtime", "timestamp")))
    total_steps = m["training"]["total_steps"]
    for name, rec in m["data_artifacts"].items():
        if not name.startswith("checkpoints/"):
            continue
        p = load_checkpoint(d / name, expected_file_sha256=rec["sha256"], expected_model_state_sha256=rec["model_state_sha256"])
        md = p["metadata"]
        okmeta = (md["scaler_statistics_sha256"] == stats.compute_hash() and md["training_config_sha256"] == tcfg.sha256() and md["encoder_config_sha256"] == enc.sha256()
                  and md["decoder_config_sha256"] == dec.sha256() and md["window_config_sha256"] == ds.config.sha256() and md["windows_run_manifest_sha256"] == inp["windows_run_manifest_sha256"]
                  and md["label"] == m["label"] and md["checkpoint_selection"] == "FINAL_EPOCH_ONLY")
        sched = p["scheduler_state"]
        chk("checkpoint_%s" % name.split("/")[-1], okmeta and p["optimizer_state"] is not None and p["epoch"] == rec["epoch"] and p["global_step"] == rec["global_step"]
            and (sched is None or sched["T_max"] == total_steps))
    final = load_checkpoint(d / m["training"]["final_checkpoint"])
    ref = stats.feature_scaler()
    chk("final_checkpoint_scaler_buffers_equal_statistics", all(torch.equal(final["model_state"]["encoder.scaler." + b], getattr(ref, b)) for b in ("node_offset", "node_scale", "edge_offset", "edge_scale")))
    chk("model_input_has_no_future_fields", not any("future" in n or "target" in n for n in [f.name for f in dataclasses.fields(EncoderInput)] + list(HISTORY_KEYS))
        and m["inputs"]["future_targets_in_model_input"] is False and m["inputs"]["offline_kinematics_used"] is False)
    if compare_with:
        o = json.loads((Path(compare_with) / "run_manifest.json").read_text(encoding="utf-8"))
        same = {k: o["data_artifacts"].get(k, {}).get("sha256") == v["sha256"] for k, v in m["data_artifacts"].items()}
        chk("deterministic_vs_second_run_all_data_artifacts_identical", all(same.values()) and set(o["data_artifacts"]) == set(m["data_artifacts"]), same)
    out["ok"] = all(v["pass"] for v in out["checks"].values())
    out["n_checks"] = len(out["checks"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare-with", help="a second training run directory that must be bitwise identical (determinism)")
    ap.add_argument("--recompute-scaler", action="store_true", help="independently recompute the scaler statistics with plain numpy")
    ap.add_argument("--report", help="write the JSON report to this NEW file (refuses to overwrite)")
    a = ap.parse_args(argv)
    try:
        rep = validate_run(a.run, a.compare_with, a.recompute_scaler)
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    if a.report:
        p = Path(a.report)
        if p.exists():
            print("error: report path exists; refusing to overwrite: %s" % p, file=sys.stderr)
            return 2
        p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    failed = [k for k, v in rep["checks"].items() if not v["pass"]]
    print("%s: %d/%d checks passed%s" % ("PASS" if rep["ok"] else "FAIL", rep["n_checks"] - len(failed), rep["n_checks"], "" if not failed else "; failed: %s" % failed))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
