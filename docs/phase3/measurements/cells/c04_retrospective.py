# ==============================================================================================
# PART F - RETROSPECTIVE ANALYSIS of what this notebook already recorded   (no new execution, no GPU)
# ==============================================================================================
# The data below were extracted programmatically from the SAVED OUTPUTS of the historical cells of this notebook
# (cell indices and the SHA-256 of the notebook they came from are inside the JSON). Nothing was typed in by hand
# and nothing was estimated. Everything computed here is labelled RETROSPECTIVE.
RETRO_JSON = r'''@@RETRO@@'''
RETRO = json.loads(RETRO_JSON)

def _per_epoch(rows):
    by = {}
    for gb, ep, loss, t in rows:
        by.setdefault(ep, []).append((loss, t))
    out = []
    for ep in sorted(by):
        l = [x[0] for x in by[ep]]; t = [x[1] for x in by[ep]]
        out.append({"epoch": ep, "n_batches": len(t), "mean_batch_loss": sum(l) / len(l), "sum_of_printed_batch_times_s": sum(t), "batch_time": ml.describe(t)})
    return out

RETRO_SUMMARY = {"label": "RETROSPECTIVE (extracted from saved notebook outputs; not a new measurement)", "source_notebook_sha256": RETRO["source_notebook_sha256"],
                 "runs": {}, "test_pass": {}, "not_reconstructable": []}
for key, run in RETRO["training_runs"].items():
    rows = run["batches"]
    times = [r[3] for r in rows]; losses = [r[2] for r in rows]
    s = {"notebook_cell_index": run["cell_index"], "execution_count": run["execution_count"], "command": run["command"], "interpreter_evidence": run["interpreter_evidence"],
         "batches_logged": len(rows), "epochs": sorted({r[1] for r in rows}), "grid_calculation_s_reported": run["grid_calculation_s"],
         "total_training_s_reported": run["total_training_s"], "sum_of_printed_batch_times_s": sum(times),
         "reported_total_minus_sum_of_batch_times_s": (run["total_training_s"] - sum(times)) if run["total_training_s"] is not None else None,
         "batch_time_printed": ml.describe(times), "batch_time_printed_excluding_first": ml.describe(times[1:]), "first_batch_time_s": times[0] if times else None,
         "loss": {"first": losses[0] if losses else None, "last": losses[-1] if losses else None}, "per_epoch": _per_epoch(rows),
         "throughput_batches_per_s_from_total": (len(rows) / run["total_training_s"]) if run["total_training_s"] else None,
         "throughput_sequences_per_s_from_total": (len(rows) * RETRO["batch_size"] / run["total_training_s"]) if run["total_training_s"] else None,
         "throughput_note": "total includes batch creation, plotting and checkpoint saving; batch_size=%d" % RETRO["batch_size"]}
    RETRO_SUMMARY["runs"][key] = s
    with open(os.path.join(MEAS_DIR, "retro_train_batches_%s.csv" % key), "x", newline="", encoding="utf-8") as fh:
        import csv as _csv
        w = _csv.writer(fh); w.writerow(["global_batch", "epoch", "train_loss", "printed_time_per_batch_s"]); w.writerows(rows)

# cross-check of the archived epoch losses (log_curve.txt) against the means of the printed batch losses of each run
ARCH = RETRO["archived_epoch_losses_log_curve"]
RETRO_SUMMARY["archived_log_curve_crosscheck"] = {"archived_epoch_losses": ARCH, "source": RETRO["archived_epoch_losses_source"], "max_abs_diff_to_run_mean_batch_loss": {}}
for key, s in RETRO_SUMMARY["runs"].items():
    if len(s["per_epoch"]) == len(ARCH):
        RETRO_SUMMARY["archived_log_curve_crosscheck"]["max_abs_diff_to_run_mean_batch_loss"][key] = max(abs(e["mean_batch_loss"] - a) for e, a in zip(s["per_epoch"], ARCH))
_d = RETRO_SUMMARY["archived_log_curve_crosscheck"]["max_abs_diff_to_run_mean_batch_loss"]
RETRO_SUMMARY["archived_log_curve_crosscheck"]["closest_run"] = min(_d, key=_d.get) if _d else None
RETRO_SUMMARY["archived_log_curve_crosscheck"]["note"] = ("printed batch losses have 3 decimals, so a difference below ~1e-3 means the archived log_curve.txt was produced by that run; "
                                                         "the other run's checkpoints/log were overwritten (same file names)")

# epoch durations derived from checkpoint modification times (cell 54 output) - an ESTIMATE with ~1 s resolution
mt = RETRO["checkpoint_mtimes_cell54"]
import datetime as _dt
_ts = [_dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in mt["epoch_0_to_4"]]
_gaps = [(_ts[i + 1] - _ts[i]).total_seconds() for i in range(len(_ts) - 1)]
RETRO_SUMMARY["epoch_duration_from_checkpoint_mtimes"] = {"label": "RETROSPECTIVE ESTIMATE", "source": mt["source"], "mtimes": mt["epoch_0_to_4"], "epoch_1_to_4_durations_s": _gaps,
    "epoch_0_duration_s_residual": (RETRO["training_runs"]["five_epoch_cell51_default_python"]["total_training_s"] - sum(_gaps)),
    "caveats": ["mtime resolution is 1 s and Drive/FUSE can delay mtimes", "epoch 0 is the total reported training time minus epochs 1-4 (it includes the first-batch warm-up)",
                "applies to the run whose checkpoints were archived (the cell-51 run, see the archived-log cross-check)"]}

# historical test pass: per-trajectory time (5 samples + all metric computations + selection)
tt = [r[1] for r in RETRO["test_pass"]["per_trajectory"]]
RETRO_SUMMARY["test_pass"] = {"command": RETRO["test_pass"]["command"], "interpreter_evidence": RETRO["test_pass"]["interpreter_evidence"], "trajectories": len(tt),
    "per_trajectory_time_s": ml.describe(tt), "sum_s": sum(tt), "throughput_trajectories_per_s": (len(tt) / sum(tt)) if tt else None,
    "WHAT_THIS_TIME_INCLUDES": "5 sampled rollouts + ground-truth scoring (ADE/FDE/MHD/speed/heading/collisions) + best-of-five selection + result bookkeeping, per trajectory",
    "NOT": "single-prediction inference latency; NOT video or real-time performance", "protocol": RETRO["test_pass"]["protocol"], "reported_metrics_final_summary": RETRO["test_pass"]["reported_metrics"],
    "metric_note": RETRO["test_pass"]["metric_note"]}
with open(os.path.join(MEAS_DIR, "retro_test_trajectory_times.csv"), "x", newline="", encoding="utf-8") as fh:
    import csv as _csv
    w = _csv.writer(fh); w.writerow(["trajectory", "time_s_5samples_plus_scoring"]); w.writerows(RETRO["test_pass"]["per_trajectory"])

RETRO_SUMMARY["environment_findings"] = RETRO["environment_findings"]
RETRO_SUMMARY["not_reconstructable"] = RETRO["not_reconstructable"]
_sha = ml.write_new_json(os.path.join(MEAS_DIR, "retrospective_summary.json"), RETRO_SUMMARY)
print("wrote retrospective_summary.json  sha256 =", _sha)
for key, s in RETRO_SUMMARY["runs"].items():
    print("\n[%s] cell %s | %d batches | total %.2f s | grid %.2f s | sum(batch times) %.2f s" % (key, s["notebook_cell_index"], s["batches_logged"], s["total_training_s_reported"],
          s["grid_calculation_s_reported"], s["sum_of_printed_batch_times_s"]))
    b = s["batch_time_printed"]
    print("   batch time: median %.3f  p90 %.3f  p95 %.3f  max %.3f  | first batch %.3f s" % (b["p50"], b["p90"], b["p95"], b["max"], s["first_batch_time_s"]))
    print("   per-epoch mean loss:", [round(e["mean_batch_loss"], 4) for e in s["per_epoch"]])
print("\narchived log_curve closest run:", RETRO_SUMMARY["archived_log_curve_crosscheck"]["closest_run"], _d)
print("epoch durations from checkpoint mtimes (epochs 1-4):", RETRO_SUMMARY["epoch_duration_from_checkpoint_mtimes"]["epoch_1_to_4_durations_s"])
t = RETRO_SUMMARY["test_pass"]["per_trajectory_time_s"]
print("test per-trajectory time (5 samples + scoring): median %.3f p90 %.3f p95 %.3f p99 %.3f s over %d trajectories" % (t["p50"], t["p90"], t["p95"], t["p99"], t["count"]))
print("\nThis is NOT single-sample inference latency. Not reconstructable from the saved outputs:")
for x in RETRO_SUMMARY["not_reconstructable"]:
    print("  -", x)
