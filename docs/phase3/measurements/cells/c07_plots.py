# ==============================================================================================
# PART G - VISUALISATIONS (six separate figures; a figure is only drawn when its data exist)
# ==============================================================================================
import csv as _csv
import matplotlib
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(MEAS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=False)
PLOTS_WRITTEN, PLOTS_SKIPPED = [], {}
RUN_LABELS = {"five_epoch_cell49_pcg_env": "cell 49 run (pcg-env, overwritten later)", "five_epoch_cell51_default_python": "cell 51 run (default python, ARCHIVED)"}

def _save(fig, name):
    p = os.path.join(PLOT_DIR, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    PLOTS_WRITTEN.append(name)

def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(_csv.DictReader(fh))

def _num(rows, key):
    out = []
    for r in rows:
        try:
            v = float(r[key])
            if v == v:
                out.append(v)
        except (KeyError, TypeError, ValueError):
            pass
    return out

NEW_TRAIN = TRAINING_OUTPUT_DIR if "TRAINING_OUTPUT_DIR" in globals() and TRAINING_OUTPUT_DIR else None
NEW_BENCH = BENCH_OUTPUT_DIR if "BENCH_OUTPUT_DIR" in globals() and BENCH_OUTPUT_DIR else None

# 1. training loss over batches --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
for key, run in RETRO["training_runs"].items():
    if key.startswith("five_epoch"):
        ax.plot([r[0] for r in run["batches"]], [r[2] for r in run["batches"]], lw=0.7, label="RETROSPECTIVE " + RUN_LABELS.get(key, key))
if NEW_TRAIN and os.path.isfile(os.path.join(NEW_TRAIN, "training_batch_log.csv")):
    rows = _read_csv(os.path.join(NEW_TRAIN, "training_batch_log.csv"))
    ax.plot(_num(rows, "global_batch")[:len(_num(rows, "train_loss"))], _num(rows, "train_loss"), lw=0.7, label="NEW monitored run")
ax.set_xlabel("global batch"); ax.set_ylabel("train loss (negative log-likelihood)"); ax.set_title("Training loss over batches (238 batches per epoch)"); ax.legend(fontsize=8)
_save(fig, "01_training_loss_over_batches.png")

# 2. batch-time distribution ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
for key, run in RETRO["training_runs"].items():
    if key.startswith("five_epoch"):
        ax.hist([r[3] for r in run["batches"]], bins=60, alpha=0.5, label="RETROSPECTIVE " + RUN_LABELS.get(key, key) + " (printed time/batch)")
if NEW_TRAIN and os.path.isfile(os.path.join(NEW_TRAIN, "training_batch_log.csv")):
    ax.hist(_num(_read_csv(os.path.join(NEW_TRAIN, "training_batch_log.csv")), "sync_batch_time_s"), bins=60, alpha=0.5, label="NEW run (CUDA-synchronised)")
ax.set_xlabel("seconds per batch"); ax.set_ylabel("batches"); ax.set_title("Training batch-time distribution"); ax.legend(fontsize=8)
_save(fig, "02_batch_time_distribution.png")

# 3. per-epoch duration -------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
w = 0.25; shown = 0
for i, (key, s) in enumerate(RETRO_SUMMARY["runs"].items()):
    if key.startswith("five_epoch"):
        ax.bar([e["epoch"] + (shown - 0.5) * w for e in s["per_epoch"]], [e["sum_of_printed_batch_times_s"] for e in s["per_epoch"]], width=w, label="RETRO sum of printed batch times: " + RUN_LABELS.get(key, key))
        shown += 1
_mt = RETRO_SUMMARY["epoch_duration_from_checkpoint_mtimes"]
ax.scatter([1, 2, 3, 4], _mt["epoch_1_to_4_durations_s"], color="k", zorder=3, label="RETRO estimate from checkpoint mtimes (epochs 1-4, ~1 s resolution)")
if NEW_TRAIN and os.path.isfile(os.path.join(NEW_TRAIN, "training_epoch_summary.csv")):
    er = _read_csv(os.path.join(NEW_TRAIN, "training_epoch_summary.csv"))
    ax.scatter(_num(er, "epoch"), _num(er, "epoch_wall_s"), marker="D", color="r", zorder=4, label="NEW run epoch wall-clock")
ax.set_xlabel("epoch"); ax.set_ylabel("seconds"); ax.set_title("Per-epoch training duration")
ax.set_ylim(0, ax.get_ylim()[1] * 1.5); ax.legend(fontsize=7, loc="upper center")
_save(fig, "03_per_epoch_duration.png")

# 4/5. GPU utilisation and memory over time: ONLY from a new monitored run's telemetry -----------------------
_rl = os.path.join(NEW_TRAIN, "training_resource_log.csv") if NEW_TRAIN else None
if _rl and os.path.isfile(_rl):
    rows = _read_csv(_rl)
    t = _num(rows, "t_s")
    util = _num(rows, "gpu_util_pct")
    if util:
        fig, ax = plt.subplots(figsize=(10, 3.5)); ax.plot([float(r["t_s"]) for r in rows if r.get("gpu_util_pct") not in (None, "")], util, lw=0.8)
        ax.set_xlabel("seconds since run start"); ax.set_ylabel("GPU utilisation (%)"); ax.set_title("GPU utilisation during training (device-wide samples)")
        _save(fig, "04_gpu_utilization_over_time.png")
    else:
        PLOTS_SKIPPED["04_gpu_utilization_over_time"] = "telemetry file has no GPU utilisation values"
    mem = _num(rows, "gpu_mem_used_mib")
    if mem or _num(rows, "torch_allocated_mib"):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for col, lab in (("gpu_mem_used_mib", "device memory used (nvidia-smi/NVML)"), ("torch_allocated_mib", "torch allocated"), ("torch_reserved_mib", "torch reserved")):
            ys = [(float(r["t_s"]), float(r[col])) for r in rows if r.get(col) not in (None, "")]
            if ys:
                ax.plot([a for a, _ in ys], [b for _, b in ys], lw=0.8, label=lab)
        ax.set_xlabel("seconds since run start"); ax.set_ylabel("MiB"); ax.set_title("GPU memory during training"); ax.legend(fontsize=8)
        _save(fig, "05_gpu_memory_over_time.png")
    else:
        PLOTS_SKIPPED["05_gpu_memory_over_time"] = "telemetry file has no memory values"
else:
    PLOTS_SKIPPED["04_gpu_utilization_over_time"] = "no GPU telemetry exists (the historical run recorded none); requires a NEW monitored run"
    PLOTS_SKIPPED["05_gpu_memory_over_time"] = "no GPU memory telemetry exists; requires a NEW monitored run"

# 6. inference latency distribution -----------------------------------------------------------------------------
if NEW_BENCH and os.path.isfile(os.path.join(NEW_BENCH, "inference_latency_samples.csv")):
    rows = _read_csv(os.path.join(NEW_BENCH, "inference_latency_samples.csv"))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(_num(rows, "pipeline_s"), bins=60, alpha=0.6, label="predictor pipeline, 1 sample (NEW)")
    ax.hist(_num(rows, "model_forward_s"), bins=60, alpha=0.6, label="model forwards only (NEW)")
    ax.set_xlabel("seconds per trajectory"); ax.set_ylabel("trajectories"); ax.set_title("PCG predictor-only latency (NOT end-to-end video)"); ax.legend(fontsize=8)
    _save(fig, "06_inference_latency_distribution.png")
else:
    PLOTS_SKIPPED["06_inference_latency_distribution"] = "no single-sample benchmark has been run (set RUN_INFERENCE_BENCHMARK = True)"
# the historical per-trajectory test time is plotted separately and labelled, because it is NOT inference latency
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist([r[1] for r in RETRO["test_pass"]["per_trajectory"]], bins=60)
ax.set_xlabel("seconds per trajectory"); ax.set_ylabel("trajectories")
ax.set_title("RETROSPECTIVE historical test time per trajectory\n(5 samples + ground-truth scoring + best-of-5 oracle selection; NOT single-prediction latency)", fontsize=9)
_save(fig, "06b_retro_test_time_NOT_inference_latency.png")

print("plots written:", PLOTS_WRITTEN)
print("plots skipped (no data):")
for k, v in PLOTS_SKIPPED.items():
    print("  -", k, ":", v)
