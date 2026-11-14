# ==============================================================================================
# PART E - SINGLE-SAMPLE PREDICTOR BENCHMARK (OPTIONAL; runs only if RUN_INFERENCE_BENCHMARK = True)
# ==============================================================================================
# SCOPE: PCG trajectory-PREDICTOR-ONLY, sample_size = 1. NOT end-to-end video performance and NOT deployment-valid:
# the baseline path still consumes future actor information, and the historical test.py runs with dropout active (net.eval() is never called).
# Why a separate benchmark: the historical test.py timing (cell 60) covers 5 sampled rollouts + ground-truth scoring + best-of-five oracle
# selection per trajectory, so it cannot be read as the cost of ONE prediction. test.py is neither modified nor run here; the baseline
# `sample` function is imported from it unchanged and its per-trajectory preparation steps are replicated in the benchmark module.
# Gradient tracking: the baseline sample() already runs under torch.no_grad(); the benchmark adds NO other context, never calls net.eval(),
# and records the observed grad mode and the model's train/eval state in the summary (gradient_tracking, model_train_eval_state).
BENCH_WORKDIR = os.path.join(WORK_ROOT, "inference")
_ck_candidates = [os.path.join(DRIVE_BACKUP, "SOCIALLSTM_lstm_model_4.tar"), os.path.join(STORE, "model", "SOCIALLSTM_lstm_model_4.tar")]
_cf_candidates = [os.path.join(DRIVE_BACKUP, "config.pkl"), os.path.join(STORE, "model", "config.pkl")]
BENCH_CKPT = next((p for p in _ck_candidates if os.path.isfile(p)), _ck_candidates[0])
BENCH_CONFIG = next((p for p in _cf_candidates if os.path.isfile(p)), _cf_candidates[0])
EXPECTED_CKPT = PREV["ckpt4"][0]
EXPECTED_CONFIG = PREV["experiment_config"][0]
_bp = bench_mod.preflight(REPO, BENCH_WORKDIR, BENCH_CKPT, BENCH_CONFIG)
print("checkpoint :", BENCH_CKPT)
print("config     :", BENCH_CONFIG)
_scope = bench_mod.sample_grad_scope(os.path.join(REPO, "test.py"))
print("%s:" % "Gradient scope of the baseline sample()", _scope["verdict"])
if not (_scope.get("uses_no_grad") and _scope.get("net_calls_inside_guard") == _scope.get("net_calls_total")):
    print("   WARNING: the baseline sample() is not fully under torch.no_grad(); the benchmark will record gradient tracking as ENABLED where it is.")
print("Pre-flight :", "OK" if not _bp else "PROBLEMS -> benchmark will NOT start")
for _p in _bp:
    print("   -", _p)

BENCH_SUMMARY = None
BENCH_OUTPUT_DIR = None
if not RUN_INFERENCE_BENCHMARK:
    print("\nRUN_INFERENCE_BENCHMARK is False: nothing was run.")
elif _bp:
    print("\nNOT started: safe isolation could not be guaranteed.")
else:
    import torch
    _hw = "%s | %s | torch %s" % (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only", platform.platform(), torch.__version__)
    os.makedirs(WORK_ROOT, exist_ok=True)
    # Run in THIS kernel (its interpreter and torch are what get recorded). To benchmark under /content/pcg-env instead, run
    #   <that python> <MEAS_DIR>/tools/inference_benchmark.py --repo ... --workdir ... --run-id ... --checkpoint ... --config ... --epoch 4
    BENCH_SUMMARY = bench_mod.run_benchmark(REPO, BENCH_WORKDIR, MEAS_RUN_ID, BENCH_CKPT, BENCH_CONFIG, epoch=4, warmup=BENCH_WARMUP,
                                            max_trajectories=BENCH_MAX_TRAJECTORIES, seed=BENCH_SEED, expected_checkpoint_sha256=EXPECTED_CKPT,
                                            expected_config_sha256=EXPECTED_CONFIG, hardware_label=_hw)
    BENCH_OUTPUT_DIR = os.path.join(MEAS_DIR, "inference_benchmark")
    shutil.copytree(os.path.join(BENCH_WORKDIR, "measurements"), BENCH_OUTPUT_DIR)
    L = BENCH_SUMMARY["latency_s"]
    print("\nSCOPE:", BENCH_SUMMARY["scope"])
    print("hashes match previously recorded: checkpoint =", BENCH_SUMMARY["hashes"]["checkpoint_matches_expected"], "| config =", BENCH_SUMMARY["hashes"]["config_matches_expected"])
    for k in ("pipeline", "model_forward_only_sum_per_trajectory"):
        d = L[k]
        print("%-42s p50 %.4f  p90 %.4f  p95 %.4f  p99 %.4f s  (n=%d)" % (k, d["p50_s"], d["p90_s"], d["p95_s"], d["p99_s"], d["count"]))
    print("throughput: %.2f sequences/s | peak GPU memory (MiB): %s" % (BENCH_SUMMARY["throughput_sequences_per_s"]["pipeline_pass"], BENCH_SUMMARY["peak_gpu_memory_mib"]))
    _G, _S = BENCH_SUMMARY["gradient_tracking"], BENCH_SUMMARY["model_train_eval_state"]
    print("gradient tracking:", _G["conclusion"])
    print("model state after run: net.training =", _S["after_all_passes"]["net_training"], "| dropout modules in training mode =",
          _S["after_all_passes"]["dropout_modules_in_training_mode"], "| unchanged during benchmark =", _S["unchanged_during_benchmark"],
          "| eval called by benchmark =", _S["eval_called_by_benchmark"])
    print("copied to:", BENCH_OUTPUT_DIR)
