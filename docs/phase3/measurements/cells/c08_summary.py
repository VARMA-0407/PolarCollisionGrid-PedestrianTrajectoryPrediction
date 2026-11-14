# ==============================================================================================
# OUTPUT INDEX - what this measurement session produced (hashes of every file under MEAS_DIR)
# ==============================================================================================
INDEX = {"measurement_run_id": MEAS_RUN_ID, "created_utc": ml.utc_now_iso(), "output_directory": MEAS_DIR, "files": [],
         "ran_monitored_training": bool(RUN_MONITORED_TRAINING and CONFIRM_RUN_ID == MEAS_RUN_ID),
         "ran_inference_benchmark": bool(RUN_INFERENCE_BENCHMARK),
         "plots_skipped": PLOTS_SKIPPED,
         "still_to_measure": [] }
if not INDEX["ran_monitored_training"]:
    INDEX["still_to_measure"].append("training resources (GPU utilisation/memory, CPU, RAM, per-epoch wall time, CUDA-synchronised batch times): requires a monitored run (Part D)")
if not INDEX["ran_inference_benchmark"]:
    INDEX["still_to_measure"].append("single-sample predictor latency (p50/p90/p95/p99, throughput, peak GPU memory): requires Part E on the GPU runtime")
INDEX["still_to_measure"] += ["end-to-end video latency (detection, tracking, decoding, forecasting): outside this notebook",
                              "non-oracle accuracy (single sample, no future actor information): needs the corrected evaluator (Phase 3)"]
for d, _, fs in os.walk(MEAS_DIR):
    for f in sorted(fs):
        p = os.path.join(d, f)
        if os.path.basename(p) == "measurement_outputs_index.json":
            continue
        INDEX["files"].append({"path": os.path.relpath(p, MEAS_DIR), "bytes": os.path.getsize(p), "sha256": ml.sha256_file(p)})
INDEX["files"].sort(key=lambda r: r["path"])
_sha = ml.write_new_json(os.path.join(MEAS_DIR, "measurement_outputs_index.json"), INDEX)
print("Output directory:", MEAS_DIR)
for r in INDEX["files"]:
    print("  %-62s %10d bytes  %s" % (r["path"], r["bytes"], r["sha256"][:12]))
print("\nStill to measure:")
for x in INDEX["still_to_measure"]:
    print("  -", x)
