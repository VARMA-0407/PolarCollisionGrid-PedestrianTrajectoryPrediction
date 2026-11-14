# ==============================================================================================
# MEASUREMENT ENHANCEMENT - SETUP (appended; nothing above this cell was modified)
# No training, no test.py run and no artifact write happens unless a flag below is set to True.
# ==============================================================================================
import os, sys, json, time, pickle, platform, subprocess, shutil, hashlib, datetime, pathlib

# ---- switches (all OFF) -------------------------------------------------------------------------
RUN_MONITORED_TRAINING = False   # Part D: starts a NEW isolated training run (GPU, minutes). Leave False unless intended.
CONFIRM_RUN_ID = ""              # to start training ALSO set this to the exact MEAS_RUN_ID printed below
RUN_INFERENCE_BENCHMARK = False  # Part E: inference-only timing of the epoch-4 checkpoint (isolated, read-only on artifacts)
TRAIN_EPOCHS_FOR_NEW_RUN = 5
PYTHON_FOR_NEW_RUNS = None       # None -> this kernel's interpreter. For comparability with the README use "/content/pcg-env/bin/python"
BENCH_WARMUP = 20                # warm-up trajectories excluded from timing
BENCH_MAX_TRAJECTORIES = None    # None -> all 1081 test sequences
BENCH_SEED = 12345

# ---- locations used by the historical notebook (read only here) ---------------------------------
REPO = "/content/PolarCollisionGrid-PedestrianTrajectoryPrediction"
DRIVE_BACKUP = "/content/drive/MyDrive/PCG_Experiments/PCG_T4_5_Epoch_Run"
STORE = os.path.join(REPO, "Store_Results")

# ---- fresh output directory for THIS measurement session -----------------------------------------
MEAS_RUN_ID = "meas_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_PARENT = "/content/drive/MyDrive/PCG_Experiments/measurements" if os.path.isdir("/content/drive/MyDrive") else "/content/pcg_measurements"
MEAS_DIR = os.path.join(OUT_PARENT, MEAS_RUN_ID)
WORK_ROOT = os.path.join("/content/pcg_runs", MEAS_RUN_ID)       # local disk (fast) for any new run's working directories
os.makedirs(OUT_PARENT, exist_ok=True)
os.makedirs(MEAS_DIR, exist_ok=False)                             # fails if it exists: never reuse or overwrite
TOOLS_DIR = os.path.join(MEAS_DIR, "tools")
os.makedirs(TOOLS_DIR, exist_ok=False)

# ---- helper sources (embedded so this notebook is self-contained; written exclusively, never over an existing file)
HELPER_SOURCES = @@HELPERS@@

HELPER_SHA256 = {}
for _name, _src in HELPER_SOURCES.items():
    _p = os.path.join(TOOLS_DIR, _name)
    with open(_p, "x", encoding="utf-8", newline="\n") as _f:
        _f.write(_src)
    HELPER_SHA256[_name] = hashlib.sha256(_src.encode("utf-8")).hexdigest()
sys.path.insert(0, TOOLS_DIR)
import measurement_lib as ml
import monitored_train_runner as runner_mod
import inference_benchmark as bench_mod

print("Measurement run id :", MEAS_RUN_ID)
print("Output directory   :", MEAS_DIR)
print("Kernel interpreter :", sys.executable, "| Python", platform.python_version())
print("Repository present :", os.path.isdir(REPO), "| Drive backup present:", os.path.isdir(DRIVE_BACKUP))
print("Helper SHA-256     :", json.dumps(HELPER_SHA256, indent=1))
print("\nSwitches: RUN_MONITORED_TRAINING =", RUN_MONITORED_TRAINING, "| RUN_INFERENCE_BENCHMARK =", RUN_INFERENCE_BENCHMARK)
