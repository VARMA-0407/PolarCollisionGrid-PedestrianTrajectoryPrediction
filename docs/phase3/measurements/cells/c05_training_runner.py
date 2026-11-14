# ==============================================================================================
# PART D - MONITORED TRAINING RUNNER (OPTIONAL; does NOT run unless you set the flags in the setup cell)
# ==============================================================================================
# A new run is started in a FRESH working directory with its own Data/ copy and its own Store_Results/, so no existing
# checkpoint, configuration, log or prediction file can be overwritten. train.py itself is run unmodified.
TRAIN_WORKDIR = os.path.join(WORK_ROOT, "training")            # does not exist yet; the runner creates it
RUNNER_PATH = os.path.join(TOOLS_DIR, "monitored_train_runner.py")
TRAIN_PY = PYTHON_FOR_NEW_RUNS or sys.executable
TRAIN_CMD = [TRAIN_PY, RUNNER_PATH, "--repo", REPO, "--workdir", TRAIN_WORKDIR, "--run-id", MEAS_RUN_ID, "--num-epochs", str(TRAIN_EPOCHS_FOR_NEW_RUN)]

_problems = runner_mod.preflight(REPO, TRAIN_WORKDIR)
_store_before = runner_mod.tree_snapshot(STORE)
print("Planned command   :", " ".join(TRAIN_CMD))
print("Interpreter       :", TRAIN_PY, "(recorded again from INSIDE the run)")
print("Pre-flight        :", "OK - isolation can be guaranteed" if not _problems else "PROBLEMS -> the run will NOT start")
for _p in _problems:
    print("   -", _p)
print("Existing Store_Results files seen (will be verified unchanged after any run):", len(_store_before))

TRAINING_OUTPUT_DIR = None
if not RUN_MONITORED_TRAINING:
    print("\nRUN_MONITORED_TRAINING is False: nothing was started. (Set it to True AND set CONFIRM_RUN_ID = %r to run.)" % MEAS_RUN_ID)
elif CONFIRM_RUN_ID != MEAS_RUN_ID:
    print("\nNOT started: CONFIRM_RUN_ID does not equal MEAS_RUN_ID (%r)." % MEAS_RUN_ID)
elif _problems:
    print("\nNOT started: safe isolation could not be guaranteed (see problems above).")
else:
    os.makedirs(WORK_ROOT, exist_ok=True)
    _r = subprocess.run(TRAIN_CMD, check=False)
    print("runner exit code:", _r.returncode)
    _src = os.path.join(TRAIN_WORKDIR, "measurements")
    if os.path.isdir(_src):
        TRAINING_OUTPUT_DIR = os.path.join(MEAS_DIR, "training_measurements")
        shutil.copytree(_src, TRAINING_OUTPUT_DIR)                      # destination does not exist: nothing is overwritten
        print("measurements copied to:", TRAINING_OUTPUT_DIR)
        print("new checkpoints/config stay in:", os.path.join(TRAIN_WORKDIR, "Store_Results"), "(copy them to Drive yourself if you want to keep them)")
        _s = json.load(open(os.path.join(TRAINING_OUTPUT_DIR, "training_resource_summary.json")))
        print("status:", _s["status"], "| isolation:", _s["isolation_check"])
    _store_after = runner_mod.tree_snapshot(STORE)
    print("existing Store_Results unchanged:", _store_before == _store_after)
