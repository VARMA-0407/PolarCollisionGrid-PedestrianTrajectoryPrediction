# ==============================================================================================
# PART B - EXPERIMENT FINGERPRINT  ->  experiment_environment.json   (read-only on the experiment)
# ==============================================================================================
import re, glob, random

def _run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else "ERROR(%d): %s" % (r.returncode, (r.stderr or r.stdout).strip()[:300])
    except Exception as exc:
        return "UNAVAILABLE: %s" % exc

def _version(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception as exc:
        return "not importable: %s" % type(exc).__name__

def _probe_other_interpreter(py):
    """Ask ANOTHER interpreter (e.g. the pcg-env venv) what it really is. Read-only."""
    if not py or not os.path.isfile(py):
        return {"path": py, "exists": False}
    code = ("import sys,json,platform\nd={'executable':sys.executable,'python':sys.version,'platform':platform.platform()}\n"
            "for m in ('torch','torchvision','numpy','pandas','scipy','sklearn','matplotlib','psutil'):\n"
            "    try:\n        mod=__import__(m); d[m]=getattr(mod,'__version__','unknown')\n    except Exception as e:\n        d[m]='not importable: '+type(e).__name__\n"
            "try:\n    import torch; d['cuda_available']=torch.cuda.is_available(); d['torch_cuda']=torch.version.cuda\n"
            "except Exception: pass\nprint(json.dumps(d))")
    out = _run([py, "-c", code], timeout=120)
    try:
        d = json.loads(out.splitlines()[-1]); d["path"] = py; d["exists"] = True
        return d
    except Exception:
        return {"path": py, "exists": True, "probe_output": out[:500]}

def _cfg_dict(path):
    if not os.path.isfile(path):
        return {"path": path, "exists": False}
    with open(path, "rb") as fh:
        ns = pickle.load(fh)                      # a Namespace written by train.py (our own artifact)
    return {"path": path, "exists": True, "sha256": ml.sha256_file(path), "args": {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v)) for k, v in vars(ns).items()}}

def collect_fingerprint():
    fp = {"fingerprint_version": "pcg-experiment-fingerprint-1.0", "measurement_run_id": MEAS_RUN_ID, "collected_utc": ml.utc_now_iso()}
    # --- interpreters (the historical notebook used two)
    fp["kernel_interpreter"] = {"executable": sys.executable, "python": sys.version, "implementation": platform.python_implementation()}
    fp["pcg_env_interpreter"] = _probe_other_interpreter("/content/pcg-env/bin/python")
    fp["interpreter_for_new_runs"] = PYTHON_FOR_NEW_RUNS or sys.executable
    # --- OS / hardware
    osr = {}
    try:
        with open("/etc/os-release") as fh:
            osr = dict(l.strip().split("=", 1) for l in fh if "=" in l)
    except Exception:
        pass
    fp["os"] = {"platform": platform.platform(), "system": platform.system(), "release": platform.release(), "os_release": {k: v.strip('"') for k, v in osr.items() if k in ("PRETTY_NAME", "VERSION_ID")}}
    cpu = _run(["lscpu"])
    fp["cpu"] = {"logical_cores": os.cpu_count(), "lscpu_model_name": next((l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.startswith("Model name")), None), "platform_processor": platform.processor()}
    try:
        import psutil
        vm = psutil.virtual_memory()
        fp["ram"] = {"total_gib": round(vm.total / 2 ** 30, 2), "available_gib_at_collection": round(vm.available / 2 ** 30, 2)}
    except Exception as exc:
        fp["ram"] = {"unavailable": str(exc)}
    # --- libraries as seen by the KERNEL interpreter
    libs = {m: _version(m) for m in ("torch", "torchvision", "numpy", "pandas", "scipy", "sklearn", "matplotlib", "psutil")}
    fp["kernel_libraries"] = libs
    try:
        import torch
        fp["cuda"] = {"cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda, "cudnn_version": (torch.backends.cudnn.version() if torch.cuda.is_available() else None)}
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            fp["gpu"] = {"name": p.name, "total_vram_mib": p.total_memory // 2 ** 20, "compute_capability": "%d.%d" % (p.major, p.minor), "multiprocessors": p.multi_processor_count}
        fp["determinism_settings_in_kernel"] = {"cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                                  "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled()}
    except Exception as exc:
        fp["cuda"] = {"error": str(exc)}
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    fp["nvidia_smi_query"] = smi
    fp["gpu_driver_version"] = smi.split(",")[-1].strip() if smi and not smi.startswith(("ERROR", "UNAVAILABLE")) else None
    # --- git
    fp["git"] = {"commit": _run(["git", "-C", REPO, "rev-parse", "HEAD"]), "branch": _run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"]),
                 "status_porcelain": _run(["git", "-C", REPO, "status", "--porcelain"]).splitlines()}
    fp["git"]["working_tree_clean"] = (len(fp["git"]["status_porcelain"]) == 0)
    # --- training configuration (the NEW run's config and the ORIGINAL CollisionGrid config)
    fp["training_configurations"] = {"training_run_config": _cfg_dict(os.path.join(STORE, "model", "config.pkl")),
                                     "collisiongrid_config_in_repo": _cfg_dict(os.path.join(STORE, "model", "CollisionGrid", "config.pkl")),
                                     "drive_backup_config": _cfg_dict(os.path.join(DRIVE_BACKUP, "config.pkl")),
                                     "drive_backup_original_collisiongrid_config": _cfg_dict(os.path.join(DRIVE_BACKUP, "CollisionGrid_original_config.pkl"))}
    # --- datasets, splits
    data_files = ["Data/HBS/hbs.csv", "Data/train/trajectories.cpkl", "Data/train/trajectories_train.cpkl", "Data/test/trajectories_test.cpkl"]
    fp["datasets"] = [ml.file_record(os.path.join(REPO, f), label=f) for f in data_files]
    fp["split_files"] = {"train_split": "Data/train/trajectories_train.cpkl", "test_split": "Data/test/trajectories_test.cpkl",
                         "note": "the historical loader has no separate validation split (see Phase 2 audit)"}
    # --- source files that define the experiment
    src_files = ["train.py", "test.py", "utils/DataLoader.py", "utils/helper.py", "utils/Interaction.py", "utils/grid.py", "model/model_CollisionGrid.py", "visualization.py", "requirements.txt"]
    fp["source_files"] = [ml.file_record(os.path.join(REPO, f), label=f) for f in src_files]
    # --- seeds: look for seeding in the code instead of assuming
    hits = []
    for f in glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True):
        try:
            for n, line in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
                if re.search(r"manual_seed|np\.random\.seed|random\.seed|set_seed|cudnn\.deterministic|use_deterministic_algorithms", line):
                    hits.append("%s:%d: %s" % (os.path.relpath(f, REPO), n, line.strip()[:120]))
        except OSError:
            pass
    fp["seeds"] = {"seed_calls_found_in_repository_python_files": hits, "conclusion": ("no seeding or determinism call found: the historical training and test were NOT seeded"
                   if not hits else "seeding calls exist; inspect them"), "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "seed_used_by_this_notebook": None,
                   "note": "new instrumented runs record any seed they set; the monitored training runner sets none (it runs unmodified train.py)"}
    fp["helpers_sha256"] = HELPER_SHA256
    return fp

FINGERPRINT = collect_fingerprint()
_sha = ml.write_new_json(os.path.join(MEAS_DIR, "experiment_environment.json"), FINGERPRINT)
print("wrote experiment_environment.json  sha256 =", _sha)
print("kernel   :", FINGERPRINT["kernel_interpreter"]["executable"], "| torch", FINGERPRINT["kernel_libraries"]["torch"])
print("pcg-env  :", {k: FINGERPRINT["pcg_env_interpreter"].get(k) for k in ("exists", "executable", "torch", "torch_cuda")})
print("git      :", FINGERPRINT["git"]["commit"], "| clean:", FINGERPRINT["git"]["working_tree_clean"])
print("seeds    :", FINGERPRINT["seeds"]["conclusion"])
