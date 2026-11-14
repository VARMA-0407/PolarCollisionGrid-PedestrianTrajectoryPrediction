"""Monitored, ISOLATED training runner for the PCG baseline (notebook companion; train.py is NOT modified).

What it does: runs the unmodified repository ``train.py`` in-process inside a FRESH working directory (its own copy of ``Data/``
and its own ``Store_Results/``), while recording per-batch, per-epoch and resource measurements. It never runs by itself:
the notebook only calls it when the user sets an explicit flag and types the run id.

Isolation (train.py writes ``Store_Results/...`` and reads ``Data/...`` relative to the current directory, so a fresh
directory is what keeps existing checkpoints, configs, logs and results safe):
* the work directory must not exist or must be empty, must not be inside the repository, and gets its own ``Data/`` copy;
* no path of the original run is used; the repository's ``Store_Results`` tree is snapshotted before and verified after;
* a pre-flight returns a list of problems; any problem means the run is NOT started.

Measurement definitions (also written into the summary):
* batch time (sync) = CUDA-synchronised wall time from the start of a batch (``Module.zero_grad`` call) to the end of
  ``optimizer.step`` of the same batch. Data slicing before zero_grad, plotting, printing and checkpoint saving are outside it.
* epoch wall time = time between consecutive "epoch beginning" lines (the last epoch ends at the final training-time line); it
  includes batch creation, plotting and the checkpoint save.
* data loading/preprocessing = timed ``DataLoader.__init__`` (filtering), ``grid_creation`` and each epoch's ``batch_creater``.
* GPU utilisation / used memory = sampled from NVML or ``nvidia-smi`` every ``interval_s`` seconds (device-wide, not per process);
  allocated / reserved / peak memory = PyTorch allocator counters of THIS process.
* peak RAM = max process RSS among samples AND (Linux) the kernel's ``ru_maxrss``.
"""

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

RUNNER_VERSION = "pcg-monitored-train-runner-1.0"
DEFAULT_REQUIRED = ("train.py", "utils/DataLoader.py", "Data/train/trajectories_train.cpkl")
DATA_FILES = ("Data/HBS/hbs.csv", "Data/train/trajectories.cpkl", "Data/train/trajectories_train.cpkl", "Data/test/trajectories_test.cpkl")
SOURCE_FILES = ("train.py", "test.py", "utils/DataLoader.py", "utils/helper.py", "utils/Interaction.py", "utils/grid.py", "visualization.py")


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pct(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s), max(1, int(math.ceil(q / 100.0 * len(s))))) - 1]


def stats(values: Sequence[float]) -> Dict[str, Any]:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not v:
        return {"count": 0}
    m = sum(v) / len(v)
    return {"count": len(v), "mean": m, "std": math.sqrt(sum((x - m) ** 2 for x in v) / len(v)), "min": min(v), "p50": _pct(v, 50),
            "p90": _pct(v, 90), "p95": _pct(v, 95), "max": max(v), "percentile_method": "nearest_rank"}


def tree_snapshot(root: str) -> Dict[str, List[Any]]:
    """path -> [size, mtime_ns] for every file under root (used to prove the original run directory was not touched)."""
    snap: Dict[str, List[Any]] = {}
    if os.path.isdir(root):
        for d, _, fs in os.walk(root):
            for f in fs:
                p = os.path.join(d, f)
                try:
                    st = os.stat(p)
                    snap[os.path.relpath(p, root)] = [st.st_size, st.st_mtime_ns]
                except OSError:
                    snap[os.path.relpath(p, root)] = None
    return snap


def preflight(repo: str, workdir: str, required_files: Sequence[str] = DEFAULT_REQUIRED) -> List[str]:
    """Return a list of problems. An empty list means isolation can be guaranteed; otherwise the run must not start."""
    problems: List[str] = []
    repo_r, work_r = os.path.realpath(repo), os.path.realpath(workdir)
    for rel in required_files:
        if not os.path.isfile(os.path.join(repo, rel)):
            problems.append("repository file missing: %s" % rel)
    if work_r == repo_r or work_r.startswith(repo_r + os.sep):
        problems.append("work directory must be OUTSIDE the repository (it is %s)" % work_r)
    if os.path.islink(workdir):
        problems.append("work directory is a symbolic link: %s" % workdir)
    if os.path.exists(workdir):
        if not os.path.isdir(workdir):
            problems.append("work directory path exists and is not a directory: %s" % workdir)
        elif os.listdir(workdir):
            problems.append("work directory is not empty (a checkpoint or log could be overwritten): %s" % workdir)
    anc = work_r
    while anc and not os.path.exists(anc) and os.path.dirname(anc) != anc:
        anc = os.path.dirname(anc)
    if os.path.exists(anc) and not os.path.isdir(anc):
        problems.append("an ancestor of the work directory is a file: %s" % anc)
    return problems


# --------------------------------------------------------------------------- GPU telemetry
class GpuProbe:
    """GPU utilisation and device memory. Tries NVML, then nvidia-smi. Reports its source honestly."""

    def __init__(self):
        self.source = "unavailable"
        self._nvml = None
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            self.source = "pynvml"
        except Exception:
            self._nvml = None
            try:
                subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5, check=True)
                self.source = "nvidia-smi"
            except Exception:
                self.source = "unavailable"

    def read(self):
        """(utilization_percent, memory_used_mib, memory_total_mib) or (None, None, None)."""
        try:
            if self._nvml is not None:
                nv, h = self._nvml
                u = nv.nvmlDeviceGetUtilizationRates(h)
                m = nv.nvmlDeviceGetMemoryInfo(h)
                return float(u.gpu), m.used / 2 ** 20, m.total / 2 ** 20
            if self.source == "nvidia-smi":
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                a, b, c = [float(x) for x in out.split(",")]
                return a, b, c
        except Exception:
            pass
        return None, None, None


def gpu_driver_version() -> Optional[str]:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=5).stdout.strip().splitlines()[0]
    except Exception:
        return None


# ---------------------------------------------------------------------------- the monitor
class Tee:
    def __init__(self, original, path, on_line):
        self._o, self._f, self._cb, self._buf = original, open(path, "x", encoding="utf-8"), on_line, ""

    def write(self, s):
        self._o.write(s)
        self._f.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._cb(line)
        return len(s)

    def flush(self):
        self._o.flush()
        self._f.flush()

    def close(self):
        if self._buf:
            self._cb(self._buf)
        self._f.close()

    def __getattr__(self, name):
        return getattr(self._o, name)


class Monitor:
    def __init__(self, torch_mod, interval_s: float = 1.0, batch_size: int = 10):
        self.torch, self.interval_s, self.batch_size = torch_mod, interval_s, batch_size
        self.cuda = bool(torch_mod.cuda.is_available())
        self.t0 = time.perf_counter()
        self.samples: List[Dict[str, Any]] = []
        self.batches: List[Dict[str, Any]] = []
        self.phases: List[Dict[str, Any]] = []
        self.markers: List[Dict[str, Any]] = []
        self.reported: List[Dict[str, Any]] = []
        self.save_times: List[float] = []
        self._batch_t0: Optional[float] = None
        self._epoch = -1
        self._global_batch = -1
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.gpu = GpuProbe()
        try:
            import psutil  # type: ignore
            self.psutil, self.proc = psutil, psutil.Process(os.getpid())
            self.proc.cpu_percent(None)
            psutil.cpu_percent(None)
        except Exception:
            self.psutil = self.proc = None

    # -- timing primitives
    def _sync(self):
        if self.cuda:
            self.torch.cuda.synchronize()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def mark(self, kind: str, **kw):
        self.markers.append(dict(kind=kind, t_s=self.now(), **kw))

    # -- stdout parsing
    def on_line(self, line: str):
        import re
        m = re.match(r"^\s*(\d+)/(\d+)\s+\(epoch\s+(\d+)\),\s*train_loss\s*=\s*(-?[\d.]+),\s*time/batch\s*=\s*([\d.]+)\s*$", line)
        if m:
            self.reported.append({"global_batch": int(m.group(1)), "epoch": int(m.group(3)), "train_loss": float(m.group(4)),
                                  "reported_time_per_batch_s": float(m.group(5)), "t_s": self.now()})
            return
        if "Training epoch beginning" in line:
            self._epoch += 1
            self.mark("epoch_begin", epoch=self._epoch)
        elif line.strip() == "Saving model":
            self.mark("saving_model", epoch=self._epoch)
        elif "The whole trainig time" in line:
            self.mark("train_done")

    # -- hooks
    def patched_zero_grad(self, orig):
        mon = self

        def zero_grad(module, *a, **k):
            mon._sync()
            mon._batch_t0 = time.perf_counter()
            return orig(module, *a, **k)
        return zero_grad

    def patched_step(self, orig):
        mon = self

        def step(opt, *a, **k):
            r = orig(opt, *a, **k)
            mon._sync()
            t1 = time.perf_counter()
            if mon._batch_t0 is not None:
                mon._global_batch += 1
                mon.batches.append({"global_batch": mon._global_batch, "epoch": max(mon._epoch, 0), "sync_batch_time_s": t1 - mon._batch_t0,
                                    "t_end_s": t1 - mon.t0})
                mon._batch_t0 = None
            return r
        return step

    def timed(self, name: str, orig):
        mon = self

        def wrapper(*a, **k):
            mon._sync()
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                mon._sync()
                mon.phases.append({"phase": name, "epoch": max(mon._epoch, 0), "duration_s": time.perf_counter() - t, "t_start_s": t - mon.t0})
        return wrapper

    def timed_save(self, orig):
        mon = self

        def save(*a, **k):
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                mon.save_times.append(time.perf_counter() - t)
        return save

    # -- sampler thread
    def start_sampler(self):
        def loop():
            while not self._stop.is_set():
                self.samples.append(self.sample())
                self._stop.wait(self.interval_s)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_sampler(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.samples.append(self.sample())

    def sample(self) -> Dict[str, Any]:
        u, used, total = self.gpu.read()
        mib = 2 ** 20
        row: Dict[str, Any] = {"t_s": round(self.now(), 3), "utc": _utc(), "epoch": max(self._epoch, 0), "global_batch": self._global_batch,
                               "gpu_util_pct": u, "gpu_mem_used_mib": used, "gpu_mem_total_mib": total}
        if self.cuda:
            row["torch_allocated_mib"] = self.torch.cuda.memory_allocated() / mib
            row["torch_reserved_mib"] = self.torch.cuda.memory_reserved() / mib
            row["torch_max_allocated_mib"] = self.torch.cuda.max_memory_allocated() / mib
        else:
            row["torch_allocated_mib"] = row["torch_reserved_mib"] = row["torch_max_allocated_mib"] = None
        if self.psutil is not None:
            row["cpu_pct_system"] = self.psutil.cpu_percent(None)
            row["cpu_pct_process"] = self.proc.cpu_percent(None)
            row["process_rss_mib"] = self.proc.memory_info().rss / mib
            vm = self.psutil.virtual_memory()
            row["ram_used_mib"], row["ram_total_mib"] = (vm.total - vm.available) / mib, vm.total / mib
        else:
            row["cpu_pct_system"] = row["cpu_pct_process"] = row["process_rss_mib"] = row["ram_used_mib"] = row["ram_total_mib"] = None
        return row


# ------------------------------------------------------------------------------- summary
def build_epoch_rows(mon: Monitor) -> List[Dict[str, Any]]:
    rows = []
    begins = [m for m in mon.markers if m["kind"] == "epoch_begin"]
    done = [m for m in mon.markers if m["kind"] == "train_done"]
    end_of_run = done[-1]["t_s"] if done else (mon.batches[-1]["t_end_s"] if mon.batches else None)
    by_epoch: Dict[int, List[Dict[str, Any]]] = {}
    for b in mon.batches:
        by_epoch.setdefault(b["epoch"], []).append(b)
    rep: Dict[int, List[Dict[str, Any]]] = {}
    for r in mon.reported:
        rep.setdefault(r["epoch"], []).append(r)
    for i, bg in enumerate(begins):
        e = bg["epoch"]
        end = begins[i + 1]["t_s"] if i + 1 < len(begins) else end_of_run
        bt = [b["sync_batch_time_s"] for b in by_epoch.get(e, [])]
        losses = [r["train_loss"] for r in rep.get(e, [])]
        wall = (end - bg["t_s"]) if end is not None else None
        compute = sum(bt)
        prep = sum(p["duration_s"] for p in mon.phases if p["phase"] == "batch_creater" and p["epoch"] == e)
        s = stats(bt)
        rows.append({"epoch": e, "n_batches": len(bt), "epoch_wall_s": wall, "sum_batch_time_s": compute, "batch_creater_s": prep,
                     "batch_time_p50_s": s.get("p50"), "batch_time_p90_s": s.get("p90"), "batch_time_p95_s": s.get("p95"), "batch_time_max_s": s.get("max"),
                     "mean_train_loss": (sum(losses) / len(losses)) if losses else None,
                     "batches_per_s_compute": (len(bt) / compute) if compute > 0 else None,
                     "sequences_per_s_compute": (len(bt) * mon.batch_size / compute) if compute > 0 else None,
                     "sequences_per_s_wall": (len(bt) * mon.batch_size / wall) if wall else None})
    return rows


def hash_tree(root: str, rels: Sequence[str]) -> Dict[str, Optional[str]]:
    return {r: (_sha(os.path.join(root, r)) if os.path.isfile(os.path.join(root, r)) else None) for r in rels}


def _write_csv(path: str, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with open(path, "x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


# --------------------------------------------------------------------------------- runner
def run_monitored(repo: str, workdir: str, run_id: str, num_epochs: int, *, method: int = 4, batch_size: int = 10, interval_s: float = 1.0,
                  extra_args: Sequence[str] = (), train_module: str = "train", required_files: Sequence[str] = DEFAULT_REQUIRED,
                  data_dir: str = "Data") -> Dict[str, Any]:
    """Run train.py monitored in a fresh work directory. Returns the summary dict (also written to disk)."""
    problems = preflight(repo, workdir, required_files)
    if problems:
        raise RuntimeError("isolation cannot be guaranteed; the run was NOT started:\n  - " + "\n  - ".join(problems))
    repo = os.path.realpath(repo)
    workdir = os.path.realpath(workdir)
    os.makedirs(workdir, exist_ok=True)               # creates missing parents; the directory itself is new or empty (checked above)
    out = os.path.join(workdir, "measurements")
    os.makedirs(out)                                              # fresh; fails if it somehow exists
    for sub in ("Store_Results/log", "Store_Results/model", "Store_Results/plot/train"):
        os.makedirs(os.path.join(workdir, sub))
    if os.path.isdir(os.path.join(repo, data_dir)):
        shutil.copytree(os.path.join(repo, data_dir), os.path.join(workdir, data_dir))     # isolated copy: train.py may write next to its data
    repo_results_before = tree_snapshot(os.path.join(repo, "Store_Results"))
    src_hash_before = hash_tree(repo, SOURCE_FILES)
    data_hash_before = hash_tree(repo, DATA_FILES)

    summary: Dict[str, Any] = {"runner_version": RUNNER_VERSION, "run_id": run_id, "started_utc": _utc(), "repo": repo, "workdir": workdir,
                               "command": {"module": train_module, "method": method, "num_epochs": num_epochs, "extra_args": list(extra_args)},
                               "status": "running"}
    old_cwd, old_argv, old_out = os.getcwd(), list(sys.argv), sys.stdout
    mon = None
    tee = None
    err = None
    try:
        os.chdir(workdir)
        sys.path.insert(0, repo)
        import torch  # noqa: WPS433
        mon = Monitor(torch, interval_s, batch_size)
        summary["environment"] = {
            "python_executable": sys.executable, "python_version": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cudnn": (torch.backends.cudnn.version() if torch.cuda.is_available() else None),
            "cuda_available": mon.cuda, "gpu_name": torch.cuda.get_device_name(0) if mon.cuda else None,
            "gpu_total_vram_mib": (torch.cuda.get_device_properties(0).total_memory / 2 ** 20) if mon.cuda else None,
            "gpu_driver": gpu_driver_version(), "gpu_telemetry_source": mon.gpu.source,
            "seeds": "none set by train.py or this runner (the baseline code sets no seed); runs are not bitwise reproducible",
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark}
        try:
            import numpy, pandas  # noqa: F401
            summary["environment"].update({"numpy": numpy.__version__, "pandas": pandas.__version__})
        except Exception:
            pass
        if mon.cuda:
            torch.cuda.reset_peak_memory_stats()
        # --- instrumentation (class-level patches; train.py itself is untouched)
        orig_zero_grad = torch.nn.Module.zero_grad
        orig_step = torch.optim.RMSprop.step
        orig_save = torch.save
        torch.nn.Module.zero_grad = mon.patched_zero_grad(orig_zero_grad)
        torch.optim.RMSprop.step = mon.patched_step(orig_step)
        torch.save = mon.timed_save(orig_save)
        dl_mod = importlib.import_module("utils.DataLoader")
        dl_cls = dl_mod.DataLoader
        patched = {}
        for name in ("__init__", "grid_creation", "batch_creater"):
            if hasattr(dl_cls, name):
                patched[name] = getattr(dl_cls, name)
                setattr(dl_cls, name, mon.timed("DataLoader." + name if name != "batch_creater" else "batch_creater", patched[name]))
        tmod = importlib.import_module(train_module)
        tee = Tee(old_out, os.path.join(out, "training_stdout.log"), mon.on_line)
        sys.stdout = tee
        sys.argv = [train_module + ".py", "--method", str(method), "--num_epochs", str(num_epochs), *list(extra_args)]    # no other option is added
        mon.start_sampler()
        t_wall0 = time.perf_counter()
        try:
            tmod.main()
        finally:
            mon._sync()
            summary["train_main_wall_s"] = time.perf_counter() - t_wall0
            mon.stop_sampler()
            sys.stdout = old_out
            tee.close()
            torch.nn.Module.zero_grad = orig_zero_grad
            torch.optim.RMSprop.step = orig_step
            torch.save = orig_save
            for name, fn in patched.items():
                setattr(dl_cls, name, fn)
        summary["status"] = "completed"
    except BaseException as exc:                                  # noqa: BLE001 - recorded, then re-raised after writing outputs
        err = exc
        summary["status"] = "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        sys.stdout = old_out
        sys.argv = old_argv
        os.chdir(old_cwd)
        if sys.path and sys.path[0] == repo:
            sys.path.pop(0)

    summary["finished_utc"] = _utc()
    cfg_path = os.path.join(workdir, "Store_Results", "model", "config.pkl")
    if os.path.isfile(cfg_path):
        try:
            import pickle
            with open(cfg_path, "rb") as fh:
                ns = pickle.load(fh)                              # the Namespace train.py itself wrote in the fresh work directory
            summary["training_args"] = {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v)) for k, v in vars(ns).items()}
            if mon is not None and isinstance(getattr(ns, "batch_size", None), int):
                mon.batch_size = ns.batch_size                    # throughput uses the batch size train.py actually used
        except Exception as exc:                                  # noqa: BLE001
            summary["training_args_error"] = str(exc)
    if mon is not None:
        epoch_rows = build_epoch_rows(mon)
        sync_t = [b["sync_batch_time_s"] for b in mon.batches]
        rep_t = [r["reported_time_per_batch_s"] for r in mon.reported]
        by_gb = {r["global_batch"]: r for r in mon.reported}
        batch_rows = [{"global_batch": b["global_batch"], "epoch": b["epoch"], "sync_batch_time_s": b["sync_batch_time_s"],
                       "reported_time_per_batch_s": by_gb.get(b["global_batch"], {}).get("reported_time_per_batch_s"),
                       "train_loss": by_gb.get(b["global_batch"], {}).get("train_loss")} for b in mon.batches]
        res_fields = ["t_s", "utc", "epoch", "global_batch", "gpu_util_pct", "gpu_mem_used_mib", "gpu_mem_total_mib", "torch_allocated_mib", "torch_reserved_mib",
                      "torch_max_allocated_mib", "cpu_pct_system", "cpu_pct_process", "process_rss_mib", "ram_used_mib", "ram_total_mib"]
        _write_csv(os.path.join(out, "training_resource_log.csv"), mon.samples, res_fields)
        _write_csv(os.path.join(out, "training_batch_log.csv"), batch_rows, ["global_batch", "epoch", "sync_batch_time_s", "reported_time_per_batch_s", "train_loss"])
        _write_csv(os.path.join(out, "training_epoch_summary.csv"), epoch_rows,
                   ["epoch", "n_batches", "epoch_wall_s", "sum_batch_time_s", "batch_creater_s", "batch_time_p50_s", "batch_time_p90_s", "batch_time_p95_s",
                    "batch_time_max_s", "mean_train_loss", "batches_per_s_compute", "sequences_per_s_compute", "sequences_per_s_wall"])
        col = lambda k: [s[k] for s in mon.samples if s.get(k) is not None]          # noqa: E731
        peak_rss_kernel = None
        try:
            import resource
            peak_rss_kernel = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            pass
        total_batch = sum(sync_t)
        summary.update({
            "batches_recorded": len(sync_t), "epochs_recorded": len(epoch_rows),
            "batch_time_sync_s": stats(sync_t), "batch_time_sync_excluding_first_s": stats(sync_t[1:]),
            "batch_time_reported_by_train_py_s": stats(rep_t),
            "throughput": {"sequences_per_s_compute": (len(sync_t) * mon.batch_size / total_batch) if total_batch > 0 else None,
                           "batches_per_s_compute": (len(sync_t) / total_batch) if total_batch > 0 else None,
                           "definition": "batches*batch_size / sum of synchronised batch times"},
            "data_preprocessing_s": {p: sum(x["duration_s"] for x in mon.phases if x["phase"] == p) for p in sorted({x["phase"] for x in mon.phases})},
            "checkpoint_save_s": {"count": len(mon.save_times), "total": sum(mon.save_times)},
            "gpu_utilisation_pct": stats(col("gpu_util_pct")), "gpu_device_memory_used_mib": stats(col("gpu_mem_used_mib")),
            "torch_allocated_mib": stats(col("torch_allocated_mib")), "torch_reserved_mib": stats(col("torch_reserved_mib")),
            "torch_peak_allocated_mib": (max(col("torch_max_allocated_mib")) if col("torch_max_allocated_mib") else None),
            "cpu_pct_system": stats(col("cpu_pct_system")), "cpu_pct_process": stats(col("cpu_pct_process")),
            "peak_process_rss_mib_sampled": (max(col("process_rss_mib")) if col("process_rss_mib") else None),
            "peak_process_rss_mib_kernel_ru_maxrss": peak_rss_kernel,
            "peak_system_ram_used_mib_sampled": (max(col("ram_used_mib")) if col("ram_used_mib") else None),
            "sampler": {"interval_s": interval_s, "n_samples": len(mon.samples), "gpu_source": mon.gpu.source},
            "markers": mon.markers,
            "caveats": ["GPU utilisation and used memory are device-wide samples, not per-process counters",
                        "sampling can miss short peaks; torch peak counters and ru_maxrss are exact for their scope",
                        "the sampler thread and CUDA synchronisation at batch boundaries add small overhead",
                        "this is baseline PCG TRAINING cost on the declared hardware; it says nothing about video or deployment performance"]})
        if tee is not None:
            summary["stdout_log"] = "training_stdout.log"
    # --- isolation verification and hashes
    repo_results_after = tree_snapshot(os.path.join(repo, "Store_Results"))
    summary["isolation_check"] = {
        "repo_store_results_unchanged": repo_results_before == repo_results_after,
        "repo_source_hashes_unchanged": src_hash_before == hash_tree(repo, SOURCE_FILES),
        "repo_data_hashes_unchanged": data_hash_before == hash_tree(repo, DATA_FILES),
        "workdir_data_hashes_equal_repo_data": data_hash_before == hash_tree(workdir, DATA_FILES)}
    summary["source_sha256"] = src_hash_before
    summary["data_sha256"] = data_hash_before
    produced = {}
    for d, _, fs in os.walk(os.path.join(workdir, "Store_Results")):
        for f in fs:
            p = os.path.join(d, f)
            produced[os.path.relpath(p, workdir)] = {"size_bytes": os.path.getsize(p), "sha256": _sha(p)}
    summary["produced_artifacts"] = produced
    with open(os.path.join(out, "training_resource_summary.json"), "x", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)
    if err is not None:
        raise err
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Monitored isolated PCG training run (never run automatically).")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--workdir", required=True, help="fresh or empty directory OUTSIDE the repository")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--num-epochs", type=int, required=True)
    ap.add_argument("--method", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--interval-s", type=float, default=1.0)
    ap.add_argument("--preflight-only", action="store_true")
    a, extra = ap.parse_known_args(argv)
    if a.preflight_only:
        probs = preflight(a.repo, a.workdir)
        print(json.dumps({"problems": probs, "safe_to_run": not probs}, indent=2))
        return 0 if not probs else 2
    try:
        s = run_monitored(a.repo, a.workdir, a.run_id, a.num_epochs, method=a.method, batch_size=a.batch_size, interval_s=a.interval_s, extra_args=extra)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"status": s["status"], "workdir": a.workdir}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
