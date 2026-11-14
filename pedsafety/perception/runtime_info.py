"""Environment / hardware capture for run manifests and performance reports (Phase 3, Task 1G).

Every import of an optional library is guarded: a missing library is recorded as ``null`` (not guessed). GPU facts come from
``torch.cuda`` only; no GPU is assumed.
"""

import importlib
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Optional

_LIBS = ("numpy", "scipy", "cv2", "torch", "torchvision", "ultralytics", "lap", "lapx", "psutil", "jsonschema", "pytest")


def _version(mod: str) -> Optional[str]:
    try:
        m = importlib.import_module(mod)
    except Exception:                                               # noqa: BLE001 - absent or broken: record as unavailable
        return None
    return str(getattr(m, "__version__", "unknown"))


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
            if out:
                return out
        except Exception:                                           # noqa: BLE001
            pass
    elif os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def collect_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {"python": sys.version.split()[0], "python_implementation": platform.python_implementation(),
                           "platform": platform.platform(), "machine": platform.machine(), "cpu": _cpu_name(), "cpu_count": os.cpu_count(),
                           "libraries": {m: _version(m) for m in _LIBS}}
    try:
        import psutil
        env["ram_total_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 2)
    except Exception:                                               # noqa: BLE001
        env["ram_total_gb"] = None
    cuda: Dict[str, Any] = {"available": False, "torch_cuda_build": None, "cudnn": None, "device_count": 0, "gpus": []}
    try:
        import torch
        cuda["torch_cuda_build"] = torch.version.cuda
        cuda["available"] = bool(torch.cuda.is_available())
        if cuda["available"]:
            cuda["cudnn"] = torch.backends.cudnn.version()
            cuda["device_count"] = torch.cuda.device_count()
            for i in range(cuda["device_count"]):
                p = torch.cuda.get_device_properties(i)
                cuda["gpus"].append({"index": i, "name": p.name, "total_memory_mib": round(p.total_memory / 2 ** 20, 1),
                                     "compute_capability": "%d.%d" % (p.major, p.minor)})
    except Exception:                                               # noqa: BLE001
        pass
    env["cuda"] = cuda
    return env


def hardware_string(env: Dict[str, Any]) -> str:
    """One-line hardware declaration for ``PerformanceRecorder.summary(hardware=...)``."""
    gpus = env.get("cuda", {}).get("gpus") or []
    gpu = "; ".join("%s %.0f MiB" % (g["name"], g["total_memory_mib"]) for g in gpus) or "no CUDA GPU"
    ram = env.get("ram_total_gb")
    return "%s (%s logical CPUs)%s; %s" % (env.get("cpu"), env.get("cpu_count"), "" if ram is None else ", %.1f GB RAM" % ram, gpu)


def git_commit(cwd: Optional[str] = None) -> Optional[str]:
    """Current commit hash if ``cwd`` is a git checkout, else None (the Colab ZIP bundle has no .git)."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, cwd=cwd)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:                                               # noqa: BLE001
        return None
