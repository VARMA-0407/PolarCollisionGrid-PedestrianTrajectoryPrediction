"""Engineering measurement helpers: wall / CPU time per stage, warm-up vs steady state, variability, peak memory, hardware description.

Peak memory is the PROCESS peak working set (monotone, via the Phase 4 helper): it is recorded after each stage and therefore bounds, but does not isolate, a stage.
CPU utilisation is ``process_time / wall_time`` (average cores busy), not an OS-level utilisation counter. GPU fields are reported as not applicable when no CUDA device exists.
"""

import math
import os
import platform
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..interaction.runner import peak_memory_bytes


def summarize(values: Sequence[float]) -> Optional[Dict[str, Any]]:
    v = [float(x) for x in values]
    if not v:
        return None
    n = len(v)
    mean = sum(v) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else None
    s = sorted(v)
    return {"n": n, "mean": mean, "std_sample": std, "min": s[0], "median": s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]), "max": s[-1],
            "cv": (std / mean) if (std is not None and mean > 0) else None}


class StageClock:
    """Records wall time, process CPU time and the process peak working set after each named stage (a stage may be entered once)."""

    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}

    @contextmanager
    def stage(self, name: str):
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            w, c = time.perf_counter() - w0, time.process_time() - c0
            self.records[name] = {"wall_s": w, "cpu_s": c, "mean_cores_busy": (c / w) if w > 0 else None, "process_peak_working_set_bytes_after": peak_memory_bytes()}


def measure_repeated(fn: Callable[[], Any], warmup: int, repeats: int) -> Dict[str, Any]:
    """Run ``fn`` ``warmup`` times (reported separately, excluded from statistics) then ``repeats`` times; return raw timings and statistics."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup >= 0 and repeats >= 1 required")
    warm = []
    for _ in range(warmup):
        t = time.perf_counter()
        fn()
        warm.append(time.perf_counter() - t)
    steady = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        steady.append(time.perf_counter() - t)
    return {"warmup_s": warm, "steady_s": steady, "steady_stats": summarize(steady)}


def _cpu_name() -> Optional[str]:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:                                                                         # noqa: BLE001 - not Windows / unavailable
        return None


def _ram_total_bytes() -> Optional[int]:
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            return int(ms.ullTotalPhys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)) else None
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:                                                                         # noqa: BLE001
        return None


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(), "processor": platform.processor(), "cpu_name": _cpu_name(),
                            "logical_cpus": os.cpu_count(), "ram_total_bytes": _ram_total_bytes(), "psutil": "unavailable",
                            "memory_method": "process peak working set via the OS API (Windows GetProcessMemoryInfo) or ru_maxrss; process-level and monotone, not per stage"}
    try:
        from ..phase5.models.neural_evaluation import torch      # torch is reached through the guarded Phase 5 module (torch imports are allowed only under phase5/models)
        info.update({"torch": torch.__version__, "torch_threads": torch.get_num_threads(), "cuda_available": bool(torch.cuda.is_available())})
        info["gpu"] = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else "NOT_APPLICABLE_NO_CUDA_DEVICE")
    except Exception as exc:                                                                  # noqa: BLE001
        info["torch"] = "unavailable: %s" % type(exc).__name__
    return info


def realtime_assessment(*, fps_target: float, frames: int, measured_seconds: float, perception_executed: bool, streaming_capable: bool, hardware_documented: bool = True) -> Dict[str, Any]:
    """A real-time claim needs ALL of: video decoding and detection/tracking inside the measurement, a causal streaming implementation, a documented configuration and a measured
    rate >= target. Anything less is reported with the measured rate and the unmet conditions."""
    rate = frames / measured_seconds if measured_seconds > 0 else None
    reasons = []
    if not perception_executed:
        reasons.append("video decoding and perception (Phase 3) were not part of this measurement")
    if not streaming_capable:
        reasons.append("the pipeline is an offline batch replay: the window builder requires the target's future frames to accept a window, so no causal online mode exists")
    if rate is None or rate < fps_target:
        reasons.append("measured rate %s < %.1f frames/s" % ("n/a" if rate is None else "%.2f" % rate, fps_target))
    if not hardware_documented:
        reasons.append("hardware/configuration not documented")
    return {"target_fps": fps_target, "measured_frames_per_s_of_measured_stages": rate, "claim_supported": not reasons, "unmet_conditions": reasons}


__all__ = ["summarize", "StageClock", "measure_repeated", "hardware_info", "realtime_assessment", "peak_memory_bytes"]
