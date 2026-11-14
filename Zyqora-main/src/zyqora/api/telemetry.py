"""Lightweight device telemetry from /proc and /sys (no extra dependencies, no fabricated values).

Every field is None when the platform does not expose it. CPU utilisation is the
system-wide busy fraction between two /proc/stat samples (the previous request's
sample, or a short 200 ms sample on the first call).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

from ..benchmark import cpu_temperature_c, device_model, read_proc_status


def _read_cpu_times() -> Optional[tuple[int, int]]:
    try:
        first = Path("/proc/stat").read_text().splitlines()[0].split()
    except (OSError, IndexError):
        return None
    if first[0] != "cpu":
        return None
    vals = [int(x) for x in first[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    return sum(vals), idle


def _meminfo() -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {"total_mb": None, "available_mb": None, "used_mb": None, "used_percent": None}
    try:
        data = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            parts = v.split()
            if parts and parts[0].isdigit():
                data[k] = int(parts[0]) / 1024.0
    except OSError:
        return out
    total, avail = data.get("MemTotal"), data.get("MemAvailable")
    if total and avail is not None:
        out.update(total_mb=total, available_mb=avail, used_mb=total - avail, used_percent=100.0 * (total - avail) / total)
    return out


def _cpufreq() -> dict:
    """Current/max clock of cpu0 and the frequency governor (explains latency on an idle Pi)."""
    base = Path("/sys/devices/system/cpu/cpu0/cpufreq")

    def read(name: str) -> Optional[str]:
        try:
            return (base / name).read_text().strip()
        except OSError:
            return None

    cur, mx = read("scaling_cur_freq"), read("cpuinfo_max_freq")
    return {"cpu_mhz": int(cur) / 1000 if cur and cur.isdigit() else None,
            "cpu_max_mhz": int(mx) / 1000 if mx and mx.isdigit() else None,
            "governor": read("scaling_governor")}


def _uptime_s() -> Optional[float]:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


class Telemetry:
    def __init__(self):
        self._lock = threading.Lock()
        self._prev: Optional[tuple[float, tuple[int, int]]] = None
        self._model = device_model()

    def snapshot_device(self) -> dict:
        """Cheap identity of the machine running inference (no sampling)."""
        import platform

        return {"device_model": self._model, "machine": platform.machine(),
                "is_raspberry_pi": bool(self._model and "Raspberry Pi" in self._model)}

    def cpu_percent(self) -> Optional[float]:
        with self._lock:
            now = time.monotonic()
            prev = self._prev
            if prev is None or now - prev[0] > 60:
                a = _read_cpu_times()
                if a is None:
                    return None
                time.sleep(0.2)
                prev = (now, a)
            b = _read_cpu_times()
            if b is None:
                return None
            self._prev = (time.monotonic(), b)
        dt, di = b[0] - prev[1][0], b[1] - prev[1][1]
        return None if dt <= 0 else max(0.0, min(100.0, 100.0 * (dt - di) / dt))

    def snapshot(self) -> dict:
        proc = read_proc_status()
        try:
            load = os.getloadavg()
        except OSError:
            load = None
        return {
            "device_model": self._model,
            "is_raspberry_pi": bool(self._model and "Raspberry Pi" in self._model),
            "cpu_percent": self.cpu_percent(),
            "cpu_count": os.cpu_count(),
            "load_average": list(load) if load else None,
            "memory": _meminfo(),
            "temperature_c": cpu_temperature_c(),
            "cpufreq": _cpufreq(),
            "uptime_s": _uptime_s(),
            "process_rss_mb": proc["vm_rss_mb"],
            "process_peak_rss_mb": proc["vm_hwm_mb"],
            "sampled_at": time.time(),
        }
