"""Measurement helpers: process memory, CPU time, temperature, platform facts (Linux)."""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Optional


def read_proc_status(path: str = "/proc/self/status") -> dict[str, Optional[float]]:
    """Current (VmRSS) and peak (VmHWM) resident memory in MB; None where unavailable."""
    out: dict[str, Optional[float]] = {"vm_rss_mb": None, "vm_hwm_mb": None}
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            parts = rest.split()
            if parts and parts[0].isdigit():
                out["vm_rss_mb" if key == "VmRSS" else "vm_hwm_mb"] = int(parts[0]) / 1024.0
    return out


def cpu_time_ns() -> int:
    """User + system CPU time of this process (all threads)."""
    return time.process_time_ns()


def cpu_temperature_c() -> Optional[float]:
    """Optional diagnostic only (Raspberry Pi thermal zone)."""
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def device_model() -> Optional[str]:
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            return Path(p).read_bytes().rstrip(b"\x00").decode(errors="replace").strip()
        except OSError:
            continue
    return None


def platform_facts() -> dict:
    model = device_model()
    return {
        "device_model": model,
        "is_raspberry_pi_5": bool(model and "Raspberry Pi 5" in model),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_temperature_c": cpu_temperature_c(),
    }
