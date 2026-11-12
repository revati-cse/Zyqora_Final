#!/usr/bin/env bash
# Collect Raspberry Pi 5 facts that IMPLEMENTATION_STATUS.md lists as unconfirmed
# (RAM, cooling/throttling, storage, audio devices, camera, OS, Python).
# Read-only; safe to run before installing anything.
set +e
echo "=== Device model ===";   tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo
echo "=== Architecture ===";   uname -m   # must be aarch64
echo "=== OS ===";             grep -E '^(PRETTY_NAME|VERSION_CODENAME)=' /etc/os-release
echo "=== Kernel ===";         uname -r
echo "=== Python ===";         python3 --version; command -v pyenv >/dev/null && pyenv versions
echo "=== RAM ===";            free -h | head -2
echo "=== Storage ===";        df -h / | tail -1
echo "=== CPU governor ===";   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
echo "=== CPU cores ===";      nproc
echo "=== Temperature ===";    vcgencmd measure_temp 2>/dev/null || cat /sys/class/thermal/thermal_zone0/temp
echo "=== Throttling ===";     vcgencmd get_throttled 2>/dev/null   # 0x0 = never throttled
echo "=== Fan (active cooler) ==="; cat /sys/devices/platform/cooling_fan/hwmon/*/fan1_input 2>/dev/null || echo "no fan sensor found"
echo "=== Audio playback devices ==="; aplay -l 2>/dev/null || echo "aplay not available"
echo "=== Camera ===";         (rpicam-hello --list-cameras 2>/dev/null || libcamera-hello --list-cameras 2>/dev/null) | head -5 || echo "no camera tool"
echo "=== Network interfaces ==="; ip -brief addr 2>/dev/null
echo "=== ExecuTorch venv ==="; ls -d ~/executorch_on_arm/.cpu_lab_venv 2>/dev/null || echo "course venv not found"
