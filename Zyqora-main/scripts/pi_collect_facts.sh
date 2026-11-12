#!/usr/bin/env bash
set +e
echo "=== Architecture ==="; uname -m
echo "=== Device model ==="; cat /proc/device-tree/model 2>/dev/null; echo
echo "=== OS ==="; cat /etc/os-release | head -n 5
echo "=== Python ==="; python3 --version
echo "=== Governor ==="; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
echo "=== Temperature ==="; vcgencmd measure_temp 2>/dev/null
echo "=== Throttling ==="; vcgencmd get_throttled 2>/dev/null
