"""Jetson Thor platform profile.

Reads the static platform facts that define the edge-inference environment:
CPU/GPU architecture, unified memory, current temperatures, and (where readable
without root) the power mode.  The point is to anchor every benchmark to the
hardware it ran on - an edge SoC with unified memory and a thermal envelope is
a different world from a discrete-GPU server.
"""

from __future__ import annotations

import os
import subprocess

import torch


def read_thermal_zones() -> dict:
    """Read all thermal_zone temperatures (millidegrees C -> C)."""
    out = {}
    for z in range(8):
        path = f"/sys/class/thermal/thermal_zone{z}"
        t = f"{path}/temp"
        name = f"{path}/type"
        if os.path.exists(t):
            try:
                temp_c = int(open(t).read().strip()) / 1000.0
                tname = open(name).read().strip() if os.path.exists(name) else f"zone{z}"
                out[tname] = temp_c
            except (ValueError, OSError):
                pass
    return out


def read_power_mode() -> str:
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown (requires root)"


def platform_profile() -> dict:
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                mem["total_kb"] = int(line.split()[1])

    return {
        "arch": os.uname().machine,
        "cpu_max_mhz": _cpu_max_mhz(),
        "gpu": props.name if props else "none",
        "gpu_sm_count": props.multi_processor_count if props else None,
        "gpu_total_memory_bytes": props.total_memory if props else None,
        "unified_memory": getattr(props, "unifiedAddressing", None),
        "ram_total_kb": mem.get("total_kb"),
        "thermal_zones_c": read_thermal_zones(),
        "power_mode": read_power_mode(),
    }


def _cpu_max_mhz() -> str:
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") as f:
            return f"{int(f.read().strip()) // 1000} MHz"
    except OSError:
        return "unknown"
