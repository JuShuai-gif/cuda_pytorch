"""tegrastats sampler and parser.

``tegrastats`` prints one line per second with RAM, CPU freq/util, per-zone
temperatures, and per-rail power.  This module runs it as a subprocess,
parses each line, and yields a dict of the metrics we care about for edge
inference: temperatures, power rails, and CPU frequency.
"""

from __future__ import annotations

import re
import subprocess
import threading
from typing import Iterator


def parse_tegrastats_line(line: str) -> dict:
    d: dict = {}

    # CPU frequency (MHz) - take the first core's frequency.
    m = re.search(r"CPU \[(\d+)%@(\d+)", line)
    if m:
        d["cpu_freq_mhz"] = int(m.group(2))

    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if m:
        d["ram_used_mb"] = int(m.group(1))

    for zone, key in [("cpu", "cpu_temp_c"), ("gpu", "gpu_temp_c"),
                      ("tj", "tj_temp_c")]:
        m = re.search(rf"{zone}@([\d.]+)C", line)
        if m:
            d[key] = float(m.group(1))

    for rail, key in [("VDD_GPU", "gpu_power_mw"), ("VIN", "total_power_mw"),
                      ("VDD_CPU_SOC_MSS", "cpu_power_mw")]:
        m = re.search(rf"{rail} (\d+)mW", line)
        if m:
            d[key] = int(m.group(1))

    return d


class TegrastatsSampler:
    """Run tegrastats and expose parsed samples."""

    def __init__(self):
        self.proc = subprocess.Popen(
            ["tegrastats"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)
        self._lock = threading.Lock()
        self.samples: list[dict] = []

    def start(self):
        def _run():
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                d = parse_tegrastats_line(line)
                if d:
                    with self._lock:
                        self.samples.append(d)

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.samples)

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)
