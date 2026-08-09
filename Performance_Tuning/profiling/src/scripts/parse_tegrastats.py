#!/usr/bin/env python3
"""将常见tegrastats字段解析成CSV；未知字段仍保留在raw列。"""
import csv
import re
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("usage: parse_tegrastats.py input.log output.csv")
patterns = {
    "ram_used_mb": r"RAM (\d+)/",
    "swap_used_mb": r"SWAP (\d+)/",
    "gr3d_pct": r"GR3D_FREQ (\d+)%",
    "emc_pct": r"EMC_FREQ (\d+)%",
    "gpu_temp_c": r"GPU@([0-9.]+)C",
    "cpu_temp_c": r"CPU@([0-9.]+)C",
    "power_mw": r"VDD_IN (\d+)mW",
}
rows = []
for index, line in enumerate(Path(sys.argv[1]).read_text(errors="replace").splitlines()):
    row = {"sample": index, "raw": line}
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        row[key] = match.group(1) if match else ""
    rows.append(row)
with Path(sys.argv[2]).open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sample", *patterns, "raw"])
    writer.writeheader(); writer.writerows(rows)
