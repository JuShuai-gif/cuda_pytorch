"""Parse ``nsys stats`` text output into a small JSON summary.

``nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum`` prints a
human-readable table.  This parser extracts the top N rows per report section
so benchmarks can record kernel durations and API overhead without opening the
GUI.  It is deliberately tolerant: if a section is missing (e.g. the run had
no NVTX ranges), the corresponding key is absent rather than fabricated.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any


def parse_stats(text: str) -> dict[str, Any]:
    sections: dict[str, list[dict[str, str]]] = {}
    current: str | None = None
    header: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]+?)\s*\(\d+\)\s*$", line)
        if m and ("Time" in raw or "Duration" in raw or "Total" in raw or "Count" in raw):
            current = m.group(1).strip()
            sections.setdefault(current, [])
            header = None
            continue
        if current is None:
            continue
        if header is None and ("%" in line and ("Time" in line or "Duration" in line)):
            header = [c.strip() for c in line.split()]
            continue
        if header is None:
            continue
        parts = line.split()
        if len(parts) < len(header):
            continue
        row = {header[i]: parts[i] for i in range(len(header))}
        sections[current].append(row)
        if len(sections[current]) >= 20:
            header = None  # stop collecting this section after 20 rows
            current = None
    return {"sections": sections}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="path to nsys stats text output")
    p.add_argument("--output", help="JSON output path (default: stdout)")
    args = p.parse_args(argv)

    with open(args.input, encoding="utf-8") as f:
        parsed = parse_stats(f.read())

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
    else:
        print(json.dumps(parsed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
