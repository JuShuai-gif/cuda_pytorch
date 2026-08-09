#!/usr/bin/env bash
set -euo pipefail
command -v perf >/dev/null || { echo "perf 未安装"; exit 0; }
bin="${1:-./build/01_cpu_hotspot}"; shift || true
perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,page-faults,context-switches,cpu-migrations "$bin" "$@"
