#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; build="${1:-$root/build}"
for demo in 20_cpu_hotspot_bad_good 21_cache_locality_bad_good 23_stream_bad_good 24_aos_soa; do
  [[ -x "$build/$demo" ]] || { echo "SKIP $demo: 未构建"; continue; }
  echo "===== $demo ====="; "$build/$demo"
done
if command -v perf >/dev/null; then perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses "$build/21_cache_locality_bad_good" || echo "perf受权限/硬件事件限制"; fi
