#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
binary="${1:-$root/build/33_numa_local_remote}"
command -v numactl >/dev/null || { echo "SKIP: numactl不存在"; exit 0; }
[[ -x "$binary" ]] || { echo "SKIP: libnuma target未构建"; exit 0; }
nodes=$(numactl --hardware | sed -n 's/^available: \([0-9][0-9]*\).*/\1/p')
if [[ -z "$nodes" || "$nodes" -lt 2 ]]; then echo "SKIP: 需要至少2个NUMA node"; exit 0; fi
echo "===== local: CPU0 / Memory0 ====="
"$binary" 0 0 256 8
echo "===== remote: CPU0 / Memory1 ====="
"$binary" 0 1 256 8
echo "===== numastat ====="
numastat
if command -v perf >/dev/null; then
  perf stat -e cycles,instructions,cache-references,cache-misses "$binary" 0 1 256 8 || true
fi
