#!/usr/bin/env bash
set -euo pipefail
command -v perf >/dev/null || { echo "perf 未安装"; exit 0; }
root="${FLAMEGRAPH_DIR:-}"; [[ -x "$root/stackcollapse-perf.pl" && -x "$root/flamegraph.pl" ]] || { echo "请设置 FLAMEGRAPH_DIR"; exit 0; }
bin="${1:-./build/01_cpu_hotspot}"; perf record -F 99 -g -- "$bin"; perf script | "$root/stackcollapse-perf.pl" | "$root/flamegraph.pl" > flamegraph.svg
echo "生成 flamegraph.svg"
