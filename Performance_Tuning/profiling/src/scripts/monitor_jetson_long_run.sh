#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
duration="${1:-1200}"
interval_ms="${2:-1000}"
shift $(( $# >= 2 ? 2 : $# ))
command -v tegrastats >/dev/null || { echo "SKIP: tegrastats不存在，此脚本仅在Jetson运行"; exit 0; }
timestamp=$(date +%Y%m%d_%H%M%S)
out="$root/jetson_long_run_$timestamp"
mkdir -p "$out"
echo "duration_s=$duration interval_ms=$interval_ms" > "$out/environment.txt"
uname -a >> "$out/environment.txt"
command -v nvpmodel >/dev/null && nvpmodel -q >> "$out/environment.txt" 2>&1 || true
tegrastats --interval "$interval_ms" > "$out/tegrastats.log" &
tegrastats_pid=$!
cleanup(){ kill "$tegrastats_pid" 2>/dev/null || true; wait "$tegrastats_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
if [[ $# -gt 0 ]]; then
  echo "application: $*" >> "$out/environment.txt"
  timeout "$duration" "$@" 2>&1 | tee "$out/application.log" || true
else
  echo "未提供应用命令，仅采集tegrastats ${duration}s"
  sleep "$duration"
fi
cleanup
trap - EXIT INT TERM
python3 "$root/scripts/parse_tegrastats.py" "$out/tegrastats.log" "$out/tegrastats.csv" || true
echo "result_dir=$out"
