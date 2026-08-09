#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; build="${1:-$root/build}"
"$build/30_realtime_jitter"
"$build/30_realtime_jitter" --interference
command -v cyclictest >/dev/null && echo "可手动运行: cyclictest -p 80 -t 4 -n -i 1000 -l 100000" || echo "cyclictest不存在：当前环境未验证"
