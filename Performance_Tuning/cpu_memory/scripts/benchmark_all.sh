#!/usr/bin/env bash
# benchmark_all.sh - run all experiments and tee output into benchmark_results/.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/src/build}"
RES_DIR="$ROOT/benchmark_results"
mkdir -p "$RES_DIR"

stamp="$(date +%Y%m%d_%H%M%S)"
dir="$RES_DIR/$stamp"
mkdir -p "$dir"

# Environment record
{
    echo "### environment $(date -Is)"
    uname -a
    lscpu 2>/dev/null | sed -n '1,20p'
    g++ --version | head -1
} > "$dir/environment.txt"

"$ROOT/scripts/run_all.sh" 2>&1 | tee "$dir/all.txt"

echo
echo "== results saved under $dir =="
