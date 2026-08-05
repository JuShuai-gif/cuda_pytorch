#!/usr/bin/env bash
# numa_test.sh - detect NUMA topology and run remote-access test if multi-node.
# Single-node machines are detected and skipped with a clear message.
set -u

if ! command -v numactl >/dev/null 2>&1; then
    echo "numactl not installed (package: numactl)." >&2
    exit 1
fi

echo "== numactl --hardware =="
numactl --hardware || true

nodes=$(numactl --hardware 2>/dev/null | awk '/^available:/{print $2}')
if [[ -z "$nodes" ]]; then
    nodes=1
fi
echo
echo "== NUMA node count: $nodes =="

if [[ "$nodes" -lt 2 ]]; then
    echo "Single NUMA node machine: remote-access tests skipped."
    echo "Not claiming NUMA test success."
    exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/src/build}"
exe="$BUILD_DIR/numa_local_remote/numa_local_remote"
if [[ -x "$exe" ]]; then
    echo "== running NUMA local/remote experiment (requires -DENABLE_NUMA_EXAMPLES=ON) =="
    "$exe"
else
    echo "numa_local_remote not built. Rebuild with:"
    echo "  BUILD_DIR=... ENABLE_NUMA_EXAMPLES=1 ./scripts/build.sh"
fi
exit 0
