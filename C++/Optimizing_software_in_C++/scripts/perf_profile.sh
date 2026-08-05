#!/usr/bin/env bash
# perf_profile.sh -- run perf stat / record / report on an executable.
#
# Usage:
#   ./scripts/perf_profile.sh ./build/08_memory_cache/08_cache_random
#
# Notes:
#   * On many systems perf needs root (perf_event_paranoid >= 1).
#   * If perf fails, try:  sudo sysctl kernel.perf_event_paranoid=1
#     or run this script with sudo.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <executable> [args...]" >&2
    exit 1
fi

TARGET="$1"
shift || true

if [ ! -x "${TARGET}" ]; then
    echo "error: not an executable: ${TARGET}" >&2
    exit 1
fi

paranoid="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 0)"
if [ "${paranoid}" -ge 2 ] && [ "$(id -u)" -ne 0 ]; then
    echo "warning: perf_event_paranoid=${paranoid}; hardware counters need root."
    echo "         options:"
    echo "           sudo sysctl kernel.perf_event_paranoid=1   (temporary)"
    echo "           sudo $0 ${TARGET} $*"
    echo ""
fi

echo "===== perf stat ====="
perf stat -e cycles,instructions,cache-misses,cache-references,branch-misses,branch-instructions \
    "${TARGET}" "$@" || echo "perf stat failed"

echo ""
echo "===== perf record ====="
perf record -g -o /tmp/perf.data "${TARGET}" "$@" || echo "perf record failed"

echo ""
echo "===== perf report (top call graph) ====="
perf report -i /tmp/perf.data --stdio || echo "perf report failed"
