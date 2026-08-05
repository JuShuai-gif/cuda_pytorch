#!/usr/bin/env bash
# perf_record.sh - record and report on a command with perf.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <command...>" >&2
    exit 2
fi

echo "== perf record (sampling, default event) =="
perf record -o /tmp/perf_cpu_memory.data "$@"
echo "== perf report (top 25) =="
perf report -i /tmp/perf_cpu_memory.data --stdio 2>/dev/null | head -40 || true
rm -f /tmp/perf_cpu_memory.data
