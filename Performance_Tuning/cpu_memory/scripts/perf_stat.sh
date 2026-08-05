#!/usr/bin/env bash
# perf_stat.sh - run perf stat with memory-relevant events on a command.
# Tolerates unsupported events by splitting into per-event invocations
# when the combined run fails, so the script never fails as a whole.
set -u

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <command...>" >&2
    exit 2
fi

common_events=(cycles instructions branches branch-misses cache-references
    cache-misses page-faults minor-faults major-faults context-switches
    cpu-migrations)
mem_events=(L1-dcache-loads L1-dcache-load-misses LLC-loads LLC-load-misses
    dTLB-loads dTLB-load-misses)

echo "== perf stat: generic events =="
if perf stat -e "$(IFS=,; echo "${common_events[*]}")" "$@" 2>&1; then
    :   # combined worked
else
    echo "(combined event set failed; trying individually)"
    for e in "${common_events[@]}"; do
        if ! perf stat -e "$e" "$@" 2>/dev/null >/dev/null; then
            echo "event not supported: $e (skipped)"
        fi
    done
fi

echo
echo "== perf stat: memory-specific events (best-effort) =="
for e in "${mem_events[@]}"; do
    if perf stat -e "$e" "$@" 2>/dev/null >/dev/null; then
        echo "--- $e ---"
        perf stat -e "$e" "$@" 2>&1 | sed -n '1,12p'
    else
        echo "event not supported: $e (skipped)"
    fi
done
exit 0
