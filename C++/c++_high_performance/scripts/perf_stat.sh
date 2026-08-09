#!/usr/bin/env bash
#
# perf_stat.sh - Run `perf stat` on a target for cycles/instructions/cache stats.
# Usage: ./scripts/perf_stat.sh <target_binary> [args...]
# Detects perf access restrictions (paranoid) and reports a friendly message.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

fct_log() {
    printf '%s\n' "[perf_stat] $*"
}

fct_fail() {
    printf '%s\n' "[perf_stat] ERROR: $*" >&2
    exit 1
}

fct_main() {
    [[ $# -ge 1 ]] || fct_fail "usage: ${0##*/} <target_binary> [args...]"
    command -v perf >/dev/null 2>&1 || fct_fail "perf not installed"
    command -v cat /proc/sys/kernel/perf_event_paranoid >/dev/null 2>&1 || true

    local target="$1"
    shift
    [[ -x "${target}" ]] || fct_fail "not executable: ${target}"

    if [[ -f /proc/sys/kernel/perf_event_paranoid ]]; then
        local paranoid
        paranoid="$(cat /proc/sys/kernel/perf_event_paranoid)"
        if [[ "${paranoid}" -ge 2 ]]; then
            fct_log "perf_event_paranoid=${paranoid} (>=2): sampling needs root. Trying perf stat (may still work)..."
        fi
    fi

    perf stat -e task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses \
        "${target}" "$@"
}

fct_main "$@"
