#!/usr/bin/env bash
#
# perf_record.sh - Sample a target with `perf record` and generate a report.
# Usage: ./scripts/perf_record.sh <target_binary> [args...]
# Requires root or perf_event_paranoid < 2 for sampling.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
OUT_DIR="${PROJECT_ROOT}/benchmark_results"

fct_log() {
    printf '%s\n' "[perf_record] $*"
}

fct_fail() {
    printf '%s\n' "[perf_record] ERROR: $*" >&2
    exit 1
}

fct_main() {
    [[ $# -ge 1 ]] || fct_fail "usage: ${0##*/} <target_binary> [args...]"
    command -v perf >/dev/null 2>&1 || fct_fail "perf not installed"

    local target="$1"
    shift
    [[ -x "${target}" ]] || fct_fail "not executable: ${target}"

    if [[ -f /proc/sys/kernel/perf_event_paranoid ]]; then
        local paranoid
        paranoid="$(cat /proc/sys/kernel/perf_event_paranoid)"
        if [[ "${paranoid}" -ge 2 ]]; then
            fct_log "perf_event_paranoid=${paranoid} (>=2): sampling requires root."
            fct_log "Try: sudo perf record -g ./build/<target>"
            exit 1
        fi
    fi

    mkdir -p "${OUT_DIR}"
    perf record -g --call-graph dwarf -o "${OUT_DIR}/perf.data" "${target}" "$@"
    fct_log "Sample written to ${OUT_DIR}/perf.data. Report with:"
    fct_log "  perf report -i ${OUT_DIR}/perf.data"
}

fct_main "$@"
