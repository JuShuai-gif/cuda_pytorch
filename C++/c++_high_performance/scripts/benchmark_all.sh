#!/usr/bin/env bash
#
# benchmark_all.sh - Run every *_benchmark executable, saving output to
#                    benchmark_results/ with a timestamp.
# Usage: ./scripts/benchmark_all.sh [build_dir]

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${1:-${PROJECT_ROOT}/build}"
RESULTS_DIR="${PROJECT_ROOT}/benchmark_results"

fct_log() {
    printf '%s\n' "[benchmark_all] $*"
}

fct_fail() {
    printf '%s\n' "[benchmark_all] ERROR: $*" >&2
    exit 1
}

fct_main() {
    [[ -d "${BUILD_DIR}" ]] || fct_fail "build dir not found: ${BUILD_DIR} (run ./scripts/build.sh first)"
    mkdir -p "${RESULTS_DIR}"

    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    local outdir="${RESULTS_DIR}/${stamp}"
    mkdir -p "${outdir}"

    fct_log "Saving results to ${outdir}"

    local failures=0
    local total=0

    while IFS= read -r -d '' bench; do
        total=$((total + 1))
        local rel="${bench#"${BUILD_DIR}/"}"
        rel="${rel//\//_}"
        local outfile="${outdir}/${rel}.txt"
        fct_log "== ${bench#"${BUILD_DIR}/"} =="
        if ! "${bench}" >"${outfile}" 2>&1; then
            fct_log "FAILED: ${bench} (see ${outfile})"
            failures=$((failures + 1))
        fi
    done < <(find "${BUILD_DIR}" -type f -name '*_benchmark' -perm -u+x -print0 2>/dev/null)

    fct_log "Ran ${total} benchmarks, ${failures} failed. Results in ${outdir}"
    [[ "${failures}" -eq 0 ]] || exit 1
}

fct_main "$@"
