#!/usr/bin/env bash
#
# run_all.sh - Run every *_example executable under the build tree.
# Usage: ./scripts/run_all.sh [build_dir]

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${1:-${PROJECT_ROOT}/build}"

fct_log() {
    printf '%s\n' "[run_all] $*"
}

fct_fail() {
    printf '%s\n' "[run_all] ERROR: $*" >&2
    exit 1
}

fct_main() {
    [[ -d "${BUILD_DIR}" ]] || fct_fail "build dir not found: ${BUILD_DIR} (run ./scripts/build.sh first)"

    local failures=0
    local total=0

    while IFS= read -r -d '' exe; do
        total=$((total + 1))
        fct_log "== ${exe#"${BUILD_DIR}/"} =="
        if ! "${exe}"; then
            fct_log "FAILED: ${exe}"
            failures=$((failures + 1))
        fi
    done < <(find "${BUILD_DIR}" -type f -name '*_example' -perm -u+x -print0 2>/dev/null)

    fct_log "Ran ${total} examples, ${failures} failed."
    [[ "${failures}" -eq 0 ]] || exit 1
}

fct_main "$@"
