#!/usr/bin/env bash
#
# sanitizer_test.sh - Build and run all *_tests with ASan+UBSan+LSan.
# Usage: ./scripts/sanitizer_test.sh [build_dir]
#   build_dir  defaults to ./build-asan

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${1:-${PROJECT_ROOT}/build-asan}"

fct_log() {
    printf '%s\n' "[sanitizer_test] $*"
}

fct_fail() {
    printf '%s\n' "[sanitizer_test] ERROR: $*" >&2
    exit 1
}

fct_main() {
    fct_log "Configuring with ASan/UBSan/LSan..."
    cmake -S "${PROJECT_ROOT}/src" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DENABLE_SANITIZERS=ON \
        -DENABLE_BENCHMARKS=OFF "$@"

    fct_log "Building..."
    cmake --build "${BUILD_DIR}" -j

    fct_log "Running tests..."
    local failures=0
    local total=0

    while IFS= read -r -d '' t; do
        total=$((total + 1))
        if ! "${t}" >/dev/null 2>&1; then
            fct_log "FAILED (or sanitizer error): ${t}"
            failures=$((failures + 1))
        fi
    done < <(find "${BUILD_DIR}" -type f -name '*_tests' -perm -u+x -print0 2>/dev/null)

    fct_log "Ran ${total} test binaries under sanitizers, ${failures} failed."
    [[ "${failures}" -eq 0 ]] || exit 1
}

fct_main "$@"
