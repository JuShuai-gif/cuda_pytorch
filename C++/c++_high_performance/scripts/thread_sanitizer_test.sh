#!/usr/bin/env bash
#
# thread_sanitizer_test.sh - Build with ThreadSanitizer and run *_tests.
# Usage: ./scripts/thread_sanitizer_test.sh [build_dir]
#   build_dir  defaults to ./build-tsan
#
# Note: on Ubuntu 24.04 with recent kernels TSan may abort with
# "unexpected memory mapping" (kernel ASLR); set vm.mmap_rnd_bits=28 or run
# under an older kernel, or use clang.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${1:-${PROJECT_ROOT}/build-tsan}"

fct_log() {
    printf '%s\n' "[thread_sanitizer_test] $*"
}

fct_fail() {
    printf '%s\n' "[thread_sanitizer_test] ERROR: $*" >&2
    exit 1
}

fct_main() {
    fct_log "Configuring with ThreadSanitizer..."
    cmake -S "${PROJECT_ROOT}/src" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DENABLE_THREAD_SANITIZER=ON \
        -DENABLE_BENCHMARKS=OFF "$@"

    fct_log "Building..."
    cmake --build "${BUILD_DIR}" -j

    fct_log "Running tests (TSan)..."
    local failures=0
    local total=0

    while IFS= read -r -d '' t; do
        total=$((total + 1))
        if ! "${t}" >/dev/null 2>&1; then
            fct_log "FAILED (or TSan error): ${t}"
            failures=$((failures + 1))
        fi
    done < <(find "${BUILD_DIR}" -type f -name '*_tests' -perm -u+x -print0 2>/dev/null)

    fct_log "Ran ${total} test binaries under TSan, ${failures} failed."
    [[ "${failures}" -eq 0 ]] || exit 1
}

fct_main "$@"
