#!/usr/bin/env bash
#
# clean_build.sh - Remove build directories and rebuild from scratch.
# Usage: ./scripts/clean_build.sh [build_dir]

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${1:-${PROJECT_ROOT}/build}"

fct_log() {
    printf '%s\n' "[clean_build] $*"
}

fct_main() {
    if [[ -d "${BUILD_DIR}" ]]; then
        fct_log "Removing ${BUILD_DIR}..."
        rm -rf "${BUILD_DIR}"
    fi
    fct_log "Rebuilding..."
    "${SCRIPT_DIR}/build.sh" "${BUILD_DIR}"
}

fct_main "$@"
