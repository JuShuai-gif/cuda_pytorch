#!/usr/bin/env bash
#
# build.sh - Configure and build the project (Release by default).
# Usage: ./scripts/build.sh [build_dir]
#   build_dir  defaults to ./build

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${1:-${PROJECT_ROOT}/build}"

fct_log() {
    printf '%s\n' "[build] $*"
}

fct_main() {
    fct_log "Configuring (Release) into ${BUILD_DIR}..."
    cmake -S "${PROJECT_ROOT}/src" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release "$@"
    fct_log "Building..."
    cmake --build "${BUILD_DIR}" -j
    fct_log "Done. Binaries under ${BUILD_DIR}/<chapter>/"
}

fct_main "$@"
