#!/usr/bin/env bash
# build.sh -- configure and build all experiments with CMake.
# Stops immediately on any error.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"

mkdir -p "${BUILD_DIR}"
cmake -S "${ROOT}/src" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" "$@"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "== build OK (${BUILD_TYPE}) =="
