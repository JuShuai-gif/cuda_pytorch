#!/usr/bin/env bash
# build.sh - configure and build all experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"
BUILD_DIR="${BUILD_DIR:-$SRC/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

extra=()
[[ -n "${ENABLE_NATIVE_OPTIMIZATION:-}" ]] && extra+=(-DENABLE_NATIVE_OPTIMIZATION=ON)
[[ -n "${ENABLE_NUMA_EXAMPLES:-}" ]] && extra+=(-DENABLE_NUMA_EXAMPLES=ON)
[[ -n "${ENABLE_AVX2_EXAMPLES:-}" ]] && extra+=(-DENABLE_AVX2_EXAMPLES=ON)
[[ -n "${ENABLE_AVX512_EXAMPLES:-}" ]] && extra+=(-DENABLE_AVX512_EXAMPLES=ON)

echo "== configure ($BUILD_TYPE) =="
cmake -S "$SRC" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" "${extra[@]}"
echo "== build =="
cmake --build "$BUILD_DIR" -j
echo "== done: binaries in $BUILD_DIR =="
