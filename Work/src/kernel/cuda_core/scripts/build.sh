#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
BUILD_DIR=${BUILD_DIR:-"$MODULE_DIR/build"}
ARCH=${CUDA_CORE_ARCH:-110}

cmake -S "$MODULE_DIR" -B "$BUILD_DIR" -DCUDA_CORE_ARCH="$ARCH" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$(nproc)"
printf 'binaries in %s/bin\n' "$BUILD_DIR"
