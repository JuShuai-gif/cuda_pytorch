#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
BIN_DIR=${BIN_DIR:-"$MODULE_DIR/build/bin"}

for exe in coalescing bank_conflict occupancy async_copy stream_overlap graph_launch; do
    echo "===== $exe ====="
    "$BIN_DIR/$exe"
    echo
done
