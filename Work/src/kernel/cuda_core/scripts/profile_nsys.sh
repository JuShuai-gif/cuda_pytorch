#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
BIN_DIR=${BIN_DIR:-"$MODULE_DIR/build/bin"}

usage() {
  cat <<'EOF'
Usage: profile_nsys.sh <exe> [--output-dir DIR]
  exe: coalescing | bank_conflict | occupancy | async_copy | stream_overlap | graph_launch
EOF
}

exe=${1:-}
[[ -n "$exe" ]] || { usage; exit 2; }
out_dir=${3:-/tmp/cuda_core_nsys}

mkdir -p "$out_dir"
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output "$out_dir/$exe" \
  "$BIN_DIR/$exe"
printf 'open with: nsys-ui %s/%s.nsys-rep\n' "$out_dir" "$exe"
