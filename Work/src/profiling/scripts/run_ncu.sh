#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

PROFILING_PYTHON=${PROFILING_PYTHON:-python3}
device=cuda
dtype=float32
hidden=1024
layers=4
batch=1
seq_len=1
steps=3
# ncu --set: basic | detailed | source | full (see `ncu --list-sets`)
set_name=basic
output_root="$MODULE_DIR/artifacts/ncu"

usage() {
  cat <<'EOF'
Usage: run_ncu.sh [options]
  --device cuda|cpu
  --dtype float32|float16|bfloat16
  --hidden N --layers N --batch N --seq-len N
  --steps N
  --set basic|detailed|source|full
  --output-root DIR

The profiled process is profile_target.py; ncu replays kernels so its whole-run
wall time is NOT a valid latency measurement.  Use it only to read kernel
counters (DRAM/L2 throughput, occupancy, Tensor Core, warp stall).

Extra profile_target.py arguments may be placed after --.
EOF
}

while (($#)); do
  case "$1" in
    --device) device=${2:?}; shift 2 ;;
    --dtype) dtype=${2:?}; shift 2 ;;
    --hidden) hidden=${2:?}; shift 2 ;;
    --layers) layers=${2:?}; shift 2 ;;
    --batch) batch=${2:?}; shift 2 ;;
    --seq-len) seq_len=${2:?}; shift 2 ;;
    --steps) steps=${2:?}; shift 2 ;;
    --set) set_name=${2:?}; shift 2 ;;
    --output-root) output_root=${2:?}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) profiling_die "unknown argument: $1" ;;
  esac
done
extra_args=("$@")

[[ "$steps" =~ ^[1-9][0-9]*$ ]] || profiling_die "invalid steps"
profiling_require_command ncu
profiling_require_command "$PROFILING_PYTHON"

label="inference_${dtype}_b${batch}_s${seq_len}_${set_name}"
run_dir=$(profiling_new_run_dir "$output_root" "$label")
mkdir -- "$run_dir/reports"
target=(
  "$PROFILING_PYTHON" "$MODULE_DIR/profile_target.py"
  --device "$device" --dtype "$dtype"
  --hidden "$hidden" --layers "$layers" --batch "$batch" --seq-len "$seq_len"
  --steps "$steps"
  "${extra_args[@]}"
)
report_path="$run_dir/reports/report"
command_line=(
  ncu
  --set "$set_name"
  --launch-count 1
  --launch-skip 1
  --export "$report_path"
  --force-overwrite
  "${target[@]}"
)
profiling_write_command "$run_dir/command.txt" "${command_line[@]}"
printf 'Nsight Compute run directory: %s\n' "$run_dir"
"${command_line[@]}"

printf 'Open the report: ncu-ui %q\n' "${report_path}.ncu-rep"
