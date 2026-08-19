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
output_root="$MODULE_DIR/artifacts/nsys"

usage() {
  cat <<'EOF'
Usage: run_nsys.sh [options]
  --device cuda|cpu
  --dtype float32|float16|bfloat16
  --hidden N --layers N --batch N --seq-len N
  --steps N
  --output-root DIR

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
    --output-root) output_root=${2:?}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) profiling_die "unknown argument: $1" ;;
  esac
done
extra_args=("$@")

[[ "$steps" =~ ^[1-9][0-9]*$ ]] || profiling_die "invalid steps"
profiling_require_command nsys
profiling_require_command "$PROFILING_PYTHON"

label="inference_${dtype}_b${batch}_s${seq_len}"
run_dir=$(profiling_new_run_dir "$output_root" "$label")
mkdir -- "$run_dir/reports" "$run_dir/analysis"
target=(
  "$PROFILING_PYTHON" "$MODULE_DIR/profile_target.py"
  --device "$device" --dtype "$dtype"
  --hidden "$hidden" --layers "$layers" --batch "$batch" --seq-len "$seq_len"
  --steps "$steps"
  "${extra_args[@]}"
)
report_base="$run_dir/reports/timeline"
command_line=(
  nsys profile
  --trace=cuda,nvtx,osrt
  --sample=none
  --cpuctxsw=none
  --force-overwrite=false
  --output "$report_base"
  "${target[@]}"
)
profiling_write_command "$run_dir/command.txt" "${command_line[@]}"
printf 'Nsight Systems run directory: %s\n' "$run_dir"
"${command_line[@]}"

report_path="${report_base}.nsys-rep"
if [[ -f "$report_path" ]]; then
  if ! nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum "$report_path" \
      >"$run_dir/analysis/stats.txt" 2>"$run_dir/analysis/stats.stderr.txt"; then
    printf 'warning: nsys stats failed; raw report is preserved at %s\n' "$report_path" >&2
  fi
fi
printf 'Open the timeline: nsys-ui %q\n' "$report_path"
