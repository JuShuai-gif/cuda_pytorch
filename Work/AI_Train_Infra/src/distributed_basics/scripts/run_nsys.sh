#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
source "$MODULE_DIR/../profiling/scripts/common.sh"

DISTRIBUTED_PYTHON=${DISTRIBUTED_PYTHON:-python3}
DISTRIBUTED_TORCHRUN=${DISTRIBUTED_TORCHRUN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
variant=baseline
output_root="$MODULE_DIR/artifacts/nsys"

while (($#)); do
  case "$1" in
    --variant) variant=${2:?}; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE=${2:?}; shift 2 ;;
    --output-root) output_root=${2:?}; shift 2 ;;
    --) shift; break ;;
    *) profiling_die "unknown argument: $1" ;;
  esac
done
extra_args=("$@")
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || profiling_die "invalid process count"
profiling_require_command nsys
profiling_require_command "$DISTRIBUTED_TORCHRUN"
profiling_require_command "$DISTRIBUTED_PYTHON"
run_dir=$(profiling_new_run_dir "$output_root" "nccl_${variant}")
mkdir -- "$run_dir/reports" "$run_dir/analysis"
report_base="$run_dir/reports/timeline"
target=(
  env "PYTHONPATH=$MODULE_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
  "$DISTRIBUTED_TORCHRUN" --standalone --nproc-per-node "$NPROC_PER_NODE"
  -m distributed_basics.profile --profiler nvtx --device cuda --backend nccl
  --variant "$variant" "${extra_args[@]}"
)
command_line=(
  nsys profile --trace=cuda,nvtx,osrt,cublas --sample=none --wait=all
  --force-overwrite=false --output "$report_base" "${target[@]}"
)
profiling_write_command "$run_dir/command.txt" "${command_line[@]}"
"${command_line[@]}"
report_path="${report_base}.nsys-rep"
[[ -s "$report_path" ]] || profiling_die "nsys produced no non-empty report"
if ! nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum "$report_path" \
  >"$run_dir/analysis/stats.txt" 2>"$run_dir/analysis/stats.stderr.txt"; then
  printf 'warning: nsys stats failed; raw report remains at %s\n' "$report_path" >&2
fi
sqlite_path="${report_base}.sqlite"
if [[ -s "$sqlite_path" ]]; then
  env "PYTHONPATH=$MODULE_DIR/..${PYTHONPATH:+:$PYTHONPATH}" \
    "$DISTRIBUTED_PYTHON" -m distributed_basics.analyze_nsys "$sqlite_path" \
    --output "$run_dir/analysis/overlap.json"
else
  printf 'warning: SQLite export is unavailable; run nsys stats or nsys export, then analyze_nsys.py\n' >&2
fi
printf 'Nsight Systems evidence: %s\n' "$run_dir"
