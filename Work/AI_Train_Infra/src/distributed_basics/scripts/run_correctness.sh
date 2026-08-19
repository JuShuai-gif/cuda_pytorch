#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
source "$MODULE_DIR/../profiling/scripts/common.sh"

DISTRIBUTED_PYTHON=${DISTRIBUTED_PYTHON:-python3}
DISTRIBUTED_TORCHRUN=${DISTRIBUTED_TORCHRUN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
device=cpu
backend=gloo
variant=baseline
output_root="$MODULE_DIR/artifacts/correctness"

while (($#)); do
  case "$1" in
    --device) device=${2:?}; shift 2 ;;
    --backend) backend=${2:?}; shift 2 ;;
    --variant) variant=${2:?}; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE=${2:?}; shift 2 ;;
    --output-root) output_root=${2:?}; shift 2 ;;
    --) shift; break ;;
    *) profiling_die "unknown argument: $1" ;;
  esac
done
extra_args=("$@")
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || profiling_die "invalid process count"
profiling_require_command "$DISTRIBUTED_TORCHRUN"
run_dir=$(profiling_new_run_dir "$output_root" "${backend}_${variant}")
command_line=(
  env "PYTHONPATH=$MODULE_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
  "$DISTRIBUTED_TORCHRUN" --standalone --nproc-per-node "$NPROC_PER_NODE"
  -m distributed_basics.correctness --device "$device" --backend "$backend"
  --variant "$variant" --output "$run_dir/result.json" "${extra_args[@]}"
)
profiling_write_command "$run_dir/command.txt" "${command_line[@]}"
"${command_line[@]}"
