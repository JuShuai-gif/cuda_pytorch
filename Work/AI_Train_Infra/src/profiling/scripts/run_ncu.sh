#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

stage=basic
kernel_regex=
tag=kernel
launch_count=1
nvtx_range=
graph_node=0
hotspot_confirmed=0
profile_from_start_off=0
output_root="$MODULE_DIR/artifacts/ncu"

usage() {
  cat <<'EOF'
Usage: run_ncu.sh --kernel REGEX [options] -- APPLICATION [ARGS...]
  --stage basic|detailed|source|full   default: basic
  --tag SAFE_LABEL                    artifact label
  --launch-count N                    default: 1
  --nvtx-range RANGE                  push/pop range; '/' is appended
  --graph-node                        add --graph-profiling node
  --profile-from-start-off            application calls cudaProfilerStart/Stop
  --hotspot-confirmed                 required beyond basic; confirms nsys + basic were reviewed
  --output-root DIR

Workflow: Nsight Systems -> NCU basic -> detailed/source only for a concrete
question -> full only when replay cost is justified.
EOF
}

while (($#)); do
  case "$1" in
    --stage) stage=${2:?}; shift 2 ;;
    --kernel) kernel_regex=${2:?}; shift 2 ;;
    --tag) tag=${2:?}; shift 2 ;;
    --launch-count) launch_count=${2:?}; shift 2 ;;
    --nvtx-range) nvtx_range=${2:?}; shift 2 ;;
    --graph-node) graph_node=1; shift ;;
    --profile-from-start-off) profile_from_start_off=1; shift ;;
    --hotspot-confirmed) hotspot_confirmed=1; shift ;;
    --output-root) output_root=${2:?}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) profiling_die "unknown argument: $1" ;;
  esac
done
application=("$@")

[[ -n "$kernel_regex" ]] || profiling_die "--kernel REGEX is required (copy the demangled hotspot name from nsys)"
[[ "$stage" =~ ^(basic|detailed|source|full)$ ]] || profiling_die "invalid stage: $stage"
[[ "$launch_count" =~ ^[1-9][0-9]*$ ]] || profiling_die "--launch-count must be a positive integer"
((${#application[@]} > 0)) || profiling_die "application command is required after --"
tag=$(profiling_safe_label "$tag")
if [[ "$stage" != basic && "$hotspot_confirmed" != 1 ]]; then
  profiling_die "--stage $stage requires --hotspot-confirmed after reviewing nsys and NCU basic"
fi
profiling_require_command ncu
profiling_require_command "${application[0]}"

run_dir=$(profiling_new_run_dir "$output_root" "${stage}_${tag}")
mkdir -- "$run_dir/reports" "$run_dir/analysis"
report_base="$run_dir/reports/${stage}_${tag}"
ncu_set=$stage
if [[ "$stage" == source ]]; then
  # NCU 2025.3 exposes no `source` set. Detailed + SourceCounters is the
  # source-attribution recipe on this host; the target also needs lineinfo.
  ncu_set=detailed
fi
ncu_args=(--set "$ncu_set")
if [[ "$stage" == source ]]; then
  ncu_args+=(--section SourceCounters)
fi
ncu_args+=(
  --kernel-name "regex:${kernel_regex}"
  --launch-count "$launch_count"
)
if [[ -n "$nvtx_range" ]]; then
  nvtx_range=${nvtx_range%/}/
  ncu_args+=(--nvtx --nvtx-include "$nvtx_range")
fi
if [[ "$graph_node" == 1 ]]; then
  ncu_args+=(--graph-profiling node)
fi
if [[ "$profile_from_start_off" == 1 ]]; then
  ncu_args+=(--profile-from-start off)
fi
ncu_args+=(-o "$report_base")
command_line=(ncu "${ncu_args[@]}" "${application[@]}")
profiling_write_command "$run_dir/command.txt" "${command_line[@]}"
printf 'Nsight Compute run directory: %s\n' "$run_dir"
"${command_line[@]}"

report_path="${report_base}.ncu-rep"
if [[ ! -s "$report_path" ]]; then
  profiling_die "ncu produced no report; verify kernel regex, NVTX filter, launch selection, and counter permissions"
fi
if ! ncu --import "$report_path" --page details >"$run_dir/analysis/details.txt" 2>"$run_dir/analysis/details.stderr.txt"; then
  printf 'warning: details export failed; raw report remains at %s\n' "$report_path" >&2
fi
printf 'Raw report: %s\n' "$report_path"
printf 'Next: enumerate action.metric_names(); unavailable metrics must remain null, not zero.\n'
