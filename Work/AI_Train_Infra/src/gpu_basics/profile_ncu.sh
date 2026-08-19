#!/usr/bin/env bash
set -euo pipefail

report_base="${1:-/tmp/gpu_basics_ncu}"
workload="${2:-gemm}"
variant="${3:-baseline}"
stage="${4:-basic}"
shift "$(( $# < 4 ? $# : 4 ))"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu is not on PATH; install Nsight Compute on the target machine." >&2
  exit 127
fi

if [[ -e "${report_base}.ncu-rep" ]]; then
  echo "Refusing to overwrite existing Nsight Compute evidence: ${report_base}.ncu-rep" >&2
  exit 2
fi

gpu_basics_python="${GPU_BASICS_PYTHON:-python3}"
command -v "$gpu_basics_python" >/dev/null 2>&1 || {
  echo "Python interpreter not found: $gpu_basics_python" >&2
  exit 127
}

case "${stage}" in
  basic) stage_args=(--set basic) ;;
  detailed) stage_args=(--set detailed) ;;
  # Nsight Compute 2025.3 on the validated Thor has no `source` set;
  # `detailed` already contains SourceCounters. Keep the user-facing stage
  # name while mapping it to flags present on this installation.
  source) stage_args=(--set detailed --section SourceCounters) ;;
  full) stage_args=(--set full) ;;
  *)
    echo "stage must be basic, detailed, source, or full" >&2
    exit 2
    ;;
esac

# Start with basic. Escalate explicitly only after Systems identifies a
# representative launch and a concrete question justifies replay overhead.
ncu \
  --target-processes all \
  "${stage_args[@]}" \
  --nvtx \
  --nvtx-include "gpu_basics_step_${workload}_${variant}/" \
  --kernel-name "regex:${NCU_KERNEL_REGEX:-.*}" \
  --launch-count "${NCU_LAUNCH_COUNT:-1}" \
  --export "${report_base}" \
  "$gpu_basics_python" -m gpu_basics.profile_workloads \
    --profiler external \
    --device cuda \
    --workload "${workload}" \
    --variant "${variant}" \
    "$@"

[[ -s "${report_base}.ncu-rep" ]] || {
  echo "ncu completed without a report; check kernel/NVTX filters and ERR_NVGPUCTRPERM." >&2
  exit 3
}
