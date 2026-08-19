#!/usr/bin/env bash
set -euo pipefail

report_base="${1:-/tmp/gpu_basics_nsys}"
shift || true

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is not on PATH; install Nsight Systems on the target machine." >&2
  exit 127
fi

if [[ -e "${report_base}.nsys-rep" || -e "${report_base}.qdrep" ]]; then
  echo "Refusing to overwrite existing Nsight Systems evidence: ${report_base}" >&2
  exit 2
fi

gpu_basics_python="${GPU_BASICS_PYTHON:-python3}"
command -v "$gpu_basics_python" >/dev/null 2>&1 || {
  echo "Python interpreter not found: $gpu_basics_python" >&2
  exit 127
}

nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=none \
  --output="${report_base}" \
  "$gpu_basics_python" -m gpu_basics.profile_workloads \
    --profiler external \
    --device cuda \
    "$@"

[[ -s "${report_base}.nsys-rep" ]] || {
  echo "nsys completed without a non-empty report: ${report_base}.nsys-rep" >&2
  exit 3
}
