#!/usr/bin/env bash
set -euo pipefail
command -v nsys >/dev/null || { echo "nsys 未安装"; exit 0; }
nsys profile -t cuda,nvtx,osrt -o profiling_pipeline --force-overwrite true "${1:-./build/cuda_12_nvtx_pipeline}"
nsys stats profiling_pipeline.nsys-rep
