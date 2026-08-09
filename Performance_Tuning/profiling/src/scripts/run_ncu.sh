#!/usr/bin/env bash
set -euo pipefail
command -v ncu >/dev/null || { echo "ncu 未安装"; exit 0; }
ncu --set full "${1:-./build/cuda_02_memory_bound}"
