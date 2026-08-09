#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; build="${1:-$root/build}"
command -v nvidia-smi >/dev/null || { echo "无NVIDIA运行环境，跳过"; exit 0; }
nvidia-smi >/dev/null 2>&1 || { echo "驱动/GPU不可用，跳过"; exit 0; }
for demo in cuda_01_vector_add cuda_02_memory_bound cuda_03_compute_bound cuda_13_kernel_hotspot_bad_good cuda_14_pipeline_overlap_bad_good; do [[ -x "$build/$demo" ]] && "$build/$demo"; done
command -v nsys >/dev/null && nsys profile -t cuda,nvtx,osrt -o "$root/cuda_pathologies" --force-overwrite true "$build/cuda_13_kernel_hotspot_bad_good" || echo "nsys不存在/运行失败"
if command -v ncu >/dev/null; then ncu --query-metrics >/dev/null && ncu --set full --kernel-name regex:slow_kernel --launch-count 1 "$build/cuda_13_kernel_hotspot_bad_good" || echo "ncu权限或GPU环境不满足"; fi
