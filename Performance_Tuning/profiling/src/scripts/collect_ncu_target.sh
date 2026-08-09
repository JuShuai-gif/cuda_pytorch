#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
binary="${1:-$root/build/cuda_13_kernel_hotspot_bad_good}"
kernel="${2:-slow_kernel}"
output="${3:-$root/ncu_slow_kernel}"
command -v ncu >/dev/null || { echo "SKIP: ncu不存在"; exit 0; }
[[ -x "$binary" ]] || { echo "SKIP: 未找到可执行文件 $binary"; exit 0; }
echo "ncu version: $(ncu --version | tail -n 1)"
echo "device/permission probe:"
if ! ncu --query-metrics > "${output}_metrics.txt" 2> "${output}_query_error.txt"; then
  cat "${output}_query_error.txt"
  echo "ncu无法查询指标。若出现ERR_NVGPUCTRPERM，请由系统管理员按目标机安全策略开放计数器；本脚本不使用sudo。"
  exit 0
fi
echo "metrics saved: ${output}_metrics.txt"
ncu --set full --kernel-name "regex:${kernel}" --launch-skip 0 --launch-count 1 \
  --export "$output" --force-overwrite "$binary"
echo "report: ${output}.ncu-rep"
