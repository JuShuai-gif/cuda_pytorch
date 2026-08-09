#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-all-safe}"

run_memory(){
  if command -v heaptrack >/dev/null; then
    heaptrack "$root/build/25_allocation_bad_good"
  else echo "SKIP heaptrack: 未安装"; fi
  if command -v valgrind >/dev/null; then
    valgrind --tool=massif --massif-out-file="$root/massif.out" "$root/build/25_allocation_bad_good"
    echo "Massif report: ms_print $root/massif.out"
  else echo "SKIP valgrind: 未安装"; fi
  echo "Leak/UAF/Overflow危险实验不会由此脚本自动运行。"
}

run_fio(){
  if command -v fio >/dev/null; then
    fio "$root/configs/fio_safe_64m.fio" --output="$root/fio_safe_result.txt"
    echo "fio result: $root/fio_safe_result.txt"
    echo "临时文件保留在/tmp/profiling-lab-fio-safe.dat，确认后可手动删除。"
  else echo "SKIP fio: 未安装"; fi
}

run_realtime(){
  if command -v cyclictest >/dev/null; then
    # 不设置实时优先级，避免普通用户权限和系统影响；目标机可经授权另测-p。
    cyclictest -t 4 -n -i 1000 -l 100000 -q > "$root/cyclictest_result.txt" || true
    echo "cyclictest result: $root/cyclictest_result.txt"
  else echo "SKIP cyclictest: 未安装"; fi
}

case "$mode" in
  memory) run_memory;;
  fio) run_fio;;
  realtime) run_realtime;;
  all-safe) run_memory; run_fio; run_realtime;;
  *) echo "usage: $0 [memory|fio|realtime|all-safe]"; exit 2;;
esac
