#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; build="${1:-$root/build}"
for demo in 22_tlb_pagefault 24_aos_soa 25_allocation_bad_good 31_memory_fragmentation; do [[ -x "$build/$demo" ]] && { echo "===== $demo ====="; "$build/$demo"; }; done
echo "危险的08_memory_leak、09_use_after_free、32_heap_buffer_overflow不会自动运行"
command -v heaptrack >/dev/null || echo "heaptrack不存在：当前环境未验证"
command -v valgrind >/dev/null || echo "valgrind不存在：当前环境未验证"
