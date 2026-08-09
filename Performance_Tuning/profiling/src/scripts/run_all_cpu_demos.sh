#!/usr/bin/env bash
set -euo pipefail
build="${1:-./build}"
for f in "$build"/[0-9][0-9]_*; do [[ -x "$f" ]] || continue; case "$(basename "$f")" in 08_memory_leak|09_use_after_free|11_io_bottleneck|32_heap_buffer_overflow) echo "SKIP dangerous/IO $(basename "$f")"; continue;; esac; echo "== $(basename "$f") =="; timeout 60s "$f" || echo "退出码 $?"; done
