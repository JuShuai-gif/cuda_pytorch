#!/usr/bin/env bash
set -euo pipefail
command -v valgrind >/dev/null || { echo "valgrind 未安装"; exit 0; }
valgrind --leak-check=full --show-leak-kinds=all "${1:-./build/08_memory_leak}"
