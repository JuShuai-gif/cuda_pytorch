#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
PROFILING_PYTHON=${PROFILING_PYTHON:-python3}
command -v "$PROFILING_PYTHON" >/dev/null 2>&1 || { printf 'error: Python not found: %s\n' "$PROFILING_PYTHON" >&2; exit 2; }

exec "$PROFILING_PYTHON" "$MODULE_DIR/benchmark.py" "$@"
