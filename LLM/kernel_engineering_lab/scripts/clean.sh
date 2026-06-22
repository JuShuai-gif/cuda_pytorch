#!/usr/bin/env bash
#
# clean.sh - Remove all build artifacts, caches, and generated files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Cleaning build artifacts from: ${PROJECT_ROOT} ==="

# Remove Python bytecode files
find "${PROJECT_ROOT}" -type f -name "*.pyc" -delete
find "${PROJECT_ROOT}" -type f -name "*.pyo" -delete
find "${PROJECT_ROOT}" -type f -name "*.pyd" -delete

# Remove __pycache__ directories
find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove pytest cache
rm -rf "${PROJECT_ROOT}/.pytest_cache"

# Remove mypy cache
rm -rf "${PROJECT_ROOT}/.mypy_cache"

# Remove ruff cache
rm -rf "${PROJECT_ROOT}/.ruff_cache"

# Remove coverage artifacts
rm -rf "${PROJECT_ROOT}/.coverage"
rm -rf "${PROJECT_ROOT}/htmlcov"

# Remove build directories
rm -rf "${PROJECT_ROOT}/build"
rm -rf "${PROJECT_ROOT}/dist"
rm -rf "${PROJECT_ROOT}/*.egg-info"

# Remove benchmark output directory
rm -rf "${PROJECT_ROOT}/bench_results"

# Remove CUDA compilation artifacts (if any)
find "${PROJECT_ROOT}" -type f -name "*.cubin" -delete
find "${PROJECT_ROOT}" -type f -name "*.ptx" -delete
find "${PROJECT_ROOT}" -type f -name "*.o" -delete

# Remove Triton kernel caches
rm -rf "${PROJECT_ROOT}/.triton"

echo "=== Clean complete ==="
