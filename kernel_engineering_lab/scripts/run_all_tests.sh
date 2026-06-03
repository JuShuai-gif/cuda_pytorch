#!/usr/bin/env bash
#
# run_all_tests.sh - Run all pytest tests in the kernel engineering lab.
# Sets PYTHONPATH to the project root for module discovery.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo " Running Kernel Engineering Lab Tests"
echo " Project root: ${PROJECT_ROOT}"
echo "============================================================"
echo ""

# Run pytest with colors and verbose output
python -m pytest "${PROJECT_ROOT}" \
    -v \
    --tb=short \
    --color=yes \
    -p no:cacheprovider \
    "$@"

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo " All tests passed."
else
    echo " Some tests failed (exit code: ${EXIT_CODE})."
fi
echo "============================================================"

exit $EXIT_CODE
