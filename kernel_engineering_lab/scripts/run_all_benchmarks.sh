#!/usr/bin/env bash
#
# run_all_benchmarks.sh - Run all benchmark scripts and generate a summary report.
# Discovers benchmark_*.py files across the project and runs each one.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="${BENCHMARK_DIR:-${PROJECT_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/bench_results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo " Running Kernel Engineering Lab Benchmarks"
echo " Project root: ${PROJECT_ROOT}"
echo " Output dir:   ${OUTPUT_DIR}"
echo " Timestamp:    ${TIMESTAMP}"
echo "============================================================"
echo ""

ALL_RESULTS_JSON="${OUTPUT_DIR}/all_results_${TIMESTAMP}.json"
COMBINED_RESULTS=""

# Collect benchmark scripts (recursively, excluding __pycache__ and hidden dirs)
BENCH_SCRIPTS=$(find "${BENCHMARK_DIR}" \
    -type f \
    -name "benchmark_*.py" \
    ! -path "*__pycache__*" \
    ! -path "*/.git/*" \
    ! -path "*/site-packages/*" \
    | sort)

if [ -z "${BENCH_SCRIPTS}" ]; then
    echo "No benchmark_*.py files found in ${BENCHMARK_DIR}."
    echo "Skipping benchmarks."
    exit 0
fi

echo "Found benchmark scripts:"
for script in ${BENCH_SCRIPTS}; do
    echo "  - ${script}"
done
echo ""

FAILED=0
TOTAL=0

for script in ${BENCH_SCRIPTS}; do
    TOTAL=$((TOTAL + 1))
    SCRIPT_NAME=$(basename "${script}" .py)
    RESULT_FILE="${OUTPUT_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.json"

    echo "--- Running: ${script} ---"
    echo "  Output: ${RESULT_FILE}"

    if python "${script}" --output "${RESULT_FILE}" 2>&1; then
        echo "  [PASS] ${script}"
        if [ -f "${RESULT_FILE}" ]; then
            COMBINED_RESULTS="${COMBINED_RESULTS} ${RESULT_FILE}"
        fi
    else
        echo "  [FAIL] ${script}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "============================================================"
echo " Benchmark Summary"
echo " Total:  ${TOTAL}"
echo " Passed: $((TOTAL - FAILED))"
echo " Failed: ${FAILED}"
echo "============================================================"

# Generate combined report if we have results
if [ -n "${COMBINED_RESULTS}" ] && [ "${FAILED}" -lt "${TOTAL}" ]; then
    echo ""
    echo "Generating combined report..."

    python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from benchmarks.report import main as report_main
import argparse

# Combine results from all runs
all_data = []
for path in '${COMBINED_RESULTS}'.split():
    if path.strip():
        import json
        with open(path.strip()) as f:
            all_data.extend(json.load(f))

import json
with open('${ALL_RESULTS_JSON}', 'w') as f:
    json.dump(all_data, f, indent=2)

print(f'Combined {len(all_data)} results into: ${ALL_RESULTS_JSON}')
"

    # Generate Markdown report
    python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from benchmarks.benchmark_utils import load_results, compare_kernels, generate_report

results = load_results('${ALL_RESULTS_JSON}')
compare_kernels(results)
generate_report(results, '${OUTPUT_DIR}/report_${TIMESTAMP}')
"
fi

exit $FAILED
