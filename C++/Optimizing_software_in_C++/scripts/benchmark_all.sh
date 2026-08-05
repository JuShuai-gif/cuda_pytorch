#!/usr/bin/env bash
# benchmark_all.sh -- run the benchmark executables and save results.
# Each run is stored under benchmark_results/<timestamp>/ with metadata.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
OUT_DIR="${ROOT}/benchmark_results/$(date +%Y%m%d_%H%M%S)"

if [ ! -d "${BUILD_DIR}" ]; then
    echo "error: no build directory found; run ./scripts/build.sh first" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

{
    echo "# benchmark results"
    echo "# host:    $(hostname)"
    echo "# cpu:     $(lscpu | grep 'Model name' | sed 's/^[^:]*: *//')"
    echo "# date:    $(date)"
    echo "# compiler: $(g++ --version | head -1)"
    echo "# perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo n/a)"
    echo
} > "${OUT_DIR}/metadata.txt"

failures=0

for exe in $(find "${BUILD_DIR}" -name "*benchmark*" -type f -perm -u+x | sort); do
    name="$(basename "${exe}")"
    echo ""
    echo "== ${name} =="
    if ! "${exe}" > "${OUT_DIR}/${name}.txt" 2>&1; then
        echo "FAILED: ${name}" | tee -a "${OUT_DIR}/summary.txt"
        failures=$((failures + 1))
        continue
    fi
    cat "${OUT_DIR}/${name}.txt"
    echo "saved -> ${OUT_DIR}/${name}.txt"
done

echo ""
echo "results saved under: ${OUT_DIR}"
if [ "${failures}" -ne 0 ]; then
    echo "${failures} benchmark(s) failed (see summary.txt)"
    exit 1
fi
echo "all benchmarks OK"
