#!/usr/bin/env bash
# run_all.sh -- run every example executable in the build tree.
# Prints each experiment name; records failures; exits nonzero if any failed.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"

if [ ! -d "${BUILD_DIR}" ]; then
    echo "error: no build directory found; run ./scripts/build.sh first" >&2
    exit 1
fi

failures=0
log=""

for exe in $(find "${BUILD_DIR}" -type f -perm -u+x | sort); do
    # skip internal CMake test binaries and libraries
    case "${exe}" in
        *.a|*.so) continue ;;
        */CMakeFiles/*) continue ;;
        */CMakeFiles) continue ;;
    esac
    name="${exe#${BUILD_DIR}/}"
    echo ""
    echo "=================================================="
    echo "RUN  ${name}"
    echo "=================================================="
    if ! "${exe}" >/tmp/run_out.txt 2>&1; then
        echo "FAILED: ${name}"
        cat /tmp/run_out.txt
        failures=$((failures + 1))
        log="${log}\nFAILED ${name}"
    else
        echo "OK    (${name})"
    fi
    # do not dump huge outputs
    tail -6 /tmp/run_out.txt
done

echo ""
echo "=================================================="
if [ "${failures}" -eq 0 ]; then
    echo "All experiments ran successfully."
    exit 0
else
    echo "${failures} experiment(s) FAILED:"
    echo -e "${log}"
    exit 1
fi
