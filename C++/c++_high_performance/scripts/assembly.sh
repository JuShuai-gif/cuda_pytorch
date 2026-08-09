#!/usr/bin/env bash
#
# assembly.sh - Generate optimized and unoptimized assembly for a source file.
# Usage: ./scripts/assembly.sh <source.cpp> [std]
#   std   defaults to c++17. Skips Clang if not installed.
#   Output: <basename>.opt.s / .noopt.s / .clang.s in the current directory.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

fct_log() {
    printf '%s\n' "[assembly] $*"
}

fct_fail() {
    printf '%s\n' "[assembly] ERROR: $*" >&2
    exit 1
}

fct_main() {
    [[ $# -ge 1 ]] || fct_fail "usage: ${0##*/} <source.cpp> [std]"
    local src="$1"
    local std="${2:-c++17}"
    [[ -f "${src}" ]] || fct_fail "file not found: ${src}"

    local base
    base="$(basename "${src}" .cpp)"
    fct_log "Generating assembly for ${src} (std=${std}) in $(pwd)..."

    g++ -std="${std}" -O2 -S "${src}" -o "${base}.opt.s"
    fct_log "Optimized (-O2):     ${base}.opt.s"

    g++ -std="${std}" -O0 -S "${src}" -o "${base}.noopt.s"
    fct_log "Unoptimized (-O0):   ${base}.noopt.s"

    if command -v clang++ >/dev/null 2>&1; then
        clang++ -std="${std}" -O2 -S "${src}" -o "${base}.clang.s"
        fct_log "Clang -O2:           ${base}.clang.s"
    else
        fct_log "Clang not installed; skipped."
    fi

    fct_log "Compare optimized vs unoptimized to see what the compiler did."
}

fct_main "$@"
