#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <binary_name>"
    echo ""
    echo "  Disassembles the given binary and filters for SIMD-related functions."
    echo "  SIMD instructions are highlighted in green, scalar in default color."
    echo ""
    echo "Examples:"
    echo "  $0 build/x86/avx2_saxpy"
    echo "  $0 build/arm/neon_saxpy"
    exit 1
fi

BINARY="$1"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found: $BINARY"
    exit 1
fi

if [ ! -x "$BINARY" ]; then
    echo "WARNING: Binary is not executable: $BINARY"
fi

if ! command -v objdump &>/dev/null; then
    echo "ERROR: objdump not found. Install binutils."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " Disassembly of: $BINARY"
echo "============================================"

# Detect binary format and choose appropriate disassembly syntax
FILE_TYPE=$(file "$BINARY")
if echo "$FILE_TYPE" | grep -q "ARM aarch64"; then
    ARCH="arm"
    DISASM_FLAGS="-d"
    echo "Architecture: ARM aarch64"
elif echo "$FILE_TYPE" | grep -q "x86-64"; then
    ARCH="x86"
    DISASM_FLAGS="-d -M intel"
    echo "Architecture: x86-64 (Intel syntax)"
else
    ARCH="unknown"
    DISASM_FLAGS="-d"
    echo "Architecture: unknown (default syntax)"
fi

echo ""
echo "--- Filtering for SIMD functions ---"

# Green = SIMD instructions, default = scalar
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Run objdump and filter for relevant function symbols
# We first find the address ranges of target functions, then print instructions within those ranges
objdump $DISASM_FLAGS "$BINARY" 2>/dev/null | awk -v green="$GREEN" -v nc="$NC" '
BEGIN {
    in_target = 0
}

# Detect start of a target function (label ending with : and matching pattern)
/^[0-9a-f]+ <(scalar_|neon_|avx2_|avx512_|sve_)/ {
    in_target = 1
    print ""
}

# Detect end of current function (next label that is not part of a target function)
/^[0-9a-f]+ </ {
    if (in_target && $0 !~ /<(scalar_|neon_|avx2_|avx512_|sve_)/) {
        in_target = 0
    }
}

# Print disassembly lines within target functions
in_target {
    # Attempt to highlight SIMD instructions
    # x86 SIMD: vaddps, vmulps, vmovaps, vfmadd, addps, mulps, movaps, etc.
    # ARM SIMD: fadd v, fmul v, fmla v, ld1 {v, st1 {v, dup v, etc.
    if ($0 ~ /(v(add|sub|mul|div|mov|fmadd|fmsub|fnmadd|fnmsub|broadcast|shuf|perm|extract|insert|zero|and|or|xor|not|cmp|min|max|sqrt|rsqrt|rcp|cvt|gather|scatter|pternlog|blend|mask|load|store|compress|expand)[psd]|[a-z]+p[sd]|fmadd|fmsub|fnmadd|fnmsub|faddp|fmulp|fminp|fmaxp|ld[0-9].*\{v|st[0-9].*\{v|fadd.*v[0-9]|fmul.*v[0-9]|fmla.*v[0-9]|fmls.*v[0-9]|dup.*v[0-9]|rev[0-9].*v[0-9]|trn[0-9].*v[0-9]|zip[0-9].*v[0-9]|uzp[0-9].*v[0-9])/) {
        printf "%s%s%s\n", green, $0, nc
    } else {
        print $0
    }
}
'

echo ""
echo "============================================"
echo " Pro Tips"
echo "============================================"
echo ""
echo "1) View source-interleaved disassembly (if compiled with -g):"
echo "   objdump -d -S -M intel \"$BINARY\" | less"
echo ""
echo "2) Compiler Explorer equivalent:"
echo "   Build with CMAKE_BUILD_TYPE=RelWithDebInfo"
echo "   Add -fverbose-asm to CFLAGS/CXXFLAGS to get variable names in comments"
echo "   Or paste the source at https://godbolt.org/"
echo ""
echo "3) Show only specific function:"
echo "   objdump -d -M intel \"$BINARY\" | sed -n '/<my_func>:/,/^$/p'"
echo ""
echo "4) Get symbol list:"
echo "   nm \"$BINARY\" | grep -E 'scalar_|neon_|avx2_|avx512_|sve_'"
