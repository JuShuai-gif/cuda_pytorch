#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

PATTERN='^(neon_|avx2_|avx512_|sve_)'

echo "============================================"
echo " SIMD Tutorial - Run All Tests"
echo "============================================"

# Collect all test binaries
binaries=()
for subdir in arm x86; do
    if [ -d "$BUILD_DIR/$subdir" ]; then
        while IFS= read -r -d '' f; do
            name=$(basename "$f")
            if [[ "$name" =~ $PATTERN ]]; then
                binaries+=("$f")
            fi
        done < <(find "$BUILD_DIR/$subdir" -maxdepth 1 -type f -executable -print0 2>/dev/null || true)
    fi
done

if [ ${#binaries[@]} -eq 0 ]; then
    echo ""
    echo "ERROR: No test binaries found in $BUILD_DIR/{arm,x86}/"
    echo "       Matching pattern: $PATTERN"
    echo "       Run ./scripts/build.sh first."
    exit 1
fi

echo "Found ${#binaries[@]} test binaries."
echo ""

passed=0
failed=0
total=${#binaries[@]}

for bin in "${binaries[@]}"; do
    name=$(basename "$bin")
    printf "  [%3d/%3d] %-50s ... " "$((passed + failed + 1))" "$total" "$name"

    if "$bin" >/dev/null 2>&1; then
        echo "PASS"
        ((passed++))
    else
        echo "FAIL"
        ((failed++))
    fi
done

echo ""
echo "============================================"
echo " Test Summary: $passed/$total passed"
echo "============================================"

if [ "$failed" -gt 0 ]; then
    echo "FAILURES: $failed test(s) failed."
    exit 1
fi

echo "All tests passed."
