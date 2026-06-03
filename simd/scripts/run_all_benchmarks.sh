#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
BENCH_DIR="$PROJECT_DIR/benchmarks"
OUTPUT_FILE="$BENCH_DIR/latest_results.txt"

PATTERN='^(neon_|avx2_|avx512_|sve_)'

echo "============================================"
echo " SIMD Tutorial - Run All Benchmarks"
echo "============================================"

# Collect all benchmark binaries
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
    echo "ERROR: No benchmark binaries found in $BUILD_DIR/{arm,x86}/"
    echo "       Matching pattern: $PATTERN"
    echo "       Run ./scripts/build.sh first."
    exit 1
fi

echo "Found ${#binaries[@]} benchmark binaries."
echo "Writing results to: $OUTPUT_FILE"

mkdir -p "$BENCH_DIR"

# Write header
{
    echo "# SIMD Benchmark Results"
    echo "# Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# Platform:  $(uname -m)"
    echo "# Kernel:    $(uname -r)"
    echo "# Hostname:  $(hostname)"
    echo ""
    echo "============================================"
} > "$OUTPUT_FILE"

passed=0
failed=0
total=${#binaries[@]}

for bin in "${binaries[@]}"; do
    name=$(basename "$bin")
    printf "  [%3d/%3d] %-50s ... " "$((passed + failed + 1))" "$total" "$name"

    {
        echo ""
        echo "--- $name ---"
        echo ""
    } >> "$OUTPUT_FILE"

    exit_code=0
    output=$("$bin" 2>&1) || exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        echo "PASS"
        ((passed++))
        echo "$output" >> "$OUTPUT_FILE"
    else
        echo "FAIL (exit=$exit_code)"
        ((failed++))
        echo "EXIT CODE: $exit_code" >> "$OUTPUT_FILE"
        echo "$output" >> "$OUTPUT_FILE"
    fi
done

echo ""
echo "============================================"
echo " Results saved to: $OUTPUT_FILE"
echo " Passed: $passed / $total"
echo "============================================"

if [ "$failed" -gt 0 ]; then
    echo "WARNING: $failed benchmark(s) failed."
    exit 1
fi
