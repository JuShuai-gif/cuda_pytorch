#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <binary_name> [args...]"
    echo ""
    echo "  Runs the binary under perf stat with relevant CPU performance counters."
    echo "  Provides interpretation of key metrics after execution."
    echo ""
    echo "Examples:"
    echo "  $0 build/x86/avx2_saxpy"
    echo "  $0 build/arm/neon_saxpy --size 1048576"
    exit 1
fi

BINARY="$1"
shift || true
BINARY_ARGS=("$@")

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found: $BINARY"
    exit 1
fi

if ! command -v perf &>/dev/null; then
    echo "ERROR: perf is not installed."
    echo "  Install with: sudo apt install linux-tools-common linux-tools-generic  (Ubuntu/Debian)"
    echo "                sudo dnf install perf                                  (Fedora)"
    exit 1
fi

# Check if perf can run (paranoid level)
if ! perf stat -e cycles:u true 2>/dev/null; then
    echo "WARNING: perf may not have sufficient permissions."
    echo "  Check /proc/sys/kernel/perf_event_paranoid (0=allow, 2=default restricted)."
    echo "  Temporarily fix: sudo sysctl kernel.perf_event_paranoid=0"
    echo "  Or run this script with sudo."
    echo ""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " perf stat: $(basename "$BINARY")"
echo "============================================"
echo "Binary:  $BINARY"
echo "Args:    ${BINARY_ARGS[*]:-(none)}"
echo ""

# --- Event selection ---
# Try advanced SIMD events first, fall back to basic set if not available

BASE_EVENTS="cycles,instructions,cache-references,cache-misses,branches,branch-misses"
MEM_EVENTS="L1-dcache-loads,L1-dcache-load-misses"

# Check if SIMD-specific events are available
HAS_SIMD_EVENTS=false
if perf list 2>/dev/null | grep -q "simd_inst_retired"; then
    HAS_SIMD_EVENTS=true
fi
if perf list 2>/dev/null | grep -q "mem_load_retired.l1_hit"; then
    HAS_MEM_L1_EVENTS=true
else
    HAS_MEM_L1_EVENTS=false
fi

SIMD_EVENTS=""
if [ "$HAS_SIMD_EVENTS" = true ]; then
    SIMD_EVENTS=",simd_inst_retired.any"

    if [ "$HAS_MEM_L1_EVENTS" = true ]; then
        SIMD_EVENTS+=",mem_load_retired.l1_hit,mem_load_retired.l1_miss"
    fi

    echo "SIMD events: AVAILABLE (full counter set)"
    EVENTS="$BASE_EVENTS,$MEM_EVENTS$SIMD_EVENTS"
else
    echo "SIMD events: NOT available (using basic counter set)"
    EVENTS="$BASE_EVENTS,$MEM_EVENTS"
fi

echo "Counters:  $EVENTS"
echo ""

# Run perf stat and capture output
PERF_OUTPUT=$(mktemp /tmp/perf_stat_output.XXXXXX)
trap 'rm -f "$PERF_OUTPUT"' EXIT

set +e
perf stat \
    -e "$EVENTS" \
    -x ';' \
    -- "$BINARY" "${BINARY_ARGS[@]}" >/dev/null 2>"$PERF_OUTPUT"
PERF_EXIT_CODE=$?
set -e

# Parse perf stat CSV output
declare -A COUNTS
while IFS=';' read -r value name _runtime; do
    name=$(echo "$name" | xargs)  # trim whitespace
    value=$(echo "$value" | xargs)
    # Skip empty lines and the "unit" lines
    if [ -n "$name" ] && [ "$value" != "<not counted>" ] && [ "$value" != "<not supported>" ]; then
        COUNTS["$name"]="$value"
    fi
done < "$PERF_OUTPUT"

# Read parsed counters
CYCLES="${COUNTS[cycles]:-0}"
INSTRUCTIONS="${COUNTS[instructions]:-0}"
CACHE_REFS="${COUNTS[cache-references]:-1}"
CACHE_MISSES="${COUNTS[cache-misses]:-0}"
BRANCHES="${COUNTS[branches]:-0}"
BRANCH_MISSES="${COUNTS[branch-misses]:-0}"
L1_LOADS="${COUNTS[L1-dcache-loads]:-0}"
L1_MISSES="${COUNTS[L1-dcache-load-misses]:-0}"
SIMD_INST="${COUNTS[simd_inst_retired.any]:-0}"
MEM_L1_HIT="${COUNTS[mem_load_retired.l1_hit]:-0}"
MEM_L1_MISS="${COUNTS[mem_load_retired.l1_miss]:-0}"

# --- Compute derived metrics ---
echo ""
echo "============================================"
echo " Metrics Calculation"
echo "============================================"
echo ""

# IPC
if [ "$CYCLES" != "0" ]; then
    IPC=$(awk "BEGIN { printf \"%.2f\", $INSTRUCTIONS / $CYCLES }")
else
    IPC="N/A"
fi
echo "  IPC (instructions per cycle):  $IPC"

# Cache miss rate
if [ "$CACHE_REFS" != "0" ]; then
    CACHE_MISS_RATE=$(awk "BEGIN { printf \"%.2f%%\", ($CACHE_MISSES / $CACHE_REFS) * 100 }")
else
    CACHE_MISS_RATE="N/A"
fi
echo "  Cache miss rate:               $CACHE_MISS_RATE"

# Branch miss rate
if [ "$BRANCHES" != "0" ]; then
    BRANCH_MISS_RATE=$(awk "BEGIN { printf \"%.2f%%\", ($BRANCH_MISSES / $BRANCHES) * 100 }")
else
    BRANCH_MISS_RATE="N/A"
fi
echo "  Branch miss rate:              $BRANCH_MISS_RATE"

# L1 load miss rate
if [ "$L1_LOADS" != "0" ]; then
    L1_MISS_RATE=$(awk "BEGIN { printf \"%.2f%%\", ($L1_MISSES / $L1_LOADS) * 100 }")
    echo "  L1 D-cache miss rate:          $L1_MISS_RATE"
fi

# SIMD ratio
if [ "$SIMD_INST" != "0" ] && [ "$INSTRUCTIONS" != "0" ]; then
    SIMD_RATIO=$(awk "BEGIN { printf \"%.2f%%\", ($SIMD_INST / $INSTRUCTIONS) * 100 }")
    echo "  SIMD instruction ratio:        $SIMD_RATIO"
fi

# L1 hit ratio for memory loads
if [ "$MEM_L1_HIT" != "0" ] || [ "$MEM_L1_MISS" != "0" ]; then
    MEM_TOTAL=$(awk "BEGIN { print $MEM_L1_HIT + $MEM_L1_MISS }")
    if [ "$MEM_TOTAL" != "0" ]; then
        MEM_L1_HIT_RATE=$(awk "BEGIN { printf \"%.2f%%\", ($MEM_L1_HIT / $MEM_TOTAL) * 100 }")
        echo "  Mem load L1 hit rate:          $MEM_L1_HIT_RATE"
    fi
fi

echo ""

# --- Interpretation ---
echo "============================================"
echo " Interpretation"
echo "============================================"
echo ""

IPC_NUM=$(echo "$IPC" | sed 's/[^0-9.]//g' || echo "0")

if [ "$IPC" = "N/A" ]; then
    echo "  [!] Could not compute IPC."
elif [ "$(echo "$IPC_NUM > 2.0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
    echo "  >> IPC > 2.0  => Compute-bound. CPU is issuing multiple instructions"
    echo "     per cycle; likely saturating execution units efficiently."
elif [ "$(echo "$IPC_NUM >= 1.0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
    echo "  >> IPC 1.0-2.0 => Moderate efficiency. Some pipeline stalls but"
    echo "     generally keeping execution units busy."
elif [ "$(echo "$IPC_NUM >= 0.5" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
    echo "  >> IPC 0.5-1.0 => Somewhat stalled. Possibly branch-heavy or"
    echo "     data-dependent chains limiting parallelism."
else
    echo "  >> IPC < 0.5   => Heavily stalled! Likely memory-bound: the CPU is"
    echo "     waiting on cache misses and main memory. Check cache miss rate below."
fi

echo ""

CACHE_MISS_NUM=$(echo "$CACHE_MISS_RATE" | sed 's/[^0-9.]//g' || echo "0")
if [ "$CACHE_MISS_RATE" != "N/A" ]; then
    if [ "$(echo "$CACHE_MISS_NUM > 10.0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        echo "  >> Cache miss rate > 10%  => Likely MEMORY-BOUND. The CPU is"
        echo "     frequently waiting on data from main memory."
    elif [ "$(echo "$CACHE_MISS_NUM > 3.0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        echo "  >> Cache miss rate 3-10%  => Moderate memory pressure."
        echo "     Consider blocking/tiling to improve locality."
    else
        echo "  >> Cache miss rate < 3%   => Good cache utilization. Data fits well"
        echo "     in caches; the workload is compute-bound or cache-friendly."
    fi
fi

echo ""

echo "  --- Classification ---"
echo ""
echo "  Compute-bound:  High IPC (>1.5), low cache misses (<3%)"
echo "                  -> Speedup from SIMD will be close to theoretical (e.g.,"
echo "                     4x for NEON f32, 8x for AVX2 f32, 16x for AVX-512 f32)."
echo ""
echo "  Memory-bound:   Low IPC (<0.5), high cache misses (>10%)"
echo "                  -> Speedup from SIMD limited (1.1x-2x). The bottleneck"
echo "                     is data movement, not computation. Consider:"
echo "                      - Software prefetching"
echo "                      - Cache blocking / tiling"
echo "                      - Non-temporal stores (streaming)"
echo "                      - Aligned allocations"
echo ""
echo "  Mixed:          Moderate IPC, moderate cache misses"
echo "                  -> Speedup typically 2x to theoretical_width."
echo "                     Benefits from both SIMD vectorization and"
echo "                     improved memory access patterns."

echo ""
echo "============================================"
echo " Raw Counters"
echo "============================================"
for event in "${!COUNTS[@]}"; do
    printf "  %-35s %s\n" "$event" "${COUNTS[$event]}"
done

echo ""
echo "Exit code: $PERF_EXIT_CODE"
echo "Done."
