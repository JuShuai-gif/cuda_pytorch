#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# profile.sh -- Comprehensive SIMD kernel profiling
#
# Usage:
#   profile.sh record   <binary> [args]     Sampling profile (perf record + report)
#   profile.sh annotate <binary> [args]     Instruction-level hotspot (perf annotate)
#   profile.sh cache    <binary> [args]     Cache simulation (cachegrind)
#   profile.sh flame    <binary> [args]     Flame graph generation
#   profile.sh mem      <binary> [args]     Memory access pattern (perf mem)
#   profile.sh topdown  <binary> [args]     Intel Top-Down microarchitecture analysis
#   profile.sh all      <binary> [args]     Run all profiling modes
#
# Prerequisites:
#   sudo apt install linux-tools-common linux-tools-generic  (perf)
#   sudo apt install valgrind                                 (cachegrind)
#   git clone https://github.com/brendangregg/FlameGraph     (flamegraph)
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROFILE_DIR="$PROJECT_DIR/benchmarks/profile_output"
FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-$HOME/tools/FlameGraph}"

usage() {
    cat <<EOF
Usage: $0 <mode> <binary> [args...]

Modes:
  record       Sampling-based profiling (perf record + report)
  annotate     Instruction-level hotspot with source (perf annotate)
  cache        Cache simulation with cachegrind
  flame        Generate flame graph (requires FlameGraph tools)
  mem          Memory access latency profiling (perf mem record/report)
  topdown      Intel Top-Down Microarchitecture Analysis (perf stat --topdown)
  all          Run record + annotate + cache in sequence

Binary should be a SIMD benchmark executable (e.g., build/x86/avx2_dot_product).

Examples:
  $0 record   build/x86/avx2_layernorm
  $0 annotate build/x86/avx2_int8_dot
  $0 cache    build/x86/avx2_dot_product
  $0 flame    build/x86/avx2_softmax_partial
  $0 all      build/x86/avx2_reduce_sum
EOF
    exit 0
}

die() { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }
info() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

check_tool() {
    command -v "$1" &>/dev/null || die "$1 is not installed. $2"
}

check_binary() {
    [ -f "$1" ] || die "Binary not found: $1"
    [ -x "$1" ] || die "Binary is not executable: $1"
}

set_perf_paranoid() {
    local level
    level=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "3")
    if [ "$level" -gt 1 ]; then
        warn "perf_event_paranoid = $level (restricted)."
        echo "  Sampling/annotation may fail. To fix temporarily:"
        echo "    sudo sysctl kernel.perf_event_paranoid=-1"
        echo "  Or run this script with sudo."
        echo ""
    fi
}

# ---- record: sampling-based profiling ----
do_record() {
    local binary="$1"; shift
    local out="$PROFILE_DIR/perf.data"
    local report_out="$PROFILE_DIR/perf_report.txt"

    mkdir -p "$PROFILE_DIR"
    check_binary "$binary"
    check_tool perf "Install: sudo apt install linux-tools-generic"
    set_perf_paranoid

    info "Recording samples for: $binary"
    echo "  Output: $out"
    echo ""

    perf record \
        -g \
        -F 999 \
        --call-graph dwarf \
        -e cycles:u,instructions:u,cache-misses:u \
        -o "$out" \
        -- "$binary" "$@"

    ok "Recording complete. Samples saved to $out"

    info "Generating report..."
    echo ""
    perf report \
        -i "$out" \
        --stdio \
        --sort comm,dso,symbol \
        --show-total-period \
        --percent-limit 1 \
        | head -100 \
        | tee "$report_out"

    echo ""
    ok "Report saved to $report_out"
    echo ""
    echo "Next steps:"
    echo "  perf report -i $out              # interactive TUI"
    echo "  perf report -i $out --stdio -n   # annotated with sample counts"
    echo "  perf script -i $out > out.perf-script  # for flame graph"
}

# ---- annotate: instruction-level analysis ----
do_annotate() {
    local binary="$1"; shift
    local out="$PROFILE_DIR/perf_annotate.txt"

    mkdir -p "$PROFILE_DIR"
    check_binary "$binary"
    check_tool perf "Install: sudo apt install linux-tools-generic"
    set_perf_paranoid

    local tmp_data="$PROFILE_DIR/perf_annotate.data"

    info "Recording with callgraph for annotation: $binary"
    perf record \
        -g \
        -F 999 \
        -e cycles:u \
        -o "$tmp_data" \
        -- "$binary" "$@"

    ok "Recording complete."

    info "Annotating hot functions..."
    echo ""

    # Find the top functions in the binary (not libc)
    perf report -i "$tmp_data" --stdio --sort symbol -n 2>/dev/null \
        | grep -E '^\s+[0-9]+\.[0-9]+%.*'"$(basename "$binary")" \
        | head -5 \
        | awk '{print $NF}' \
        | while read -r func; do
            echo "--- $func ---"
            perf annotate -i "$tmp_data" --stdio -l "$func" 2>/dev/null | head -80
            echo ""
        done | tee "$out"

    echo ""
    ok "Annotation saved to $out"
    echo ""
    echo "For interactive annotation:"
    echo "  perf annotate -i $tmp_data    # TUI mode"
    echo ""
    echo "For objdump cross-reference:"
    echo "  objdump -d -S --disassemble='$(basename "$binary")' $binary | less"
}

# ---- cache: cachegrind simulation ----
do_cache() {
    local binary="$1"; shift
    local out="$PROFILE_DIR/cachegrind.txt"

    mkdir -p "$PROFILE_DIR"
    check_binary "$binary"
    check_tool valgrind "Install: sudo apt install valgrind"

    info "Running cachegrind on: $binary"
    echo "  This simulates L1/L2/LL cache behavior (may be 10-50x slower)."
    echo ""

    valgrind \
        --tool=cachegrind \
        --cachegrind-out-file="$PROFILE_DIR/cachegrind.out.%p" \
        --cache-sim=yes \
        --branch-sim=yes \
        --log-file="$out" \
        "$binary" "$@"

    local cgout
    cgout=$(ls -t "$PROFILE_DIR"/cachegrind.out.* 2>/dev/null | head -1)

    if [ -n "$cgout" ] && command -v cg_annotate &>/dev/null; then
        echo ""
        info "Annotating cachegrind output..."
        cg_annotate --auto=yes "$cgout" | head -120 >> "$out"
    fi

    ok "Cache simulation complete."
    echo "  Log: $out"
    echo ""
    echo "Interpreting results:"
    echo "  D1mr / D1mw  = L1 data cache read/write misses"
    echo "  DLmr / DLmw  = Last-level cache (LLC) read/write misses"
    echo "  High D1mr (>3%) with low DLmr => L2 resident, consider blocking for L1"
    echo "  High DLmr (>5%)           => Memory-bound, needs access pattern redesign"
}

# ---- flame: flame graph generation ----
do_flame() {
    local binary="$1"; shift
    mkdir -p "$PROFILE_DIR"
    check_binary "$binary"
    check_tool perf "Install: sudo apt install linux-tools-generic"
    set_perf_paranoid

    local data="$PROFILE_DIR/perf_flame.data"
    local folded="$PROFILE_DIR/perf_folded.txt"
    local flame="$PROFILE_DIR/flamegraph.svg"

    info "Recording for flame graph: $binary"
    perf record -g -F 99 -e cycles:u -o "$data" -- "$binary" "$@"

    if [ ! -f "$data" ]; then
        die "perf record failed to produce data."
    fi

    ok "Recording complete."

    info "Generating flame graph..."
    if [ ! -f "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" ]; then
        warn "FlameGraph tools not found at $FLAMEGRAPH_DIR"
        echo "  Clone them first:"
        echo "    git clone https://github.com/brendangregg/FlameGraph.git $FLAMEGRAPH_DIR"
        echo ""
        echo "  You can still do manual steps:"
        echo "    perf script -i $data > $PROFILE_DIR/perf.script"
        echo "    stackcollapse-perf.pl $PROFILE_DIR/perf.script | flamegraph.pl > flame.svg"
        return 0
    fi

    perf script -i "$data" \
        | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" \
        > "$folded"

    "$FLAMEGRAPH_DIR/flamegraph.pl" \
        --title "$(basename "$binary") CPU Flame Graph" \
        --width 1200 \
        "$folded" \
        > "$flame"

    ok "Flame graph generated: $flame"
    echo "  Open with: xdg-open $flame  or  firefox $flame"
}

# ---- mem: memory access profiling ----
do_mem() {
    local binary="$1"; shift
    local out="$PROFILE_DIR/perf_mem.txt"

    mkdir -p "$PROFILE_DIR"
    check_binary "$binary"
    check_tool perf "Install: sudo apt install linux-tools-generic"
    set_perf_paranoid

    info "Recording memory access latency: $binary"
    echo "  Note: requires sudo or perf_event_paranoid=0 on most systems"
    echo ""

    local data="$PROFILE_DIR/perf_mem.data"

    perf mem record -o "$data" -- "$binary" "$@" 2>/dev/null || {
        warn "perf mem record failed (may need root). Falling back to perf stat -d."
        perf stat -d -- "$binary" "$@" 2>&1 | tee "$out"
        return 0
    }

    ok "Memory recording complete."

    perf mem report -i "$data" --stdio 2>/dev/null | head -80 | tee "$out"

    ok "Memory latency report saved to $out"
}

# ---- topdown: Intel Top-Down analysis ----
do_topdown() {
    local binary="$1"; shift
    local out="$PROFILE_DIR/perf_topdown.txt"

    mkdir -p "$PROFILE_DIR"
    check_binary "$binary"
    check_tool perf "Install: sudo apt install linux-tools-generic"
    set_perf_paranoid

    info "Running Intel Top-Down Microarchitecture Analysis"
    echo ""

    # Try Top-Down L1 (available on Intel since ICL, AMD Zen3+)
    if perf stat --topdown -a true 2>/dev/null; then
        perf stat --topdown -- "$binary" "$@" 2>&1 | tee "$out"
    else
        warn "Top-Down not supported on this CPU. Using basic metric group."
        # Fallback: use metric groups
        perf stat \
            -M TopdownL1 \
            -- "$binary" "$@" 2>&1 | tee "$out" || {
            warn "Top-Down metrics unavailable. See perf list metric."
            # Last resort: basic bottleneck analysis
            perf stat \
                -e cycles,instructions, stalled-cycles-frontend,stalled-cycles-backend \
                -- "$binary" "$@" 2>&1 | tee "$out"
            echo ""
            echo "Frontend bound = stalled-cycles-frontend / cycles"
            echo "Backend bound  = stalled-cycles-backend / cycles"
            echo "Retiring        = 1 - (frontend + backend)"
        }
    fi

    ok "Top-Down analysis saved to $out"
}

# ---- all: run everything ----
do_all() {
    local binary="$1"; shift
    echo "============================================"
    echo " Full Profiling Pipeline"
    echo " Binary: $binary"
    echo "============================================"

    do_record "$binary" "$@"
    echo ""
    echo "============================================"
    do_annotate "$binary" "$@"
    echo ""
    echo "============================================"
    do_cache "$binary" "$@"
    echo ""
    echo "============================================"
    echo "All profiles saved to: $PROFILE_DIR"
    echo ""
    echo "Summary of what to look at:"
    echo "  1. perf_report.txt  -- which functions take the most CPU time"
    echo "  2. perf_annotate.txt -- which instructions within hot functions are costly"
    echo "  3. cachegrind.txt    -- cache miss rates (L1/L2/LLC)"
    echo ""
    echo "Decision matrix:"
    echo "  High IPC + low cache misses  --> COMPUTE bound (SIMD gives linear speedup)"
    echo "  Low IPC + high cache misses  --> MEMORY bound (fix data layout first)"
    echo "  Frontend stalls               --> I-cache, branch predictor, or decode issues"
    echo "  Backend stalls                --> Data cache misses or execution port pressure"
}

# ---- main ----
[ $# -lt 2 ] && usage
MODE="$1"; BINARY="$2"; shift 2 || true

case "$MODE" in
    record)   do_record   "$BINARY" "$@" ;;
    annotate) do_annotate "$BINARY" "$@" ;;
    cache)    do_cache    "$BINARY" "$@" ;;
    flame)    do_flame    "$BINARY" "$@" ;;
    mem)      do_mem      "$BINARY" "$@" ;;
    topdown)  do_topdown  "$BINARY" "$@" ;;
    all)      do_all      "$BINARY" "$@" ;;
    -h|--help|help) usage ;;
    *) die "Unknown mode: $MODE. Use --help for usage." ;;
esac
