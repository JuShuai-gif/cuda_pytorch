#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# llvm_mca.sh -- Static machine code throughput/latency analyzer wrapper
#
# Usage:
#   llvm_mca.sh <binary> <function_name>        Analyze function from binary
#   llvm_mca.sh --asm <file.s>                  Analyze assembly file
#   llvm_mca.sh --source <file.cpp> <function>    Compile + analyze from source
#   llvm_mca.sh --demo                           Run the built-in demo
#
# llvm-mca (Machine Code Analyzer) predicts execution throughput, latency,
# and port pressure WITHOUT running the code. Essential for SIMD optimization
# because you can iterate on intrinsic sequences without running on hardware.
#
# Prerequisites:
#   sudo apt install llvm-dev llvm  (provides llvm-mca)
#   llvm-mca-18 | llvm-mca-17 | ... or just llvm-mca
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MCA_OUTPUT_DIR="${MCA_OUTPUT_DIR:-/tmp/llvm_mca_output}"
DEMO_ASM="$PROJECT_DIR/x86/src/llvm_mca_demo.s"

usage() {
    cat <<EOF
Usage: $0 <mode> [args...]

Modes:
  <binary> <function>             Extract function from binary, run llvm-mca
  --asm <file.s>                  Run llvm-mca on annotated assembly file
  --asm <file.s> <region>         Analyze only a named LLVM-MCA-BEGIN/END region
  --source <file.cpp> <function>  Compile source, extract function, run llvm-mca
  --demo                          Run the built-in annotated demo (3 regions)
  --iterations <N>                Override default iteration count (default: 100)
  --mcpu <arch>                   Override CPU microarchitecture (see below)
  --help                          Show this message

CPU microarchitecture names (-mcpu / auto-detected):
  x86:
    skylake, skylake-avx512, icelake-server, icelake-client,
    alderlake, raptorlake, znver3, znver4
  ARM:
    cortex-a76, neoverse-n1, neoverse-v1, apple-m1

Examples:
  $0 build/x86/avx2_dot_product   avx2_dot_product_f32
  $0 --asm x86/src/llvm_mca_demo.s
  $0 --asm x86/src/llvm_mca_demo.s fast_dot
  $0 --source x86/src/avx2_dot_product.cpp  avx2_dot_product_f32
  $0 --demo --iterations 500
  $0 --asm foo.s --mcpu znver4
EOF
    exit 0
}

die() { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }
info() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
header() { echo -e "${BOLD}${CYAN}$*${NC}"; }

# ---------------------------------------------------------------------------
# Tool discovery -- find llvm-mca on the system
# ---------------------------------------------------------------------------
find_llvm_mca() {
    for candidate in \
        llvm-mca-20 llvm-mca-19 llvm-mca-18 llvm-mca-17 llvm-mca-16 \
        llvm-mca-15 llvm-mca-14 llvm-mca \
        /usr/lib/llvm-*/bin/llvm-mca; do
        if command -v "$candidate" &>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

check_tool() {
    command -v "$1" &>/dev/null || die "$1 is not installed. $2"
}

# ---------------------------------------------------------------------------
# CPU microarchitecture auto-detection
# ---------------------------------------------------------------------------
detect_mcpu() {
    # Strategy 1: use llvm-mca --mcpu=native if supported
    local mca
    mca=$(find_llvm_mca 2>/dev/null || true)
    if [ -n "$mca" ]; then
        if "$mca" --mcpu=native --version &>/dev/null; then
            echo "native"
            return 0
        fi
    fi

    # Strategy 2: use gcc -march=native detection, then map to LLVM name
    if command -v gcc &>/dev/null; then
        local march
        march=$(gcc -march=native -Q --help=target 2>/dev/null \
            | grep -E '^\s+-march=' \
            | head -1 \
            | awk '{print $2}' || true)
        if [ -n "$march" ] && [ "$march" != "native" ]; then
            local mapped
            mapped=$(map_gcc_march_to_mcpu "$march")
            if [ -n "$mapped" ]; then
                echo "$mapped"
                return 0
            fi
        fi
    fi

    # Strategy 3: parse /proc/cpuinfo and lscpu
    if [ -f /proc/cpuinfo ]; then
        local vendor model_name family model
        vendor=$(grep -m1 "vendor_id" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || echo "")
        model_name=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs || echo "")

        if echo "$vendor" | grep -qi "GenuineIntel"; then
            detect_intel_mcpu "$model_name"
            return $?
        elif echo "$vendor" | grep -qi "AuthenticAMD"; then
            detect_amd_mcpu "$model_name"
            return $?
        else
            detect_arm_mcpu
            return $?
        fi
    fi

    warn "Cannot auto-detect CPU microarchitecture."
    echo "  Use --mcpu <name> to specify one manually."
    echo "  Common choices: skylake, icelake-client, alderlake, znver3, znver4"
    echo "                  cortex-a76, neoverse-n1, apple-m1"
    return 1
}

map_gcc_march_to_mcpu() {
    local march="$1"
    case "$march" in
        skylake|skylake-avx512)         echo "skylake" ;;
        icelake-client|icelake-server)  echo "icelake-server" ;;
        alderlake|raptorlake)           echo "alderlake" ;;
        sapphirerapids|graniterapids)   echo "sapphirerapids" ;;
        znver3)                         echo "znver3" ;;
        znver4)                         echo "znver4" ;;
        neoverse-n1)                    echo "neoverse-n1" ;;
        neoverse-v1)                    echo "neoverse-v1" ;;
        *)                              echo "" ;;
    esac
}

detect_intel_mcpu() {
    local name="$1"
    # Map Intel CPU model numbers to microarchitecture names.
    # The model number isn't in 'model name', so we parse /proc/cpuinfo further.
    local family model stepping
    family=$(grep -m1 "^cpu family" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || echo "6")
    model=$(grep -m1 "^model\b" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || echo "0")

    if [ "$family" = "6" ]; then
        case "$model" in
            # Broadwell
            61|71|79|86) echo "broadwell" ; return 0 ;;
            # Skylake / Kaby Lake / Coffee Lake / Comet Lake (client)
            78|94|142|158|165|166) echo "skylake" ; return 0 ;;
            # Skylake-SP / Cascade Lake-SP
            85) echo "skylake-avx512" ; return 0 ;;
            # Ice Lake client
            126) echo "icelake-client" ; return 0 ;;
            # Ice Lake server
            106|108) echo "icelake-server" ; return 0 ;;
            # Tiger Lake
            140|141) echo "tigerlake" ; return 0 ;;
            # Alder Lake / Raptor Lake
            151|154|183|186) echo "alderlake" ; return 0 ;;
            # Sapphire Rapids
            143) echo "sapphirerapids" ; return 0 ;;
            # Granite Rapids
            173) echo "graniterapids" ; return 0 ;;
            # Lunar Lake
            189) echo "lunarlake" ; return 0 ;;
        esac
    fi

    # Fallback: guess from marketing name
    if echo "$name" | grep -qiE "i[3579]-[0-9]+"; then
        local gen
        gen=$(echo "$name" | grep -oP 'i[3579]-\K[0-9]+' | head -1 || echo "0")
        if [ "$gen" -le 7 ]; then echo "skylake"     ; return 0; fi
        if [ "$gen" -le 10 ]; then echo "skylake"     ; return 0; fi
        if [ "$gen" -le 11 ]; then echo "tigerlake"   ; return 0; fi
        if [ "$gen" -ge 12 ]; then echo "alderlake"   ; return 0; fi
    fi

    echo "skylake"  # safest default for Intel
    return 0
}

detect_amd_mcpu() {
    local name="$1"
    local family model
    family=$(grep -m1 "^cpu family" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || echo "23")
    model=$(grep -m1 "^model\b" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || echo "0")

    if [ "$family" = "23" ]; then
        # Zen 1 / Zen+
        echo "znver1"
        return 0
    elif [ "$family" = "25" ]; then
        if [ "$model" -le 47 ]; then
            echo "znver3"   # Zen 3
        else
            echo "znver4"   # Zen 4
        fi
        return 0
    elif [ "$family" = "26" ]; then
        echo "znver5"   # Zen 5
        return 0
    fi

    # Fallback for detectable in model name
    if echo "$name" | grep -qiE "Ryzen [789]"; then
        echo "znver4"
        return 0
    elif echo "$name" | grep -qiE "Ryzen [56]"; then
        echo "znver3"
        return 0
    fi

    echo "znver3"  # safest AMD default
    return 0
}

detect_arm_mcpu() {
    local impl part
    impl=$(grep -m1 "CPU implementer" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || echo "")
    part=$(grep -m1 "CPU part" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || echo "")

    if [ "$impl" = "0x41" ]; then  # ARM
        case "$part" in
            0xd0b) echo "cortex-a76"    ; return 0 ;;
            0xd0c) echo "neoverse-n1"   ; return 0 ;;
            0xd40) echo "neoverse-v1"   ; return 0 ;;
            0xd4f) echo "neoverse-n2"   ; return 0 ;;
            0xd0d) echo "cortex-a77"    ; return 0 ;;
            0xd41) echo "cortex-a78"    ; return 0 ;;
            0xd44) echo "cortex-x1"     ; return 0 ;;
            0xd47) echo "cortex-a510"   ; return 0 ;;
            0xd48) echo "cortex-a715"   ; return 0 ;;
            0xd4e) echo "cortex-a720"   ; return 0 ;;
            0xd81) echo "cortex-a720"   ; return 0 ;;
            0xd85) echo "cortex-x925"   ; return 0 ;;
        esac
    elif [ "$impl" = "0x61" ]; then  # Apple
        case "$part" in
            0x022|0x023|0x024|0x025|0x028) echo "apple-m1" ; return 0 ;;
            0x032|0x033|0x034)             echo "apple-m2" ; return 0 ;;
            0x042)                          echo "apple-m4" ; return 0 ;;
        esac
        echo "apple-m1"
        return 0
    fi

    # Fallback from Features line
    if grep -q "asimd" /proc/cpuinfo 2>/dev/null; then
        echo "cortex-a76"
        return 0
    fi

    echo "cortex-a76"
    return 0
}

# ---------------------------------------------------------------------------
# Assembly extraction from binary (objdump)
# ---------------------------------------------------------------------------
extract_function_asm() {
    local binary="$1"
    local func="$2"

    [ -f "$binary" ] || die "Binary not found: $binary"
    [ -x "$binary" ] || warn "Binary is not executable: $binary"
    check_tool objdump "Install: sudo apt install binutils"

    local arch_flag
    if file "$binary" | grep -qi "x86-64"; then
        arch_flag="-M intel"
    else
        arch_flag=""
    fi

    # Extract the function: from the function label to the next label or end of section
    objdump -d $arch_flag "$binary" 2>/dev/null \
        | awk -v func="$func" '
            BEGIN { found=0; }
            $0 ~ "<" func ">:" {
                found=1;
                print "# LLVM-MCA-BEGIN " func;
                next;
            }
            found && /^$/ { found=0; print "# LLVM-MCA-END"; print ""; exit; }
            found { print; }
        ' || die "Could not extract function '$func' from $binary."

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        die "objdump failed on $binary."
    fi

    echo "# LLVM-MCA-END"
}

# ---------------------------------------------------------------------------
# Assembly extraction from source (compile + objdump)
# ---------------------------------------------------------------------------
compile_and_extract() {
    local src="$1"
    local func="$2"

    [ -f "$src" ] || die "Source file not found: $src"

    local tmpdir
    tmpdir=$(mktemp -d "/tmp/llvm_mca_compile.XXXXXX")
    trap 'rm -rf "$tmpdir"' EXIT

    local obj="$tmpdir/$(basename "$src" .cpp).o"
    local binary="$tmpdir/$(basename "$src" .cpp)"

    local compile_flags="-O3 -mavx2 -mfma -g"
    info "Compiling $src with: $compile_flags"

    g++ -std=c++17 $compile_flags -c "$src" -o "$obj" 2>&1 || \
        die "Compilation of $src failed."
    g++ $compile_flags "$obj" -o "$binary" 2>&1 || \
        die "Linking of $obj failed."

    ok "Compilation successful."
    extract_function_asm "$binary" "$func"
}

# ---------------------------------------------------------------------------
# Run llvm-mca and capture output
# ---------------------------------------------------------------------------
run_mca() {
    local asm_file="$1"
    local region="$2"
    local iterations="$3"
    local mcpu="$4"

    local mca
    mca=$(find_llvm_mca) || die \
        "llvm-mca not found on PATH."$'\n' \
        "  Install via: sudo apt install llvm-dev llvm"$'\n' \
        "  Or check: https://apt.llvm.org/ for newer versions."

    local mca_args=()

    if [ -n "$mcpu" ]; then
        if [ "$mcpu" = "native" ]; then
            mca_args+=("--mcpu=native")
        else
            mca_args+=("--mcpu=$mcpu")
        fi
    fi

    mca_args+=("--iterations=$iterations")
    mca_args+=("--timeline")
    mca_args+=("--bottleneck-analysis")

    if [ -n "$region" ] && [ "$region" != "__all__" ]; then
        mca_args+=("--region=$region")
    fi

    info "Running: $mca ${mca_args[*]} $asm_file"
    echo ""

    "$mca" "${mca_args[@]}" "$asm_file" 2>&1
}

# ---------------------------------------------------------------------------
# Parse and interpret llvm-mca output
# ---------------------------------------------------------------------------
interpret_mca_output() {
    local output="$1"

    echo ""
    header "=== llvm-mca Output Interpretation ==="
    echo ""

    # Extract key metrics
    local iterations
    iterations=$(echo "$output" | grep -oP 'Iterations:\s*\K[0-9]+' | head -1 || echo "?")

    local total_cycles
    total_cycles=$(echo "$output" | grep -oP 'Total Cycles:\s*\K[0-9]+' | head -1 || echo "?")

    local total_uops
    total_uops=$(echo "$output" | grep -oP 'Total uOps:\s*\K[0-9]+' | head -1 || echo "?")

    local dispatch_width
    dispatch_width=$(echo "$output" | grep -oP 'Dispatch Width:\s*\K[0-9]+' | head -1 || echo "?")

    local uops_per_cycle
    uops_per_cycle=$(echo "$output" | grep -oP 'uOps Per Cycle:\s*\K[0-9.]+' | head -1 || echo "?")

    local ipc
    ipc=$(echo "$output" | grep -oP '^IPC:\s*\K[0-9.]+' | head -1 || echo "?")

    local block_rthroughput
    block_rthroughput=$(echo "$output" | grep -oP 'Block RThroughput:\s*\K[0-9.]+' | head -1 || echo "?")

    echo "  Iterations simulated:       $iterations"
    echo "  Total Cycles:               $total_cycles"
    echo "  Dispatch Width (max issue): $dispatch_width"
    echo "  uOps Per Cycle:             $uops_per_cycle"
    echo "  IPC (Instructions/Cycle):   $ipc"
    echo "  Reciprocal Throughput:      $block_rthroughput"
    echo ""

    # Interpretation
    echo "--- What these numbers mean ---"
    echo ""

    if [ "$iterations" != "?" ] && [ "$iterations" -gt 0 ]; then
        echo "  Iterations: llvm-mca simulates the loop body $iterations times"
        echo "  to get statistically stable results. Higher = more accurate but slower."
        echo ""
    fi

    if [ "$dispatch_width" != "?" ]; then
        echo "  Dispatch Width ($dispatch_width): maximum instructions the CPU can"
        echo "  decode/issue per cycle. Skylake=4, Icelake+=6, Zen3=6, ARM varies."
        echo ""
    fi

    if [ "$ipc" != "?" ] && [ "$dispatch_width" != "?" ]; then
        local dw_num
        dw_num=$(echo "$dispatch_width" | sed 's/[^0-9]//g')

        if [ "$(echo "$ipc > $dispatch_width * 0.8" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
            echo "  IPC ($ipc) is close to Dispatch Width ($dispatch_width)"
            echo "  => EXCELLENT: This loop keeps the frontend nearly saturated."
            echo "     The CPU is issuing almost every cycle."
        elif [ "$(echo "$ipc > $dispatch_width * 0.5" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
            echo "  IPC ($ipc) is moderate vs Dispatch Width ($dispatch_width)"
            echo "  => GOOD: There is some ILP but dependencies limit full issue."
        else
            echo "  IPC ($ipc) is low vs Dispatch Width ($dispatch_width)"
            echo "  => POOR: The loop is bottlenecked - check Resource Pressure below."
        fi
        echo ""
    fi

    if [ "$block_rthroughput" != "?" ]; then
        echo "  Block RThroughput ($block_rthroughput cycles/iteration):"
        echo "  The minimum number of cycles needed per loop iteration."
        echo "  This is the theoretical lower bound dictated by the most"
        echo "  congested execution resource (a specific port, the frontend,"
        echo "  or memory). If TotalCycles/Iterations >> RThroughput, there"
        echo "  are other bottlenecks (e.g., long dependency chains)."
        echo ""
    fi

    # ---- Resource pressure analysis ----
    echo "--- Resource Pressure (Port) Analysis ---"
    echo ""

    # Extract resource pressure table
    local pressure_section
    pressure_section=$(echo "$output" | awk '
        /Resource pressure per iteration:/ { found=1; next }
        found && /^$/ { exit }
        found { print }
    ' || true)

    if [ -n "$pressure_section" ]; then
        echo "  This shows how many uOps each execution port receives per iteration."
        echo "  The port with the highest pressure is your BOTTLENECK."
        echo ""
        echo "  Skylake port reference:"
        echo "    Port 0: ALU, FMA, MUL, DIV, shifts"
        echo "    Port 1: ALU, FMA, MUL, fast LEA, shifts"
        echo "    Port 2: Load address generation"
        echo "    Port 3: Load address generation"
        echo "    Port 4: Store data"
        echo "    Port 5: ALU, shuffle, permute, branch, HADD"
        echo "    Port 6: Integer ALU, branch"
        echo "    Port 7: Store address generation"
        echo ""

        echo "$pressure_section"

        # Find the bottleneck port
        local max_pressure=0
        local bottleneck_port=""
        local port_idx=0
        while read -r line; do
            # Parse port pressures from the line
            local all_vals
            IFS=' ' read -r -a all_vals <<< "$line"
            for val in "${all_vals[@]}"; do
                val=$(echo "$val" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                if [ "$val" = "-" ]; then continue; fi
                if [ "$(echo "$val > $max_pressure" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
                    max_pressure="$val"
                    bottleneck_port="$port_idx"
                fi
                ((port_idx++)) || true
            done
        done <<< "$pressure_section"

        echo ""
        if [ -n "$bottleneck_port" ]; then
            echo "  Bottleneck: Port $bottleneck_port ($max_pressure uOps/iteration)"
            echo "  => To improve: redistribute operations away from Port $bottleneck_port"
            echo "     (e.g., replace shuffles with different sequences, use different"
            echo "     instruction forms that schedule to other ports)."
        fi
    else
        echo "  (No resource pressure data found in output)"
        echo "  Run with --bottleneck-analysis flag to see port breakdown."
    fi

    # ---- Bottleneck analysis section ----
    echo ""
    local bottleneck_section
    bottleneck_section=$(echo "$output" | awk '
        /Cycles with backend pressure increase/ { found=1; print; next }
        found && /^$/ { found=0 }
        found { print }
    ' || true)
    if [ -n "$bottleneck_section" ]; then
        echo "--- Backend Pressure Summary ---"
        echo "$bottleneck_section"
    fi

    echo ""
    echo "--- Decision Guide ---"
    echo ""
    echo "  High IPC, port pressure evenly distributed"
    echo "    => COMPUTE BOUND, well-scheduled. Further speedup requires"
    echo "       wider vectors (e.g., AVX2 -> AVX-512) or higher frequency."
    echo ""
    echo "  High port pressure on Port 5"
    echo "    => SHUFFLE BOTTLENECK. Reduce horizontal operations,"
    echo "       change data layout to avoid permutes, or unroll to"
    echo "       amortize reduction overhead across more FMAs."
    echo ""
    echo "  High port pressure on Port 2/3, low ALU port pressure"
    echo "    => MEMORY BOUND (load-bandwidth). Optimize data layout,"
    echo "       use prefetching (PREFETCHT0), or cache-block the data."
    echo ""
    echo "  Low IPC, no single port saturated"
    echo "    => DEPENDENCY CHAIN limited. Look for long latency"
    echo "       instructions (DIV, long FP chains) and break chains"
    echo "       with multi-accumulator unrolling."
    echo ""
}

# ---------------------------------------------------------------------------
# Demo mode -- run the built-in annotated assembly
# ---------------------------------------------------------------------------
do_demo() {
    local iterations="$1"
    local mcpu="$2"

    [ -f "$DEMO_ASM" ] || die "Demo assembly not found: $DEMO_ASM"$'\n' \
        "  Expected location: $DEMO_ASM"

    header "============================================"
    header "  llvm-mca SIMD Analysis Demo"
    header "  File: $DEMO_ASM"
    header "============================================"
    echo ""

    if [ -n "$mcpu" ]; then
        echo "  CPU specified: $mcpu"
    else
        echo "  CPU: $(detect_mcpu || echo 'unknown (default)')"
    fi
    echo "  Iterations: $iterations"
    echo ""

    for region in fast_dot slow_dot vector_add; do
        header "----------------------------------------"
        header "  Region: $region"
        header "----------------------------------------"
        echo ""

        case "$region" in
            fast_dot)
                echo "  CODE: Well-optimized 4-way unrolled AVX2 FMA dot product"
                echo "  Key: 4 independent accumulator chains hide FMA latency (4 cycles)"
                echo "       Loads spread across ports 2/3, FMAs on 0/1 = good balance"
                echo ""
                ;;
            slow_dot)
                echo "  CODE: Reduction every 8 elements using vhaddps"
                echo "  Key: vhaddps executes ONLY on Port 5 on Skylake"
                echo "       Each iteration does a horizontal reduction = Port 5 bottleneck"
                echo "       Compare IPC and throughput with fast_dot"
                echo ""
                ;;
            vector_add)
                echo "  CODE: Simple memory-bound vector add (streaming)"
                echo "  Key: Two loads + one store per add. Bound by load bandwidth."
                echo "       High dispatch width but low IPC because memory is the limit"
                echo ""
                ;;
        esac

        local output
        output=$(run_mca "$DEMO_ASM" "$region" "$iterations" "${mcpu:-}")

        local raw_out="$MCA_OUTPUT_DIR/${region}_raw.txt"
        mkdir -p "$MCA_OUTPUT_DIR"
        echo "$output" > "$raw_out"

        interpret_mca_output "$output"

        echo ""
        ok "Raw output saved to: $raw_out"
        echo ""

        # Brief pause for readability between regions
        if [ "$region" != "vector_add" ]; then
            echo ""
        fi
    done

    header "============================================"
    header "  Summary"
    header "============================================"
    echo ""
    echo "  fast_dot:    Best IPC, balanced port use, compute-bound"
    echo "  slow_dot:    Lower IPC, Port 5 bottleneck (hadd/shuffle)"
    echo "  vector_add:  Lowest IPC, memory-bound (load ports saturated)"
    echo ""
    echo "  Takeaway: llvm-mca lets you compare these without running a single"
    echo "  cycle of actual execution. Iterate on intrinsics, then verify with"
    echo "  perf stat on real hardware."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    local MODE=""
    local INPUT=""
    local FUNC=""
    local REGION="__all__"
    local ITERATIONS=100
    local MCPU=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --demo)
                MODE="demo"
                shift
                ;;
            --asm)
                MODE="asm"
                INPUT="$2"
                shift 2
                ;;
            --source)
                MODE="source"
                INPUT="$2"
                FUNC="$3"
                shift 3
                ;;
            --iterations)
                ITERATIONS="$2"
                shift 2
                ;;
            --mcpu)
                MCPU="$2"
                shift 2
                ;;
            -h|--help|help)
                usage
                ;;
            *)
                if [ "$MODE" = "" ]; then
                    MODE="binary"
                    INPUT="$1"
                    if [ $# -gt 1 ]; then
                        FUNC="$2"
                        shift
                    fi
                    shift
                elif [ "$MODE" = "asm" ] && [ "$INPUT" != "" ] && [ "$REGION" = "__all__" ]; then
                    REGION="$1"
                    shift
                else
                    die "Unknown argument: $1"
                fi
                ;;
        esac
    done

    # ---- Handle each mode ----
    case "$MODE" in
        demo)
            if [ -z "$MCPU" ]; then
                MCPU=$(detect_mcpu 2>/dev/null || echo "")
            fi
            do_demo "$ITERATIONS" "$MCPU"
            ;;

        asm)
            [ -f "$INPUT" ] || die "Assembly file not found: $INPUT"
            info "Analyzing assembly file: $INPUT"
            if [ -z "$MCPU" ]; then
                MCPU=$(detect_mcpu 2>/dev/null || echo "")
            fi
            OUTPUT=$(run_mca "$INPUT" "$REGION" "$ITERATIONS" "${MCPU:-}")
            mkdir -p "$MCA_OUTPUT_DIR"
            echo "$OUTPUT" > "$MCA_OUTPUT_DIR/analysis_raw.txt"
            echo "$OUTPUT"
            echo ""
            interpret_mca_output "$OUTPUT"
            ok "Raw output saved to: $MCA_OUTPUT_DIR/analysis_raw.txt"
            ;;

        source)
            [ -n "$FUNC" ] || die "Function name required for --source mode."
            tmp_asm=$(mktemp "/tmp/llvm_mca_src.XXXXXX.s")
            trap 'rm -f "$tmp_asm"' EXIT
            compile_and_extract "$INPUT" "$FUNC" > "$tmp_asm"
            info "Extracted assembly ($(wc -l < "$tmp_asm") lines)"
            if [ -z "$MCPU" ]; then
                MCPU=$(detect_mcpu 2>/dev/null || echo "")
            fi
            OUTPUT=$(run_mca "$tmp_asm" "$REGION" "$ITERATIONS" "${MCPU:-}")
            mkdir -p "$MCA_OUTPUT_DIR"
            echo "$OUTPUT" > "$MCA_OUTPUT_DIR/analysis_raw.txt"
            echo "$OUTPUT"
            echo ""
            interpret_mca_output "$OUTPUT"
            ;;

        binary)
            [ -n "$FUNC" ] || die "Function name required for binary mode."
            tmp_asm=$(mktemp "/tmp/llvm_mca_bin.XXXXXX.s")
            trap 'rm -f "$tmp_asm"' EXIT
            extract_function_asm "$INPUT" "$FUNC" > "$tmp_asm"
            info "Extracted assembly for '$FUNC' ($(wc -l < "$tmp_asm") lines)"
            if [ -z "$MCPU" ]; then
                MCPU=$(detect_mcpu 2>/dev/null || echo "")
            fi
            OUTPUT=$(run_mca "$tmp_asm" "$REGION" "$ITERATIONS" "${MCPU:-}")
            mkdir -p "$MCA_OUTPUT_DIR"
            echo "$OUTPUT" > "$MCA_OUTPUT_DIR/analysis_raw.txt"
            echo "$OUTPUT"
            echo ""
            interpret_mca_output "$OUTPUT"
            ;;

        *)
            usage
            ;;
    esac
}

main "$@"
