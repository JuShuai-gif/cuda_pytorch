#!/usr/bin/env bash
# compare_versions.sh -- A/B performance regression comparison for CI.
#
# Runs two executables (or one executable in two builds) interleaved for
# N rounds and reports min/median wall time plus the relative change.
# Exits nonzero when the second executable is significantly slower than the
# first, which makes it usable as a CI performance gate.
#
# Usage:
#   ./scripts/compare_versions.sh <exeA> [--other <exeB>] [-n 7] [--threshold 0.05]
#
#   <exeA>              baseline executable (required)
#   --other <exeB>      candidate executable (default: same as A, stability run)
#   -n <rounds>         rounds per executable (default 7)
#   --threshold <frac>  regression threshold, e.g. 0.05 = 5% (default 0.05)
#
# Note: timing the whole executable includes startup; use it for benchmarks
# whose workload dominates startup (as in this project). Interleaving A/B
# averages out CPU-frequency and OS noise between the two runs.
set -uo pipefail

ROUNDS=7
THRESHOLD=0.05
EXE_A=""
EXE_B=""

while [ $# -gt 0 ]; do
    case "$1" in
        --other)    EXE_B="$2"; shift 2 ;;
        -n)         ROUNDS="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        -h|--help)  echo "usage: $0 <exeA> [--other <exeB>] [-n ROUNDS] [--threshold FLOAT]"; exit 0 ;;
        *)          EXE_A="$1"; shift ;;
    esac
done

[ -n "${EXE_A}" ] || { echo "error: missing <exeA>" >&2; exit 2; }
[ -x "${EXE_A}" ] || { echo "error: not executable: ${EXE_A}" >&2; exit 2; }
[ -n "${EXE_B}" ] || EXE_B="${EXE_A}"
[ -x "${EXE_B}" ] || { echo "error: not executable: ${EXE_B}" >&2; exit 2; }

# measure in milliseconds using date (portable, no external timer needed)
time_ms() {
    local exe="$1"
    local t0 t1
    t0=$(date +%s%N)
    "${exe}" >/dev/null 2>&1
    t1=$(date +%s%N)
    echo $(( (t1 - t0) / 1000000 ))
}

# warm up both once so caches and CPU frequency settle
time_ms "${EXE_A}" >/dev/null
[ "${EXE_A}" != "${EXE_B}" ] && time_ms "${EXE_B}" >/dev/null

# collect interleaved samples to average out drift
A_MS=()
B_MS=()
for i in $(seq 1 "${ROUNDS}"); do
    A_MS+=("$(time_ms "${EXE_A}")")
    B_MS+=("$(time_ms "${EXE_B}")")
done

stats() {
    # $1 = array name; prints "min median" in ms
    local arr=("$@")
    local sorted min median
    sorted=($(printf '%s\n' "${arr[@]}" | sort -n))
    min="${sorted[0]}"
    median="${sorted[$(( ${#sorted[@]} / 2 ))]}"
    echo "${min} ${median}"
}

read -r A_min A_med <<< "$(stats "${A_MS[@]}")"
read -r B_min B_med <<< "$(stats "${B_MS[@]}")"

# relative change of the median (B vs A); positive = B slower
change=$(awk -v a="$A_med" -v b="$B_med" 'BEGIN { printf "%.4f", (b-a)/a }')

echo "rounds          = ${ROUNDS}"
echo "baseline (A)    = ${EXE_A}"
echo "candidate (B)   = ${EXE_B}"
printf "A min/median    = %s / %s ms\n" "${A_min}" "${A_med}"
printf "B min/median    = %s / %s ms\n" "${B_min}" "${B_med}"
printf "relative change = %+.2f%%\n" "$(awk -v c="$change" 'BEGIN{print c*100}')"

# verdict: candidate (B) significantly slower than baseline (A)?
if awk -v c="$change" -v t="$THRESHOLD" 'BEGIN{exit !(c > t)}'; then
    echo "VERDICT: REGRESSION (candidate slower by more than threshold)"
    exit 1
else
    echo "VERDICT: OK (within threshold or faster)"
    exit 0
fi
