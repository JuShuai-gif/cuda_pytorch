#!/usr/bin/env bash
# ============================================================================
# profiling.sh - 封装 perf stat 用于程序性能分析。
#
# 用法：
#   ./profiling.sh <程序> [参数...]
#   ./profiling.sh --events <事件列表> <程序> [参数...]
#   ./profiling.sh --list              显示可用的硬件事件
#
# 前置条件：perf（linux-tools-common / linux-tools-generic）
# ============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------
# 默认硬件事件 - 覆盖 IPC、缓存、分支和操作系统级指标
# ----------------------------------------------------------------------------
DEFAULT_EVENTS=(
    "cycles"
    "instructions"
    "cache-references"
    "cache-misses"
    "branch-instructions"
    "branch-misses"
    "context-switches"
    "cpu-migrations"
    "page-faults"
    "cpu-clock"
    "task-clock"
    "L1-dcache-loads"
    "L1-dcache-load-misses"
    "LLC-loads"
    "LLC-load-misses"
)

# ----------------------------------------------------------------------------
# 格式化打印用法信息
# ----------------------------------------------------------------------------
usage() {
    cat << 'EOF'
用法：profiling.sh [选项] <程序> [程序参数...]

封装 perf stat 以分析程序性能。报告 IPC、缓存
未命中率、分支预测失败率以及操作系统级指标。

选项：
  --events <csv>     逗号分隔的 perf 事件列表（覆盖默认值）
  --repeat <N>       重复测量的次数（默认：1）
  --output <文件>    将原始 perf stat 输出写入文件
  --list             显示可用的硬件和软件事件
  -h, --help         显示此帮助信息

示例：
  ./profiling.sh ./my_app
  ./profiling.sh --repeat 5 ./my_app --flag1
  ./profiling.sh --events cycles,instructions ./my_app
  ./profiling.sh --list

默认监控事件：
  cycles、instructions、cache-references、cache-misses、
  branch-instructions、branch-misses、context-switches、
  cpu-migrations、page-faults、L1-dcache-{loads,load-misses}、
  LLC-{loads,load-misses}
EOF
}

# ----------------------------------------------------------------------------
# 检查 perf 是否可用
# ----------------------------------------------------------------------------
check_perf() {
    if ! command -v perf &>/dev/null; then
        echo "[错误] 在 PATH 中未找到 'perf'。" >&2
        echo "安装命令：sudo apt-get install linux-tools-generic" >&2
        exit 1
    fi

    # 验证 perf stat 是否可用（某些容器可能限制访问）
    if ! perf stat true 2>/dev/null; then
        echo "[错误] 'perf stat' 失败。请检查 kernel.perf_event_paranoid 设置：" >&2
        echo "  cat /proc/sys/kernel/perf_event_paranoid" >&2
        echo "可尝试：sudo sysctl kernel.perf_event_paranoid=-1" >&2
        exit 1
    fi
}

# ----------------------------------------------------------------------------
# 列出可用事件
# ----------------------------------------------------------------------------
list_events() {
    echo "=== 硬件事件 ==="
    perf list hardware 2>/dev/null || true
    echo ""
    echo "=== 软件事件 ==="
    perf list software 2>/dev/null || true
    echo ""
    echo "=== 缓存事件 ==="
    perf list cache 2>/dev/null || true
}

# ----------------------------------------------------------------------------
# 解析参数并提取程序和参数
# ----------------------------------------------------------------------------
parse_args() {
    EVENT_LIST=""
    REPEAT=1
    OUTPUT_FILE=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --events)
                shift
                EVENT_LIST="$1"
                shift
                ;;
            --repeat)
                shift
                REPEAT="$1"
                shift
                ;;
            --output)
                shift
                OUTPUT_FILE="$1"
                shift
                ;;
            --list)
                list_events
                exit 0
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --)
                shift
                break
                ;;
            -*)
                echo "[错误] 未知选项：$1" >&2
                usage
                exit 1
                ;;
            *)
                break
                ;;
        esac
    done

    if [[ $# -eq 0 ]]; then
        echo "[错误] 未指定程序。" >&2
        usage
        exit 1
    fi

    PROGRAM_ARGS=("$@")
}

# ----------------------------------------------------------------------------
# 构建 perf stat 的事件字符串
# ----------------------------------------------------------------------------
build_event_string() {
    if [[ -n "$EVENT_LIST" ]]; then
        echo "$EVENT_LIST"
    else
        local IFS=","
        echo "${DEFAULT_EVENTS[*]}"
    fi
}

# ----------------------------------------------------------------------------
# 运行 perf stat 并收集结果
# ----------------------------------------------------------------------------
run_perf_stat() {
    local events="$1"
    local repeat="$2"
    local prog=("${@:3}")

    local perf_args=(-e "$events")

    if [[ "$repeat" -gt 1 ]]; then
        perf_args+=(-r "$repeat")
    fi

    if [[ -n "$OUTPUT_FILE" ]]; then
        echo "[信息] 正在运行：perf stat ${perf_args[*]} ${prog[*]}"
        echo "[信息] 正在将原始输出写入：$OUTPUT_FILE"
        perf stat "${perf_args[@]}" -- "${prog[@]}" 2>&1 | tee "$OUTPUT_FILE"
    else
        echo "[信息] 正在运行：perf stat ${perf_args[*]} ${prog[*]}"
        perf stat "${perf_args[@]}" -- "${prog[@]}" 2>&1
    fi
}

# ----------------------------------------------------------------------------
# 后处理：从 perf stat 输出计算衍生指标
# ----------------------------------------------------------------------------
compute_derived_metrics() {
    local perf_output
    perf_output=$(perf stat -x ',' -e "instructions,cycles,cache-references,cache-misses,branch-instructions,branch-misses" -- "${PROGRAM_ARGS[@]}" 2>&1 || true)

    # 从类 CSV 的 perf 输出中提取值（-x ','）
    local instructions cycles cache_refs cache_misses branch_ins branch_misses
    instructions=$(echo "$perf_output" | grep ",instructions," | cut -d',' -f1 | head -1)
    cycles=$(echo "$perf_output" | grep ",cycles," | grep -v "ref-cycles" | cut -d',' -f1 | head -1)
    cache_refs=$(echo "$perf_output" | grep ",cache-references," | cut -d',' -f1 | head -1)
    cache_misses=$(echo "$perf_output" | grep ",cache-misses," | cut -d',' -f1 | head -1)
    branch_ins=$(echo "$perf_output" | grep ",branch-instructions," | cut -d',' -f1 | head -1)
    branch_misses=$(echo "$perf_output" | grep ",branch-misses," | cut -d',' -f1 | head -1)

    echo ""
    echo "================================================================"
    echo "  衍生指标"
    echo "================================================================"

    if [[ -n "$cycles" && -n "$instructions" && "$cycles" != "0" ]]; then
        local ipc
        ipc=$(echo "scale=4; $instructions / $cycles" | bc -l 2>/dev/null || echo "N/A")
        echo "  IPC（每周期指令数）：               $ipc"
        if command -v bc &>/dev/null; then
            if (( $(echo "$ipc < 0.7" | bc -l) )); then
                echo "    -> 可能是内存瓶颈（IPC 较低）"
            elif (( $(echo "$ipc > 2.0" | bc -l) )); then
                echo "    -> 可能是计算瓶颈（流水线效率较高）"
            else
                echo "    -> IPC 适中"
            fi
        fi
    fi

    if [[ -n "$cache_refs" && -n "$cache_misses" && "$cache_refs" != "0" ]]; then
        local cache_miss_rate
        cache_miss_rate=$(echo "scale=2; $cache_misses * 100 / $cache_refs" | bc -l 2>/dev/null || echo "N/A")
        echo "  缓存未命中率：                      ${cache_miss_rate}%"
    fi

    if [[ -n "$branch_ins" && -n "$branch_misses" && "$branch_ins" != "0" ]]; then
        local branch_miss_rate
        branch_miss_rate=$(echo "scale=2; $branch_misses * 100 / $branch_ins" | bc -l 2>/dev/null || echo "N/A")
        echo "  分支预测失败率：                    ${branch_miss_rate}%"
        if command -v bc &>/dev/null && [[ "$branch_miss_rate" != "N/A" ]]; then
            if (( $(echo "$branch_miss_rate > 10" | bc -l) )); then
                echo "    -> 分支预测失败率较高 - 可考虑使用 PGO 或分支提示"
            fi
        fi
    fi
    echo "================================================================"
}

# ----------------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------------
main() {
    check_perf
    parse_args "$@"

    echo "================================================================"
    echo "  性能分析：${PROGRAM_ARGS[0]}"
    echo "================================================================"
    echo "  程序：    ${PROGRAM_ARGS[*]}"
    echo "  重复：    $REPEAT"
    echo "================================================================"
    echo ""

    local events
    events=$(build_event_string)
    run_perf_stat "$events" "$REPEAT" "${PROGRAM_ARGS[@]}"

    echo ""
    compute_derived_metrics
}

main "$@"
