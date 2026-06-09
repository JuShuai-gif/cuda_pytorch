#!/usr/bin/env bash
# ============================================================================
# flamegraph.sh - 录制 perf 数据并生成 CPU 火焰图。
#
# 用法：
#   ./flamegraph.sh --pid <PID> [--duration <秒数>] [--frequency <Hz>]
#   ./flamegraph.sh --cmd <程序> [参数...]
#
# 前置条件：
#   - perf（linux-tools-generic）
#   - perl（用于 FlameGraph 脚本）
#   - FlameGraph 工具（https://github.com/brendangregg/FlameGraph）
#
#   如果在本地找不到 FlameGraph 脚本，本脚本将自动下载，
#   或者你也可以手动安装：
#     git clone https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph
# ============================================================================
set -euo pipefail

FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-/opt/FlameGraph}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$THIS_DIR/flamegraph_output}"

# ----------------------------------------------------------------------------
# 用法说明
# ----------------------------------------------------------------------------
usage() {
    cat << 'EOF'
用法：flamegraph.sh [选项]

使用 Linux perf 生成 CPU 火焰图。

选项：
  --pid <PID>         按 PID 分析正在运行的进程
  --cmd <命令>        运行并分析一个命令（剩余参数将传递给该命令）
  --duration <N>      采样持续时间，单位秒（默认：30）
  --frequency <N>     采样频率，单位 Hz（默认：99）
  --output <目录>      输出目录（默认：./flamegraph_output）
  --offcpu            生成 off-CPU 火焰图（而非 on-CPU）
  -h, --help          显示此帮助信息

示例：
  # 以 99Hz 对正在运行的进程采样 30 秒
  ./flamegraph.sh --pid 12345 --duration 30

  # 分析一个命令
  ./flamegraph.sh --cmd ./my_program arg1 arg2

  # Off-CPU 分析（进程在哪里阻塞？）
  ./flamegraph.sh --pid 12345 --offcpu

输出：
  flamegraph_output/cpu_flamegraph.svg  （或 offcpu_flamegraph.svg）
  flamegraph_output/perf.data           （原始 perf 数据）
EOF
}

# ----------------------------------------------------------------------------
# 下载或定位 FlameGraph 脚本
# ----------------------------------------------------------------------------
ensure_flamegraph_scripts() {
    local stackcollapse="$FLAMEGRAPH_DIR/stackcollapse-perf.pl"
    local flamegraph="$FLAMEGRAPH_DIR/flamegraph.pl"

    if [[ -x "$stackcollapse" ]] && [[ -x "$flamegraph" ]]; then
        return 0
    fi

    echo "[信息] 在 $FLAMEGRAPH_DIR 中未找到 FlameGraph 脚本"
    echo "[信息] 正在尝试下载到 $THIS_DIR/.flamegraph_tmp ..."

    local tmp_dir="$THIS_DIR/.flamegraph_tmp"
    mkdir -p "$tmp_dir"

    if [[ ! -f "$tmp_dir/stackcollapse-perf.pl" ]]; then
        curl -sSLo "$tmp_dir/stackcollapse-perf.pl" \
            "https://raw.githubusercontent.com/brendangregg/FlameGraph/master/stackcollapse-perf.pl"
        chmod +x "$tmp_dir/stackcollapse-perf.pl"
    fi

    if [[ ! -f "$tmp_dir/flamegraph.pl" ]]; then
        curl -sSLo "$tmp_dir/flamegraph.pl" \
            "https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl"
        chmod +x "$tmp_dir/flamegraph.pl"
    fi

    # 覆盖 FLAMEGRAPH_DIR 以使用临时位置
    FLAMEGRAPH_DIR="$tmp_dir"
    echo "[信息] 正在使用 FlameGraph 脚本，路径：$FLAMEGRAPH_DIR"
}

# ----------------------------------------------------------------------------
# 检查依赖项
# ----------------------------------------------------------------------------
check_deps() {
    if ! command -v perf &>/dev/null; then
        echo "[错误] 未找到 'perf'。安装命令：sudo apt-get install linux-tools-generic" >&2
        exit 1
    fi
    if ! command -v perl &>/dev/null; then
        echo "[错误] 未找到 'perl'。安装命令：sudo apt-get install perl" >&2
        exit 1
    fi
    ensure_flamegraph_scripts
}

# ----------------------------------------------------------------------------
# 使用 perf 录制数据
# ----------------------------------------------------------------------------
record_perf() {
    local output="$1"
    local duration="$2"
    local freq="$3"
    local mode="$4"
    shift 4

    local perf_cmd=(
        perf record
        -F "$freq"
        -g
        -o "$output"
    )

    if [[ "$mode" == "offcpu" ]]; then
        # Off-CPU：追踪调度器事件以捕获阻塞/等待
        perf_cmd+=(
            -e "sched:sched_switch,sched:sched_stat_sleep,sched:sched_stat_blocked,sched:sched_stat_wait"
        )
    fi

    if [[ $# -gt 0 ]]; then
        if [[ "$1" == "--pid" ]]; then
            perf_cmd+=(-p "$2" -- sleep "$duration")
        else
            perf_cmd+=(-- "$@")
        fi
    fi

    echo "[信息] 正在录制：${perf_cmd[*]}"
    "${perf_cmd[@]}"
}

# ----------------------------------------------------------------------------
# 从 perf.data 生成火焰图 SVG
# ----------------------------------------------------------------------------
generate_flamegraph() {
    local perf_data="$1"
    local output_svg="$2"
    local title="$3"

    local stackcollapse="$FLAMEGRAPH_DIR/stackcollapse-perf.pl"
    local flamegraph="$FLAMEGRAPH_DIR/flamegraph.pl"

    echo "[信息] 正在折叠调用栈..."
    perf script -i "$perf_data" | "$stackcollapse" > "$OUT_DIR/perf.folded"

    echo "[信息] 正在生成火焰图：$output_svg"
    "$flamegraph" \
        --title "$title" \
        --width 1600 \
        --colors java \
        "$OUT_DIR/perf.folded" \
        > "$output_svg"

    echo "[信息] 火焰图已保存至：$output_svg"
}

# ----------------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------------
main() {
    local pid=""
    local cmd_args=()
    local duration=30
    local frequency=99
    local mode="oncpu"

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --pid)
                shift; pid="$1"; shift ;;
            --cmd)
                shift
                while [[ $# -gt 0 && "$1" != --* ]]; do
                    cmd_args+=("$1"); shift
                done
                ;;
            --duration)
                shift; duration="$1"; shift ;;
            --frequency)
                shift; frequency="$1"; shift ;;
            --output)
                shift; OUT_DIR="$1"; shift ;;
            --offcpu)
                mode="offcpu"; shift ;;
            -h|--help)
                usage; exit 0 ;;
            *)
                echo "[错误] 未知选项：$1" >&2
                usage; exit 1 ;;
        esac
    done

    if [[ -z "$pid" ]] && [[ ${#cmd_args[@]} -eq 0 ]]; then
        echo "[错误] 需要指定 --pid 或 --cmd。" >&2
        usage
        exit 1
    fi

    check_deps
    mkdir -p "$OUT_DIR"

    local perf_data="$OUT_DIR/perf.data"
    rm -f "$perf_data" "$OUT_DIR/perf.folded"

    # 确定输出文件名
    local svg_name="cpu_flamegraph.svg"
    local title="CPU 火焰图"
    if [[ "$mode" == "offcpu" ]]; then
        svg_name="offcpu_flamegraph.svg"
        title="Off-CPU 火焰图"
    fi

    if [[ -n "$pid" ]]; then
        echo "[信息] 正在以 ${frequency}Hz 对 PID $pid 采样 ${duration}s（模式：$mode）"
        record_perf "$perf_data" "$duration" "$frequency" "$mode" --pid "$pid"
    else
        echo "[信息] 正在分析命令：${cmd_args[*]}（模式：$mode）"
        record_perf "$perf_data" "$duration" "$frequency" "$mode" "${cmd_args[@]}"
    fi

    generate_flamegraph "$perf_data" "$OUT_DIR/$svg_name" "$title"

    echo ""
    echo "================================================================"
    echo "  输出文件："
    echo "    原始数据：  $perf_data"
    echo "    SVG 图表：  $OUT_DIR/$svg_name"
    echo "================================================================"
}

main "$@"
