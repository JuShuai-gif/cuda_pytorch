#!/usr/bin/env python3
"""
自动化 GPU Kernel Profiling 运行器

用法：
  # 对特定 kernel 做 ncu 分析
  python profile_runner.py --kernel flash_attention --tool ncu

  # 对所有已注册的 kernel 做 nsys 分析
  python profile_runner.py --all --tool nsys

  # 对 matmul 做 CPU 端 perf 分析
  python profile_runner.py --kernel matmul --tool perf

  # 导出结果到指定目录
  python profile_runner.py --kernel softmax --tool ncu --output-dir ./reports/

  # 只使用 ncu basic section（快速扫描）
  python profile_runner.py --kernel rmsnorm --tool ncu --set basic

  # 生成 roofline 分析
  python profile_runner.py --kernel matmul --tool ncu --set roofline

  # 指定 GPU 设备
  python profile_runner.py --kernel flash_attention --tool nsys --gpu 0

  # 指定 kernel 启动次数（只分析第 1 次 launch）
  python profile_runner.py --kernel flash_attention --tool ncu --launch-count 1

  # 对比两个 kernel 版本
  python profile_runner.py --kernel matmul --tool ncu --compare baseline.ncu-rep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 项目根目录
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Kernel 注册表
# ---------------------------------------------------------------------------

KERNEL_REGISTRY = {
    "flash_attention": {
        "module": "01_cuda_basics.benchmark_cuda_basics",
        "func": "bench_flash_attention",
        "description": "FlashAttention（对比 torch.sdpa）",
        "tags": ["attention", "memory-bound"],
    },
    "paged_attention": {
        "module": "01_cuda_basics.benchmark_cuda_basics",
        "func": "bench_paged_attention",
        "description": "PagedAttention（KV cache 分页访问）",
        "tags": ["attention", "memory-bound"],
    },
    "rmsnorm": {
        "module": "01_cuda_basics.benchmark_cuda_basics",
        "func": "bench_rmsnorm",
        "description": "RMSNorm CUDA vs PyTorch 手动实现 vs torch.compile",
        "tags": ["normalization", "memory-bound"],
    },
    "layernorm": {
        "module": "01_cuda_basics.benchmark_cuda_basics",
        "func": "bench_layernorm",
        "description": "LayerNorm CUDA vs torch.nn.LayerNorm vs torch.compile",
        "tags": ["normalization", "memory-bound"],
    },
    "fused_residual_norm": {
        "module": "01_cuda_basics.benchmark_cuda_basics",
        "func": "bench_fused_residual_norm",
        "description": "融合残差 + LayerNorm vs 顺序 add+LayerNorm",
        "tags": ["fusion", "memory-bound"],
    },
    "silu": {
        "module": "01_cuda_basics.benchmark_cuda_basics",
        "func": "bench_activations",
        "description": "SiLU/GELU/SwiGLU 激活函数",
        "tags": ["activation", "memory-bound"],
    },
    "fused_bias_activation": {
        "module": "01_cuda_basics.benchmark_cuda_basics",
        "func": "bench_fused_bias_activation",
        "description": "融合 bias+activation vs 顺序计算",
        "tags": ["fusion", "memory-bound"],
    },
    "softmax": {
        "module": "01_cuda_basics.benchmark_cuda_basics",
        "func": "bench_softmax",
        "description": "Online Softmax vs torch.softmax",
        "tags": ["math", "latency-bound"],
    },
    "reduction": {
        "module": "01_cuda_basics.benchmark_cuda_basics",
        "func": "bench_reduction",
        "description": "Warp reduce / Naive reduce / torch.sum",
        "tags": ["reduction", "memory-bound"],
    },
    "vector_add": {
        "module": "01_cuda_basics.benchmark_cuda_basics",
        "func": "bench_vector_add",
        "description": "CUDA vector_add vs torch.add（纯 memory-bound 基准）",
        "tags": ["elementwise", "memory-bound"],
    },
    "matmul": {
        "module": "01_cuda_basics.benchmark_cuda_basics",
        "func": "bench_matmul",
        "description": "CUDA tiled matmul vs torch.matmul vs torch.compile",
        "tags": ["matmul", "compute-bound"],
    },
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class ProfileResult:
    """单次 profiling 的结果元数据。"""

    kernel_name: str
    tool: str
    timestamp: str
    output_path: Path
    exit_code: int
    duration_seconds: float
    extra_info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def find_tool(tool_name: str) -> Optional[str]:
    """在 PATH 中查找 profiling 工具，返回绝对路径或 None。"""
    return shutil.which(tool_name)


def ensure_output_dir(output_dir: Path) -> None:
    """确保输出目录存在。"""
    output_dir.mkdir(parents=True, exist_ok=True)


def list_kernels() -> str:
    """列出所有注册的 kernel 及其描述，返回格式化字符串。"""
    lines = ["已注册的 Kernel：", ""]
    for name, info in KERNEL_REGISTRY.items():
        tags = ", ".join(info["tags"])
        lines.append(f"  {name:<25s}  {info['description']}")
        lines.append(f"  {'':25s}  标签: {tags}")
    return "\n".join(lines)


def _build_module_script(kernel_name: str) -> str:
    """
    构建一个自包含的 Python 脚本字符串，只运行指定的 benchmark。

    这个脚本会被写入临时文件，然后传递给 ncu/nsys/perf。
    """
    info = KERNEL_REGISTRY[kernel_name]
    module_path = PROJECT_ROOT / info["module"].replace(".", "/") + ".py"
    module_path = str(module_path)

    return f'''#!/usr/bin/env python3
"""Auto-generated profiling script for: {kernel_name}"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, "{PROJECT_ROOT}")

# 导入 benchmark 函数
import importlib.util
spec = importlib.util.spec_from_file_location(
    "benchmark_module",
    "{module_path}"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["benchmark_module"] = mod
spec.loader.exec_module(mod)

# 运行指定的 benchmark 函数
if not hasattr(mod, "{info["func"]}"):
    print(f"错误: 找不到函数 {{'{info["func"]}'}}")
    sys.exit(1)

mod._ensure_kernels_built()
mod.{info["func"]}()
'''


def _write_temp_script(kernel_name: str) -> Path:
    """将 profiling 脚本写入临时文件，返回文件路径。"""
    content = _build_module_script(kernel_name)
    fd, path = tempfile.mkstemp(suffix=".py", prefix=f"profile_{kernel_name}_")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)
    return Path(path)


# ---------------------------------------------------------------------------
# Profiling 命令构建
# ---------------------------------------------------------------------------


def build_ncu_command(
    script_path: Path,
    output_dir: Path,
    kernel_name: str,
    section_set: str = "full",
    launch_skip: Optional[int] = None,
    launch_count: Optional[int] = None,
    kernel_filter: Optional[str] = None,
    gpu_device: int = 0,
    compare_baseline: Optional[str] = None,
) -> list[str]:
    """
    构建 Nsight Compute 命令行参数。

    返回一个命令列表，可直接传给 subprocess.run()。
    """
    output_base = output_dir / f"{kernel_name}_ncu"
    cmd = [
        find_tool("ncu") or "ncu",
        "--set",
        section_set,
        "-o",
        str(output_base),
        "--target-processes",
        "all",
    ]

    if launch_skip is not None:
        cmd.extend(["--launch-skip", str(launch_skip)])
    if launch_count is not None:
        cmd.extend(["--launch-count", str(launch_count)])
    if kernel_filter:
        cmd.extend(["--kernel-name", kernel_filter])
    if compare_baseline:
        cmd.extend(["--import-source", "yes", "--compare", compare_baseline])

    cmd.append(str(script_path))
    return cmd


def build_nsys_command(
    script_path: Path,
    output_dir: Path,
    kernel_name: str,
    gpu_device: int = 0,
    enable_nvtx: bool = True,
) -> list[str]:
    """
    构建 Nsight Systems 命令行参数。

    返回一个命令列表。
    """
    output_base = output_dir / f"{kernel_name}_nsys"

    trace_options = "cuda,osrt"
    if enable_nvtx:
        trace_options = "cuda,nvtx,osrt"

    return [
        find_tool("nsys") or "nsys",
        "profile",
        "--trace=" + trace_options,
        "--output=" + str(output_base),
        "--force-overwrite=true",
        "--stats=true",
        str(script_path),
    ]


def build_perf_command(
    script_path: Path,
    output_dir: Path,
    kernel_name: str,
    perf_mode: str = "stat",
) -> list[str]:
    """
    构建 perf 命令行参数。

    perf_mode 可选值: stat, record, mem
    """
    assert perf_mode in ("stat", "record", "mem"), f"无效的 perf 模式: {perf_mode}"

    if perf_mode == "stat":
        return [
            find_tool("perf") or "perf",
            "stat",
            "-d",
            "-r",
            "3",
            "python",
            str(script_path),
        ]
    elif perf_mode == "record":
        output_file = output_dir / f"{kernel_name}_perf.data"
        return [
            find_tool("perf") or "perf",
            "record",
            "-F",
            "99",
            "-g",
            "-o",
            str(output_file),
            "python",
            str(script_path),
        ]
    else:  # mem
        output_file = output_dir / f"{kernel_name}_perf_mem.data"
        return [
            find_tool("perf") or "perf",
            "mem",
            "record",
            "-o",
            str(output_file),
            "python",
            str(script_path),
        ]


# ---------------------------------------------------------------------------
# Profiling 执行
# ---------------------------------------------------------------------------


def run_profile(
    kernel_name: str,
    tool: str,
    output_dir: Path,
    **kwargs,
) -> ProfileResult:
    """
    对指定 kernel 运行一次 profiling。

    Args:
        kernel_name: KERNEL_REGISTRY 中注册的 kernel 名称
        tool: 使用的工具（ncu / nsys / perf）
        output_dir: 输出目录
        **kwargs: 传递给命令构建函数的额外参数

    Returns:
        ProfileResult 包含 profiling 元数据
    """
    if kernel_name not in KERNEL_REGISTRY:
        raise KeyError(f"未知 kernel: {kernel_name}。\n{list_kernels()}")

    ensure_output_dir(output_dir)

    # 检查工具是否可用
    if tool != "perf":
        tool_path = find_tool(tool)
        if tool_path is None:
            print(f"警告: {tool} 未在 PATH 中找到，尝试直接使用 '{tool}' 命令")

    # 生成临时脚本
    script_path = _write_temp_script(kernel_name)
    print(f"临时脚本: {script_path}")

    # 构建命令
    if tool == "ncu":
        cmd = build_ncu_command(script_path, output_dir, kernel_name, **kwargs)
    elif tool == "nsys":
        cmd = build_nsys_command(script_path, output_dir, kernel_name, **kwargs)
    elif tool == "perf":
        perf_mode = kwargs.pop("perf_mode", "stat")
        cmd = build_perf_command(script_path, output_dir, kernel_name, perf_mode=perf_mode)
    else:
        raise ValueError(f"不支持的工具: {tool}")

    # 执行
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 70}")
    print(f"[{timestamp}] 运行 {tool} profiling: {kernel_name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    result = subprocess.run(
        cmd,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(kwargs.get("gpu_device", 0))},
        capture_output=False,  # 实时输出到终端
    )

    elapsed = time.time() - start_time

    # 清理临时脚本
    try:
        script_path.unlink()
    except OSError:
        pass

    profile_result = ProfileResult(
        kernel_name=kernel_name,
        tool=tool,
        timestamp=timestamp,
        output_path=output_dir,
        exit_code=result.returncode,
        duration_seconds=elapsed,
        extra_info={"command": " ".join(cmd)},
    )

    # 打印结果
    status = "成功" if result.returncode == 0 else "失败"
    print(f"\n  ── Profiling {status}（耗时: {elapsed:.1f}s，退出码: {result.returncode}）")

    return profile_result


def run_all_kernels(
    tool: str,
    output_dir: Path,
    **kwargs,
) -> list[ProfileResult]:
    """
    对所有注册的 kernel 运行 profiling。

    Returns:
        ProfileResult 列表
    """
    results = []
    kernel_names = list(KERNEL_REGISTRY.keys())

    print(f"将对 {len(kernel_names)} 个 kernel 运行 {tool} profiling:")
    for name in kernel_names:
        print(f"  - {name}")

    for i, name in enumerate(kernel_names, 1):
        print(f"\n[{i}/{len(kernel_names)}] 正在分析: {name}")
        try:
            result = run_profile(name, tool, output_dir, **kwargs)
            results.append(result)
        except Exception as exc:
            print(f"错误: {name} profiling 失败: {exc}")
            continue

    return results


def generate_summary_report(results: list[ProfileResult], output_dir: Path) -> None:
    """
    生成 profiling 汇总报告（JSON 格式）。
    """
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_kernels": len(results),
        "successful": sum(1 for r in results if r.exit_code == 0),
        "failed": sum(1 for r in results if r.exit_code != 0),
        "results": [
            {
                "kernel_name": r.kernel_name,
                "tool": r.tool,
                "timestamp": r.timestamp,
                "exit_code": r.exit_code,
                "duration_seconds": round(r.duration_seconds, 1),
            }
            for r in results
        ],
    }

    report_path = output_dir / f"profile_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n汇总报告已保存: {report_path}")
    print(f"  - 总计: {report['total_kernels']} 个 kernel")
    print(f"  - 成功: {report['successful']} 个")
    print(f"  - 失败: {report['failed']} 个")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="自动化 GPU Kernel Profiling 运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对 flash_attention 做 ncu 完整分析
  python profile_runner.py --kernel flash_attention --tool ncu

  # 对所有 kernel 做快速 nsys 扫描
  python profile_runner.py --all --tool nsys

  # 对 matmul 做 CPU perf stat 分析
  python profile_runner.py --kernel matmul --tool perf

  # 使用 ncu roofline section 做算术强度分析
  python profile_runner.py --kernel matmul --tool ncu --set roofline

  # 只分析第 1 次 kernel launch
  python profile_runner.py --kernel softmax --tool ncu --launch-count 1

  # 列出所有可用 kernel
  python profile_runner.py --list
        """,
    )

    # ── 主要参数 ──
    parser.add_argument(
        "--kernel",
        "-k",
        type=str,
        default=None,
        help="要分析的 kernel 名称（从 KERNEL_REGISTRY 中选择）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="分析所有已注册的 kernel",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有已注册的 kernel 并退出",
    )

    # ── 工具选择 ──
    parser.add_argument(
        "--tool",
        "-t",
        type=str,
        default="ncu",
        choices=["ncu", "nsys", "perf"],
        help="使用的 profiling 工具（默认: ncu）",
    )

    # ── ncu 专用参数 ──
    parser.add_argument(
        "--set",
        type=str,
        default="full",
        help="ncu section 集合（默认: full，可选: basic, full, roofline, memory, compute, occupancy）",
    )
    parser.add_argument(
        "--launch-skip",
        type=int,
        default=None,
        help="跳过前 N 次 kernel launch",
    )
    parser.add_argument(
        "--launch-count",
        type=int,
        default=None,
        help="只分析 N 次 kernel launch",
    )
    parser.add_argument(
        "--kernel-filter",
        type=str,
        default=None,
        help='按 kernel 名称过滤（支持正则，如 regex:"attention|matmul"）',
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="与指定的 .ncu-rep 基线文件做对比分析",
    )

    # ── perf 专用参数 ──
    parser.add_argument(
        "--perf-mode",
        type=str,
        default="stat",
        choices=["stat", "record", "mem"],
        help="perf 分析模式（默认: stat）",
    )

    # ── 通用参数 ──
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="使用的 GPU 设备 ID（默认: 0）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="输出目录（默认: ./profile_output/<kernel_name>/）",
    )

    args = parser.parse_args()

    # ── 列出所有 kernel ──
    if args.list:
        print(list_kernels())
        return

    # ── 验证参数 ──
    if not args.all and args.kernel is None:
        parser.error("请指定 --kernel <name> 或 --all")
    if args.all and args.kernel is not None:
        parser.error("--kernel 和 --all 不能同时使用")

    # ── 验证 kernel 名称 ──
    if args.kernel and args.kernel not in KERNEL_REGISTRY:
        print(f"错误: 未知 kernel '{args.kernel}'")
        print(list_kernels())
        sys.exit(1)

    # ── 确定输出目录 ──
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "13_kernel_profile" / "profile_output"
    ensure_output_dir(output_dir)

    # ── 收集额外参数 ──
    extra_kwargs = {}
    if args.tool == "ncu":
        extra_kwargs["section_set"] = args.set
        if args.launch_skip is not None:
            extra_kwargs["launch_skip"] = args.launch_skip
        if args.launch_count is not None:
            extra_kwargs["launch_count"] = args.launch_count
        if args.kernel_filter:
            extra_kwargs["kernel_filter"] = args.kernel_filter
        if args.compare:
            extra_kwargs["compare_baseline"] = args.compare
    elif args.tool == "perf":
        extra_kwargs["perf_mode"] = args.perf_mode

    extra_kwargs["gpu_device"] = args.gpu

    # ── 运行 profiling ──
    start_all = time.time()

    if args.all:
        results = run_all_kernels(args.tool, output_dir, **extra_kwargs)
    else:
        result = run_profile(args.kernel, args.tool, output_dir, **extra_kwargs)
        results = [result]

    total_elapsed = time.time() - start_all

    # ── 生成汇总报告 ──
    if len(results) > 0:
        generate_summary_report(results, output_dir)

    print(f"\n全部完成。总耗时: {total_elapsed:.1f}s")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
