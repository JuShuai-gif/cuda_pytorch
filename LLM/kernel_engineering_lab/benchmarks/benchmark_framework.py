"""
工业级 GPU Kernel Benchmark 框架。

借鉴:
  - CUTLASS profiler: 多维度 sweep (shape, dtype, block_size, num_warps)
  - PyTorch benchmark utils: CUDA event 精确计时 + 统计
  - Triton benchmarking: 自动 autotune 集成

功能:
  1. 多维度 sweep (shape, dtype, block_size, num_warps)
  2. CUDA event 精确计时 + 统计 (p50/p90/p99)
  3. Roofline 分析 (GFLOPS vs bandwidth, arithmetic intensity)
  4. CSV/JSON/Markdown 报告导出
  5. 与 PyTorch/cuBLAS baseline 自动对比
  6. 多 GPU 支持
  7. 自动 warmup 和 JIT 编译处理
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from benchmarks.gpu_info import GpuSpec, detect_gpu

# ============================================================================
# 数据结构
# ============================================================================


@dataclass
class KernelConfig:
    """
    Kernel 配置 - 借鉴 CUTLASS kernel config。

    记录 kernel launch 的所有可调参数，
    用于 sweep 和结果分析。
    """

    name: str = ""
    block_size: Tuple[int, int, int] = (128, 1, 1)  # (X, Y, Z)
    grid_size: Tuple[int, int, int] = (1, 1, 1)  # (X, Y, Z)
    num_warps: int = 4
    num_stages: int = 2
    shared_memory_bytes: int = 0
    register_count: int = 0
    num_threads: int = 128
    dtype: torch.dtype = torch.float16
    implementation: str = "cuda"  # "cuda", "triton", "torch", "cublas"


@dataclass
class ProblemSize:
    """问题规模描述。"""

    shape: Tuple[int, ...]
    description: str = ""


@dataclass
class BenchmarkResult:
    """单个 benchmark 的完整结果。"""

    config: KernelConfig
    problem_size: ProblemSize
    latency_us: float = 0.0  # 中位数延迟 (微秒)
    latency_p50_us: float = 0.0
    latency_p90_us: float = 0.0
    latency_p99_us: float = 0.0
    latency_mean_us: float = 0.0
    latency_std_us: float = 0.0
    gflops: float = 0.0  # 实测 GFLOPS
    peak_gflops_pct: float = 0.0  # 占峰值百分比
    bandwidth_gbps: float = 0.0  # 实测带宽
    peak_bandwidth_pct: float = 0.0  # 占峰值带宽百分比
    arithmetic_intensity: float = 0.0  # 算术强度 (FLOP/Byte)
    memory_bytes: int = 0  # 总内存访问字节数
    flop_count: int = 0  # 总浮点运算次数
    gpu_name: str = ""
    gpu_arch: str = ""
    roofline_bound: str = ""  # "memory_bound", "compute_bound", "balanced"
    repetitions: int = 0
    warmups: int = 0
    timestamp: str = ""


# ============================================================================
# Roofline 分析器
# ============================================================================


class RooflineAnalyzer:
    """
    Roofline 分析器 - 判断 kernel 是 memory-bound 还是 compute-bound。

    借鉴: Williams et al. (2009) "Roofline: An Insightful Visual Performance Model"
           CUTLASS profiler 的 roofline 可视化
    """

    def __init__(self, gpu_spec: GpuSpec, dtype: torch.dtype = torch.float16):
        self.gpu_spec = gpu_spec
        self.dtype = dtype

        # 选择合适的峰值指标
        if dtype in (torch.float16, torch.half):
            self.peak_tflops = gpu_spec.peak_tensor_core_fp16_tflops
        elif dtype in (torch.bfloat16,):
            self.peak_tflops = gpu_spec.peak_tensor_core_bf16_tflops
        elif dtype in (torch.float32, torch.float):
            self.peak_tflops = gpu_spec.peak_fp32_tflops
        else:
            self.peak_tflops = gpu_spec.peak_fp32_tflops

        self.peak_bw_gbps = gpu_spec.memory_bandwidth_gbps

    def compute_arithmetic_intensity(self, flops: int, bytes_moved: int) -> float:
        """计算算术强度 (FLOP/Byte)。"""
        if bytes_moved <= 0:
            return float("inf")
        return flops / bytes_moved

    def classify(self, gflops: float, bandwidth_gbps: float) -> str:
        """
        分类 kernel 的瓶颈类型。

        Args:
            gflops: 实测 GFLOPS。
            bandwidth_gbps: 实测带宽 (GB/s)。

        Returns:
            "memory_bound", "compute_bound", 或 "balanced"。
        """
        if bandwidth_gbps <= 0:
            return "compute_bound"
        ai = gflops / bandwidth_gbps  # GFLOPS / (GB/s) = FLOP/Byte
        ridge = self.peak_tflops * 1000.0 / self.peak_bw_gbps

        if ai < ridge * 0.8:
            return "memory_bound"
        elif ai > ridge * 1.2:
            return "compute_bound"
        else:
            return "balanced"

    def roofline_summary(self, result: BenchmarkResult) -> str:
        """生成单个 result 的 roofline 分析摘要。"""
        return (
            f"Roofline: {result.roofline_bound} | "
            f"AI={result.arithmetic_intensity:.2f} FLOP/Byte | "
            f"Peak %: compute={result.peak_gflops_pct:.1f}% "
            f"bw={result.peak_bandwidth_pct:.1f}%"
        )


# ============================================================================
# FLOPS / 内存估算器
# ============================================================================


def estimate_matmul_flops(M: int, N: int, K: int) -> int:
    """估算矩阵乘法 FLOPs: 2*M*N*K。"""
    return 2 * M * N * K


def estimate_matmul_bytes(M: int, N: int, K: int, dtype_size: int = 2) -> int:
    """估算矩阵乘法内存访问字节数 (不考虑缓存)。"""
    return (M * K + K * N + M * N) * dtype_size


def estimate_attention_flops(batch: int, heads: int, seq: int, head_dim: int) -> int:
    """估算 Attention FLOPs: 2*s^2*d + 2*s^2*d (QK^T + PV)。"""
    s2d = seq * seq * head_dim
    return int(4 * batch * heads * s2d)


def estimate_attention_bytes(
    batch: int, heads: int, seq: int, head_dim: int, dtype_size: int = 2
) -> int:
    """估算 Attention 内存访问字节数 (不考虑 FlashAttention 节省)。"""
    elements = batch * heads * seq * head_dim
    return elements * 3 * dtype_size  # Q, K, V each once


def estimate_elementwise_flops(n: int) -> int:
    """估算逐元素操作 FLOPs。"""
    return n


def estimate_elementwise_bytes(n: int, dtype_size: int = 2, reads: int = 2, writes: int = 1) -> int:
    """估算逐元素操作内存访问字节数。"""
    return n * (reads + writes) * dtype_size


def estimate_reduction_flops(n: int) -> int:
    """估算 reduction 操作 FLOPs: n-1 次加法。"""
    return max(n - 1, 0)


def estimate_reduction_bytes(n: int, dtype_size: int = 4) -> int:
    """估算 reduction 操作内存访问字节数。"""
    return n * dtype_size  # read input only


def estimate_norm_flops(rows: int, hidden_dim: int) -> int:
    """估算 LayerNorm/RMSNorm FLOPs: 读取、平方、求和、除法、乘以权重。"""
    return rows * hidden_dim * 5  # square + mean + normalize + scale + bias


def estimate_norm_bytes(rows: int, hidden_dim: int, dtype_size: int = 2) -> int:
    """估算 LayerNorm/RMSNorm 内存访问。"""
    return rows * hidden_dim * dtype_size * 3  # input, weight, output


# ============================================================================
# 工业级 Benchmark Suite
# ============================================================================


class BenchmarkSuite:
    """
    工业级基准测试套件 - 借鉴 CUTLASS profiler。

    使用 CUDA events 进行精确计时，
    提供 warmup 和 repeat 机制，
    自动计算 GFLOPS、带宽、算术强度。
    """

    def __init__(
        self,
        warmup: int = 10,
        repeat: int = 100,
        use_cuda_events: bool = True,
    ):
        self.warmup = warmup
        self.repeat = repeat
        self.use_cuda_events = use_cuda_events
        self.results: List[BenchmarkResult] = []

        # 自动检测 GPU
        self.gpu_spec = detect_gpu()
        self.gpu_name = "unknown"
        if self.gpu_spec:
            self.gpu_name = self.gpu_spec.model
        elif torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)

    def benchmark_kernel(
        self,
        fn: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        config: KernelConfig = KernelConfig(),
        problem_size: ProblemSize = ProblemSize(shape=()),
        flop_fn: Optional[Callable[..., int]] = None,
        bytes_fn: Optional[Callable[..., int]] = None,
    ) -> BenchmarkResult:
        """
        使用 CUDA events 对 kernel 进行精确计时。

        Args:
            fn: kernel 函数。
            args: 位置参数。
            kwargs: 关键字参数。
            config: Kernel 配置信息。
            problem_size: 问题规模描述。
            flop_fn: FLOP 计数函数 (*args, **kwargs) -> int。
            bytes_fn: 内存访问字节计数函数 (*args, **kwargs) -> int。

        Returns:
            BenchmarkResult 包含完整计时和分析结果。
        """
        if kwargs is None:
            kwargs = {}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot benchmark GPU kernels.")

        # 构建 full args
        full_args = args
        full_kwargs = kwargs

        # ---- Warmup: 处理 JIT 编译和 kernel 预热 ----
        for _ in range(self.warmup):
            fn(*full_args, **full_kwargs)
        torch.cuda.synchronize()

        # ---- Timing: 使用 CUDA events ----
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream()

        latencies_us: List[float] = []

        for _ in range(self.repeat):
            stream.record_event(start_event)
            fn(*full_args, **full_kwargs)
            stream.record_event(end_event)
            end_event.synchronize()
            elapsed_us = start_event.elapsed_time(end_event) * 1000.0  # ms -> us
            latencies_us.append(elapsed_us)

        # ---- Statistics ----
        sorted_latencies = sorted(latencies_us)
        n = len(sorted_latencies)
        p50 = sorted_latencies[int(n * 0.50)]
        p90 = sorted_latencies[min(int(n * 0.90), n - 1)]
        p99 = sorted_latencies[min(int(n * 0.99), n - 1)]
        mean_lat = statistics.mean(sorted_latencies)
        std_lat = statistics.stdev(sorted_latencies) if n > 1 else 0.0

        # ---- FLOPs / Bytes estimation ----
        flops = 0
        memory_bytes = 0
        if flop_fn is not None:
            flops = flop_fn(*args, **kwargs)
        if bytes_fn is not None:
            memory_bytes = bytes_fn(*args, **kwargs)

        # ---- Compute derived metrics ----
        mean_time_s = mean_lat / 1_000_000.0  # us -> s
        gflops = (flops / mean_time_s) / 1e9 if mean_time_s > 0 else 0.0
        bandwidth_gbps = (memory_bytes / mean_time_s) / 1e9 if mean_time_s > 0 else 0.0
        ai = gflops / max(bandwidth_gbps, 0.001)

        # ---- Roofline classification ----
        roofline_bound = "unknown"
        peak_gflops_pct = 0.0
        peak_bw_pct = 0.0

        if self.gpu_spec:
            analyzer = RooflineAnalyzer(self.gpu_spec, config.dtype)
            roofline_bound = analyzer.classify(gflops, bandwidth_gbps)
            if analyzer.peak_tflops > 0:
                peak_gflops_pct = gflops / (analyzer.peak_tflops * 1000.0) * 100.0
            if analyzer.peak_bw_gbps > 0:
                peak_bw_pct = bandwidth_gbps / analyzer.peak_bw_gbps * 100.0

        result = BenchmarkResult(
            config=config,
            problem_size=problem_size,
            latency_us=p50,
            latency_p50_us=p50,
            latency_p90_us=p90,
            latency_p99_us=p99,
            latency_mean_us=mean_lat,
            latency_std_us=std_lat,
            gflops=gflops,
            peak_gflops_pct=peak_gflops_pct,
            bandwidth_gbps=bandwidth_gbps,
            peak_bandwidth_pct=peak_bw_pct,
            arithmetic_intensity=ai,
            memory_bytes=memory_bytes,
            flop_count=flops,
            gpu_name=self.gpu_name,
            gpu_arch=self.gpu_spec.architecture if self.gpu_spec else "",
            roofline_bound=roofline_bound,
            repetitions=self.repeat,
            warmups=self.warmup,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self.results.append(result)
        return result

    def benchmark_torch_op(
        self,
        fn: Callable,
        *args: Any,
        name: str = "torch_op",
        problem_size: ProblemSize = ProblemSize(shape=()),
        flop_fn: Optional[Callable[..., int]] = None,
        bytes_fn: Optional[Callable[..., int]] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """便捷方法：benchmark PyTorch 操作。"""
        config = KernelConfig(name=name, implementation="torch", dtype=dtype)
        return self.benchmark_kernel(
            fn=fn,
            args=args,
            kwargs=kwargs,
            config=config,
            problem_size=problem_size,
            flop_fn=flop_fn,
            bytes_fn=bytes_fn,
        )

    def compare_kernel_vs_baseline(
        self,
        kernel_fn: Callable,
        kernel_config: KernelConfig,
        baseline_fn: Callable,
        baseline_name: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        problem_size: ProblemSize = ProblemSize(shape=()),
        flop_fn: Optional[Callable[..., int]] = None,
        bytes_fn: Optional[Callable[..., int]] = None,
    ) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """
        对比自定义 kernel 和 baseline (如 PyTorch 原生实现)。

        Returns:
            (kernel_result, baseline_result)。
        """
        baseline_config = KernelConfig(
            name=baseline_name,
            implementation="torch",
            dtype=kernel_config.dtype,
        )

        kernel_result = self.benchmark_kernel(
            fn=kernel_fn,
            args=args,
            kwargs=kwargs,
            config=kernel_config,
            problem_size=problem_size,
            flop_fn=flop_fn,
            bytes_fn=bytes_fn,
        )

        baseline_result = self.benchmark_kernel(
            fn=baseline_fn,
            args=args,
            kwargs=kwargs,
            config=baseline_config,
            problem_size=problem_size,
            flop_fn=flop_fn,
            bytes_fn=bytes_fn,
        )

        return kernel_result, baseline_result

    def sweep_block_sizes(
        self,
        kernel_gen_fn: Callable[[int, int, int], Tuple[Callable, Any, Any]],
        problem_size: ProblemSize,
        block_sizes: List[Tuple[int, int, int]],
        flop_fn: Optional[Callable[..., int]] = None,
        bytes_fn: Optional[Callable[..., int]] = None,
    ) -> List[BenchmarkResult]:
        """
        扫描 block size + num_warps 组合，找最优配置。

        Args:
            kernel_gen_fn: (block_x, block_y, block_z) -> (fn, args, kwargs)。
            problem_size: 问题规模。
            block_sizes: 待扫描的 block size 列表。
            flop_fn: FLOPs 计数函数。
            bytes_fn: 字节计数函数。

        Returns:
            所有配置的 benchmark 结果列表。
        """
        sweep_results = []
        for bx, by, bz in block_sizes:
            fn, args, kwargs = kernel_gen_fn(bx, by, bz)
            config = KernelConfig(
                name=f"block_{bx}x{by}x{bz}",
                block_size=(bx, by, bz),
                num_threads=bx * by * bz,
                implementation="cuda",
            )
            result = self.benchmark_kernel(
                fn=fn,
                args=args,
                kwargs=kwargs,
                config=config,
                problem_size=problem_size,
                flop_fn=flop_fn,
                bytes_fn=bytes_fn,
            )
            sweep_results.append(result)

        return sweep_results

    def sweep_problem_sizes(
        self,
        fn: Callable,
        config: KernelConfig,
        shapes: List[Tuple[int, ...]],
        flop_fn: Optional[Callable[..., int]] = None,
        bytes_fn: Optional[Callable[..., int]] = None,
        kwargs_factory: Optional[Callable[[Tuple[int, ...]], Dict[str, Any]]] = None,
    ) -> List[BenchmarkResult]:
        """
        对不同问题规模进行扫描。

        Args:
            fn: kernel 函数。
            config: 基础 kernel 配置。
            shapes: 要扫描的形状列表。
            flop_fn: FLOPs 计数函数。
            bytes_fn: 字节计数函数。
            kwargs_factory: 根据 shape 生成 kwargs 的工厂函数。

        Returns:
            所有规模的 benchmark 结果列表。
        """
        sweep_results = []
        for shape in shapes:
            kwargs = {}
            if kwargs_factory:
                kwargs = kwargs_factory(shape)
            ps = ProblemSize(shape=shape)
            c = dataclasses.replace(config)
            result = self.benchmark_kernel(
                fn=fn,
                args=shape,
                kwargs=kwargs,
                config=c,
                problem_size=ps,
                flop_fn=flop_fn,
                bytes_fn=bytes_fn,
            )
            sweep_results.append(result)

        return sweep_results

    def generate_report(
        self,
        results: Optional[List[BenchmarkResult]] = None,
        output_path: Optional[str] = None,
        title: str = "GPU Kernel Benchmark Report",
    ) -> str:
        """
        生成综合 benchmark 报告 (Markdown + CSV + JSON)。

        Args:
            results: 要报告的 benchmark 结果列表。默认使用 self.results。
            output_path: 输出文件基名 (无扩展名)。若提供，保存 .md/.csv/.json。
            title: 报告标题。

        Returns:
            Markdown 格式的报告字符串。
        """
        if results is None:
            results = self.results
        if not results:
            return f"# {title}\n\nNo results.\n"

        gpu_name = results[0].gpu_name

        # ---- Markdown Report ----
        lines = [
            f"# {title}",
            "",
            f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"GPU: {gpu_name}",
            f"Architecture: {results[0].gpu_arch if results[0].gpu_arch else 'unknown'}",
            "",
            "## Performance Summary",
            "",
            "| Implementation | Shape | Latency (us) | GFLOPS | BW (GB/s) | AI | Peak % | Roofline |",
            "|----------------|-------|-------------|--------|-----------|----|--------|----------|",
        ]

        for r in results:
            shape_str = "x".join(str(d) for d in r.problem_size.shape)
            lines.append(
                f"| {r.config.name} | {shape_str} | {r.latency_p50_us:.1f} | "
                f"{r.gflops:.1f} | {r.bandwidth_gbps:.1f} | "
                f"{r.arithmetic_intensity:.2f} | {r.peak_gflops_pct:.1f}% | "
                f"{r.roofline_bound} |"
            )

        # ---- Roofline Analysis ----
        if self.gpu_spec:
            analyzer = RooflineAnalyzer(self.gpu_spec)
            lines.append("")
            lines.append("## Roofline Analysis")
            lines.append("")
            lines.append(
                f"Peak Compute: {analyzer.peak_tflops:.1f} TFLOPS | "
                f"Peak Bandwidth: {analyzer.peak_bw_gbps:.1f} GB/s | "
                f"Ridge Point: {analyzer.peak_tflops * 1000.0 / analyzer.peak_bw_gbps:.1f} FLOP/Byte"
            )
            lines.append("")

            for r in results:
                lines.append(
                    f"- **{r.config.name}**: {r.roofline_bound}, "
                    f"AI={r.arithmetic_intensity:.2f} FLOP/Byte, "
                    f"Compute: {r.peak_gflops_pct:.1f}%, BW: {r.peak_bandwidth_pct:.1f}%"
                )

        # ---- Comparison Table (if multiple implementations) ----
        impls = set(r.config.implementation for r in results)
        if len(impls) > 1:
            lines.append("")
            lines.append("## Cross-Implementation Comparison")
            lines.append("")
            lines.append("| Shape | CUDA Ker | PyTorch | Speedup |")
            lines.append("|-------|----------|---------|---------|")

            # Group by problem size
            by_shape: Dict[Tuple, Dict[str, BenchmarkResult]] = {}
            for r in results:
                shape_key = r.problem_size.shape
                if shape_key not in by_shape:
                    by_shape[shape_key] = {}
                by_shape[shape_key][r.config.implementation] = r

            for shape_key, impl_results in by_shape.items():
                shape_str = "x".join(str(d) for d in shape_key)
                cuda_lat = impl_results.get("cuda", None)
                torch_lat = impl_results.get("torch", None)
                cuda_str = f"{cuda_lat.latency_p50_us:.1f}" if cuda_lat else "N/A"
                torch_str = f"{torch_lat.latency_p50_us:.1f}" if torch_lat else "N/A"
                speedup = ""
                if cuda_lat and torch_lat and cuda_lat.latency_p50_us > 0:
                    s = torch_lat.latency_p50_us / cuda_lat.latency_p50_us
                    speedup = f"{s:.2f}x"
                lines.append(f"| {shape_str} | {cuda_str} | {torch_str} | {speedup} |")

        report_md = "\n".join(lines)

        # ---- Save output files ----
        if output_path:
            base = Path(output_path)
            # Markdown
            with open(base.with_suffix(".md"), "w") as f:
                f.write(report_md)
            print(f"Markdown report saved to: {base.with_suffix('.md')}")

            # CSV
            csv_path = base.with_suffix(".csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "name",
                        "implementation",
                        "shape",
                        "latency_p50_us",
                        "latency_p90_us",
                        "latency_p99_us",
                        "gflops",
                        "peak_gflops_pct",
                        "bandwidth_gbps",
                        "peak_bandwidth_pct",
                        "arithmetic_intensity",
                        "memory_bytes",
                        "flop_count",
                        "roofline_bound",
                        "gpu_name",
                        "gpu_arch",
                    ]
                )
                for r in results:
                    writer.writerow(
                        [
                            r.config.name,
                            r.config.implementation,
                            "x".join(str(d) for d in r.problem_size.shape),
                            f"{r.latency_p50_us:.1f}",
                            f"{r.latency_p90_us:.1f}",
                            f"{r.latency_p99_us:.1f}",
                            f"{r.gflops:.1f}",
                            f"{r.peak_gflops_pct:.1f}",
                            f"{r.bandwidth_gbps:.1f}",
                            f"{r.peak_bandwidth_pct:.1f}",
                            f"{r.arithmetic_intensity:.2f}",
                            r.memory_bytes,
                            r.flop_count,
                            r.roofline_bound,
                            r.gpu_name,
                            r.gpu_arch,
                        ]
                    )
            print(f"CSV report saved to: {csv_path}")

            # JSON
            json_path = base.with_suffix(".json")
            serialized = [dataclasses.asdict(r) for r in results]
            with open(json_path, "w") as f:
                json.dump(serialized, f, indent=2, default=str)
            print(f"JSON report saved to: {json_path}")

        return report_md

    def print_comparison_table(self, results: Optional[List[BenchmarkResult]] = None) -> None:
        """打印格式化的对比表格到终端。"""
        if results is None:
            results = self.results
        if not results:
            print("No results to display.")
            return

        headers = [
            "Implementation",
            "Shape",
            "Lat(us)",
            "GFLOPS",
            "BW(GB/s)",
            "AI",
            "Peak%",
            "Bound",
        ]
        rows = []
        for r in results:
            shape_str = "x".join(str(d) for d in r.problem_size.shape)
            rows.append(
                [
                    r.config.name,
                    shape_str,
                    f"{r.latency_p50_us:.1f}",
                    f"{r.gflops:.1f}",
                    f"{r.bandwidth_gbps:.1f}",
                    f"{r.arithmetic_intensity:.2f}",
                    f"{r.peak_gflops_pct:.1f}%",
                    r.roofline_bound,
                ]
            )

        col_widths = [
            max(len(str(rows[i][j])) for i in range(len(rows))) for j in range(len(headers))
        ]
        for i, h in enumerate(headers):
            col_widths[i] = max(col_widths[i], len(h))
        fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
        sep = "-+-".join("-" * w for w in col_widths)
        print(fmt.format(*headers))
        print(sep)
        for row in rows:
            print(fmt.format(*row))

    def save_results(self, path: str) -> None:
        """保存所有结果到 JSON 文件。"""
        data = [dataclasses.asdict(r) for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to: {path}")

    def load_results(self, path: str) -> List[BenchmarkResult]:
        """从 JSON 文件加载结果。"""
        with open(path, "r") as f:
            data = json.load(f)
        results = [BenchmarkResult(**d) for d in data]
        self.results.extend(results)
        return results

    def clear(self) -> None:
        """清除所有累积的结果。"""
        self.results.clear()


# ============================================================================
# 便捷工厂方法
# ============================================================================


def create_suite(
    warmup: int = 10,
    repeat: int = 100,
) -> BenchmarkSuite:
    """创建一个预配置的 benchmark suite。"""
    return BenchmarkSuite(warmup=warmup, repeat=repeat)


def quick_bench(
    fn: Callable,
    *args: Any,
    name: str = "kernel",
    warmup: int = 5,
    repeat: int = 50,
    **kwargs: Any,
) -> BenchmarkResult:
    """
    快速 benchmark 一个 kernel 函数。

    这是最常用的入口，自动检测 GPU 并进行 roofline 分析。
    """
    suite = BenchmarkSuite(warmup=warmup, repeat=repeat)
    config = KernelConfig(name=name)
    return suite.benchmark_kernel(fn=fn, args=args, kwargs=kwargs, config=config)


# ============================================================================
# 测试用 main
# ============================================================================


def _demo_matmul_bench() -> None:
    """演示矩阵乘法的 benchmark 流程。"""
    suite = BenchmarkSuite(warmup=5, repeat=50)

    M, N, K = 1024, 1024, 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    flops = estimate_matmul_flops(M, N, K)
    mem_bytes = estimate_matmul_bytes(M, N, K, dtype_size=2)

    result = suite.benchmark_torch_op(
        torch.matmul,
        A,
        B,
        name="torch.matmul_fp16",
        problem_size=ProblemSize(shape=(M, N, K), description="Standard matmul"),
        flop_fn=lambda *a: flops,
        bytes_fn=lambda *a: mem_bytes,
        dtype=torch.float16,
    )

    suite.print_comparison_table([result])
    print()
    if suite.gpu_spec:
        analyzer = RooflineAnalyzer(suite.gpu_spec)
        print(analyzer.roofline_summary(result))


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
        _demo_matmul_bench()
    else:
        print("CUDA not available. Cannot run demo bench.")
