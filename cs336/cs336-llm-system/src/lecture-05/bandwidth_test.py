"""
第 05 讲 — GPU 架构：HBM bandwidth 与 matmul 吞吐量分析。

定义了分析函数，用于基于 GPU 规格估算可达到的带宽和 matmul
吞吐量。不执行实际的 GPU 基准测试（这些是离线分析工具）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 分析模型
# ---------------------------------------------------------------------------


@dataclass
class BandwidthEstimates:
    """给定 GPU 配置的分析带宽估算。"""

    gpu_name: str
    peak_hbm_bw_gbs: float  # 理论峰值 GB/s
    achievable_bw_gbs: float  # 优化良好的 kernel 约达到峰值的 70–85%
    efficiency: float  # achievable / peak


def estimate_hbm_bandwidth(
    gpu_name: str,
    peak_bw_gbs: float,
    efficiency: float = 0.80,
) -> BandwidthEstimates:
    """估算可实现的 HBM 带宽。

    真实场景的 kernel 中，简单流式 kernel 通常可以达到峰值带宽的
    70–85%；更复杂的访存模式则可达到 60–80%。
    """
    return BandwidthEstimates(
        gpu_name=gpu_name,
        peak_hbm_bw_gbs=peak_bw_gbs,
        achievable_bw_gbs=peak_bw_gbs * efficiency,
        efficiency=efficiency,
    )


# ---------------------------------------------------------------------------
# Matmul 吞吐量模型
# ---------------------------------------------------------------------------


@dataclass
class MatmulThroughput:
    """给定形状和 GPU 的 matmul 吞吐量估算。"""

    gpu_name: str
    M: int
    N: int
    K: int
    dtype: str  # fp16, bf16, fp8
    peak_tflops: float  # 该数据类型的理论峰值
    achieved_tflops: float
    efficiency: float
    time_us: float  # 估算的计算时间，单位微秒
    data_moved_gib: float
    arithmetic_intensity: float  # FLOP / Byte


def estimate_matmul_throughput(
    M: int,
    N: int,
    K: int,
    gpu_name: str = "A100",
    dtype: str = "bf16",
    efficiency: float = 0.75,
) -> MatmulThroughput:
    """分析估算 matmul 吞吐量。

    Parameters
    ----------
    M, N, K : int
        矩阵维度：C[M,N] = A[M,K] × B[K,N]。
    gpu_name : str
        'V100'、'A100' 或 'H100' 之一。
    dtype : str
        计算数据类型（'fp16'、'bf16'、'fp8'）。
    efficiency : float
        预期可达到峰值的比例（0..1）。

    Returns
    -------
    带有详细估算的 MatmulThroughput。
    """
    peak_map: Dict[str, Dict[str, float]] = {
        "V100": {"fp16": 125.0, "bf16": 0.0, "fp8": 0.0},
        "A100": {"fp16": 312.0, "bf16": 312.0, "fp8": 0.0},
        "H100": {"fp16": 989.0, "bf16": 989.0, "fp8": 1979.0},
    }
    bw_map = {"V100": 900.0, "A100": 2039.0, "H100": 3350.0}

    peak_tflops = peak_map.get(gpu_name, {}).get(dtype, 0.0)
    bw_gbs = bw_map.get(gpu_name, 3350.0)

    elem_size_map = {"fp16": 2, "bf16": 2, "fp8": 1, "fp32": 4}
    elem_size = elem_size_map.get(dtype, 2)

    # FLOPs
    flops = 2.0 * M * N * K  # 融合乘加

    # 数据传输量
    data_read_gib = (M * K + K * N) * elem_size / (1024**3)
    data_write_gib = (M * N) * 4 / (1024**3)  # 输出通常为 fp32 累加器
    data_moved = data_read_gib + data_write_gib

    ai = flops / max(data_moved * (1024**3), 1)  # FLOP / Byte

    # 可达到的 TFLOPS
    achieved_tflops = peak_tflops * efficiency

    # 计算时间
    time_seconds = (
        flops / (achieved_tflops * 1e12) if achieved_tflops > 0 else float("inf")
    )

    return MatmulThroughput(
        gpu_name=gpu_name,
        M=M,
        N=N,
        K=K,
        dtype=dtype,
        peak_tflops=peak_tflops,
        achieved_tflops=achieved_tflops,
        efficiency=efficiency,
        time_us=time_seconds * 1e6,
        data_moved_gib=data_moved,
        arithmetic_intensity=ai,
    )


# ---------------------------------------------------------------------------
# Roofline 分析
# ---------------------------------------------------------------------------


def matmul_roofline_analysis(
    M: int,
    N: int,
    K: int,
    dtype: str = "bf16",
) -> List[Tuple[str, float, float, str]]:
    """对给定 matmul 形状在多种 GPU 上运行 roofline 分析。

    返回 (gpu_name, attained_tflops, memory_bw_tflops, bound) 的列表。
    """
    results: List[Tuple[str, float, float, str]] = []
    for gpu in ["V100", "A100", "H100"]:
        est = estimate_matmul_throughput(M, N, K, gpu, dtype, efficiency=1.0)
        if est.peak_tflops == 0:
            continue

        bw_map = {"V100": 900.0, "A100": 2039.0, "H100": 3350.0}
        mem_bound_tflops = est.arithmetic_intensity * bw_map[gpu] / 1e3
        bound = "compute" if est.peak_tflops <= mem_bound_tflops else "memory"
        attained = min(est.peak_tflops, mem_bound_tflops)
        results.append((gpu, attained, mem_bound_tflops, bound))
    return results


# ---------------------------------------------------------------------------
# 带宽受限操作模型
# ---------------------------------------------------------------------------


@dataclass
class BandwidthLimitedOp:
    """带宽受限操作的特性。"""

    name: str  # 例如 "ReLU", "GeLU", "Dropout", "LayerNorm"
    elements: int  # 涉及的元素数量
    bytes_read: int
    bytes_write: int
    arithmetic_intensity: float  # FLOP / Byte
    estimated_time_us: float  # 按可达带宽估算
    memory_bound: bool  # 若 AI 低于 ridge point 则为 True


def analyze_bandwidth_op(
    name: str,
    batch_size: int,
    seq_len: int,
    dim: int,
    flops_per_elem: float,
    bytes_per_elem: int = 2,
    gpu_bw_gbs: float = 2039.0,  # A100
    ridge_point: float = 150.0,  # A100 bf16 的近似值
) -> BandwidthLimitedOp:
    """分析一个带宽受限的逐元素操作。

    Parameters
    ----------
    name : str
        操作名称。
    batch_size, seq_len, dim : int
        张量维度。
    flops_per_elem : float
        每个元素的 FLOPs（例如 GeLU 为 5，ReLU 为 1）。
    bytes_per_elem : int
        每个元素的字节数。
    gpu_bw_gbs : float
        可达到的 GPU 内存带宽。
    ridge_point : float
        Roofline ridge point（FLOP / Byte）。
    """
    elements = batch_size * seq_len * dim
    bytes_read = elements * bytes_per_elem
    bytes_write = elements * bytes_per_elem
    ai = flops_per_elem * elements / max(bytes_read + bytes_write, 1)

    total_bytes = bytes_read + bytes_write
    time_s = total_bytes / (gpu_bw_gbs * 1e9)
    memory_bound = ai < ridge_point

    return BandwidthLimitedOp(
        name=name,
        elements=elements,
        bytes_read=bytes_read,
        bytes_write=bytes_write,
        arithmetic_intensity=ai,
        estimated_time_us=time_s * 1e6,
        memory_bound=memory_bound,
    )


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== HBM Bandwidth Estimates ===\n")
    for gpu, peak in [("V100", 900.0), ("A100", 2039.0), ("H100", 3350.0)]:
        est = estimate_hbm_bandwidth(gpu, peak)
        print(
            f"  {gpu}: peak={est.peak_hbm_bw_gbs:.0f} GB/s, achievable≈{est.achievable_bw_gbs:.0f} GB/s ({est.efficiency:.0%})"
        )

    print("\n=== Matmul Throughput Estimates ===\n")
    for shape in [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192)]:
        M, N, K = shape
        est = estimate_matmul_throughput(M, N, K, "H100", "bf16")
        print(f"  {M}×{N}×{K} on {est.gpu_name} ({est.dtype}):")
        print(
            f"    AI={est.arithmetic_intensity:.0f} FLOP/Byte, time≈{est.time_us:.1f} µs, eff≈{est.efficiency:.0%}"
        )

    print("\n=== Roofline Analysis ===\n")
    for gpu, atflops, mbw_tflops, bound in matmul_roofline_analysis(8192, 8192, 8192):
        print(
            f"  {gpu}: attained≈{atflops:.0f} TFLOPS, mem-bw limit≈{mbw_tflops:.0f} TFLOPS → {bound}-bound"
        )

    print("\n=== Bandwidth-Limited Operations ===\n")
    for op_name, flops_per in [("ReLU", 1.0), ("GeLU", 5.0), ("Dropout", 1.0)]:
        op = analyze_bandwidth_op(op_name, 1, 2048, 4096, flops_per)
        print(
            f"  {op.name:10s}: AI={op.arithmetic_intensity:.4f}, time≈{op.estimated_time_us:.1f} µs, mem-bound={op.memory_bound}"
        )

    print("\nAll checks passed.")
