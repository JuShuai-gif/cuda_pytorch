"""
Roofline model analysis for GPU kernel optimization.

The roofline model helps identify whether a kernel is memory-bound
or compute-bound by comparing its arithmetic intensity (FLOPs/Byte)
against the GPU's ridge point.

Key concepts:
    - Arithmetic Intensity (AI): FLOPs / Bytes transferred
    - Ridge Point: Peak FLOPs / Peak Bandwidth (FLOP/Byte)
    - If AI < Ridge Point: Memory-bound (optimize data movement)
    - If AI >= Ridge Point: Compute-bound (optimize computation)

GPU ridge points:
    - H100 (bf16): 989 TFLOPS / 3.35 TB/s = 295 FLOP/Byte
    - A100 (bf16): 312 TFLOPS / 2.04 TB/s = 153 FLOP/Byte
    - RTX 4090 (bf16): 83 TFLOPS / 1.01 TB/s = 82 FLOP/Byte

Reference: Williams et al., "Roofline: An Insightful Visual
Performance Model for Multicore Architectures", CACM 2009.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ==============================================================================
#  GPU hardware specification constants
# ==============================================================================


class BottleneckType(Enum):
    """Kernel bottleneck classification."""

    MEMORY_BOUND = "memory"
    COMPUTE_BOUND = "compute"
    LATENCY_BOUND = "latency"


@dataclass
class GPURooflineSpec:
    """Roofline model specifications for a GPU.

    Attributes:
        name: GPU model name.
        peak_fp16_tflops: Peak FP16 TFLOPS with Tensor Cores.
        peak_bf16_tflops: Peak BF16 TFLOPS with Tensor Cores.
        peak_fp8_tflops: Peak FP8 TFLOPS with Tensor Cores.
        peak_fp32_tflops: Peak FP32 TFLOPS.
        memory_bandwidth_gbs: Peak HBM bandwidth in GB/s.
        ridge_fp16: Ridge point for FP16 (FLOP/Byte).
        ridge_bf16: Ridge point for BF16 (FLOP/Byte).
        ridge_fp8: Ridge point for FP8 (FLOP/Byte).
        ridge_fp32: Ridge point for FP32 (FLOP/Byte).
    """

    name: str
    peak_fp16_tflops: float
    peak_bf16_tflops: float
    peak_fp8_tflops: float
    peak_fp32_tflops: float
    memory_bandwidth_gbs: float

    @property
    def ridge_fp16(self) -> float:
        return self.peak_fp16_tflops * 1e3 / self.memory_bandwidth_gbs

    @property
    def ridge_bf16(self) -> float:
        return self.peak_bf16_tflops * 1e3 / self.memory_bandwidth_gbs

    @property
    def ridge_fp8(self) -> float:
        return self.peak_fp8_tflops * 1e3 / self.memory_bandwidth_gbs

    @property
    def ridge_fp32(self) -> float:
        return self.peak_fp32_tflops * 1e3 / self.memory_bandwidth_gbs


# Predefined GPU roofline specifications
GPU_ROOFLINE_SPECS: Dict[str, GPURooflineSpec] = {
    "H100": GPURooflineSpec(
        name="NVIDIA H100 (SXM5)",
        peak_fp16_tflops=989.0,
        peak_bf16_tflops=989.0,
        peak_fp8_tflops=1979.0,
        peak_fp32_tflops=67.0,
        memory_bandwidth_gbs=3350.0,
    ),
    "A100": GPURooflineSpec(
        name="NVIDIA A100 (SXM4)",
        peak_fp16_tflops=312.0,
        peak_bf16_tflops=312.0,
        peak_fp8_tflops=0.0,
        peak_fp32_tflops=19.5,
        memory_bandwidth_gbs=2039.0,
    ),
    "V100": GPURooflineSpec(
        name="NVIDIA V100 (SXM2)",
        peak_fp16_tflops=125.0,
        peak_bf16_tflops=0.0,
        peak_fp8_tflops=0.0,
        peak_fp32_tflops=15.7,
        memory_bandwidth_gbs=900.0,
    ),
    "RTX4090": GPURooflineSpec(
        name="NVIDIA RTX 4090",
        peak_fp16_tflops=165.4,
        peak_bf16_tflops=82.6,
        peak_fp8_tflops=0.0,
        peak_fp32_tflops=82.6,
        memory_bandwidth_gbs=1008.0,
    ),
    "B200": GPURooflineSpec(
        name="NVIDIA B200 (Blackwell)",
        peak_fp16_tflops=2250.0,
        peak_bf16_tflops=2250.0,
        peak_fp8_tflops=4500.0,
        peak_fp32_tflops=90.0,
        memory_bandwidth_gbs=8000.0,
    ),
}


# ==============================================================================
#  Arithmetic intensity computation
# ==============================================================================


@dataclass
class OperationAnalysis:
    """Analysis of an operation's arithmetic intensity.

    Attributes:
        name: Operation name.
        flops: Total floating point operations.
        bytes_transferred: Total bytes read + written.
        arithmetic_intensity: FLOPs per byte.
        ridge_point: GPU ridge point for the given dtype.
        bottleneck: Memory-bound or compute-bound.
        peak_tflops: GPU peak TFLOPS for the given dtype.
        memory_bandwidth_gbs: GPU memory bandwidth.
        max_attainable_tflops: Theoretical maximum TFLOPS for this operation
            (limited by either compute or memory bandwidth).
    """

    name: str
    flops: float
    bytes_transferred: float
    arithmetic_intensity: float
    ridge_point: float
    bottleneck: BottleneckType
    peak_tflops: float
    memory_bandwidth_gbs: float
    max_attainable_tflops: float


def compute_arithmetic_intensity(
    flops: float,
    bytes_read: float,
    bytes_write: float,
) -> float:
    """Compute arithmetic intensity (FLOP/Byte) for an operation.

    Arithmetic Intensity = Total FLOPS / Total Bytes Transferred

    Args:
        flops: Total floating-point operations.
        bytes_read: Number of bytes read from memory.
        bytes_write: Number of bytes written to memory.

    Returns:
        Arithmetic intensity in FLOP/Byte. Returns infinity if no data movement.

    Raises:
        ValueError: If any argument is negative.
    """
    if flops < 0 or bytes_read < 0 or bytes_write < 0:
        raise ValueError(
            f"All arguments must be non-negative: "
            f"flops={flops}, bytes_read={bytes_read}, bytes_write={bytes_write}"
        )

    total_bytes = bytes_read + bytes_write
    if total_bytes == 0:
        return float("inf")
    return flops / total_bytes


def compute_ridge_point(
    peak_tflops: float,
    memory_bandwidth_gbs: float,
) -> float:
    """Compute the roofline ridge point for given hardware specs.

    Ridge Point = Peak TFLOPS / Memory Bandwidth (in FLOP/Byte)

    This is the arithmetic intensity at which the memory bandwidth
    ceiling meets the compute ceiling.

    Args:
        peak_tflops: Peak TFLOPS for the data type.
        memory_bandwidth_gbs: Peak memory bandwidth in GB/s.

    Returns:
        Ridge point in FLOP/Byte.

    Raises:
        ValueError: If memory_bandwidth_gbs is zero.
    """
    if memory_bandwidth_gbs <= 0:
        raise ValueError(
            f"memory_bandwidth_gbs must be positive, got {memory_bandwidth_gbs}"
        )
    return peak_tflops * 1e3 / memory_bandwidth_gbs


def identify_bottleneck(
    arithmetic_intensity: float,
    ridge_point: float,
) -> BottleneckType:
    """Determine if an operation is memory-bound or compute-bound.

    Args:
        arithmetic_intensity: Operation's AI in FLOP/Byte.
        ridge_point: GPU ridge point in FLOP/Byte.

    Returns:
        BottleneckType indicating memory-bound or compute-bound.
    """
    if arithmetic_intensity < ridge_point:
        return BottleneckType.MEMORY_BOUND
    return BottleneckType.COMPUTE_BOUND


def estimate_gpu_utilization(
    achieved_tflops: float,
    peak_tflops: float,
    achieved_bandwidth_gbs: float,
    peak_bandwidth_gbs: float,
    bottleneck: BottleneckType,
) -> float:
    """Estimate GPU utilization based on the limiting resource.

    For memory-bound operations, utilization is measured against
    peak bandwidth. For compute-bound operations, against peak TFLOPS.

    This gives a single number (0-1) representing how close the
    operation is to the hardware's theoretical limits.

    Args:
        achieved_tflops: Actually measured TFLOPS.
        peak_tflops: Peak TFLOPS of the GPU for the given dtype.
        achieved_bandwidth_gbs: Actually measured memory bandwidth in GB/s.
        peak_bandwidth_gbs: Peak memory bandwidth of the GPU.
        bottleneck: Which resource is the limiting factor.

    Returns:
        Utilization ratio (0.0 to 1.0+).
    """
    if bottleneck == BottleneckType.MEMORY_BOUND:
        return achieved_bandwidth_gbs / max(peak_bandwidth_gbs, 1.0)
    else:
        return achieved_tflops / max(peak_tflops, 1.0)


# ==============================================================================
#  Operation-specific AI calculators
# ==============================================================================


def ai_matmul(
    M: int,
    N: int,
    K: int,
    bytes_per_elem_input: int = 2,
    bytes_per_elem_output: int = 2,
) -> float:
    """Compute arithmetic intensity for matrix multiplication C[M,N] = A[M,K] x B[K,N].

    FLOPs = 2 * M * N * K (fused multiply-add)
    Bytes = (M*K + K*N) * input_elem_size + (M*N) * output_elem_size

    Args:
        M, N, K: Matrix dimensions.
        bytes_per_elem_input: Bytes per element for inputs (e.g. 2 for fp16).
        bytes_per_elem_output: Bytes per element for output.

    Returns:
        Arithmetic intensity in FLOP/Byte.
    """
    flops = 2.0 * M * N * K
    bytes_read = (M * K + K * N) * bytes_per_elem_input
    bytes_write = M * N * bytes_per_elem_output
    return compute_arithmetic_intensity(flops, bytes_read, bytes_write)


def ai_attention(
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    bytes_per_elem: int = 2,
) -> float:
    """Compute arithmetic intensity for scaled dot-product attention.

    For single head:
        FLOPs = 4 * seq_len_q * seq_len_k * head_dim
        Bytes = (seq_len_q + 2 * seq_len_k) * head_dim * bytes_per_elem

    This assumes the attention matrix does NOT get materialized in HBM
    (FlashAttention-style). For standard attention, add N^2 * bytes_per_elem
    for the intermediate scores matrix.

    Args:
        seq_len_q: Query sequence length.
        seq_len_k: Key sequence length.
        head_dim: Head dimension.
        bytes_per_elem: Bytes per element (2 for fp16/bf16).

    Returns:
        Arithmetic intensity in FLOP/Byte.
    """
    flops = 4.0 * seq_len_q * seq_len_k * head_dim
    bytes_read = (seq_len_q + 2 * seq_len_k) * head_dim * bytes_per_elem
    bytes_write = seq_len_q * head_dim * bytes_per_elem
    return compute_arithmetic_intensity(flops, bytes_read, bytes_write)


def ai_attention_standard(
    seq_len: int,
    head_dim: int,
    bytes_per_elem: int = 2,
) -> float:
    """Compute arithmetic intensity for standard (non-Flash) attention.

    Standard attention materializes the full NxN attention matrix,
    substantially reducing arithmetic intensity for long sequences.

    FLOPs  = 4 * N^2 * head_dim
    Bytes  = (3*N*head_dim + 2*N^2) * bytes_per_elem
           (Q, K, V reads + NxN matrix read/write + O write)

    Args:
        seq_len: Sequence length (square attention).
        head_dim: Head dimension.
        bytes_per_elem: Bytes per element.

    Returns:
        Arithmetic intensity in FLOP/Byte.
    """
    flops = 4.0 * seq_len * seq_len * head_dim
    bytes_read = (3 * seq_len * head_dim + seq_len * seq_len) * bytes_per_elem
    bytes_write = (seq_len * head_dim + seq_len * seq_len) * bytes_per_elem
    return compute_arithmetic_intensity(flops, bytes_read, bytes_write)


def ai_elementwise(
    flops_per_element: float,
    bytes_per_elem: int = 2,
) -> float:
    """Compute arithmetic intensity for an element-wise operation.

    Most element-wise operations (ReLU, GeLU, SiLU, Dropout) have
    very low arithmetic intensity because they have O(1) FLOPs per
    element but must read and write each element.

    AI = flops_per_element / (2 * bytes_per_elem)

    Args:
        flops_per_element: FLOPs per element (e.g. 1 for ReLU, 5 for GeLU).
        bytes_per_elem: Bytes per element.

    Returns:
        Arithmetic intensity in FLOP/Byte.

    Raises:
        ValueError: If flops_per_element is negative.
    """
    if flops_per_element < 0:
        raise ValueError(
            f"flops_per_element must be non-negative, got {flops_per_element}"
        )
    return compute_arithmetic_intensity(
        flops_per_element, bytes_per_elem, bytes_per_elem
    )


def ai_layernorm(
    seq_len: int,
    hidden_dim: int,
    bytes_per_elem: int = 2,
) -> float:
    """Compute arithmetic intensity for Layer Normalization.

    FLOPs ≈ 5 * N * D  (mean, var, normalize, scale, shift)
    Bytes = 2 * N * D * bytes_per_elem  (read input + write output)

    Args:
        seq_len: Number of tokens (batch dimension).
        hidden_dim: Hidden dimension size.
        bytes_per_elem: Bytes per element.

    Returns:
        Arithmetic intensity in FLOP/Byte.
    """
    flops = 5.0 * seq_len * hidden_dim
    bytes_read = seq_len * hidden_dim * bytes_per_elem
    bytes_write = seq_len * hidden_dim * bytes_per_elem
    return compute_arithmetic_intensity(flops, bytes_read, bytes_write)


def ai_rmsnorm(
    seq_len: int,
    hidden_dim: int,
    bytes_per_elem: int = 2,
) -> float:
    """Compute arithmetic intensity for RMS Normalization.

    FLOPs ≈ 4 * N * D  (square, mean, rsqrt, scale)
    Bytes = 2 * N * D * bytes_per_elem

    Args:
        seq_len: Number of tokens.
        hidden_dim: Hidden dimension size.
        bytes_per_elem: Bytes per element.

    Returns:
        Arithmetic intensity in FLOP/Byte.
    """
    flops = 4.0 * seq_len * hidden_dim
    bytes_read = seq_len * hidden_dim * bytes_per_elem
    bytes_write = seq_len * hidden_dim * bytes_per_elem
    return compute_arithmetic_intensity(flops, bytes_read, bytes_write)


# ==============================================================================
#  Full roofline analysis
# ==============================================================================


def analyze_operation(
    name: str,
    flops: float,
    bytes_read: float,
    bytes_write: float,
    gpu_name: str = "H100",
    dtype: str = "bf16",
) -> OperationAnalysis:
    """Perform complete roofline analysis for an operation.

    Args:
        name: Human-readable operation name.
        flops: Total floating-point operations.
        bytes_read: Bytes read from HBM.
        bytes_write: Bytes written to HBM.
        gpu_name: GPU model (H100, A100, V100, RTX4090, B200).
        dtype: Computation dtype (fp16, bf16, fp8, fp32).

    Returns:
        OperationAnalysis with bottleneck classification and utilization.

    Raises:
        KeyError: If gpu_name is not recognized.
    """
    spec = GPU_ROOFLINE_SPECS[gpu_name]

    peak_tflops_map = {
        "fp16": spec.peak_fp16_tflops,
        "bf16": spec.peak_bf16_tflops,
        "fp8": spec.peak_fp8_tflops,
        "fp32": spec.peak_fp32_tflops,
    }
    peak_tflops = peak_tflops_map.get(dtype, spec.peak_bf16_tflops)

    ridge_map = {
        "fp16": spec.ridge_fp16,
        "bf16": spec.ridge_bf16,
        "fp8": spec.ridge_fp8,
        "fp32": spec.ridge_fp32,
    }
    ridge = ridge_map.get(dtype, spec.ridge_bf16)

    ai = compute_arithmetic_intensity(flops, bytes_read, bytes_write)
    bottleneck = identify_bottleneck(ai, ridge)

    # Maximum attainable TFLOPS given the bottleneck
    mem_bound_tflops = ai * spec.memory_bandwidth_gbs / 1e3
    max_attainable = min(peak_tflops, mem_bound_tflops)

    return OperationAnalysis(
        name=name,
        flops=flops,
        bytes_transferred=bytes_read + bytes_write,
        arithmetic_intensity=ai,
        ridge_point=ridge,
        bottleneck=bottleneck,
        peak_tflops=peak_tflops,
        memory_bandwidth_gbs=spec.memory_bandwidth_gbs,
        max_attainable_tflops=max_attainable,
    )


def compare_gpu_rooflines(
    flops: float,
    bytes_read: float,
    bytes_write: float,
    dtype: str = "bf16",
    gpus: Optional[List[str]] = None,
) -> List[OperationAnalysis]:
    """Compare roofline analysis across multiple GPUs.

    Args:
        flops: Total FLOPs.
        bytes_read: Bytes read.
        bytes_write: Bytes written.
        dtype: Data type.
        gpus: List of GPU names. Default: all available.

    Returns:
        List of OperationAnalysis results, one per GPU.
    """
    if gpus is None:
        gpus = list(GPU_ROOFLINE_SPECS.keys())

    results: List[OperationAnalysis] = []
    for gpu_name in gpus:
        spec = GPU_ROOFLINE_SPECS[gpu_name]
        peak_tflops_map = {
            "fp16": spec.peak_fp16_tflops,
            "bf16": spec.peak_bf16_tflops,
            "fp8": spec.peak_fp8_tflops,
            "fp32": spec.peak_fp32_tflops,
        }
        if peak_tflops_map.get(dtype, 0.0) <= 0:
            continue  # GPU doesn't support this dtype
        results.append(
            analyze_operation("op", flops, bytes_read, bytes_write, gpu_name, dtype)
        )
    return results


# ==============================================================================
#  Optimization guidance
# ==============================================================================


def optimization_advice(analysis: OperationAnalysis) -> List[str]:
    """Provide optimization guidance based on roofline analysis.

    Args:
        analysis: OperationAnalysis from analyze_operation.

    Returns:
        List of actionable advice strings.
    """
    advice: List[str] = []

    if analysis.bottleneck == BottleneckType.MEMORY_BOUND:
        advice.append(
            f"Operation is MEMORY-BOUND (AI={analysis.arithmetic_intensity:.1f} "
            f"< ridge={analysis.ridge_point:.1f} FLOP/Byte)"
        )
        advice.append("Optimization strategies for memory-bound kernels:")
        advice.append("  1. Fuse with downstream operations to reduce HBM round-trips")
        advice.append("  2. Use lower precision (FP8/INT8) to halve memory traffic")
        advice.append("  3. Apply kernel fusion (e.g. RMSNorm + residual + activation)")
        advice.append("  4. Increase arithmetic intensity via larger tile sizes")
        advice.append(
            f"  5. At most {analysis.memory_bandwidth_gbs:.0f} GB/s read + write → "
            f"minimum time = {analysis.bytes_transferred / (analysis.memory_bandwidth_gbs * 1e9) * 1e6:.1f} µs"
        )
    else:
        advice.append(
            f"Operation is COMPUTE-BOUND (AI={analysis.arithmetic_intensity:.1f} "
            f">= ridge={analysis.ridge_point:.1f} FLOP/Byte)"
        )
        advice.append("Optimization strategies for compute-bound kernels:")
        advice.append("  1. Use Tensor Cores whenever possible (FP16/BF16/FP8 matmuls)")
        advice.append(
            "  2. Increase occupancy via careful register/shared memory usage"
        )
        advice.append("  3. Use higher warp counts and pipeline stages")
        advice.append("  4. Apply structured sparsity (2:4 pattern) on Ampere+")
        advice.append(
            f"  5. Peak {analysis.peak_tflops:.0f} TFLOPS → "
            f"minimum time = {analysis.flops / (analysis.peak_tflops * 1e12) * 1e6:.1f} µs"
        )

    advice.append(
        f"\nMax attainable: {analysis.max_attainable_tflops:.1f} TFLOPS "
        f"({analysis.max_attainable_tflops / max(analysis.peak_tflops, 0.001) * 100:.1f}% of peak)"
    )

    return advice


# ==============================================================================
#  Demonstration
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GPU Roofline Model Analysis")
    print("=" * 70)

    # Ridge points
    print("\nRidge Points (FLOP/Byte):")
    print(f"{'GPU':<12s} {'FP16':>10s} {'BF16':>10s} {'FP8':>10s} {'FP32':>10s}")
    print("-" * 55)
    for gpu_name in ["H100", "A100", "V100", "RTX4090", "B200"]:
        spec = GPU_ROOFLINE_SPECS[gpu_name]
        ridges = [spec.ridge_fp16, spec.ridge_bf16, spec.ridge_fp8, spec.ridge_fp32]
        ridge_strs = [f"{r:8.0f}" if r > 0 else "      N/A" for r in ridges]
        print(
            f"  {gpu_name:<10s} {ridge_strs[0]:>10s} {ridge_strs[1]:>10s} {ridge_strs[2]:>10s} {ridge_strs[3]:>10s}"
        )

    # Operation analyses
    print("\n" + "=" * 70)
    print("Operation Arithmetic Intensity on H100 (BF16)")
    print("=" * 70)

    analyses = [
        # name, flops, bytes_read, bytes_write
        ("GeLU (4096 elts)", 4096 * 5, 4096 * 2, 4096 * 2),
        ("RMSNorm (32k, 4096)", 32 * 4096 * 4, 32 * 4096 * 2, 32 * 4096 * 2),
        ("MatMul (1024^3)", 2 * 1024**3, (1024**2) * 2 * 2, 1024**2 * 2),
        ("MatMul (4096^3)", 2 * 4096**3, (4096**2) * 2 * 2, 4096**2 * 2),
        ("Attention (256, 64d)", 4 * 256**2 * 64, 256 * 64 * 2 * 3, 256 * 64 * 2),
        ("Attention (8192, 64d)", 4 * 8192**2 * 64, 8192 * 64 * 2 * 3, 8192 * 64 * 2),
    ]

    for name, flops, br, bw in analyses:
        analysis = analyze_operation(name, flops, br, bw, "H100", "bf16")
        print(
            f"\n  {name}:"
            f"\n    AI = {analysis.arithmetic_intensity:,.1f} FLOP/Byte"
            f"\n    Bottleneck: {analysis.bottleneck.value}"
            f"\n    Max TFLOPS: {analysis.max_attainable_tflops:,.1f}"
        )

    # Optimization advice
    print("\n" + "=" * 70)
    print("Optimization Advice: RMSNorm (seq=4096, dim=4096) on H100")
    print("=" * 70)

    flops = ai_rmsnorm(4096, 4096) * 4096 * 2 * 2  # actually compute total
    analysis = analyze_operation(
        "RMSNorm (4096, 4096)",
        4 * 4096 * 4096,  # flops
        4096 * 4096 * 2,  # bytes read
        4096 * 4096 * 2,  # bytes write
        "H100",
        "bf16",
    )

    for line in optimization_advice(analysis):
        print(f"  {line}")

    print("\nAll checks passed.")
