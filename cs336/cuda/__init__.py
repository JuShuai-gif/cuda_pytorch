"""
cs336.cuda - Production-grade CUDA/Triton kernel module.

Provides fused GPU kernels for neural network operations used in
large language model training and inference.

Submodules:
    kernels:     Fused activation and normalization kernels (GeLU, SiLU, RMSNorm, LayerNorm)
    flash_attention: FlashAttention v2 forward/backward pass
    matmul:      Tiled matrix multiplication with autotuning
    paged_attention: PagedAttention for efficient LLM inference
    quantization: FP8/INT8 quantization kernels for inference optimization
    profiler:    GPU profiling utilities (timing, bandwidth, occupancy)
    roofline:    Roofline model analysis for kernel optimization
"""

from __future__ import annotations

from cs336.cuda import (
    flash_attention,
    kernels,
    matmul,
    paged_attention,
    profiler,
    quantization,
    roofline,
)

# Fused activation and normalization kernels
from cs336.cuda.kernels import (
    fused_gelu,
    fused_layernorm,
    fused_linear_activation,
    fused_rms_norm,
    fused_silu_mul,
)

# FlashAttention
from cs336.cuda.flash_attention import (
    flash_attention_backward,
    flash_attention_forward,
)

# Matrix multiplication
from cs336.cuda.matmul import (
    batch_matmul,
    tiled_matmul,
)

# Paged attention
from cs336.cuda.paged_attention import (
    PagedAttentionManager,
    block_table_lookup,
    create_block_table,
    paged_attention_kernel,
)

# Quantization
from cs336.cuda.quantization import (
    fp8_dequantize,
    fp8_quantize,
    int8_dequantize,
    int8_quantize,
)

# Profiling
from cs336.cuda.profiler import (
    CUDAEventTimer,
    KernelProfiler,
    OccupancyReport,
    TimingResult,
    benchmark_kernel,
    calculate_occupancy,
    compute_memory_bandwidth,
    estimate_bandwidth_utilization,
    find_optimal_block_size,
)

# Roofline analysis
from cs336.cuda.roofline import (
    BottleneckType,
    OperationAnalysis,
    ai_attention,
    ai_attention_standard,
    ai_elementwise,
    ai_layernorm,
    ai_matmul,
    ai_rmsnorm,
    analyze_operation,
    compare_gpu_rooflines,
    compute_arithmetic_intensity,
    compute_ridge_point,
    estimate_gpu_utilization,
    identify_bottleneck,
    optimization_advice,
)

HAS_TRITON_KERNELS = True
try:
    import triton  # type: ignore[import-untyped]
except ImportError:
    HAS_TRITON_KERNELS = False

__all__ = [
    # Submodules
    "flash_attention",
    "kernels",
    "matmul",
    "paged_attention",
    "profiler",
    "quantization",
    "roofline",
    # Kernels
    "batch_matmul",
    "fused_gelu",
    "fused_layernorm",
    "fused_linear_activation",
    "fused_rms_norm",
    "fused_silu_mul",
    "tiled_matmul",
    # FlashAttention
    "flash_attention_backward",
    "flash_attention_forward",
    # PagedAttention
    "PagedAttentionManager",
    "block_table_lookup",
    "create_block_table",
    "paged_attention_kernel",
    # Quantization
    "fp8_dequantize",
    "fp8_quantize",
    "int8_dequantize",
    "int8_quantize",
    # Profiling
    "CUDAEventTimer",
    "KernelProfiler",
    "OccupancyReport",
    "TimingResult",
    "benchmark_kernel",
    "calculate_occupancy",
    "compute_memory_bandwidth",
    "estimate_bandwidth_utilization",
    "find_optimal_block_size",
    # Roofline
    "BottleneckType",
    "OperationAnalysis",
    "ai_attention",
    "ai_attention_standard",
    "ai_elementwise",
    "ai_layernorm",
    "ai_matmul",
    "ai_rmsnorm",
    "analyze_operation",
    "compare_gpu_rooflines",
    "compute_arithmetic_intensity",
    "compute_ridge_point",
    "estimate_gpu_utilization",
    "identify_bottleneck",
    "optimization_advice",
]
