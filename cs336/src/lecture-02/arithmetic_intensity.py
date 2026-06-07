"""
第 02 讲 — 资源核算：算术强度与 Roofline 模型。

针对常见深度学习原语计算算术强度（FLOPs / 内存传输字节数），
并提供一个简单的 roofline 模型分析器。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Tuple


# ---------------------------------------------------------------------------
# 算术强度辅助函数
# ---------------------------------------------------------------------------


def _bytes_per_elem(dtype_str: str) -> int:
    """根据 dtype 字符串（如 'fp32' / 'bf16'）返回元素字节大小。"""
    mapping = {
        "fp32": 4,
        "tf32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp8": 1,
        "int8": 1,
        "int4": 1,  # 压缩存储，粗略近似
    }
    return mapping.get(dtype_str.lower(), 4)


# ---------------------------------------------------------------------------
# 矩阵乘法
# ---------------------------------------------------------------------------


def matmul_arithmetic_intensity(
    M: int,
    N: int,
    K: int,
    dtype_a: str = "bf16",
    dtype_b: str = "bf16",
    dtype_c: str = "fp32",
) -> float:
    """计算 GEMM C[M,N] = A[M,K] × B[K,N] 的算术强度。

    FLOPs  = 2 * M * N * K（融合乘加算作两次运算）
    内存   = read(A) + read(B) + write(C)
    """
    flops = 2.0 * M * N * K
    bytes_read = M * K * _bytes_per_elem(dtype_a) + K * N * _bytes_per_elem(dtype_b)
    bytes_write = M * N * _bytes_per_elem(dtype_c)
    return flops / max(bytes_read + bytes_write, 1)


# ---------------------------------------------------------------------------
# Attention（缩放点积）
# ---------------------------------------------------------------------------


def attention_arithmetic_intensity(
    batch: int,
    heads: int,
    seq_len: int,
    d_head: int,
    dtype: str = "bf16",
) -> dict[str, float]:
    """多头自注意力的算术强度分解。

    计算三个阶段：
        1. QK^T → scores  [B, H, S, S]
        2. Softmax → attn  [B, H, S, S]
        3. AV     → output [B, H, S, d]
    """
    bw = _bytes_per_elem(dtype)
    S, d = seq_len, d_head

    # 阶段 1：QK^T  [B,H,S,d] × [B,H,S,d]^T
    flops_qk = 2.0 * batch * heads * S * S * d
    bytes_qk = batch * heads * (2 * S * d * bw + S * S * bw)  # Q, K 读取 + scores 写入
    ai_qk = flops_qk / max(bytes_qk, 1)

    # 阶段 2：Softmax — 每个元素约 5 次 FLOPs（exp、加法等）
    flops_sm = 5.0 * batch * heads * S * S
    bytes_sm = 2 * batch * heads * S * S * bw  # 读取 scores，写入 attn
    ai_sm = flops_sm / max(bytes_sm, 1)

    # 阶段 3：AV  [B,H,S,S] × [B,H,S,d]
    flops_av = 2.0 * batch * heads * S * S * d
    bytes_av = (
        batch * heads * (S * S * bw + S * d * bw + S * d * bw)
    )  # attn, V 读取 + O 写入
    ai_av = flops_av / max(bytes_av, 1)

    total_flops = flops_qk + flops_sm + flops_av
    total_bytes = bytes_qk + bytes_sm + bytes_av
    return {
        "ai_qk": ai_qk,
        "ai_softmax": ai_sm,
        "ai_av": ai_av,
        "ai_overall": total_flops / max(total_bytes, 1),
    }


# ---------------------------------------------------------------------------
# 逐元素操作
# ---------------------------------------------------------------------------


def elementwise_arithmetic_intensity(
    num_elements: int = 1 << 20,
    flops_per_element: float = 1.0,
    input_bytes_per_elem: int = 4,
    output_bytes_per_elem: int = 4,
) -> float:
    """逐元素操作（加法、gelu 等）的算术强度。

    读取 N 个元素并写入 N 个元素 → FLOPs 极少，因此 AI 通常
    << 1.0 — 严格受内存带宽限制。
    """
    flops = flops_per_element * num_elements
    bytes_moved = num_elements * (input_bytes_per_elem + output_bytes_per_elem)
    return flops / max(bytes_moved, 1)


# ---------------------------------------------------------------------------
# Roofline 模型
# ---------------------------------------------------------------------------


@dataclass
class GPUPeak:
    """GPU 的峰值能力（以 TFLOPS 和 GB/s 表示）。"""

    name: str
    peak_tflops: float  # 理论峰值（如 fp16 tensor-core）
    peak_bw_gbs: float  # HBM 带宽


def roofline_performance(
    ai: float,
    gpu: GPUPeak,
) -> Tuple[float, str]:
    """返回可达到的 TFLOPS 值以及限制资源。

    若 AI >= ridge_point → 计算受限（可达峰值 TFLOPS）
    否则               → 内存受限（可达 AI * bandwidth）
    """
    ridge_point = gpu.peak_tflops * 1e3 / gpu.peak_bw_gbs  # FLOP / Byte
    if ai >= ridge_point:
        return gpu.peak_tflops, "compute"
    attainable = ai * gpu.peak_bw_gbs / 1e3  # TFLOPS
    return attainable, "memory"


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


def _demo_rooflines() -> list[GPUPeak]:
    return [
        GPUPeak("V100 (fp16 TC)", 112.0, 900.0),
        GPUPeak("A100 (bf16 TC)", 312.0, 2039.0),
        GPUPeak("H100 (fp8 TC)", 1979.0, 3350.0),
    ]


if __name__ == "__main__":
    # Matmul 算术强度
    ai_mm = matmul_arithmetic_intensity(2048, 2048, 2048)
    print(f"Matmul 2048³ AI   = {ai_mm:.1f} FLOP/Byte")

    # Attention 算术强度
    attn_ai = attention_arithmetic_intensity(1, 32, 2048, 128)
    for k, v in attn_ai.items():
        print(f"  {k}: {v:.1f} FLOP/Byte")

    # 逐元素操作
    ew_ai = elementwise_arithmetic_intensity(1 << 20, flops_per_element=5)
    print(f"GeLU-like elem-wise AI = {ew_ai:.4f} FLOP/Byte")

    # Roofline 分析
    print("\nRoofline analysis:")
    for gpu in _demo_rooflines():
        for label, ai_val in [("Matmul", ai_mm), ("Element-wise", ew_ai)]:
            perf, bound = roofline_performance(ai_val, gpu)
            print(f"  {gpu.name:20s} | {label:12s} → {perf:6.1f} TFLOPS ({bound})")

    print("\nAll checks passed.")
