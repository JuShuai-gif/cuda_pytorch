"""
LLM 推理的算术强度分析。

算术强度（Arithmetic Intensity）= FLOPS / 传输字节数（FLOPS/Byte）

这个比值决定了计算属于哪一种瓶颈：
  - 计算瓶颈（Compute-bound）：  受限于处理速度（算术强度高）
  - 内存瓶颈（Memory-bound）：   受限于数据传输速度（算术强度低）

Prefill 和 Decode 阶段的算术强度有显著差异：
  - Prefill：一次性处理完整 prompt → 算术强度高
  - Decode：   每次只处理一个 token → 算术强度非常低

理解这一点对优化推理至关重要：
  - Prefill 受益于高计算吞吐量（更多 FLOPS）
  - Decode 受益于高内存带宽（更快的 KV cache 读取）
"""

from __future__ import annotations

import math


# =========================================================================
# 算术强度计算
# =========================================================================


def attention_flops(seq_len_q: int, seq_len_k: int, head_dim: int) -> int:
    """
    计算单个注意力头的 FLOPS。

    Q @ K^T:  seq_len_q × seq_len_k × head_dim × 2（乘加）
    P @ V:    seq_len_q × head_dim × seq_len_k × 2
    总计:     4 × seq_len_q × seq_len_k × head_dim
    """
    return 4 * seq_len_q * seq_len_k * head_dim


def attention_bytes(
    seq_len_q: int, seq_len_k: int, head_dim: int, dtype_bytes: int = 2
) -> int:
    """
    估算单个注意力头的传输字节数。

    读取：Q (seq_len_q × head_dim)、K (seq_len_k × head_dim)、V (seq_len_k × head_dim)
    写入：O (seq_len_q × head_dim)
    无 KV cache 时：还需要读写 N×N 的注意力矩阵。

    有 KV cache 时（decode）：Q 为 1 个 token，K/V 为完整缓存的序列。
    """
    reads = (seq_len_q + 2 * seq_len_k) * head_dim * dtype_bytes
    writes = seq_len_q * head_dim * dtype_bytes
    return reads + writes


def attention_arithmetic_intensity(
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> float:
    """计算注意力的算术强度。"""
    flops = attention_flops(seq_len_q, seq_len_k, head_dim)
    bytes_val = attention_bytes(seq_len_q, seq_len_k, head_dim, dtype_bytes)
    return flops / max(bytes_val, 1)


def ffn_flops(seq_len: int, hidden_size: int, intermediate_size: int) -> int:
    """
    计算 FFN 层的 FLOPS。

    标准 FFN：x → FC1 → activation → FC2
    FLOPS ≈ 2 × seq_len × hidden_size × intermediate_size × 2（两次矩阵乘法）
    对于 SwiGLU：3 次矩阵乘法
    """
    return 2 * seq_len * hidden_size * intermediate_size * 2


def ffn_bytes(
    seq_len: int, hidden_size: int, intermediate_size: int, dtype_bytes: int = 2
) -> int:
    """估算 FFN 的传输字节数。"""
    # 读取输入 (seq×hidden)、权重 (hidden×intermed)、中间激活值、输出
    reads = seq_len * hidden_size * dtype_bytes * 2  # 输入 + 输出
    reads += hidden_size * intermediate_size * dtype_bytes * 2  # 两个权重矩阵
    writes = seq_len * hidden_size * dtype_bytes
    return reads + writes


# =========================================================================
# 硬件参数（示例）
# =========================================================================

# NVIDIA A100（80GB）
A100_FP16_TFLOPS = 312  # TFLOPS（Tensor Core）
A100_MEM_BW_GBPS = 2039  # GB/s（HBM2e）

# NVIDIA H100
H100_FP16_TFLOPS = 990  # TFLOPS
H100_MEM_BW_GBPS = 3350  # GB/s（HBM3）

# NVIDIA RTX 4090
RTX4090_FP16_TFLOPS = 82.6  # TFLOPS（Tensor Core）
RTX4090_MEM_BW_GBPS = 1008  # GB/s


def compute_bound_threshold(tflops: float, mem_bw_gbps: float) -> float:
    """
    计算算术强度的阈值，超过该阈值后，
    给定硬件上的操作将变为计算瓶颈。

    threshold = compute_throughput / memory_bandwidth
    （两者使用相同单位：FLOPS 和 bytes/s）
    """
    return (tflops * 1e12) / (mem_bw_gbps * 1e9)


# =========================================================================
# 分析
# =========================================================================


def analyze_prefill(
    seq_len: int,
    head_dim: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
) -> None:
    """分析 prefill 阶段的算术强度。"""
    # Prefill: Q 和 K/V 的序列长度相同
    attn_ai = attention_arithmetic_intensity(seq_len, seq_len, head_dim)
    attn_flops_total = (
        attention_flops(seq_len, seq_len, head_dim) * num_heads * num_layers
    )
    ffn_flops_total = ffn_flops(seq_len, hidden_size, intermediate_size) * num_layers

    total_flops = attn_flops_total + ffn_flops_total

    # 估算总字节数
    kv_cache_bytes = 2 * num_layers * seq_len * num_heads * head_dim * 2  # fp16
    weight_bytes = (
        _estimate_model_weights(hidden_size, intermediate_size, num_layers, num_heads)
        * 2
    )
    total_bytes = kv_cache_bytes + weight_bytes
    overall_ai = total_flops / max(total_bytes, 1)

    return attn_ai, overall_ai, total_flops, total_bytes


def analyze_decode(
    seq_len: int,
    head_dim: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
) -> None:
    """分析 decode 阶段的算术强度。"""
    # Decode: Q 有 1 个 token，K/V 的序列长度为完整历史
    attn_ai = attention_arithmetic_intensity(1, seq_len, head_dim)
    attn_flops_total = attention_flops(1, seq_len, head_dim) * num_heads * num_layers
    ffn_flops_total = ffn_flops(1, hidden_size, intermediate_size) * num_layers

    total_flops = attn_flops_total + ffn_flops_total

    # Decode 阶段从内存中读取完整的 KV cache
    kv_cache_bytes = 2 * num_layers * seq_len * num_heads * head_dim * 2  # fp16
    weight_bytes = (
        _estimate_model_weights(hidden_size, intermediate_size, num_layers, num_heads)
        * 2
    )
    # 在 decode 阶段，每个新 token 都需要从 HBM 读取 KV cache
    total_bytes = kv_cache_bytes + weight_bytes * 0.01  # 权重大多缓存在 L2 中
    overall_ai = total_flops / max(total_bytes, 1) if total_bytes > 0 else 0

    return attn_ai, overall_ai, total_flops, total_bytes


def _estimate_model_weights(
    hidden_size: int, intermediate_size: int, num_layers: int, num_heads: int
) -> int:
    """估算模型总权重参数量。"""
    # 每层的粗略估算
    # 注意力：4 个投影（Q、K、V、O）= 4 × hidden²
    # FFN：2 个投影（SwiGLU 时为 3 个）= 2 × hidden × intermediate
    attn_params = 4 * hidden_size * hidden_size
    ffn_params = 2 * hidden_size * intermediate_size
    return (attn_params + ffn_params) * num_layers


def main() -> None:
    print("=" * 70)
    print("Arithmetic Intensity Analysis for LLM Inference")
    print("=" * 70)

    # 硬件计算瓶颈阈值
    print("\nHardware Compute-Bound Thresholds:")
    print(
        f"  A100: {compute_bound_threshold(A100_FP16_TFLOPS, A100_MEM_BW_GBPS):.1f} FLOPS/Byte"
    )
    print(
        f"  H100: {compute_bound_threshold(H100_FP16_TFLOPS, H100_MEM_BW_GBPS):.1f} FLOPS/Byte"
    )
    print(
        f"  4090: {compute_bound_threshold(RTX4090_FP16_TFLOPS, RTX4090_MEM_BW_GBPS):.1f} FLOPS/Byte"
    )

    # 模型配置（按 LLaMA-2 7B 规模，简化版）
    hidden_size = 4096
    intermediate_size = 11008
    num_layers = 32
    num_heads = 32
    head_dim = hidden_size // num_heads  # 128

    print(
        f"\nModel Config: {num_layers} layers, hidden={hidden_size}, intermediate={intermediate_size}, {num_heads} heads"
    )
    print(
        f"{'Phase':<12} {'Seq Len':<10} {'Attn AI':<12} {'Overall AI':<14} {'Total FLOPS':<16} {'Total Bytes':<16} {'Bound By':<14}"
    )
    print("-" * 100)

    for phase_name, analyze_fn in [
        ("Prefill", analyze_prefill),
        ("Decode", analyze_decode),
    ]:
        for seq_len in [128, 256, 512, 1024, 2048, 4096, 8192]:
            attn_ai, overall_ai, total_flops, total_bytes = analyze_fn(
                seq_len,
                head_dim,
                hidden_size,
                intermediate_size,
                num_layers,
                num_heads,
            )
            # 判断是计算瓶颈还是内存瓶颈
            threshold = compute_bound_threshold(A100_FP16_TFLOPS, A100_MEM_BW_GBPS)
            if overall_ai > threshold:
                bound = "COMPUTE"
            else:
                bound = "MEMORY"

            print(
                f"{phase_name:<12} {seq_len:<10} {attn_ai:>8.1f}    "
                f"{overall_ai:>10.2f}    {total_flops / 1e9:>10.2f} G  "
                f"{total_bytes / 1e9:>10.2f} GB  {bound:<14}"
            )

    # 详细分解
    print("\n" + "=" * 70)
    print("详细分解：为什么 Decode 阶段是内存瓶颈")
    print("=" * 70)
    print("""
    Prefill 阶段（处理完整 prompt）：
      - Q 和 K 的长度都等于 prompt_length（例如 512 个 token）
      - Q @ K^T 产生一个 (512, 512) 的注意力矩阵
      - 该矩阵乘法具有较高的算术强度，
        因为 FLOPS = O(N² × d) 而 Bytes = O(N × d)
      - AI ≈ O(N) → 对于较长序列，算术强度高
      - 结论：对于合理的序列长度，Prefill 是计算瓶颈（COMPUTE-BOUND）

    Decode 阶段（逐 token 生成）：
      - Q 的长度为 1，K 的长度 = full_history（例如 4096 个 token）
      - Q @ K^T 产生一个 (1, 4096) 向量 → FLOPS 非常少
      - 但必须读取整个 K cache（4096 × d 个元素）
      - AI ≈ O(d) ≈ 常数，非常小
      - 算力：每个 KV 元素约 2 × head_dim FLOPS
      - 内存：每个 KV 元素约 2 字节（从 HBM 加载）
      - AI ≈ 2-4 FLOPS/Byte → 远低于硬件阈值
      - 结论：Decode 是内存瓶颈（MEMORY-BOUND）

    推论：
      - Prefill 延迟 ≈ 模型 FLOPS / GPU TFLOPS
      - Decode 延迟  ≈ KV cache 大小 / GPU 内存带宽

    实际影响：
      - 加速 Prefill：需要更强的计算能力（更高 TFLOPS 的 GPU）
      - 加速 Decode：需要更高的内存带宽（HBM3）
      - 这就是为什么量化（int8/int4 KV cache）对 decode 帮助如此之大：
        它减少了内存带宽瓶颈
      - 这也是为什么 batch-size=1 的 decode 如此缓慢：
        你为仅仅一个 token 付出了读取完整 KV cache 的代价
    """)


if __name__ == "__main__":
    main()
