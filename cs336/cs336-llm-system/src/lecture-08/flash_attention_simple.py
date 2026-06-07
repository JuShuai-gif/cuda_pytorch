"""
FlashAttention 概念的简化实现。
演示核心思想：分块计算 softmax，
通过在线重缩放避免物化完整的 N x N 注意力矩阵。

这是一个教学实现，使用 PyTorch 在 tile 上的循环，
并非真正的 FlashAttention 算法（它使用 SRAM 感知的 CUDA kernel）。
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F


# =========================================================================
# 朴素 SDPA（参照实现）
# =========================================================================


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    朴素注意力：先计算完整的 Q @ K^T，再做 softmax，最后 @ V。

    内存占用：注意力分数矩阵为 O(N^2)。
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # [B, H, N, N]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


# =========================================================================
# Flash Attention - 在线 Softmax 重缩放
# =========================================================================


def flash_attention_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    causal: bool = False,
) -> torch.Tensor:
    """
    使用在线 softmax 重缩放的分块（tiled）注意力。

    FlashAttention 的核心思想：
    1. 沿序列维度将 Q 分割成 block。
    2. 对每个 Q 块，遍历 K/V 块。
    3. 对每个 block 计算部分 softmax。
    4. 使用在线重缩放组合结果，无需存储完整的 N x N 注意力矩阵。

    这样避免了在 HBM 中物化完整注意力分数矩阵，
    将峰值内存从 O(N^2) 降至 O(N * block_size)。

    Args:
        q: Query， shape 为 (batch, num_heads, seq_len, head_dim)
        k: Key，   shape 为 (batch, num_heads, seq_len, head_dim)
        v: Value， shape 为 (batch, num_heads, seq_len, head_dim)
        block_size: 内层循环的 tile 大小
        causal: 是否应用因果 mask

    Returns:
        与 Q 相同 shape 的输出
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # 输出缓冲区
    output = torch.zeros_like(q)

    # 沿序列维度将 Q 分割成 block
    num_q_blocks = (seq_len + block_size - 1) // block_size

    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start + block_size, seq_len)
        q_block = q[:, :, q_start:q_end, :]  # (B, H, Bq, D)
        q_block_len = q_end - q_start

        # 此 Q 块的在线 softmax 统计量
        # m：运行中的最大值（保证数值稳定性）
        # l：运行中的 exp(scores - m) 之和
        m = torch.full(
            (batch, num_heads, q_block_len, 1),
            float("-inf"),
            device=q.device,
            dtype=q.dtype,
        )
        l = torch.zeros(
            batch, num_heads, q_block_len, 1, device=q.device, dtype=q.dtype
        )
        o = torch.zeros(
            batch, num_heads, q_block_len, head_dim, device=q.device, dtype=q.dtype
        )

        # 遍历 K/V 块
        num_kv_blocks = (seq_len + block_size - 1) // block_size
        for kv_start in range(0, seq_len, block_size):
            kv_end = min(kv_start + block_size, seq_len)
            k_block = k[:, :, kv_start:kv_end, :]  # (B, H, Bk, D)
            v_block = v[:, :, kv_start:kv_end, :]  # (B, H, Bk, D)

            # Causal：Q 只能关注位置 <= Q 位置的 K
            if causal:
                # 若所有 Q 位置都在所有 K 位置之前，跳过
                if q_start >= kv_end:
                    continue
                # 若所有 K 位置都在此 Q 块最后一个位置之后，跳过
                # （仍需为重叠的部分计算部分结果）
                # 因果 mask 在下面按元素应用，通过 S_ij 实现

            # 计算此 tile 的分数：(B, H, Bq, Bk)
            s_ij = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            # 在 tile 内应用因果 mask
            if causal:
                # 对每个元素 (i, j)，若 j > i 则 mask
                # q_start + local_q_idx 是全局 query 索引
                # kv_start + local_kv_idx 是全局 key 索引
                q_indices = torch.arange(q_start, q_end, device=q.device).view(
                    1, 1, -1, 1
                )
                kv_indices = torch.arange(kv_start, kv_end, device=q.device).view(
                    1, 1, 1, -1
                )
                causal_mask = kv_indices <= q_indices
                s_ij = s_ij.masked_fill(~causal_mask, float("-inf"))

            # 在线 softmax 更新（FlashAttention 论文中的算法 1）
            # m_new = max(m, rowmax(s_ij))
            # l_new = exp(m - m_new) * l + sum(exp(s_ij - m_new))
            # o_new = exp(m - m_new) * o + exp(s_ij - m_new) @ V

            m_ij = torch.max(s_ij, dim=-1, keepdim=True).values  # (B, H, Bq, 1)
            m_new = torch.maximum(m, m_ij)

            # 旧累加器的重缩放因子
            exp_m_diff = torch.exp(m - m_new)
            l = exp_m_diff * l + torch.sum(
                torch.exp(s_ij - m_new), dim=-1, keepdim=True
            )

            # 更新输出
            p_ij = torch.exp(s_ij - m_new)  # (B, H, Bq, Bk)
            o = exp_m_diff * o + torch.matmul(p_ij, v_block)

            m = m_new

        # 此 Q 块的最终输出：O / L
        output[:, :, q_start:q_end, :] = o / l

    return output


# =========================================================================
# 原理说明与分析
# =========================================================================


def explain_flash_attention() -> None:
    """解释 FlashAttention 为何能减少 HBM 访问。"""
    print("=" * 70)
    print("FlashAttention: Why Tiling Reduces HBM Access")
    print("=" * 70)
    print("""
    Standard Attention Memory Access Pattern:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Read Q, K from HBM                                           │
    │ 2. Compute S = Q @ K^T → write S (N x N) to HBM                 │
    │ 3. Read S from HBM → compute P = softmax(S) → write P to HBM    │
    │ 4. Read P, V from HBM → compute O = P @ V → write O to HBM      │
    └─────────────────────────────────────────────────────────────────┘

    HBM Reads:  Q (Nd) + K (Nd) + S (N^2) + P (N^2) + V (Nd)
    HBM Writes: S (N^2) + P (N^2) + O (Nd)
    Total HBM R/W: ≈ 4Nd + 2N^2  → O(N^2) dominated

    FlashAttention Memory Access Pattern:
    ┌─────────────────────────────────────────────────────────────────┐
    │ For each Q block (Bq x d):                                      │
    │   Load Q_block into SRAM                                        │
    │   Initialize O_block, m, l in SRAM                              │
    │   For each KV block (Bk x d):                                   │
    │     Load K_block, V_block into SRAM                             │
    │     Compute S_block = Q_block @ K_block^T (in SRAM)             │
    │     Compute P_block = softmax_update(S_block, m, l) (in SRAM)   │
    │     Update O_block += P_block @ V_block (in SRAM)               │
    │     Update m, l (in SRAM)                                       │
    │   Write O_block to HBM                                          │
    └─────────────────────────────────────────────────────────────────┘

    HBM Reads:  Q (Nd) + K (Nd) + V (Nd)  (KV loaded T_r times)
              = Nd + T_r * (2 * Bk * d) per head
    HBM Writes: O (Nd)

    With T_r = ceil(N / Bk), Bk = block_size:
      HBM R/W ≈ O(Nd * N/Bk) ← linear in N, quadratic would be O(N^2)

    Key Insight:
    - The N x N attention matrix NEVER leaves SRAM
    - Online softmax rescaling eliminates the need to store S and P
    - Memory complexity: O(N * block_size) vs O(N^2)
    """)

    # 数值示例
    print("\nMemory Comparison (seq_len=4096, head_dim=64, bf16):")
    seq_len = 4096
    head_dim = 64
    bytes_per_elem = 2  # bf16

    naive_mem = seq_len * seq_len * bytes_per_elem
    flash_mem = seq_len * 64 * bytes_per_elem  # block_size=64

    print(f"  Naive (S matrix): {naive_mem / 1e6:.1f} MB")
    print(f"  Flash (tiled):    {flash_mem / 1e6:.1f} MB")
    print(f"  Reduction:        {naive_mem / flash_mem:.0f}x")

    print("\nNumerical Stability (Online Softmax):")
    print("  Standard softmax: p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))")
    print("  Online softmax:   maintain running m and l, rescale on the fly")
    print("  Both are mathematically equivalent (within floating point)")
    print("  Online version requires 2 passes but no large intermediate storage")


# =========================================================================
# 正确性验证
# =========================================================================


def check_correctness() -> None:
    """验证分块注意力与朴素注意力的结果是否一致。"""
    print("\n" + "=" * 70)
    print("Correctness Check: Tiled vs Naive Attention")
    print("=" * 70)

    torch.manual_seed(42)
    batch, num_heads, seq_len, head_dim = 1, 4, 128, 64

    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    v = torch.randn(batch, num_heads, seq_len, head_dim)

    # 朴素实现
    naive_out = naive_attention(q, k, v, causal=True)

    # 分块实现
    tiled_out = flash_attention_tiled(q, k, v, block_size=32, causal=True)

    # 对比
    max_diff = (naive_out - tiled_out).abs().max().item()
    mean_diff = (naive_out - tiled_out).abs().mean().item()

    print(f"  Max absolute difference:  {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(
        f"  Allclose (rtol=1e-3):     {torch.allclose(naive_out, tiled_out, rtol=1e-3, atol=1e-4)}"
    )
    print(
        f"  Allclose (rtol=1e-2):     {torch.allclose(naive_out, tiled_out, rtol=1e-2, atol=1e-3)}"
    )

    if max_diff < 1e-2:
        print("\n  ✓ Tiled attention matches naive attention within tolerance.")
    else:
        print("\n  ✗ Significant deviation. Check numerical precision.")


# =========================================================================
# 性能基准测试
# =========================================================================


def benchmark() -> None:
    """对比朴素与分块注意力的速度。"""
    print("\n" + "=" * 70)
    print("Speed Comparison (may not reflect real FlashAttention performance)")
    print("=" * 70)

    configs = [
        (1, 4, 256, 64),
        (1, 4, 512, 64),
        (1, 4, 1024, 64),
        (1, 4, 2048, 64),
    ]

    for batch, num_heads, seq_len, head_dim in configs:
        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)

        # 预热
        for _ in range(3):
            _ = naive_attention(q, k, v)
            _ = flash_attention_tiled(q, k, v, block_size=64)

        # 计时：朴素实现
        start = time.perf_counter()
        for _ in range(5):
            _ = naive_attention(q, k, v)
        naive_t = (time.perf_counter() - start) / 5

        # 计时：分块实现
        start = time.perf_counter()
        for _ in range(5):
            _ = flash_attention_tiled(q, k, v, block_size=64)
        tiled_t = (time.perf_counter() - start) / 5

        print(
            f"\n  seq_len={seq_len:<5} | Naive: {naive_t * 1000:.2f}ms | Tiled: {tiled_t * 1000:.2f}ms | Ratio: {naive_t / tiled_t:.2f}x"
        )

    print("\n  Note: This Python tiled implementation is slower than naive due to")
    print("  Python loop overhead. The real FlashAttention CUDA kernel is 2-4x faster")
    print("  than standard attention for long sequences because it avoids HBM R/W.")
    print("  This implementation only demonstrates the algorithm, not its performance.")


def main() -> None:
    explain_flash_attention()
    check_correctness()
    benchmark()


if __name__ == "__main__":
    main()
