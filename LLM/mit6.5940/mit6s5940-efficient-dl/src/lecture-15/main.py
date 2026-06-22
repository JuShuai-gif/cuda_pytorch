"""
第 15 讲：长上下文注意力优化
=================================================
实现并比较以下方法：
  (1) 完整缩放点积注意力               -- O(n^2) 内存
  (2) 滑动窗口注意力                   -- O(n * w) 内存
  (3) 带窗口缓存的流式注意力           -- 分块滑动窗口
  (4) 带 NTK 感知缩放的 RoPE           -- 通过频率缩放扩展上下文
  (5) KV 缓存淘汰策略                   -- 保留首尾 token

所有实现在 CPU 上运行。依赖：torch、numpy、math（标准库）。
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 辅助工具函数
# ---------------------------------------------------------------------------


def _print_memory(title: str, matrix_shape: Tuple[int, ...], elements: int) -> None:
    """格式化打印单行内存占用信息。"""
    print(f"  {title:40s} shape={str(matrix_shape):24s}  elements={elements:>12,d}")


def _divider(char: str = "=", width: int = 100) -> None:
    """打印分隔线。"""
    print(char * width)


# ===========================================================================
# 1. 完整缩放点积注意力  -  O(n^2) 内存
# ===========================================================================


def full_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> Tuple[torch.Tensor, int]:
    """
    标准缩放点积注意力。
    返回 (输出, 注意力矩阵中的元素数量)。
    """
    d_k = q.size(-1)
    # 分数矩阵：(batch, heads, seq_q, seq_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    elements = scores.numel()  # O(seq^2) 个元素

    if mask is not None:
        # 将掩码为 0 的位置设为 -inf，使 softmax 输出为零
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, elements


# ===========================================================================
# 2. 滑动窗口注意力  -  O(n * w) 内存
# ===========================================================================


def sliding_window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, int]:
    """
    每个查询位置 i 只能关注键 [max(0, i-window_size+1) .. i]。

    我们通过一个因果掩码来实现，该掩码进一步被窗口限制，
    因此有效的注意力矩阵最多有 n * window_size 个
    未被掩蔽的条目。返回 (输出, 未掩蔽元素数量)。
    """
    seq_len = q.size(2)
    device = q.device

    # 因果掩码：行 = 查询，列 = 键；1 = 允许关注，0 = 掩蔽
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device))

    # 窗口掩码：仅允许在 `window_size` 距离内的位置
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    window_mask = (col_idx >= row_idx - window_size + 1).int()

    mask = causal * window_mask  # 取交集
    elements = mask.sum().item()  # 未被掩蔽（可关注）的位置数

    # 应用掩码：将不允许关注的位置设为 -inf
    attn_mask = (
        mask.float().masked_fill(mask == 0, float("-inf")).unsqueeze(0).unsqueeze(0)
    )

    return full_attention(q, k, v, mask=1 - mask.unsqueeze(0).unsqueeze(0))[0], int(
        elements
    )


# ===========================================================================
# 3. 流式注意力（分块滑动窗口）
# ===========================================================================


def streaming_attention(
    q_full: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    chunk_size: int = 128,
    window_size: int = 256,
) -> Tuple[torch.Tensor, int]:
    """
    按*块*处理序列。对于每个块，我们维护一个 KV 缓存，其大小
    受 `window_size` 个 token 限制。
    每个块的注意力矩阵为 (chunk_size) x (cache_size)，
    因此总内存大致为 n_chunks * chunk_size * window_size。

    这展示了相对于序列长度的恒定每步内存。返回 (拼接输出, 所有块中的总元素数)。
    """
    batch, heads, seq_len, d_k = q_full.shape
    device = q_full.device

    # 我们建模一个*因果*流式场景：位置 t 只能向后
    # 查看最多 `window_size` 个 token。
    out_chunks: list[torch.Tensor] = []
    total_elements = 0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        # 确定缓存窗口 [cache_start, end)
        cache_start = max(0, end - window_size)

        q_chunk = q_full[:, :, start:end, :]  # (B, H, chunk, d)
        k_cache = k_full[:, :, cache_start:end, :]  # (B, H, cache, d)
        v_cache = v_full[:, :, cache_start:end, :]

        # 注意力分数：Q_chunk @ K_cache^T / sqrt(d_k)
        scores = torch.matmul(q_chunk, k_cache.transpose(-2, -1)) / math.sqrt(d_k)
        total_elements += scores.numel()

        attn_weights = F.softmax(scores, dim=-1)
        out_chunks.append(torch.matmul(attn_weights, v_cache))

    # 将各块的输出沿序列维度拼接
    output = torch.cat(out_chunks, dim=2)
    return output, total_elements


# ===========================================================================
# 4. 旋转位置嵌入（RoPE）与 NTK 感知缩放
# ===========================================================================


def _compute_rope_frequencies(dim: int, base: float = 10000.0) -> torch.Tensor:
    """返回 RoPE 的逆频率向量（形状：dim//2）。"""
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    theta = base ** (-i / dim)  # θ_i = base^(-2i/d)
    return theta  # 形状 (dim//2,)


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0
) -> torch.Tensor:
    """
    对张量 x 应用旋转位置嵌入。

    x:  (..., seq_len, dim)   -- 通常是 query 或 key
    positions: (seq_len,)         -- 绝对位置索引
    """
    *prefix, seq_len, dim = x.shape
    assert dim % 2 == 0, "RoPE 要求偶数维度。"

    theta = _compute_rope_frequencies(dim, base)  # (dim//2,)

    # 为每个 (position, dimension-pair) 计算 cos/sin
    # 形状：(1, 1, seq_len, dim//2)  -- 可在 batch & heads 上广播
    pos = positions.float().unsqueeze(-1)  # (seq_len, 1)
    freqs = pos * theta.unsqueeze(0)  # (seq_len, dim//2)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)

    # 将 x 重塑为成对格式以进行旋转
    x_pairs = x.reshape(*prefix, seq_len, dim // 2, 2)  # 最后一维 = (real, imag)

    # 应用 2D 旋转：将每对 (a, b) 通过角度 θ 旋转
    # x' = a*cos(θ) - b*sin(θ)
    # y' = a*sin(θ) + b*cos(θ)
    x_out = torch.empty_like(x_pairs)
    x_out[..., 0] = x_pairs[..., 0] * cos - x_pairs[..., 1] * sin
    x_out[..., 1] = x_pairs[..., 0] * sin + x_pairs[..., 1] * cos

    return x_out.reshape(*prefix, seq_len, dim)


def ntk_aware_rope_base(
    dim: int,
    original_max_seq_len: int = 2048,
    target_max_seq_len: int = 8192,
    original_base: float = 10000.0,
) -> float:
    """
    NTK 感知缩放：调整 RoPE 基础频率，使得高频（低维）对几乎保持不变，
    而低频（高维）对被"拉伸"以适应更长的上下文。

    缩放因子 s 从 NTK（神经正切核）直觉导出：
    令 s = (target / original) ^ (dim / (dim-2))。
    则 new_base = original_base * s。

    参考："NTK-Aware Scaled RoPE" (bloc97, 2023)
           https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
    """
    if target_max_seq_len <= original_max_seq_len:
        return original_base  # 无需缩放

    scale = target_max_seq_len / original_max_seq_len
    # 指数 dim/(dim-2) 确保最高频率分量（小 dim 索引）几乎不缩放，
    # 而最低频率得到完整的缩放因子。
    exponent = dim / (dim - 2)
    ntk_factor = scale**exponent
    new_base = original_base * ntk_factor

    return new_base


# ===========================================================================
# 5. KV 缓存淘汰（保留前 k + 后 m，淘汰中间部分）
# ===========================================================================


def kv_cache_eviction(
    k: torch.Tensor,
    v: torch.Tensor,
    keep_first: int,
    keep_last: int,
    cache_capacity: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    StreamLLM 风格的淘汰策略：当 KV 缓存超过 `cache_capacity` 时，
    保留前 `keep_first` 个 token（注意力槽）和后 `keep_last` 个 token（近期上下文），
    丢弃中间部分。

    返回 (淘汰后的_k, 淘汰后的_v, 已淘汰的_token数)。
    """
    seq_len = k.size(2)
    if seq_len <= cache_capacity:
        return k, v, 0  # 缓存未满，无需淘汰

    keep = keep_first + keep_last
    if keep >= cache_capacity:
        # 空间不足；尽可能多地保留尾部 token
        k_evicted = k[:, :, -cache_capacity:, :]
        v_evicted = v[:, :, -cache_capacity:, :]
        evicted = seq_len - cache_capacity
    else:
        # 保留：前 keep_first 个 + 最后 keep_last 个 + 中间若干以填满容量
        middle_keep = cache_capacity - keep
        idx = list(range(keep_first)) + list(
            range(seq_len - keep_last - middle_keep, seq_len)
        )
        k_evicted = k[:, :, idx, :]
        v_evicted = v[:, :, idx, :]
        evicted = seq_len - cache_capacity

    return k_evicted, v_evicted, evicted


# ===========================================================================
# 比较驱动函数
# ===========================================================================


def compare_methods() -> None:
    """
    在序列长度 [256, 512, 1024, 2048, 4096] 上运行全部五种注意力变体，
    并打印比较表。
    """
    seq_lengths = [256, 512, 1024, 2048, 4096]
    batch, heads, dim = 1, 4, 64  # 演示用的小维度
    window_size = 128  # 滑动窗口大小
    chunk_size = 128  # 流式注意力的块大小
    keep_first, keep_last, kv_capacity = 4, 128, 128  # KV 淘汰参数

    print("\n" + "=" * 100)
    print("  第 15 讲：长上下文注意力优化  --  内存占用比较")
    print("=" * 100)
    print(
        f"  配置：batch={batch}  heads={heads}  d_model={dim}  window={window_size}  chunk={chunk_size}"
    )
    print()

    # ------------------------------------------------------------------
    # 表 1：完整注意力 vs 滑动窗口（理论比较）
    # ------------------------------------------------------------------
    print("  表 1 -- 内存占用（注意力分数矩阵中的元素数量）")
    print(
        f"  {'序列长度':>8s}  {'完整 (n^2)':>15s}  {'窗口 (n*w)':>15s}  {'缩减比':>12s}"
    )
    print(f"  {'-' * 8}  {'-' * 15}  {'-' * 15}  {'-' * 12}")

    for seq_len in seq_lengths:
        full_el = seq_len * seq_len  # O(n^2)
        window_el = seq_len * window_size  # O(n * w)
        ratio = full_el / window_el if window_el > 0 else float("inf")
        print(f"  {seq_len:>8d}  {full_el:>15,d}  {window_el:>15,d}  {ratio:>11.1f}x")

    # ------------------------------------------------------------------
    # 构建实际张量并运行各方法
    # ------------------------------------------------------------------
    print(f"\n  {'=' * 70}")
    print("  表 2 -- 实测元素数量（实际张量运行）+ KV 缓存淘汰统计")
    print(f"  {'=' * 70}")

    torch.manual_seed(42)  # 固定随机种子以确保可重复性

    for seq_len in seq_lengths:
        _divider("-", 90)
        print(f"  序列长度 = {seq_len}")
        _divider("-", 90)

        # 创建随机 Q/K/V 张量
        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        # ---- 1. 完整注意力 ----
        _, full_el = full_attention(q, k, v)
        _print_memory("1. 完整注意力", (seq_len, seq_len), full_el)

        # ---- 2. 滑动窗口注意力 ----
        _, sw_el = sliding_window_attention(q, k, v, window_size)
        _print_memory("2. 滑动窗口", (seq_len, window_size), sw_el)

        # ---- 3. 流式注意力 ----
        _, stream_el = streaming_attention(q, k, v, chunk_size, window_size)
        _print_memory("3. 流式（分块）", (seq_len, window_size), stream_el)

        # ---- 4. NTK 感知 RoPE 缩放 ----
        # 演示基础频率的变化
        original_base = 10000.0
        new_base = ntk_aware_rope_base(dim, 2048, 8192, original_base)
        positions = torch.arange(seq_len)
        # 应用 RoPE 到 Q 和 K 上
        q_rope = apply_rope(q, positions, base=original_base)
        k_rope = apply_rope(k, positions, base=original_base)
        # RoPE 的注意力内存与原始相同，但记录缩放信息
        _, rope_el = full_attention(q_rope, k_rope, v)
        print(
            f"  4. RoPE（内存与完整注意力相同）   base: {original_base:.0f}"
            f"  ->  用于 4x 上下文的 NTK 缩放 base: {new_base:.1f}"
        )

        # ---- 5. KV 缓存淘汰 ----
        evicted_k, evicted_v, num_evicted = kv_cache_eviction(
            k, v, keep_first, keep_last, kv_capacity
        )
        orig_tokens = k.size(2)
        remaining = evicted_k.size(2)
        print(f"  5. KV 缓存淘汰")
        print(
            f"       keep_first={keep_first}  keep_last={keep_last}  容量={kv_capacity}"
        )
        print(
            f"       原始 token 数={orig_tokens:>5d}"
            f"  剩余={remaining:>5d}"
            f"  已淘汰={num_evicted:>5d}"
            f"  节省的内存={orig_tokens - remaining:>5d} 个 token"
        )
        print()

    # ------------------------------------------------------------------
    # 附加：不同目标长度下的 NTK 感知缩放演示
    # ------------------------------------------------------------------
    _divider("=", 100)
    print("  附加：RoPE 的 NTK 感知 base 缩放")
    print(f"  原始 base = 10000.0，训练上下文长度 = 2048")
    print(
        f"  {'目标长度':>12s}  {'扩展比例':>18s}  {'新 Base':>15s}  {'Log2(Base)':>12s}"
    )
    print(f"  {'-' * 12}  {'-' * 18}  {'-' * 15}  {'-' * 12}")

    targets = [4096, 8192, 16384, 32768, 65536, 131072]
    for target in targets:
        nb = ntk_aware_rope_base(dim, 2048, target)
        print(
            f"  {target:>12,d}  {target / 2048:>17.1f}x  {nb:>15.1f}  {math.log2(nb):>11.2f}"
        )

    _divider("=", 100)
    print("\n  完成。所有比较已在 CPU 上完成。\n")


# ===========================================================================
# 入口点
# ===========================================================================

if __name__ == "__main__":
    compare_methods()
