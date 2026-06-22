"""
Rotary Position Embedding (RoPE) 实现。

基于 Su et al., 2021 的 "RoFormer: Enhanced Transformer with Rotary
Position Embedding" 中描述的 RoPE 方法。RoPE 通过根据其绝对位置旋转
query 和 key 向量来编码位置信息。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)。

    为所有位置预计算 cos 和 sin 频率表，最大支持 max_seq_len 个位置。

    Args:
        dim: head 维度（必须为偶数）。
        max_seq_len: 预计算的最大序列长度。
        theta: rotary embedding 的基础频率（默认: 10000.0）。
    """

    def __init__(
        self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        self.theta: float = theta

        # 计算逆频率: 1 / (theta^(2i/d))，其中 i 取 [0, dim/2)
        freqs: torch.Tensor = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        # 形状: [max_seq_len, dim/2]
        positions: torch.Tensor = torch.arange(max_seq_len, dtype=torch.float32)
        angles: torch.Tensor = torch.outer(positions, freqs)  # [max_seq_len, dim/2]

        # 预计算 cos 和 sin 表
        # 每个值重复两次后扩展为 [max_seq_len, dim]
        emb: torch.Tensor = torch.cat([angles, angles], dim=-1)  # [max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device | str = "cpu",
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回请求序列长度对应的缓存的 (cos, sin) 张量。

        在增量解码过程中，使用 start_pos=cached_len 调用以获取
        正确绝对位置的编码。

        Args:
            seq_len: 所需的位置数量。
            device: 目标设备。
            start_pos: 起始位置偏移量（用于增量解码）。

        Returns:
            (cos, sin) 元组，每个形状为 [1, seq_len, 1, dim]。
        """
        return (
            self.cos_cached[start_pos : start_pos + seq_len]
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0),
            self.sin_cached[start_pos : start_pos + seq_len]
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0),
        )


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 query 和 key 张量应用 rotary position embedding。

    旋转操作的计算方式为：
        x_rotated = x * cos + rotate_half(x) * sin

    Args:
        q: query 张量，形状为 [batch, num_heads, seq_len, head_dim]。
        k: key 张量，形状为 [batch, num_kv_heads, seq_len, head_dim]。
        cos: cosine 表，形状为 [1, 1, seq_len, head_dim]。
        sin: sine 表，形状为 [1, 1, seq_len, head_dim]。

    Returns:
        (rotated_q, rotated_k) 元组，形状与输入相同。
    """

    # 旋转一半的维度: 将后一半交换并取反
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1: torch.Tensor = x[..., : x.shape[-1] // 2]
        x2: torch.Tensor = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    # 在 head 维度上广播 cos/sin
    q_rotated: torch.Tensor = (q * cos) + (rotate_half(q) * sin)
    k_rotated: torch.Tensor = (k * cos) + (rotate_half(k) * sin)

    return q_rotated, k_rotated


# 快速测试
if __name__ == "__main__":
    dim: int = 64
    max_len: int = 128
    batch: int = 2
    n_heads: int = 8
    seq: int = 16

    rope = RotaryEmbedding(dim=dim, max_seq_len=max_len, theta=10000.0)
    cos, sin = rope.forward(seq)

    q = torch.randn(batch, n_heads, seq, dim)
    k = torch.randn(batch, n_heads, seq, dim)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_rot.shape == q.shape, f"Shape mismatch: {q_rot.shape} != {q.shape}"
    assert k_rot.shape == k.shape, f"Shape mismatch: {k_rot.shape} != {k.shape}"
    assert not torch.allclose(q, q_rot), "RoPE should change query values"
    print(f"RoPE test passed! Shapes: q={q_rot.shape}, k={k_rot.shape}")
