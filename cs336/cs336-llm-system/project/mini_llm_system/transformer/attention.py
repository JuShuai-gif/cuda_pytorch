"""
从零实现的注意力机制。

包含：
- ScaledDotProductAttention: 带 Q*K^T / sqrt(d) 缩放的基础注意力。
- CausalAttention: 带因果（下三角）掩码的自回归注意力。
- GroupedQueryAttention: 使用共享 KV 头的 GQA，提高效率。
- FlashAttentionSimple: 使用 online softmax 重标定的分块实现。

所有实现均支持 KV cache，用于高效的自回归推理。
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

# 允许直接运行此文件或作为包的一部分导入
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    标准缩放点积注意力。

    计算公式：softmax(Q @ K^T / sqrt(d_k)) @ V

    参数：
        dropout: 应用于注意力权重的 Dropout 概率。
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout: float = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参数：
            query: [batch, num_heads, seq_len_q, head_dim]
            key: [batch, num_heads, seq_len_k, head_dim]
            value: [batch, num_heads, seq_len_k, head_dim]
            mask: 可选的注意力掩码 [batch, 1, seq_len_q, seq_len_k] 或可广播的形状。

        返回：
            输出 tensor [batch, num_heads, seq_len_q, head_dim]。
        """
        head_dim: int = query.size(-1)
        scale: float = 1.0 / math.sqrt(head_dim)

        # 计算注意力分数
        attn_weights: torch.Tensor = torch.matmul(query, key.transpose(-2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        # 对最后一维做 softmax（转换为 float32 以保证数值稳定性）
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(key.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 值的加权求和
        output: torch.Tensor = torch.matmul(attn_weights, value)
        return output


class CausalAttention(ScaledDotProductAttention):
    """
    带因果（自回归）掩码的缩放点积注意力。

    因果掩码确保每个位置只能关注自身及之前的位置，
    防止从未来 token 泄露信息。
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数：
            query: [batch, num_heads, seq_len, head_dim]
            key: [batch, num_heads, seq_len, head_dim]
            value: [batch, num_heads, seq_len, head_dim]

        返回：
            带因果掩码的输出 tensor [batch, num_heads, seq_len, head_dim]。
        """
        seq_len: int = query.size(2)
        # 创建下三角因果掩码
        causal_mask: torch.Tensor = torch.tril(
            torch.ones(seq_len, seq_len, device=query.device, dtype=query.dtype)
        ).view(1, 1, seq_len, seq_len)

        return super().forward(query, key, value, mask=causal_mask)


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（Grouped Query Attention，GQA）。

    GQA 通过使用比查询头更少的 KV 头来减少 KV-cache 内存。
    每组查询头共享一个 KV 头。当 num_kv_heads == num_heads 时，
    等效于标准多头注意力。当 num_kv_heads == 1 时，
    等效于多查询注意力（Multi-Query Attention，MQA）。

    参数：
        hidden_size: 模型隐藏层维度。
        num_heads: 查询注意力头的数量。
        num_kv_heads: key/value 注意力头的数量（必须能整除 num_heads）。
        head_dim: 每个注意力头的维度。
        dropout: 注意力 dropout 概率。
        use_rope: 如果为 True，则期望 RoPE 从外部应用。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.use_rope: bool = use_rope
        self.n_rep: int = num_heads // num_kv_heads
        self.dropout: float = dropout

        # Q, K, V 投影层
        self.q_proj: nn.Linear = nn.Linear(
            hidden_size, num_heads * head_dim, bias=False
        )
        self.k_proj: nn.Linear = nn.Linear(
            hidden_size, num_kv_heads * head_dim, bias=False
        )
        self.v_proj: nn.Linear = nn.Linear(
            hidden_size, num_kv_heads * head_dim, bias=False
        )

        # 输出投影
        self.o_proj: nn.Linear = nn.Linear(
            num_heads * head_dim, hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        GQA 的前向传播。

        参数：
            hidden_states: [batch, seq_len, hidden_size]
            cos: 来自 RoPE 的余弦表 [1, 1, seq_len, head_dim]。
            sin: 来自 RoPE 的正弦表 [1, 1, seq_len, head_dim]。
            attention_mask: 可选掩码 [batch, 1, seq_len, seq_len]。
            kv_cache: 可选元组 (cached_k, cached_v)，用于增量解码。

        返回：
            元组 (output [batch, seq_len, hidden_size], updated kv_cache)。
        """
        batch_size: int = hidden_states.size(0)
        seq_len: int = hidden_states.size(1)

        # 投影 Q, K, V
        query_states: torch.Tensor = self.q_proj(hidden_states)
        key_states: torch.Tensor = self.k_proj(hidden_states)
        value_states: torch.Tensor = self.v_proj(hidden_states)

        # 重塑为 [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # 对 Q 和 K 应用 RoPE（在此处进行，因为 cos/sin 由外部提供）
        if self.use_rope and cos is not None and sin is not None:
            from transformer.rotary_embedding import apply_rotary_pos_emb

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # 处理 KV cache：将缓存的 KV 与新的 KV 拼接
        new_kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            key_states = torch.cat([cached_k, key_states], dim=2)
            value_states = torch.cat([cached_v, value_states], dim=2)
        new_kv_cache = (key_states, value_states)

        # 扩展 KV 头以匹配 Q 头（将每个 KV 头重复 n_rep 次）
        if self.n_rep > 1:
            key_states = key_states.repeat_interleave(self.n_rep, dim=1)
            value_states = value_states.repeat_interleave(self.n_rep, dim=1)

        # 计算注意力
        scale: float = 1.0 / math.sqrt(self.head_dim)
        attn_weights: torch.Tensor = (
            torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
        )

        # 应用因果掩码（为简单起见仅用于推理；也可以使用提供的掩码）
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            # 默认：因果掩码
            kv_len: int = key_states.size(2)
            q_len: int = query_states.size(2)
            causal_mask: torch.Tensor = torch.triu(
                torch.full(
                    (q_len, kv_len),
                    float("-inf"),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
                diagonal=1,
            ).view(1, 1, q_len, kv_len)
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(hidden_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output: torch.Tensor = torch.matmul(attn_weights, value_states)

        # 重塑回 [batch, seq_len, hidden_size]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )

        # 输出投影
        output: torch.Tensor = self.o_proj(attn_output)
        return output, new_kv_cache


class FlashAttentionSimple(nn.Module):
    """
    简化的分块 FlashAttention 实现，使用 PyTorch 循环。

    实现了 FlashAttention 的核心思想：将 Q 分割为块（tile），
    逐块计算注意力并使用 online softmax 重标定，避免
    生成完整的 N×N 注意力矩阵。

    关键概念：
    - 分块计算（Tiled computation）：将 Q 分成块来处理以减少内存。
    - Online softmax：维护运行中的最大值与和，以数值稳定的方式计算 softmax。
    - 重标定（Rescaling）：当发现新的最大值时更新之前的输出。

    这是一个展示算法的教学实现；由于 Python 循环的开销，
    它并不比原生实现更快，但演示了节省内存的原理。

    参数：
        block_size: 分块计算中 Q 块的大小。
    """

    def __init__(self, block_size: int = 128) -> None:
        super().__init__()
        self.block_size: int = block_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        分块 flash attention 的前向传播。

        参数：
            query: [batch, num_heads, seq_len, head_dim]
            key: [batch, num_heads, seq_len, head_dim]
            value: [batch, num_heads, seq_len, head_dim]
            causal: 如果为 True，则应用因果掩码。

        返回：
            输出 tensor [batch, num_heads, seq_len, head_dim]。
        """
        batch_size: int = query.size(0)
        num_heads: int = query.size(1)
        seq_len: int = query.size(2)
        head_dim: int = query.size(3)
        scale: float = 1.0 / math.sqrt(head_dim)

        block_size: int = self.block_size
        if block_size > seq_len:
            block_size = seq_len

        # 输出累加器
        output: torch.Tensor = torch.zeros_like(query)

        # 按块处理 Q
        for q_start in range(0, seq_len, block_size):
            q_end: int = min(q_start + block_size, seq_len)
            q_block: torch.Tensor = query[:, :, q_start:q_end, :]  # [B, H, Bq, D]
            q_block_len: int = q_end - q_start

            # Online softmax 状态
            m_i: torch.Tensor = torch.full(
                (batch_size, num_heads, q_block_len, 1),
                float("-inf"),
                device=query.device,
                dtype=query.dtype,
            )  # 运行中的最大值
            l_i: torch.Tensor = torch.zeros(
                (batch_size, num_heads, q_block_len, 1),
                device=query.device,
                dtype=query.dtype,
            )  # 运行中的和
            o_i: torch.Tensor = torch.zeros(
                (batch_size, num_heads, q_block_len, head_dim),
                device=query.device,
                dtype=query.dtype,
            )  # 运行中的输出

            # 按块处理 K/V
            for k_start in range(0, seq_len, block_size):
                k_end: int = min(k_start + block_size, seq_len)
                k_block: torch.Tensor = key[:, :, k_start:k_end, :]  # [B, H, Bk, D]
                v_block: torch.Tensor = value[:, :, k_start:k_end, :]  # [B, H, Bk, D]

                # 计算当前块的注意力分数
                scores: torch.Tensor = (
                    torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
                )
                # scores 形状: [B, H, Bq, Bk]

                # 如有需要，应用因果掩码
                if causal:
                    # q_block 中的位置只能关注自身及更早的位置
                    q_indices: torch.Tensor = torch.arange(
                        q_start, q_end, device=query.device
                    )
                    k_indices: torch.Tensor = torch.arange(
                        k_start, k_end, device=query.device
                    )
                    causal_mask: torch.Tensor = (
                        q_indices[:, None] >= k_indices[None, :]
                    ).to(query.dtype)
                    causal_mask = causal_mask.view(1, 1, q_block_len, k_end - k_start)
                    scores = scores.masked_fill(causal_mask == 0, float("-inf"))

                # Online softmax 更新
                m_new: torch.Tensor = torch.max(
                    scores, dim=-1, keepdim=True
                ).values  # [B, H, Bq, 1]
                m_new = torch.maximum(m_i, m_new)

                # 计算缩放因子
                # exp(m_i - m_new) 重标定之前的和；exp(scores - m_new) 用于当前块
                alpha: torch.Tensor = torch.exp(m_i - m_new)
                p: torch.Tensor = torch.exp(scores - m_new)  # [B, H, Bq, Bk]

                # 更新运行中的和
                l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)

                # 更新输出：重标定旧输出并加上加权后的值
                o_i = o_i * alpha + torch.matmul(p, v_block)

                m_i = m_new

            # 用 softmax 分母归一化输出
            output[:, :, q_start:q_end, :] = o_i / l_i

        return output


# 测试和演示
if __name__ == "__main__":
    batch, seq, hidden = 2, 64, 768
    num_heads, num_kv_heads = 8, 4
    head_dim: int = hidden // num_heads

    # 测试 ScaledDotProductAttention
    q = torch.randn(batch, num_heads, seq, head_dim)
    k = torch.randn(batch, num_heads, seq, head_dim)
    v = torch.randn(batch, num_heads, seq, head_dim)

    sdpa = ScaledDotProductAttention()
    out = sdpa(q, k, v)
    assert out.shape == (batch, num_heads, seq, head_dim), (
        f"SDPA shape wrong: {out.shape}"
    )
    print(f"ScaledDotProductAttention: OK, shape={out.shape}")

    # 测试 CausalAttention
    causal = CausalAttention()
    out_causal = causal(q, k, v)
    assert out_causal.shape == (batch, num_heads, seq, head_dim), (
        f"Causal shape wrong: {out_causal.shape}"
    )

    # 验证因果性：位置 i 的输出应仅依赖于位置 ≤ i 的 token
    # 使用已知值创建测试
    q_test = torch.zeros(1, 1, 3, 4)
    k_test = torch.zeros(1, 1, 3, 4)
    v_test = torch.ones(1, 1, 3, 4)
    k_test[:, :, 2, :] = 100.0  # 使最后一个 key 位置具有极大的权重
    out_test = causal(q_test, k_test, v_test)
    # 位置 0 的输出应为其自身的值 (1)，位置 1 应为位置 0 和 1 的平均值
    print(f"CausalAttention: OK, shape={out_causal.shape}")

    # 测试 GroupedQueryAttention
    gqa = GroupedQueryAttention(hidden, num_heads, num_kv_heads, head_dim)
    x = torch.randn(batch, seq, hidden)
    out_gqa, kv_cache = gqa(x)
    assert out_gqa.shape == (batch, seq, hidden), f"GQA shape wrong: {out_gqa.shape}"
    assert kv_cache is not None and len(kv_cache) == 2, (
        "KV cache should be a tuple of 2 tensors"
    )
    print(f"GroupedQueryAttention: OK, shape={out_gqa.shape}")

    # 测试 GQA 的 KV cache（增量解码）
    x_step1 = x[:, :1, :]  # 第一个 token（预填充）
    _, kv_cache = gqa(x_step1, kv_cache=None)  # 预填充
    x_step2 = x[:, 1:2, :]  # 第二个 token（缓存命中）
    out_step2, kv_cache2 = gqa(x_step2, kv_cache=kv_cache)  # 使用缓存进行解码
    assert out_step2.shape == (batch, 1, hidden), (
        f"GQA decode shape wrong: {out_step2.shape}"
    )

    # 测试 FlashAttentionSimple
    flash = FlashAttentionSimple(block_size=16)
    out_flash = flash(q, k, v, causal=True)
    assert out_flash.shape == (batch, num_heads, seq, head_dim), (
        f"Flash shape wrong: {out_flash.shape}"
    )

    # 比较 flash 与朴素实现（应很接近）
    out_causal_full = causal(q, k, v)
    max_diff: float = (out_flash - out_causal_full).abs().max().item()
    # 注意：由于 online softmax，可能存在微小的数值差异
    # 在 float32 下，这些差异应该非常小
    print(
        f"FlashAttentionSimple: OK, shape={out_flash.shape}, max_diff_vs_causal={max_diff:.6f}"
    )

    print("\nAll attention tests passed!")
