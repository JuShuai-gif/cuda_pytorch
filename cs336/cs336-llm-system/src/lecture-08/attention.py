"""
仅使用 PyTorch 张量操作从零实现的注意力变体。

具体实现：
  - scaled_dot_product_attention：朴素 SDPA
  - Multi-Head Attention (MHA)：标准 transformer 注意力
  - Causal Attention：带因果（上三角）mask
  - Grouped Query Attention (GQA)：KV 头的数量少于 Q 头
  - Multi-Query Attention (MQA)：所有 Q 头共享单个 KV 头
  - Sliding Window Attention：每个 token 只关注局部窗口
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Scaled Dot-Product Attention（朴素实现）
# =========================================================================


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    朴素 scaled dot-product attention。

    计算公式：softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        q: Query 张量，shape 为 (..., seq_len_q, d_k)
        k: Key 张量，  shape 为 (..., seq_len_k, d_k)
        v: Value 张量，shape 为 (..., seq_len_k, d_v)
        mask: 可选的 mask，shape 为 (..., seq_len_q, seq_len_k)。
              值为 -inf 的位置将被遮蔽
        dropout_p: Dropout 概率

    Returns:
        输出张量，shape 为 (..., seq_len_q, d_v)
    """
    d_k = q.size(-1)
    # 计算注意力分数：Q @ K^T / sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用 mask（在 softmax 之前将被遮蔽的位置设为 -inf）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 在 key 维度上做 softmax
    attn_weights = F.softmax(scores, dim=-1)

    # 应用 dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # 对 value 加权求和
    output = torch.matmul(attn_weights, v)
    return output


# =========================================================================
# Multi-Head Attention
# =========================================================================


class MultiHeadAttention(nn.Module):
    """
    标准 Multi-Head Attention (MHA)。

    所有头的维度相同。Q、K、V 从输入投影后，
    分割为多个头，然后每个头独立计算注意力。

    默认设置（num_kv_heads == num_heads）产生标准 MHA。
    设置 num_kv_heads < num_heads 可以启用 GQA/MQA。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.dropout = dropout

        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # 投影层
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_size, q_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=bias)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量，shape 为 (batch, seq_len, hidden_size)
            mask: 可选的注意力 mask

        Returns:
            输出张量，shape 为 (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = x.shape

        # 投影并重塑
        q = (
            self.q_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 若为 GQA/MQA，则扩展 KV 头（为每个 query 组重复对应的 KV 头）
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # 计算注意力
        attn_output = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout
        )
        # attn_output: (batch, num_heads, seq_len, head_dim)

        # 将头合并回去
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)


# =========================================================================
# Causal Attention
# =========================================================================


def create_causal_mask(
    seq_len: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """
    创建因果（下三角）注意力 mask。

    返回一个布尔 mask，True 表示允许关注。
    Shape 为 (1, 1, seq_len, seq_len)，方便广播。
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.view(1, 1, seq_len, seq_len)


class CausalAttention(MultiHeadAttention):
    """
    带因果 mask 的 multi-head attention。
    每个 token i 只能关注 j <= i 的 token。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        causal_mask = create_causal_mask(seq_len, x.device)
        return super().forward(x, mask=causal_mask)


# =========================================================================
# Grouped Query Attention (GQA)
# =========================================================================


class GroupedQueryAttention(MultiHeadAttention):
    """
    Grouped Query Attention：Q 头分组，每组共享一个 KV 头。

    典型配置：num_kv_heads = num_heads // group_size。
    例如：num_heads=32, num_kv_heads=8 → 4 个 query 头共享一对 KV。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ):
        assert num_kv_heads < num_heads, "GQA requires num_kv_heads < num_heads"
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
        )


# =========================================================================
# Multi-Query Attention (MQA)
# =========================================================================


class MultiQueryAttention(MultiHeadAttention):
    """
    Multi-Query Attention：所有 Q 头共享单个 KV 头。

    这是 GQA 在 num_kv_heads = 1 时的极端情况。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=head_dim,
            dropout=dropout,
        )


# =========================================================================
# Sliding Window Attention
# =========================================================================


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    is_causal: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    创建 sliding window 注意力 mask。

    若为 causal，每个 token i 可以关注 [max(0, i-window+1), i] 范围内的 token。
    若为非 causal，窗口居中：[i-window//2, i+window//2]。

    Returns:
        布尔 mask，shape 为 (1, 1, seq_len, seq_len)。
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

    for i in range(seq_len):
        if is_causal:
            start = max(0, i - window_size + 1)
            end = i + 1  # 包含
        else:
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = True

    return mask.view(1, 1, seq_len, seq_len)


class SlidingWindowAttention(MultiHeadAttention):
    """
    Sliding window attention：每个 token 只关注局部窗口内的 token。

    常见于 Mistral 和 Longformer 等架构中。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 4096,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        mask = create_sliding_window_mask(
            seq_len, self.window_size, is_causal=True, device=x.device
        )
        return super().forward(x, mask=mask)


# =========================================================================
# 辅助函数：根据配置创建注意力模块
# =========================================================================


def create_attention(
    variant: str,
    hidden_size: int = 512,
    num_heads: int = 8,
    **kwargs: object,
) -> nn.Module:
    """
    工厂函数，创建指定的注意力变体。

    Args:
        variant: 可选 'mha'、'causal'、'gqa'、'mqa'、'sliding_window' 之一
        hidden_size: 模型隐藏层大小
        num_heads: 注意力头数
        **kwargs: 特定变体的额外参数

    Returns:
        初始化好的注意力模块
    """
    variant = variant.lower()
    if variant == "mha":
        return MultiHeadAttention(hidden_size, num_heads, **kwargs)
    elif variant == "causal":
        return CausalAttention(hidden_size, num_heads, **kwargs)
    elif variant == "gqa":
        num_kv_heads = kwargs.get("num_kv_heads", num_heads // 4)
        return GroupedQueryAttention(hidden_size, num_heads, num_kv_heads, **kwargs)
    elif variant == "mqa":
        return MultiQueryAttention(hidden_size, num_heads, **kwargs)
    elif variant == "sliding_window":
        window_size = kwargs.get("window_size", 4096)
        return SlidingWindowAttention(hidden_size, num_heads, window_size, **kwargs)
    else:
        raise ValueError(f"Unknown attention variant: {variant}")


# =========================================================================
# 演示
# =========================================================================


def main() -> None:
    print("=" * 60)
    print("Attention Variants Demo")
    print("=" * 60)

    batch, seq_len, hidden = 2, 16, 512
    x = torch.randn(batch, seq_len, hidden)

    variants = {
        "MHA": create_attention("mha", hidden),
        "Causal": create_attention("causal", hidden),
        "GQA (4 KV heads)": create_attention("gqa", hidden, num_kv_heads=4),
        "MQA (1 KV head)": create_attention("mqa", hidden),
        "Sliding Window (w=8)": create_attention(
            "sliding_window", hidden, window_size=8
        ),
    }

    for name, attn in variants.items():
        attn.eval()
        with torch.no_grad():
            out = attn(x)
        params = sum(p.numel() for p in attn.parameters())
        print(f"\n{name}:")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Parameters:   {params:,}")

    # 演示原始 SDPA
    print("\n--- Raw Scaled Dot-Product Attention ---")
    q = torch.randn(1, 8, 16, 64)  # (batch, heads, seq, head_dim)
    k = torch.randn(1, 8, 16, 64)
    v = torch.randn(1, 8, 16, 64)
    causal_mask = create_causal_mask(16)
    out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
    print(f"  Q shape: {q.shape}")
    print(f"  Output shape: {out.shape}")


if __name__ == "__main__":
    main()
