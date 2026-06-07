"""
Attention variants implemented from scratch using only PyTorch tensor operations.

Implementations:
  - scaled_dot_product_attention: Naive SDPA
  - Multi-Head Attention (MHA): Standard transformer attention
  - Causal Attention: With causal (upper triangular) mask
  - Grouped Query Attention (GQA): Fewer KV heads than Q heads
  - Multi-Query Attention (MQA): Single KV head for all Q heads
  - Sliding Window Attention: Each token attends to a local window
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Scaled Dot-Product Attention (naive)
# =========================================================================


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Naive scaled dot-product attention.

    Computes: softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        q: Query tensor, shape (..., seq_len_q, d_k)
        k: Key tensor,   shape (..., seq_len_k, d_k)
        v: Value tensor, shape (..., seq_len_k, d_v)
        mask: Optional mask, shape (..., seq_len_q, seq_len_k).
              -inf entries will be masked out
        dropout_p: Dropout probability

    Returns:
        Output tensor of shape (..., seq_len_q, d_v)
    """
    d_k = q.size(-1)
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (add -inf to masked positions before softmax)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax over the key dimension
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # Weighted sum of values
    output = torch.matmul(attn_weights, v)
    return output


# =========================================================================
# Multi-Head Attention
# =========================================================================


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention (MHA).

    All heads have same dimension. Q, K, V are projected from input,
    split into heads, then attention is computed independently per head.

    The default setup (num_kv_heads == num_heads) produces standard MHA.
    Setting num_kv_heads < num_heads enables GQA/MQA.
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

        # Projections
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
            x: Input tensor, shape (batch, seq_len, hidden_size)
            mask: Optional attention mask

        Returns:
            Output tensor, shape (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = x.shape

        # Project and reshape
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

        # Expand KV heads if GQA/MQA (repeat each KV head for its query group)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Compute attention
        attn_output = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout
        )
        # attn_output: (batch, num_heads, seq_len, head_dim)

        # Merge heads back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)


# =========================================================================
# Causal Attention
# =========================================================================


def create_causal_mask(
    seq_len: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """
    Create a causal (lower triangular) attention mask.

    Returns a boolean mask where True = allowed to attend.
    Shape: (1, 1, seq_len, seq_len) for broadcasting.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.view(1, 1, seq_len, seq_len)


class CausalAttention(MultiHeadAttention):
    """
    Multi-head attention with a causal mask.
    Each token i can only attend to tokens j <= i.
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
    Grouped Query Attention: Q heads are grouped, each group shares one KV head.

    Typical config: num_kv_heads = num_heads // group_size.
    For example: num_heads=32, num_kv_heads=8 → 4 query heads share one KV pair.
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
    Multi-Query Attention: All Q heads share a single KV head.

    This is the extreme case of GQA with num_kv_heads = 1.
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
    Create a sliding window attention mask.

    Each token i can attend to tokens in [max(0, i-window+1), i] (if causal).
    For non-causal, window is centered: [i-window//2, i+window//2].

    Returns:
        Boolean mask, shape (1, 1, seq_len, seq_len).
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

    for i in range(seq_len):
        if is_causal:
            start = max(0, i - window_size + 1)
            end = i + 1  # inclusive
        else:
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = True

    return mask.view(1, 1, seq_len, seq_len)


class SlidingWindowAttention(MultiHeadAttention):
    """
    Sliding window attention: each token only attends to tokens within a local window.

    Commonly used in architectures like Mistral and Longformer.
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
# Helper: create attention from config
# =========================================================================


def create_attention(
    variant: str,
    hidden_size: int = 512,
    num_heads: int = 8,
    **kwargs: object,
) -> nn.Module:
    """
    Factory to create the requested attention variant.

    Args:
        variant: One of 'mha', 'causal', 'gqa', 'mqa', 'sliding_window'
        hidden_size: Model hidden size
        num_heads: Number of attention heads
        **kwargs: Additional arguments for specific variants

    Returns:
        Initialized attention module
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
# Demo
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

    # Demonstrate raw SDPA
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
