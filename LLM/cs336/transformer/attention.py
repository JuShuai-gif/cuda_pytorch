"""
Production-grade attention implementations with flash attention support.

Provides:
- MultiHeadAttention (standard MHA)
- GroupedQueryAttention (GQA with configurable group size)
- MultiQueryAttention (MQA, extreme case of GQA)

All implementations support:
- KV cache for efficient autoregressive inference
- Causal masking
- Flash attention backend (via torch.nn.functional.scaled_dot_product_attention)
- Mixed precision (fp16/bf16/fp32)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_flash_attention_available() -> bool:
    """Check if flash attention is available via torch's SDPA backend."""
    return hasattr(F, "scaled_dot_product_attention")


def _compute_causal_mask(
    query_len: int,
    key_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a causal mask of shape [1, 1, query_len, key_len].

    Returns a boolean mask where True = attend (keep), False = mask (set to -inf).
    """
    return torch.tril(
        torch.ones(query_len, key_len, device=device, dtype=torch.bool)
    ).view(1, 1, query_len, key_len)


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (MHA).

    All query heads have dedicated key/value heads. Use this when GQA group size = 1.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        dropout: Attention dropout probability.
        use_rope: If True, expects RoPE to be applied externally.
        use_flash: If True, uses flash attention backend when available.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_heads
        self.head_dim: int = head_dim
        self.use_rope: bool = use_rope
        self.use_flash: bool = use_flash and _is_flash_attention_available()
        self.n_rep: int = 1

        self.q_proj: nn.Linear = nn.Linear(
            hidden_size, num_heads * head_dim, bias=False
        )
        self.k_proj: nn.Linear = nn.Linear(
            hidden_size, num_heads * head_dim, bias=False
        )
        self.v_proj: nn.Linear = nn.Linear(
            hidden_size, num_heads * head_dim, bias=False
        )
        self.o_proj: nn.Linear = nn.Linear(
            num_heads * head_dim, hidden_size, bias=False
        )

        self.attn_dropout: float = dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV cache.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos: RoPE cosine table [1, 1, seq_len, head_dim].
            sin: RoPE sine table [1, 1, seq_len, head_dim].
            attention_mask: Optional mask [batch, 1, seq_len, kv_len].
            kv_cache: Optional tuple (cached_k, cached_v) for incremental decoding.

        Returns:
            (output [batch, seq_len, hidden_size], updated_kv_cache) tuple.
        """
        batch_size: int = hidden_states.size(0)
        seq_len: int = hidden_states.size(1)

        # Project Q, K, V
        query_states: torch.Tensor = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states: torch.Tensor = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states: torch.Tensor = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE
        if self.use_rope and cos is not None and sin is not None:
            from .rope import apply_rotary_pos_emb

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # KV cache
        new_kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            key_states = torch.cat([cached_k, key_states], dim=2)
            value_states = torch.cat([cached_v, value_states], dim=2)
        new_kv_cache = (key_states, value_states)

        # Compute attention
        key_len: int = key_states.size(2)
        query_len: int = query_states.size(2)

        if self.use_flash and attention_mask is None:
            # Build causal mask for flash attention
            causal_mask = _compute_causal_mask(
                query_len, key_len, hidden_states.device, hidden_states.dtype
            )
            attn_output: torch.Tensor = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            # Manual attention computation
            scale: float = 1.0 / math.sqrt(self.head_dim)
            attn_weights: torch.Tensor = (
                torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
            )

            if attention_mask is None:
                causal_mask = _compute_causal_mask(
                    query_len, key_len, hidden_states.device, hidden_states.dtype
                )
                attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
            else:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights.float(), dim=-1).to(
                hidden_states.dtype
            )
            attn_weights = F.dropout(
                attn_weights, p=self.attn_dropout, training=self.training
            )
            attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        output: torch.Tensor = self.o_proj(attn_output)
        return output, new_kv_cache


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA).

    Uses fewer key/value heads than query heads to reduce KV-cache memory.
    Each group of query heads shares one KV head.

    When num_kv_heads == num_heads: equivalent to MHA.
    When num_kv_heads == 1: equivalent to MQA.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (must divide num_heads).
        head_dim: Dimension per attention head.
        dropout: Attention dropout probability.
        use_rope: If True, expects RoPE to be applied externally.
        use_flash: If True, uses flash attention backend when available.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.use_rope: bool = use_rope
        self.use_flash: bool = use_flash and _is_flash_attention_available()
        self.n_rep: int = num_heads // num_kv_heads
        self.attn_dropout: float = dropout

        self.q_proj: nn.Linear = nn.Linear(
            hidden_size, num_heads * head_dim, bias=False
        )
        self.k_proj: nn.Linear = nn.Linear(
            hidden_size, num_kv_heads * head_dim, bias=False
        )
        self.v_proj: nn.Linear = nn.Linear(
            hidden_size, num_kv_heads * head_dim, bias=False
        )
        self.o_proj: nn.Linear = nn.Linear(
            num_heads * head_dim, hidden_size, bias=False
        )

    def _repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match Q heads for GQA.

        Input:  [batch, num_kv_heads, seq_len, head_dim]
        Output: [batch, num_heads,      seq_len, head_dim]
        """
        if self.n_rep == 1:
            return kv
        batch, n_kv, seq, d = kv.shape
        kv = kv[:, :, None, :, :].expand(batch, n_kv, self.n_rep, seq, d)
        return kv.reshape(batch, n_kv * self.n_rep, seq, d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV cache.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos: RoPE cosine table [1, 1, seq_len, head_dim].
            sin: RoPE sine table [1, 1, seq_len, head_dim].
            attention_mask: Optional mask [batch, 1, seq_len, kv_len].
            kv_cache: Optional tuple (cached_k, cached_v).

        Returns:
            (output [batch, seq_len, hidden_size], updated_kv_cache) tuple.
        """
        batch_size: int = hidden_states.size(0)
        seq_len: int = hidden_states.size(1)

        query_states: torch.Tensor = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states: torch.Tensor = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states: torch.Tensor = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE
        if self.use_rope and cos is not None and sin is not None:
            from .rope import apply_rotary_pos_emb

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # KV cache
        new_kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            key_states = torch.cat([cached_k, key_states], dim=2)
            value_states = torch.cat([cached_v, value_states], dim=2)
        new_kv_cache = (key_states, value_states)

        # Expand KV heads to match Q heads for GQA
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        key_len: int = key_states.size(2)
        query_len: int = query_states.size(2)

        if self.use_flash and attention_mask is None:
            causal_mask = _compute_causal_mask(
                query_len, key_len, hidden_states.device, hidden_states.dtype
            )
            attn_output: torch.Tensor = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            scale: float = 1.0 / math.sqrt(self.head_dim)
            attn_weights: torch.Tensor = (
                torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
            )

            if attention_mask is None:
                causal_mask = _compute_causal_mask(
                    query_len, key_len, hidden_states.device, hidden_states.dtype
                )
                attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
            else:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights.float(), dim=-1).to(
                hidden_states.dtype
            )
            attn_weights = F.dropout(
                attn_weights, p=self.attn_dropout, training=self.training
            )
            attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        output: torch.Tensor = self.o_proj(attn_output)
        return output, new_kv_cache


class MultiQueryAttention(GroupedQueryAttention):
    """Multi-Query Attention (MQA).

    An extreme case of GQA where all query heads share a single KV head.
    Maximally reduces KV-cache footprint.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of query attention heads.
        head_dim: Dimension per attention head.
        dropout: Attention dropout probability.
        use_rope: If True, expects RoPE to be applied externally.
        use_flash: If True, uses flash attention backend when available.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_flash: bool = True,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=head_dim,
            dropout=dropout,
            use_rope=use_rope,
            use_flash=use_flash,
        )


# Quick test
if __name__ == "__main__":
    batch, seq, hidden = 2, 64, 768
    num_heads, num_kv_heads = 12, 4
    head_dim = hidden // num_heads

    # Test MHA
    mha = MultiHeadAttention(hidden, num_heads, head_dim)
    x = torch.randn(batch, seq, hidden)
    out, cache = mha(x)
    assert out.shape == (batch, seq, hidden), f"MHA shape: {out.shape}"
    print(f"MultiHeadAttention: OK, shape={out.shape}")

    # Test GQA
    gqa = GroupedQueryAttention(hidden, num_heads, num_kv_heads, head_dim)
    out, cache = gqa(x)
    assert out.shape == (batch, seq, hidden), f"GQA shape: {out.shape}"
    print(f"GroupedQueryAttention: OK, shape={out.shape}")

    # Test MQA
    mqa = MultiQueryAttention(hidden, num_heads, head_dim)
    out, cache = mqa(x)
    assert out.shape == (batch, seq, hidden), f"MQA shape: {out.shape}"
    print(f"MultiQueryAttention: OK, shape={out.shape}")

    # Test KV cache (incremental decoding)
    x_step1 = x[:, :1, :]
    _, kv_cache = gqa(x_step1, kv_cache=None)
    x_step2 = x[:, 1:2, :]
    out_step2, _ = gqa(x_step2, kv_cache=kv_cache)
    assert out_step2.shape == (batch, 1, hidden)
    print(f"GQA KV cache decode: OK")

    print("\nAll attention tests passed!")
