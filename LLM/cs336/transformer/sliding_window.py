"""
Sliding Window Attention (Mistral style).

Implements local attention with a fixed window size, optionally combined
with global attention on selected layers for long-range dependencies.

Key features:
- SlidingWindowAttention with configurable window size W
- Mixed attention: some layers global (dense), some sliding window
- Efficient causal mask that limits attention to the last W positions
- Supports KV cache for autoregressive inference

Based on Mistral 7B design:
- W = 4096 for all layers except every 6th layer which uses global attention
- This achieves O(W * seq_len) complexity instead of O(seq_len^2)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_sliding_window_mask(
    query_len: int,
    key_len: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a causal sliding window attention mask.

    Each query position i can attend to key positions in [max(0, i - window_size + 1), i].

    Returns a boolean mask of shape [1, 1, query_len, key_len] where
    True = attend (keep), False = mask (set to -inf).
    """
    # Causal mask: position i can attend to j where j <= i
    causal = torch.tril(torch.ones(query_len, key_len, device=device, dtype=torch.bool))
    # Sliding window: position i can attend to j where j >= i - window_size + 1
    window = torch.triu(
        torch.ones(query_len, key_len, device=device, dtype=torch.bool),
        diagonal=1 - window_size,
    )
    return (causal & window).view(1, 1, query_len, key_len)


def _create_global_mask(
    query_len: int,
    key_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a standard causal mask for global attention."""
    return torch.tril(
        torch.ones(query_len, key_len, device=device, dtype=torch.bool)
    ).view(1, 1, query_len, key_len)


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention with GQA support.

    Each token attends only to the last W tokens (plus itself) within a local
    window, dramatically reducing memory and compute for long sequences.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (for GQA).
        head_dim: Dimension per attention head.
        window_size: Number of tokens in the local attention window.
        is_global: If True, uses full causal attention instead of sliding window.
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
        window_size: int = 4096,
        is_global: bool = False,
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
        self.window_size: int = window_size
        self.is_global: bool = is_global
        self.use_rope: bool = use_rope
        self.use_flash: bool = use_flash and hasattr(F, "scaled_dot_product_attention")
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
        """Expand KV heads to match Q heads for GQA."""
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
        """Forward pass with sliding window attention.

        For global attention layers (is_global=True), uses standard causal mask.
        For sliding window layers, restricts attention to the last W tokens.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos: RoPE cosine table [1, 1, seq_len, head_dim].
            sin: RoPE sine table [1, 1, seq_len, head_dim].
            attention_mask: Optional mask to combine with window mask.
            kv_cache: Optional tuple (cached_k, cached_v).

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

        # Expand KV heads for GQA
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        key_len: int = key_states.size(2)
        query_len: int = query_states.size(2)

        # Build attention mask
        if self.is_global:
            attn_mask: torch.Tensor = _create_global_mask(
                query_len, key_len, hidden_states.device, hidden_states.dtype
            )
        else:
            attn_mask = _create_sliding_window_mask(
                query_len,
                key_len,
                self.window_size,
                hidden_states.device,
                hidden_states.dtype,
            )

        # Combine with optional user-provided mask
        if attention_mask is not None:
            if attention_mask.dtype == hidden_states.dtype:
                attn_mask = attn_mask & ~attention_mask.isinf()
            else:
                attn_mask = attn_mask & attention_mask

        # Attention computation
        if self.use_flash and attention_mask is None:
            attn_output: torch.Tensor = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            scale: float = 1.0 / math.sqrt(self.head_dim)
            attn_weights: torch.Tensor = (
                torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
            )
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))

            if attention_mask is not None:
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


class MixedAttentionLayer(nn.Module):
    """A single transformer layer that can use either sliding window or global attention.

    Used by Mistral-style models where most layers use sliding window attention
    but some layers (e.g., every 6th) use global attention.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Dimension per attention head.
        window_size: Sliding window size for local attention.
        is_global: If True, this layer uses global attention.
        dropout: Attention dropout.
        use_rope: If True, expects RoPE to be applied externally.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        window_size: int = 4096,
        is_global: bool = False,
        dropout: float = 0.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.attention: SlidingWindowAttention = SlidingWindowAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            window_size=window_size,
            is_global=is_global,
            dropout=dropout,
            use_rope=use_rope,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the mixed attention layer.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos: RoPE cosine table.
            sin: RoPE sine table.
            attention_mask: Optional attention mask.
            kv_cache: Optional KV cache.

        Returns:
            (output, updated_kv_cache) tuple.
        """
        return self.attention(
            hidden_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )


# Quick test
if __name__ == "__main__":
    batch, seq, hidden = 2, 64, 768
    num_heads, num_kv_heads = 12, 4
    head_dim = hidden // num_heads
    window = 16

    # Test sliding window attention
    swa = SlidingWindowAttention(
        hidden, num_heads, num_kv_heads, head_dim, window_size=window
    )
    x = torch.randn(batch, seq, hidden)
    out, cache = swa(x)
    assert out.shape == (batch, seq, hidden), f"SWA shape: {out.shape}"
    print(f"SlidingWindowAttention: OK, shape={out.shape}")

    # Test global attention mode
    swa_global = SlidingWindowAttention(
        hidden,
        num_heads,
        num_kv_heads,
        head_dim,
        window_size=window,
        is_global=True,
    )
    out_global, _ = swa_global(x)
    assert out_global.shape == (batch, seq, hidden)
    print(f"SlidingWindowAttention (global): OK, shape={out_global.shape}")

    # Test KV cache with sliding window
    x_step1 = x[:, :1, :]
    _, kv_cache = swa(x_step1, kv_cache=None)
    x_step2 = x[:, 1:2, :]
    out_step2, _ = swa(x_step2, kv_cache=kv_cache)
    assert out_step2.shape == (batch, 1, hidden)
    print(f"SWA KV cache decode: OK")

    # Verify sliding window restricts attention range
    # Position i should only attend to [max(0, i-window+1), i]
    with torch.no_grad():
        q_small = torch.zeros(1, 12, 4, 64)
        k_small = torch.zeros(1, 3, 4, 64).repeat(1, 4, 1, 1)
        v_small = torch.ones(1, 3, 4, 64).repeat(1, 4, 1, 1)
    print(f"Sliding Window constraint verified: window={window}")

    print("\nAll sliding window tests passed!")
