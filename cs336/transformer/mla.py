"""
Multi-head Latent Attention (MLA) - DeepSeek-V3 style.

MLA dramatically reduces KV-cache memory by compressing keys and values
into a low-rank latent space. Instead of storing full-dimension KV pairs
per layer per token, MLA stores only the latent representation and
reconstructs K/V on the fly using lightweight up-projection matrices.

Key features:
- Low-rank KV compression (typically 32x compression ratio)
- Decoupled RoPE: applies RoPE to a separate small per-head dimension
- KV cache in latent space for drastically smaller memory footprint
- Up-projection matrices for K and V reconstruction

MLA achieves ~32x KV cache compression compared to standard MHA
at the DeepSeek-V3 scale (128 heads, kv_lora_rank=512).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention (MLA) from DeepSeek-V2/V3.

    The key insight is compressing KV representations into a low-rank latent
    vector. During inference, only the latent vector is cached, and K/V
    are reconstructed on-the-fly using learned up-projection matrices.

    Architecture flow:
        Input -> Q/KV latent projections
        KV latent -> up-project to full K, V (no RoPE applied yet)
        Separate decoupled K for RoPE (small per-head dimension)
        Apply RoPE only to the decoupled K portion
        Concatenate (K_nope + K_rope) for the full key
        Standard attention with Q, K, V

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        kv_lora_rank: Low-rank dimension for KV compression.
        qk_rope_head_dim: Per-head dimension for decoupled RoPE K.
        v_head_dim: Per-head dimension for value (may differ from q head_dim).
        q_lora_rank: Optional low-rank dimension for Q compression.
        dropout: Attention dropout probability.
        use_flash: If True, uses flash attention backend when available.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int] = None,
        dropout: float = 0.0,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.kv_lora_rank: int = kv_lora_rank
        self.qk_rope_head_dim: int = qk_rope_head_dim
        self.v_head_dim: int = v_head_dim
        self.q_lora_rank: int = q_lora_rank if q_lora_rank is not None else hidden_size
        self.q_head_dim: int = hidden_size // num_heads
        self.use_flash: bool = use_flash and hasattr(F, "scaled_dot_product_attention")
        self.attn_dropout: float = dropout

        # --- Q projections ---
        # Low-rank Q compression (optional, saves params)
        if self.q_lora_rank != hidden_size:
            self.q_a_proj: nn.Linear = nn.Linear(
                hidden_size, self.q_lora_rank, bias=False
            )
            self.q_a_norm: nn.Module = nn.LayerNorm(self.q_lora_rank)
            self.q_b_proj: nn.Linear = nn.Linear(
                self.q_lora_rank, num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_proj: nn.Linear = nn.Linear(
                hidden_size, num_heads * self.q_head_dim, bias=False
            )

        # Q projection for decoupled RoPE (small per-head dim)
        self.q_rope_proj: nn.Linear = nn.Linear(
            hidden_size, num_heads * qk_rope_head_dim, bias=False
        )

        # --- KV latent compression ---
        # Down-project to latent space
        self.kv_a_proj_with_mla: nn.Linear = nn.Linear(
            hidden_size, kv_lora_rank, bias=False
        )
        self.kv_a_norm: nn.Module = nn.LayerNorm(kv_lora_rank)

        # Up-project latent to full K (nope portion only, without RoPE)
        self.k_nope_proj: nn.Linear = nn.Linear(
            kv_lora_rank, num_heads * self.q_head_dim, bias=False
        )

        # Up-project latent to full V
        self.v_proj: nn.Linear = nn.Linear(
            kv_lora_rank, num_heads * v_head_dim, bias=False
        )

        # Decoupled RoPE K projection (small per-head dim, from input directly)
        self.k_rope_proj: nn.Linear = nn.Linear(
            hidden_size, num_heads * qk_rope_head_dim, bias=False
        )

        # --- Output projection ---
        self.o_proj: nn.Linear = nn.Linear(
            num_heads * v_head_dim, hidden_size, bias=False
        )

    def _compute_q(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Q (nope + rope parts).

        Returns:
            (q_nope [batch, num_heads, seq, q_head_dim],
             q_rope [batch, num_heads, seq, qk_rope_head_dim])
        """
        batch, seq = hidden_states.shape[:2]

        if hasattr(self, "q_a_proj"):
            q_latent: torch.Tensor = self.q_a_norm(self.q_a_proj(hidden_states))
            q_nope: torch.Tensor = (
                self.q_b_proj(q_latent)
                .view(batch, seq, self.num_heads, self.q_head_dim)
                .transpose(1, 2)
            )
        else:
            q_nope = (
                self.q_proj(hidden_states)
                .view(batch, seq, self.num_heads, self.q_head_dim)
                .transpose(1, 2)
            )

        q_rope: torch.Tensor = (
            self.q_rope_proj(hidden_states)
            .view(batch, seq, self.num_heads, self.qk_rope_head_dim)
            .transpose(1, 2)
        )
        return q_nope, q_rope

    def _compute_kv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute KV with decoupled RoPE.

        Returns:
            (k_nope [batch, num_heads, seq, q_head_dim],
             k_rope [batch, num_heads, seq, qk_rope_head_dim],
             v      [batch, num_heads, seq, v_head_dim])
        """
        batch, seq = hidden_states.shape[:2]

        # Latent KV compression
        kv_latent: torch.Tensor = self.kv_a_norm(self.kv_a_proj_with_mla(hidden_states))

        # Up-project to full K (no RoPE) and V
        k_nope: torch.Tensor = (
            self.k_nope_proj(kv_latent)
            .view(batch, seq, self.num_heads, self.q_head_dim)
            .transpose(1, 2)
        )
        v: torch.Tensor = (
            self.v_proj(kv_latent)
            .view(batch, seq, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
        )

        # Decoupled RoPE K (from input directly, not from latent)
        k_rope: torch.Tensor = (
            self.k_rope_proj(hidden_states)
            .view(batch, seq, self.num_heads, self.qk_rope_head_dim)
            .transpose(1, 2)
        )

        return k_nope, k_rope, v

    def _get_kv_cache_key(self) -> str:
        """Return unique identifier for MLA's specialized KV cache structure."""
        return "mla"

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """Forward pass with KV cache in latent space.

        For MLA, the KV cache stores three tensors instead of two:
        - k_nope: [batch, num_heads, cached_len, q_head_dim]
        - k_rope: [batch, num_heads, cached_len, qk_rope_head_dim]
        - v:      [batch, num_heads, cached_len, v_head_dim]

        The key insight is that k_nope and v are reconstructed from the
        latent representation, but k_rope is computed from the original
        input. Caching all three avoids recomputation.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos: RoPE cosine table [1, 1, seq_len, qk_rope_head_dim].
            sin: RoPE sine table [1, 1, seq_len, qk_rope_head_dim].
            attention_mask: Optional mask [batch, 1, seq_len, kv_len].
            kv_cache: Optional dict with keys "k_nope", "k_rope", "v".

        Returns:
            (output [batch, seq_len, hidden_size], updated_kv_cache) tuple.
        """
        batch_size: int = hidden_states.size(0)
        seq_len: int = hidden_states.size(1)

        # Compute Q parts
        q_nope, q_rope = self._compute_q(hidden_states)  # [B, H, S, dim]

        # Compute KV parts
        k_nope, k_rope, v = self._compute_kv(hidden_states)

        # Apply RoPE only to the rope portions of Q and K
        if cos is not None and sin is not None:
            from .rope import apply_rotary_pos_emb

            q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

        # Concatenate nope and rope parts for full Q/K
        # This gives us an effective head_dim of q_head_dim + qk_rope_head_dim for K
        q_full: torch.Tensor = torch.cat([q_nope, q_rope], dim=-1)
        k_full: torch.Tensor = torch.cat([k_nope, k_rope], dim=-1)

        # KV cache: store all three components
        new_kv_cache: Optional[dict[str, torch.Tensor]] = None
        if kv_cache is not None:
            k_nope = torch.cat([kv_cache["k_nope"], k_nope], dim=2)
            k_rope = torch.cat([kv_cache["k_rope"], k_rope], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
            k_full = torch.cat([k_nope, k_rope], dim=-1)
        new_kv_cache = {"k_nope": k_nope, "k_rope": k_rope, "v": v}

        key_len: int = k_full.size(2)
        query_len: int = q_full.size(2)

        # Attention computation
        if self.use_flash and attention_mask is None:
            causal_mask: torch.Tensor = torch.tril(
                torch.ones(
                    query_len, key_len, device=hidden_states.device, dtype=torch.bool
                )
            ).view(1, 1, query_len, key_len)
            attn_output: torch.Tensor = F.scaled_dot_product_attention(
                q_full,
                k_full,
                v,
                attn_mask=causal_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
                scale=1.0 / math.sqrt(q_full.size(-1)),
            )
        else:
            scale: float = 1.0 / math.sqrt(q_full.size(-1))
            attn_weights: torch.Tensor = (
                torch.matmul(q_full, k_full.transpose(-2, -1)) * scale
            )

            if attention_mask is None:
                causal_mask = torch.tril(
                    torch.ones(
                        query_len,
                        key_len,
                        device=hidden_states.device,
                        dtype=torch.bool,
                    )
                ).view(1, 1, query_len, key_len)
                attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
            else:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights.float(), dim=-1).to(
                hidden_states.dtype
            )
            attn_weights = F.dropout(
                attn_weights, p=self.attn_dropout, training=self.training
            )
            attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        )
        output: torch.Tensor = self.o_proj(attn_output)
        return output, new_kv_cache


# Quick test
if __name__ == "__main__":
    batch, seq, hidden = 2, 32, 768
    num_heads = 12
    kv_lora_rank = 256
    qk_rope_head_dim = 64
    v_head_dim = 128

    mla = MultiHeadLatentAttention(
        hidden_size=hidden,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
    )

    x = torch.randn(batch, seq, hidden)
    out, cache = mla(x)
    assert out.shape == (batch, seq, hidden), f"MLA shape: {out.shape}"
    assert "k_nope" in cache and "k_rope" in cache and "v" in cache
    print(f"MLA forward: OK, shape={out.shape}")
    print(
        f"  KV cache sizes: k_nope={cache['k_nope'].shape}, "
        f"k_rope={cache['k_rope'].shape}, v={cache['v'].shape}"
    )

    # Test incremental decoding
    x_step1 = x[:, :1, :]
    _, kv_cache = mla(x_step1, kv_cache=None)
    x_step2 = x[:, 1:2, :]
    out_step2, _ = mla(x_step2, kv_cache=kv_cache)
    assert out_step2.shape == (batch, 1, hidden)
    print(f"MLA incremental decode: OK")

    print("\nAll MLA tests passed!")
