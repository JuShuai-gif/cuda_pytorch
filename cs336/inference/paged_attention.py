"""
PagedAttention forward implementation with non-contiguous KV cache blocks.

Computes scaled dot-product attention by gathering K/V from
non-contiguous physical blocks using a block table for address
translation. Supports:
  - Variable block sizes (16, 32, 64 tokens per block)
  - GQA (Grouped-Query Attention) with MQA fallback
  - PyTorch fallback when Triton is unavailable
  - Integration with KVCacheManager block tables
  - Optimized gather/scatter for paged memory access

For the compute-bound prefill phase, standard FlashAttention is preferred.
PagedAttention is most beneficial during the memory-bound decode phase
(arith intensity = 52 FLOP/byte << 295 FLOP/byte GPU ridge point).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ==============================================================================
#  PagedAttention class
# ==============================================================================


class PagedAttention:
    """Manages paged attention computation with block table translation.

    Handles the gather step: for each query, collects K/V from the
    non-contiguous blocks specified by the block table, then computes
    standard scaled dot-product attention.

    Args:
        block_size: Number of tokens per KV cache block.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value attention heads (for GQA).
        head_dim: Dimension per attention head.
        sm_scale: Softmax scale factor (default: 1/sqrt(head_dim)).
        device: Device for tensor operations.
    """

    def __init__(
        self,
        block_size: int = 16,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        sm_scale: float | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.block_size = block_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)
        self.device = torch.device(device)

    def forward(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute paged attention for batched queries.

        Args:
            q: Query tensor of shape (batch, n_heads, head_dim) for decode
               (single query token per sequence).
            k_pool: Paged K cache of shape
                    (num_blocks, n_kv_heads, block_size, head_dim).
            v_pool: Paged V cache of shape
                    (num_blocks, n_kv_heads, block_size, head_dim).
            block_table: Block table of shape (batch, max_blocks_per_seq).
                         Maps logical block -> physical block. -1 for unused.
            seq_lens: Sequence lengths of shape (batch,). Total tokens stored
                      in KV cache for each sequence.

        Returns:
            Output tensor of shape (batch, n_heads, head_dim).

        Note:
            Uses the Triton kernel from cs336.cuda.paged_attention when
            available, falling back to a PyTorch reference implementation.
        """
        return paged_attention_forward(
            q=q,
            k_pool=k_pool,
            v_pool=v_pool,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=self.block_size,
            sm_scale=self.sm_scale,
        )

    def prefill_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard contiguous attention for the prefill phase.

        During prefill, KV tensors are newly computed and contiguous,
        so paged gather is unnecessary. Use standard scaled dot-product
        attention or FlashAttention.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim).
            k: Key tensor of shape (batch, n_kv_heads, kv_seq_len, head_dim).
            v: Value tensor of shape (batch, n_kv_heads, kv_seq_len, head_dim).
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch, n_heads, seq_len, head_dim).
        """
        batch, n_heads, q_len, head_dim = q.shape
        _, n_kv_heads, kv_len, _ = k.shape

        # Handle GQA: expand KV heads to match Q heads
        if n_heads != n_kv_heads:
            ratio = n_heads // n_kv_heads
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.sm_scale

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


# ==============================================================================
#  PyTorch fallback implementation
# ==============================================================================


def _paged_attention_pytorch(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """PyTorch reference for paged attention with block table translation.

    For each sequence in the batch:
      1. Iterate through block table entries
      2. Gather K/V chunks from physical blocks
      3. Concatenate into contiguous K/V tensors
      4. Compute scaled dot-product attention
      5. Handle GQA by expanding KV heads

    Args:
        q: Query (batch, n_heads, head_dim) -- single token per sequence.
        k_pool: Paged K cache (num_blocks, n_kv_heads, block_size, head_dim).
        v_pool: Paged V cache (num_blocks, n_kv_heads, block_size, head_dim).
        block_table: (batch, max_blocks_per_seq).
        seq_lens: (batch,).
        block_size: Tokens per KV cache block.
        sm_scale: Softmax scale.

    Returns:
        Output (batch, n_heads, head_dim).
    """
    batch, n_heads, head_dim = q.shape
    n_kv_heads = k_pool.shape[1]
    device = q.device
    dtype = q.dtype

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    max_blocks = block_table.shape[1]
    outputs: list[torch.Tensor] = []

    for b in range(batch):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            outputs.append(torch.zeros(n_heads, head_dim, device=device, dtype=dtype))
            continue

        # Gather K/V from blocks
        k_chunks: list[torch.Tensor] = []
        v_chunks: list[torch.Tensor] = []
        remaining = seq_len

        for block_idx in range(max_blocks):
            if remaining <= 0:
                break
            phys_block = int(block_table[b, block_idx].item())
            if phys_block < 0:
                break

            take = min(block_size, remaining)
            k_chunks.append(k_pool[phys_block, :, :take, :])
            v_chunks.append(v_pool[phys_block, :, :take, :])
            remaining -= take

        if not k_chunks:
            outputs.append(torch.zeros(n_heads, head_dim, device=device, dtype=dtype))
            continue

        k_seq = torch.cat(k_chunks, dim=1)  # (n_kv_heads, seq_len, head_dim)
        v_seq = torch.cat(v_chunks, dim=1)

        # GQA expansion
        if n_heads != n_kv_heads:
            ratio = n_heads // n_kv_heads
            k_seq = k_seq.repeat_interleave(ratio, dim=0)
            v_seq = v_seq.repeat_interleave(ratio, dim=0)

        q_b = q[b : b + 1]  # (1, n_heads, head_dim)

        scores = torch.matmul(q_b, k_seq.transpose(-2, -1)) * sm_scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_seq)
        outputs.append(out.squeeze(0))

    return torch.stack(outputs, dim=0)


# ==============================================================================
#  Main forward entry point
# ==============================================================================


def paged_attention_forward(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Compute paged attention with automatic kernel selection.

    Attempts to use the optimized Triton kernel from cs336.cuda.paged_attention.
    Falls back to PyTorch reference implementation when Triton is unavailable.

    Args:
        q: Query tensor (batch, n_heads, head_dim).
        k_pool: Paged key cache (num_blocks, n_kv_heads, block_size, head_dim).
        v_pool: Paged value cache (num_blocks, n_kv_heads, block_size, head_dim).
        block_table: Block mapping (batch, max_blocks). -1 for unused.
        seq_lens: Sequence lengths (batch,).
        block_size: Tokens per cache block.
        sm_scale: Softmax scale (default: 1/sqrt(head_dim)).

    Returns:
        Attention output of shape (batch, n_heads, head_dim).
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))

    # Try the cs336.cuda Triton kernel first
    try:
        from cs336.cuda.paged_attention import paged_attention_kernel

        return paged_attention_kernel(
            q=q,
            k_cache=k_pool,
            v_cache=v_pool,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=block_size,
            sm_scale=sm_scale,
        )
    except (ImportError, Exception):
        pass

    return _paged_attention_pytorch(
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
        sm_scale=sm_scale,
    )
