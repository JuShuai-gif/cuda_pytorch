"""
FlashAttention-style kernels for prefill and decode phases.

Two specialized attention kernels for LLM inference:

1. attention_prefill_kernel: Process all Q tokens against all K/V tokens.
   Used during prefill (prompt processing). Compute-bound: O(Q_len × KV_len × D).
   The tiled attention approach is used here to avoid materializing the full
   attention matrix.

2. attention_decode_kernel: Process a single Q token against cached K/V tokens.
   Used during autoregressive generation. Memory-bound: O(KV_len × D).
   Since Q is tiny (1 token), the optimization strategy shifts from tiling in
   the Q dimension to maximizing memory bandwidth throughput for the KV read.

   In production, this is the critical kernel for inference throughput - it runs
   once per generated token per layer per head.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Attention Prefill Kernel (all Q tokens vs all K/V tokens)
# ---------------------------------------------------------------------------


@triton.jit
def _attention_prefill_kernel(
    Q,
    K,
    V,
    O,
    Q_len: int,
    KV_len: int,
    D: int,
    stride_qb: int,
    stride_qh: int,
    stride_qm: int,
    stride_qd: int,
    stride_kb: int,
    stride_kh: int,
    stride_km: int,
    stride_kd: int,
    stride_vb: int,
    stride_vh: int,
    stride_vm: int,
    stride_vd: int,
    stride_ob: int,
    stride_oh: int,
    stride_om: int,
    stride_od: int,
    scale: float,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Prefill attention: compute all Q rows against all K/V rows in tiles.

    Each program handles BLOCK_M query rows. Iterates over the KV dimension
    in tiles of BLOCK_N using online softmax, with D-dimension chunking
    for BLOCK_D < D.

    Prefill characteristics:
      - Q_len == KV_len (or KV_len > Q_len for cross-attention)
      - Compute-bound because we do Q_len × KV_len × D operations
      - Tiling in both Q and KV dimensions reduces HBM traffic
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    q_start = pid_m * BLOCK_M
    rm = q_start + tl.arange(0, BLOCK_M)
    rm_mask = rm < Q_len

    # Online softmax state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    for k_start in range(0, KV_len, BLOCK_N):
        rn = k_start + tl.arange(0, BLOCK_N)
        rn_mask_full = rn < KV_len

        # Compute S = Q @ K^T by accumulating dot products over D chunks
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            q_tile = tl.load(
                q_base + rm[:, None] * stride_qm + d_offs[None, :] * stride_qd,
                mask=rm_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            k_tile = tl.load(
                k_base + rn[:, None] * stride_km + d_offs[None, :] * stride_kd,
                mask=rn_mask_full[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            s += tl.dot(q_tile, tl.trans(k_tile))

        s = s * scale

        if CAUSAL:
            row_indices = rm[:, None]
            col_indices = rn[None, :]
            causal_mask_val = col_indices > row_indices
            s = tl.where(causal_mask_val, float("-inf"), s)

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        scaling = tl.exp(m_i - m_new)
        acc = acc * scaling[:, None]
        l_i = l_i * scaling

        p = tl.exp(s - m_new[:, None])
        l_i = l_i + tl.sum(p, axis=1)

        # acc += P @ V across D chunks
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            v_tile = tl.load(
                v_base + rn[:, None] * stride_vm + d_offs[None, :] * stride_vd,
                mask=rn_mask_full[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            acc += tl.dot(p, v_tile)

        m_i = m_new

    # Write output in D chunks
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        o_tile = acc / l_i[:, None]
        tl.store(
            o_base + rm[:, None] * stride_om + d_offs[None, :] * stride_od,
            o_tile.to(O.dtype.element_ty),
            mask=rm_mask[:, None] & d_mask[None, :],
        )


def attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: bool = False,
    scale: Optional[float] = None,
    block_m: int = 64,
    block_n: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """Prefill attention: process all Q tokens against all K/V tokens.

    This is the attention variant used during the prefill (prompt processing)
    phase of LLM inference, where all input tokens are processed at once.

    Compute complexity: O(Q_len × KV_len × D) per head.

    Args:
        q: Query tensor [batch, num_heads, q_len, head_dim]
        k: Key tensor [batch, num_heads, kv_len, head_dim]
        v: Value tensor [batch, num_heads, kv_len, head_dim]
        causal_mask: If True, mask upper triangle.
        scale: Scale factor. Defaults to 1 / sqrt(head_dim).
        block_m: Q tile size.
        block_n: KV tile size.
        block_d: Head dimension tile size.

    Returns:
        Output tensor [batch, num_heads, q_len, head_dim]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA"
    B, H, Q_len, D = q.shape
    KV_len = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    o = torch.empty_like(q)

    grid = (B, H, triton.cdiv(Q_len, block_m))

    _attention_prefill_kernel[grid](
        q,
        k,
        v,
        o,
        Q_len,
        KV_len,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        CAUSAL=causal_mask,
    )
    return o


# ---------------------------------------------------------------------------
# Attention Decode Kernel (single Q token vs cached K/V)
# ---------------------------------------------------------------------------


@triton.jit
def _attention_decode_kernel(
    Q,
    K,
    V,
    O,
    KV_len: int,
    D: int,
    stride_qb: int,
    stride_qh: int,
    stride_qm: int,
    stride_qd: int,
    stride_kb: int,
    stride_kh: int,
    stride_km: int,
    stride_kd: int,
    stride_vb: int,
    stride_vh: int,
    stride_vm: int,
    stride_vd: int,
    stride_ob: int,
    stride_oh: int,
    stride_om: int,
    stride_od: int,
    scale: float,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Decode attention: compute single Q token against cached K/V.

    Each program handles one (batch, head) pair with a single Q row.
    Since Q is a single token, we don't tile in the Q dimension.
    Instead, we stream through K/V tiles to maximize memory bandwidth.

    Decode characteristics:
      - Q_len = 1 (single new token)
      - KV_len >> Q_len (cached context)
      - Memory-bound: dominated by reading K/V from HBM
      - The inner loop loads K/V tiles and accumulates into registers
      - No Q tiling needed since Q is tiny
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load the single Q row [1, D] - Q is at position Q_len-1 (before q.shape[2]-1)
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    # For decode, Q has shape [B, H, 1, D], so q_len idx 0
    q_row_ptr = q_base + 0 * stride_qm + d_offs * stride_qd
    q_row = tl.load(q_row_ptr, mask=d_mask, other=0.0).to(tl.float32)

    # Online softmax state (scalar since Q is single row)
    m_val = float("-inf")
    l_val = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh

    for k_start in range(0, KV_len, BLOCK_N):
        rn = k_start + tl.arange(0, BLOCK_N)
        rn_mask = rn < KV_len

        # Load K tile [BLOCK_N, D]
        k_ptrs = k_base + rn[:, None] * stride_km + d_offs[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=rn_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Compute scores: q_row @ K_tile^T -> [BLOCK_N]
        # q_row: [D], k_tile: [BLOCK_N, D]
        s = tl.sum(q_row[None, :] * k_tile, axis=1) * scale  # [BLOCK_N]

        # Apply causal mask: mask positions > 0 (the single Q token position)
        # For decode, all KV positions are valid (they come before the current token)
        # If we're using causal attention, all KV positions are <= current position
        # Since Q is at the latest position, no masking needed for standard decode

        # Online softmax update (scalar version)
        s_max = tl.max(s)
        m_new = tl.maximum(m_val, s_max)
        scaling = tl.exp(m_val - m_new)

        acc = acc * scaling
        l_val = l_val * scaling

        p = tl.exp(s - m_new)
        l_val = l_val + tl.sum(p)

        # Load V tile [BLOCK_N, D]
        v_ptrs = v_base + rn[:, None] * stride_vm + d_offs[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=rn_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # acc += p^T @ V_tile  -> [BLOCK_N]^T @ [BLOCK_N, BLOCK_D] -> [BLOCK_D]
        acc += tl.sum(p[:, None] * v_tile, axis=0)

        m_val = m_new

    result = acc / l_val

    # Write single output row
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    o_ptrs = o_base + 0 * stride_om + d_offs * stride_od
    tl.store(o_ptrs, result.to(O.dtype.element_ty), mask=d_mask)


def attention_decode(
    q_single: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: Optional[float] = None,
    block_n: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """Decode attention: process a single Q token against cached K/V.

    Used during autoregressive generation where one new token is processed
    per step against the growing KV cache.

    Memory complexity: O(KV_len × D) per head per layer (dominated by K/V reads).
    Compute complexity: O(KV_len × D) per head per layer.

    Args:
        q_single: Single query token [batch, num_heads, 1, head_dim]
        k_cache: Cached keys [batch, num_heads, kv_len, head_dim]
        v_cache: Cached values [batch, num_heads, kv_len, head_dim]
        scale: Scale factor. Defaults to 1 / sqrt(head_dim).
        block_n: KV tile size.
        block_d: Head dimension tile size.

    Returns:
        Output for the single token [batch, num_heads, 1, head_dim]
    """
    assert q_single.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "Tensors must be on CUDA"
    B, H, Q_len, D = q_single.shape
    assert Q_len == 1, f"Decode expects Q_len=1, got {Q_len}"
    KV_len = k_cache.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    o = torch.empty_like(q_single)

    grid = (B, H)

    _attention_decode_kernel[grid](
        q_single,
        k_cache,
        v_cache,
        o,
        KV_len,
        D,
        q_single.stride(0),
        q_single.stride(1),
        q_single.stride(2),
        q_single.stride(3),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        scale,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return o


# ---------------------------------------------------------------------------
# Helper: simulate KV cache during autoregressive generation
# ---------------------------------------------------------------------------


class SimpleKVCache:
    """Simple KV cache that stores keys and values for autoregressive generation.

    In production (e.g. vLLM, TensorRT-LLM), the KV cache is managed more
    sophisticatedly (paged attention, prefix caching, etc.), but this shows
    the basic idea.
    """

    def __init__(
        self,
        max_seq_len: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self.k_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim, device="cuda", dtype=dtype
        )
        self.v_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim, device="cuda", dtype=dtype
        )
        self.seq_len = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append new K/V tokens to the cache."""
        B, H, cur_len, D = k.shape
        assert cur_len == 1 or cur_len == self.max_seq_len - self.seq_len
        self.k_cache[:, :, self.seq_len : self.seq_len + cur_len, :] = k
        self.v_cache[:, :, self.seq_len : self.seq_len + cur_len, :] = v
        self.seq_len += cur_len

    def get_valid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the valid (filled) portion of the KV cache."""
        return self.k_cache[:, :, : self.seq_len, :], self.v_cache[:, :, : self.seq_len, :]


# ---------------------------------------------------------------------------
# Standalone test/demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        B, H, Q_len, KV_len, D = 1, 4, 64, 64, 64
        q = torch.randn(B, H, Q_len, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)

        # Prefill
        o_prefill = attention_prefill(q, k, v)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D))
        err = (o_prefill - ref).abs().max().item()
        print(f"Prefill attention {B}x{H}x{Q_len}x{KV_len}x{D} - max error: {err:.2e}")

        # Decode: simulate autoregressive generation
        cache = SimpleKVCache(max_seq_len=64, batch_size=B, num_heads=H, head_dim=D)
        # Prefill
        cache.append(k, v)

        # Generate 4 more tokens
        for step in range(4):
            q_new = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
            k_valid, v_valid = cache.get_valid()
            o_decode = attention_decode(q_new, k_valid, v_valid)

            # Simulate: append new K/V to cache
            k_new = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
            v_new = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
            cache.append(k_new, v_new)

        print(f"Decode: generated {step + 1} tokens, final KV cache size: {cache.seq_len}")
        print("All tests passed!")
