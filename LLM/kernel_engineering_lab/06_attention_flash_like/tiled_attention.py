"""
Tiled attention with online softmax in Triton.

Implements the core idea behind FlashAttention: process attention in tiles
(blocks of Q rows × blocks of K/V columns) using online softmax to avoid
materializing the full N×N attention matrix in HBM.

The online softmax algorithm maintains running maximum and sum for numerically
stable softmax computation across tiles. This is the breakthrough that made
FlashAttention possible.

Algorithm sketch:
  1. Split Q into tiles of BLOCK_M rows each
  2. For each Q tile:
     a. Load Q tile into shared memory
     b. Initialize running max m (size BLOCK_M), running sum l (size BLOCK_M),
        output accumulator acc (BLOCK_M x D)
     c. For each KV tile of size BLOCK_N:
        i.   Load K tile, V tile into shared memory
        ii.  Compute S = Q_tile @ K_tile^T (BLOCK_M x BLOCK_N)
        iii. Apply causal mask if needed
        iv.  Update m_new = max(m_old, row_max(S))
        v.   Rescale acc: acc *= exp(m_old - m_new)
        vi.  Compute P = exp(S - m_new), l_new = l * exp(m_old - m_new) + sum(P, dim=1)
        vii. Update acc: acc += P @ V_tile
        viii.Update m = m_new, l = l_new
     d. Normalize output: O_tile = acc / l[:, None]
     e. Write O_tile to HBM

Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness", NeurIPS 2022.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _tiled_attention_forward_kernel(
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
    """Tiled attention with online softmax.

    Each program handles BLOCK_M query rows for one (batch, head) pair.
    Iterates over KV_len in tiles of BLOCK_N, with an inner D-dimension
    loop in BLOCK_D chunks, updating running softmax statistics without
    ever materializing the full N×N attention matrix.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Compute the range of query rows this program handles
    q_start = pid_m * BLOCK_M
    rm = q_start + tl.arange(0, BLOCK_M)
    rm_mask = rm < Q_len

    # Base pointers
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    # Initialize online softmax state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Iterate over KV_len in tiles of BLOCK_N
    for k_start in range(0, KV_len, BLOCK_N):
        rn = k_start + tl.arange(0, BLOCK_N)
        rn_mask_full = rn < KV_len

        # Compute S = Q @ K^T * scale by accumulating dot products over D chunks
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            combined_mask = rm_mask[:, None] & d_mask[None, :]

            # Q tile: [BLOCK_M, BLOCK_D]
            q_tile = tl.load(
                q_base + rm[:, None] * stride_qm + d_offs[None, :] * stride_qd,
                mask=combined_mask,
                other=0.0,
            ).to(tl.float32)

            # K tile: [BLOCK_N, BLOCK_D]
            k_tile = tl.load(
                k_base + rn[:, None] * stride_km + d_offs[None, :] * stride_kd,
                mask=rn_mask_full[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Partial dot product: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            s += tl.dot(q_tile, tl.trans(k_tile))

        s = s * scale

        # Apply causal mask if needed
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

        # V @ P accumulation: iterate over D chunks
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            v_tile = tl.load(
                v_base + rn[:, None] * stride_vm + d_offs[None, :] * stride_vd,
                mask=rn_mask_full[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # p: [BLOCK_M, BLOCK_N], v_tile: [BLOCK_N, BLOCK_D]
            # acc += p @ v_tile -> [BLOCK_M, BLOCK_D]
            acc += tl.dot(p, v_tile)

        m_i = m_new

    # Final normalization and write output in D chunks
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask_chunk = d_offs < D

        o_tile = acc / l_i[:, None]
        tl.store(
            o_base + rm[:, None] * stride_om + d_offs[None, :] * stride_od,
            o_tile.to(O.dtype.element_ty),
            mask=rm_mask[:, None] & d_mask_chunk[None, :],
        )


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: bool = False,
    scale: Optional[float] = None,
    block_m: int = 64,
    block_n: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """Tiled attention with online softmax.

    Processes attention in tiles to avoid materializing the full attention
    matrix. Uses online softmax for numerical stability.

    Args:
        q: Query tensor [batch, num_heads, q_len, head_dim]
        k: Key tensor [batch, num_heads, kv_len, head_dim]
        v: Value tensor [batch, num_heads, kv_len, head_dim]
        causal_mask: If True, mask upper triangle (j > i).
        scale: Scale factor. Defaults to 1 / sqrt(head_dim).
        block_m: Tile size for Q rows (BLOCK_M).
        block_n: Tile size for KV positions (BLOCK_N).
        block_d: Tile size for head_dim (BLOCK_D).

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

    _tiled_attention_forward_kernel[grid](
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
# Reference implementation for verification
# ---------------------------------------------------------------------------


def _scaled_dot_product_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference implementation using torch.nn.functional.scaled_dot_product_attention."""
    is_causal = causal_mask and q.shape[2] == k.shape[2]
    attn_mask = None
    if causal_mask and not is_causal:
        Q_len = q.shape[2]
        KV_len = k.shape[2]
        m = torch.ones(Q_len, KV_len, device=q.device, dtype=torch.bool)
        m = torch.triu(m, diagonal=KV_len - Q_len + 1 if KV_len > Q_len else 1)
        attn_mask = m

    if attn_mask is not None:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=scale, dropout_p=0.0
        )
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, scale=scale, dropout_p=0.0
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        B, H, Q_len, KV_len, D = 2, 4, 128, 128, 64
        q = torch.randn(B, H, Q_len, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)

        o_tiled = tiled_attention(q, k, v)
        o_ref = _scaled_dot_product_attention_ref(q, k, v)
        err = (o_tiled - o_ref).abs().max().item()
        print(f"Tiled attention {B}x{H}x{Q_len}x{KV_len}x{D} - max error vs torch: {err:.2e}")

        # Causal
        o_tiled_c = tiled_attention(q, k, v, causal_mask=True)
        o_ref_c = _scaled_dot_product_attention_ref(q, k, v, causal_mask=True)
        err_c = (o_tiled_c - o_ref_c).abs().max().item()
        print(f"Tiled causal attention - max error vs torch: {err_c:.2e}")

        # Decode pattern: Q_len=1, KV_len=128
        q_dec = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
        o_dec = tiled_attention(q_dec, k, v)
        o_dec_ref = _scaled_dot_product_attention_ref(q_dec, k, v)
        err_dec = (o_dec - o_dec_ref).abs().max().item()
        print(f"Tiled decode attention Q=1 KV=128 - max error vs torch: {err_dec:.2e}")
