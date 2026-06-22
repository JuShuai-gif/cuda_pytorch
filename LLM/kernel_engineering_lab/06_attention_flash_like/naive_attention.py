"""
Naive attention implementations without tiling or memory optimization.

Provides both pure PyTorch and Triton naive implementations that materialize
the full NxN attention matrix in HBM. This is the baseline that FlashAttention
and tiled attention improve upon.

The standard attention formula:
    S = Q @ K^T / sqrt(d)
    P = softmax(S)
    O = P @ V

Both implementations materialize S (seq_len x seq_len), making them O(N^2)
in memory. For a 4096-token sequence with fp32, S is 4096 * 4096 * 4 = 64 MB
per head. With 32 heads, that's 2 GB just for the attention matrix.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pure PyTorch naive attention
# ---------------------------------------------------------------------------


def naive_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Naive PyTorch attention: Q @ K^T -> softmax -> @ V.

    Materializes the full NxN attention matrix in HBM.

    Args:
        q: Query tensor [batch, num_heads, q_len, head_dim]
        k: Key tensor [batch, num_heads, kv_len, head_dim]
        v: Value tensor [batch, num_heads, kv_len, head_dim]
        causal_mask: If True, mask the upper triangle (i > j).
        scale: Scale factor for Q @ K^T. Defaults to 1 / sqrt(head_dim).

    Returns:
        Output tensor [batch, num_heads, q_len, head_dim]
    """
    B, H, Q_len, D = q.shape
    _, _, KV_len, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # S = Q @ K^T * scale  -> [B, H, Q_len, KV_len]
    s = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal_mask:
        causal = torch.triu(
            torch.ones(Q_len, KV_len, device=s.device, dtype=torch.bool),
            diagonal=KV_len - Q_len + 1 if KV_len > Q_len else 1,
        )
        s = s.masked_fill(causal, float("-inf"))

    # P = softmax(S)
    p = torch.softmax(s, dim=-1)

    # O = P @ V
    o = torch.matmul(p, v)
    return o


# ---------------------------------------------------------------------------
# Triton naive attention (materialize full attention matrix)
# ---------------------------------------------------------------------------


@triton.jit
def _naive_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
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
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Naive Triton attention: each program computes one head-row.

    This kernel processes a single Q row against all K/V rows using
    online softmax. The dot product is accumulated across all D
    dimensions (iterating in BLOCK_D chunks), then softmax is applied
    to the full score. V accumulation is done in BLOCK_D chunks.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    q_offs = pid_b * stride_qb + pid_h * stride_qh + pid_m * stride_qm
    k_base = k_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = v_ptr + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh + pid_m * stride_om

    # Online softmax state (scalar: score is a single number after full D dot product)
    running_max = float("-inf")
    exp_sum = 0.0

    # Output accumulator: we need D elements, collect across KV iterations
    # Use a separate inner loop to accumulate V contributions per D chunk
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # The V-weighted output for this D chunk
        weighted_v_chunk = tl.zeros([BLOCK_D], dtype=tl.float32)

        for j in range(0, KV_len):
            # Compute full dot product score = sum_d(Q[m,d] * K[j,d]) * scale
            score = 0.0
            for dd_start in range(0, D, BLOCK_D):
                dd_offs = dd_start + tl.arange(0, BLOCK_D)
                dd_mask = dd_offs < D
                q_chunk = tl.load(q_ptr + q_offs + dd_offs * stride_qd, mask=dd_mask, other=0.0).to(
                    tl.float32
                )
                k_chunk = tl.load(
                    k_base + j * stride_km + dd_offs * stride_kd, mask=dd_mask, other=0.0
                ).to(tl.float32)
                score += tl.sum(q_chunk * k_chunk, axis=0)
            score = score * scale

            if CAUSAL:
                score = tl.where(j > pid_m, float("-inf"), score)

            score_f = score.to(tl.float32)
            prev_max = running_max
            running_max = tl.maximum(running_max, score_f)

            # Rescale previous accumulation
            correction = tl.exp(prev_max - running_max)
            weighted_v_chunk = weighted_v_chunk * correction
            exp_sum = exp_sum * correction

            exp_score = tl.exp(score_f - running_max)
            exp_sum += exp_score

            # Load V slice for the current D chunk and accumulate
            v_chunk = tl.load(
                v_base + j * stride_vm + d_offs * stride_vd, mask=d_mask, other=0.0
            ).to(tl.float32)
            weighted_v_chunk += v_chunk * exp_score

        # Normalize and write this D chunk
        result_chunk = weighted_v_chunk / exp_sum
        tl.store(o_base + d_offs * stride_od, result_chunk.to(O.dtype.element_ty), mask=d_mask)


def naive_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: bool = False,
    scale: Optional[float] = None,
    BLOCK_D: int = 64,
) -> torch.Tensor:
    """Naive Triton attention that materializes full attention weights per row.

    Each program handles one query position. It loads all K and V positions
    sequentially and maintains the running softmax with online rescaling,
    but still processes the full KV_len span per query row within a single
    program. This demonstrates the O(N^2) per-head computation pattern
    without tiling in the Q dimension.

    Args:
        q: Query tensor [batch, num_heads, q_len, head_dim]
        k: Key tensor [batch, num_heads, kv_len, head_dim]
        v: Value tensor [batch, num_heads, kv_len, head_dim]
        causal_mask: If True, mask the upper triangle.
        scale: Scale factor. Defaults to 1 / sqrt(head_dim).
        BLOCK_D: Block size for the head_dim dimension.

    Returns:
        Output tensor [batch, num_heads, q_len, head_dim]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA"
    B, H, Q_len, D = q.shape
    KV_len = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    o = torch.empty_like(q)

    grid = (B, H, Q_len)

    _naive_attention_kernel[grid](
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
        BLOCK_D=BLOCK_D,
        CAUSAL=causal_mask,
    )
    return o


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        B, H, Q_len, KV_len, D = 2, 4, 64, 64, 64
        q = torch.randn(B, H, Q_len, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)

        o_torch = naive_attention_torch(q, k, v)
        o_triton = naive_attention_triton(q, k, v)
        err = (o_triton - o_torch).abs().max().item()
        print(f"Naive attention {B}x{H}x{Q_len}x{KV_len}x{D} - max error: {err:.2e}")

        # With causal mask
        o_torch_c = naive_attention_torch(q, k, v, causal_mask=True)
        o_triton_c = naive_attention_triton(q, k, v, causal_mask=True)
        err_c = (o_triton_c - o_torch_c).abs().max().item()
        print(f"Naive causal attention - max error: {err_c:.2e}")
