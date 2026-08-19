"""Attention: naive (materialized) vs flash (online softmax).

The key insight of flash attention is that the N x N attention matrix
``softmax(Q @ K^T / sqrt(d))`` is never materialized.  Instead the kernel
streams over K/V tiles and keeps a running max/denominator (online softmax),
so memory stays O(N) instead of O(N^2).  This is what makes long-sequence
inference feasible on a single GPU.

The Triton kernel here is a from-scratch forward pass.  ``reference`` uses
PyTorch's ``scaled_dot_product_attention`` (which internally calls an
optimized flash kernel) as the correctness oracle.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from . import Op

BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, 64


@triton.jit
def _flash_fwd_kernel(
    Q, K, V, Out,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
    stride_on, stride_od,
    N_CTX, D,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)  # batch * heads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + off_hz * stride_qn * N_CTX + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_d[None, :] < D, other=0.0)  # (BLOCK_M, BLOCK_D)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_cur = start_n + offs_n
        k_ptrs = K + off_hz * stride_kn * N_CTX + offs_n_cur[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_d[None, :] < D, other=0.0)  # (BLOCK_N, BLOCK_D)

        s = tl.dot(q, tl.trans(k)) * sm_scale  # (BLOCK_M, BLOCK_N)

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = V + off_hz * stride_vn * N_CTX + offs_n_cur[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_d[None, :] < D, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    out_ptrs = Out + off_hz * stride_on * N_CTX + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_d[None, :] < D)


def triton_flash(q, k, v, sm_scale):
    B, N, D = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(N, BLOCK_M), B)
    _flash_fwd_kernel[grid](
        q, k, v, out,
        q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1), v.stride(2),
        out.stride(1), out.stride(2),
        N, D, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return out


def build():
    def inputs(device, dtype):
        B, N, D = 4, 512, 64
        q = torch.randn(B, N, D, device=device, dtype=dtype)
        k = torch.randn(B, N, D, device=device, dtype=dtype)
        v = torch.randn(B, N, D, device=device, dtype=dtype)
        return q, k, v

    def reference(q, k, v):
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale)

    def triton_fn(q, k, v):
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        return triton_flash(q, k, v, sm_scale)

    return Op(
        name="flash_attention",
        triton=triton_fn,
        reference=reference,
        inputs=inputs,
        note="online softmax, O(N) memory vs O(N^2) naive; N=%d D=%d" % (512, 64),
    )
