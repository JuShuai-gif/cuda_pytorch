"""Fused vs unfused operator implementations.

Four cases that cover the classic fusion wins:

1. bias+relu     : linear followed by ReLU (elementwise op folded into gemm)
2. residual+rmsnorm : x + r followed by RMSNorm (avoid materializing the sum)
3. gemm+bias     : matmul with the bias added inside the accumulation epilogue
4. dequant+gemm  : int8 weight dequantized in SRAM, then tensor-core dot

Each ``unfused`` is eager PyTorch; each ``fused`` is a Triton kernel that does
the whole thing in one pass.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import FusionCase

BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
BLOCK = 1024


# --------------------------------------------------------------------------
# 1. bias + relu
# --------------------------------------------------------------------------
@triton.jit
def _linear_relu_kernel(
    a_ptr, w_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_wk, stride_wn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)  # relu
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def fused_linear_relu(a, w, b):
    M, K = a.shape
    N = w.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    wt = w.t()  # F.linear's w is (N, K); the kernel expects (K, N) like gemm
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _linear_relu_kernel[grid](
        a, wt, b, c, M, N, K,
        a.stride(0), a.stride(1), wt.stride(0), wt.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


# --------------------------------------------------------------------------
# 2. residual + rmsnorm
# --------------------------------------------------------------------------
@triton.jit
def _residual_rmsnorm_kernel(
    x_ptr, r_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = x_ptr + row * n_cols
    x = tl.load(base + cols, mask=mask, other=0.0)
    r = tl.load(r_ptr + row * n_cols + cols, mask=mask, other=0.0)
    y = x + r
    ms = tl.sum(y * y, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * n_cols + cols, y * rstd * w, mask=mask)


def fused_residual_rmsnorm(x, r, w, eps):
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _residual_rmsnorm_kernel[(n_rows,)](x, r, w, out, n_cols, eps, BLOCK=BLOCK)
    return out


# --------------------------------------------------------------------------
# 3. gemm + bias
# --------------------------------------------------------------------------
@triton.jit
def _gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def fused_gemm_bias(a, b, bias):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _gemm_bias_kernel[grid](
        a, b, bias, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


# --------------------------------------------------------------------------
# 4. dequant + gemm (weight-only int8)
# --------------------------------------------------------------------------
@triton.jit
def _dequant_gemm_kernel(
    a_ptr, wq_ptr, ws_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_wk, stride_wn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = wq_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        wq = tl.load(w_ptrs, mask=offs_k[:, None] < K - k, other=0.0).to(tl.float32)
        # per-channel scale broadcast across K, dequantized to activation dtype
        ws = tl.load(ws_ptr + offs_n, mask=offs_n < N, other=1.0)
        w = (wq * ws[None, :]).to(a_ptr.dtype.element_ty)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def fused_dequant_gemm(a, wq, ws):
    M, K = a.shape
    N = wq.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _dequant_gemm_kernel[grid](
        a, wq, ws, c, M, N, K,
        a.stride(0), a.stride(1), wq.stride(0), wq.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
