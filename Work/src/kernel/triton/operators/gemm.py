"""Tiled GEMM: the canonical Triton kernel.

The tiling strategy that makes Triton worth learning, in miniature:

* the output C (M x N) is split into BLOCK_M x BLOCK_N tiles, one per program;
* the K dimension is walked in BLOCK_K chunks;
* each chunk loads an A tile and a B tile into SRAM and accumulates with
  ``tl.dot`` (which lowers to Tensor Core mma on fp16/bf16/tf32);
* C is stored once, at the end.

Contrast with a naive triple loop, which re-reads A/B from global memory per
output element.  The whole point is to keep the O(M*N*K) compute in registers
and only move O(M*N + K*(M+N)) data across the memory hierarchy.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import Op

BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
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
        # input_precision="ieee" keeps fp32 exactness (default "tf32" drops
        # mantissa bits).  For fp16/bf16 inputs tl.dot uses tensor cores and
        # this flag is ignored.
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def build():
    def inputs(device, dtype):
        M, N, K = 1024, 1024, 1024
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N, device=device, dtype=dtype)
        return a, b

    return Op(
        name="gemm",
        triton=triton_matmul,
        reference=lambda a, b: a @ b,
        inputs=inputs,
        note="tiled matmul %dx%dx%d, tl.dot over BLOCK_K=%d" % (BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_K),
    )
