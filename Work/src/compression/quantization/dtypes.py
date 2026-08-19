"""Numeric format comparison: FP32 / TF32 / FP16 / BF16.

Two things matter for a format: its representable range (min/max normal) and
its precision (how many mantissa bits), plus the hardware path it takes in a
GEMM.  This module measures (1) the format metadata and (2) the GEMM
accuracy/speed of each format on a fixed matrix product, so "why INT8/FP16 is
sometimes faster and sometimes not" has concrete numbers behind it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from common.measure import cuda_event_latency


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr,
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
        if IEEE:
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def triton_gemm(a: torch.Tensor, b: torch.Tensor, ieee: bool = False) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    _gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, IEEE=ieee,
    )
    return c


def format_metadata() -> list[dict]:
    # mantissa/exponent split is fixed per IEEE format (not exposed by finfo).
    specs = {
        "fp32": (torch.float32, 23, 8),
        "fp16": (torch.float16, 10, 5),
        "bf16": (torch.bfloat16, 7, 8),
    }
    out = []
    for name, (dt, mantissa, exponent) in specs.items():
        info = torch.finfo(dt)
        out.append({
            "name": name,
            "bits": info.bits,
            "mantissa_bits": mantissa,
            "exponent_bits": exponent,
            "max": info.max,
            "min_positive": info.smallest_normal,
            "eps": info.eps,
        })
    # TF32 is not a torch dtype; document it explicitly.
    out.append({
        "name": "tf32", "bits": 32, "mantissa_bits": 10, "exponent_bits": 8,
        "max": 3.4e38, "min_positive": 1.2e-38, "eps": 2**-10,
        "note": "fp32 with 10 mantissa bits; tensor-core format",
    })
    return out


def gemm_precision_speed(device: torch.device, M=1024, N=1024, K=1024,
                         warmup=20, iterations=100) -> list[dict]:
    """GEMM accuracy and speed per format, vs an fp64 reference."""
    torch.manual_seed(0)
    a = torch.randn(M, K, device=device, dtype=torch.float32)
    b = torch.randn(K, N, device=device, dtype=torch.float32)
    ref = (a.double() @ b.double())  # fp64 reference

    rows = []
    # tf32 vs fp32-ieee via the Triton kernel (fp32 inputs).
    for label, ieee in [("fp32-ieee", True), ("tf32", False)]:
        c = triton_gemm(a, b, ieee=ieee)
        max_err = (c.float() - ref.float()).abs().max().item()
        t = cuda_event_latency(lambda: triton_gemm(a, b, ieee=ieee), device=device,
                               warmup=warmup, iterations=iterations)
        rows.append({"format": label, "max_abs_err": max_err, "event_us": t.mean,
                     "speedup_vs_fp32_ieee": None})

    # fp16 and bf16 via torch.matmul (cuBLAS tensor core).
    fp32_ieee_us = rows[0]["event_us"]
    for label, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        ah, bh = a.to(dt), b.to(dt)
        c = (ah @ bh).float()
        max_err = (c - ref.float()).abs().max().item()
        t = cuda_event_latency(lambda: ah @ bh, device=device, warmup=warmup,
                               iterations=iterations)
        rows.append({"format": label, "max_abs_err": max_err, "event_us": t.mean,
                     "speedup_vs_fp32_ieee": fp32_ieee_us / t.mean})

    rows[0]["speedup_vs_fp32_ieee"] = 1.0
    rows[1]["speedup_vs_fp32_ieee"] = fp32_ieee_us / rows[1]["event_us"]
    return rows
