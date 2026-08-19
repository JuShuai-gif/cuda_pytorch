"""INT8 quantization / dequantization (per-tensor, symmetric).

Quantization maps fp32 to int8 with a single scale:

    x_q = clamp(round(x / scale), -127, 127)
    x   ~= scale * x_q

This module shows the two kernels and the round-trip error.  The calibration
side (choosing ``scale = max(|x|) / 127``) and the fused dequant+GEMM are in
the fusion module.  The important lesson here is that quantize/dequant is a
memory-bound elementwise pair: its cost is dominated by moving bytes, not by
arithmetic, which is why "dequant kernel becomes the bottleneck" shows up in
real weight-only INT8 GEMMs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import Op

BLOCK = 1024


@triton.jit
def quantize_kernel(x_ptr, q_ptr, scale, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    q = tl.extra.cuda.libdevice.rint(x / scale)
    q = tl.clamp(q, -127.0, 127.0)
    tl.store(q_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def dequant_kernel(q_ptr, x_ptr, scale, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    q = tl.load(q_ptr + offs, mask=mask).to(tl.float32)
    tl.store(x_ptr + offs, q * scale, mask=mask)


def triton_quantize(x: torch.Tensor, scale: float) -> torch.Tensor:
    q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    n = x.numel()
    quantize_kernel[(triton.cdiv(n, BLOCK),)](x, q, scale, n, BLOCK=BLOCK)
    return q


def triton_dequant(q: torch.Tensor, scale: float) -> torch.Tensor:
    x = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    n = q.numel()
    dequant_kernel[(triton.cdiv(n, BLOCK),)](q, x, scale, n, BLOCK=BLOCK)
    return x


def build():
    def inputs(device, dtype):
        n = 1 << 22
        x = torch.randn(n, device=device, dtype=torch.float32)
        return (x,)

    def reference(x):
        q = (x / 0.02).round().clamp(-127, 127).to(torch.int8)
        return q.float() * 0.02

    def triton_fn(x):
        q = triton_quantize(x, 0.02)
        return triton_dequant(q, 0.02)

    return Op(
        name="int8_quant_dequant",
        triton=triton_fn,
        reference=reference,
        inputs=inputs,
        note="per-tensor symmetric int8 round-trip, scale=0.02",
    )
