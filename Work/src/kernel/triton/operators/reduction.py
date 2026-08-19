"""Reduction: sum of a 1D tensor.

A two-level reduction: each program reduces BLOCK elements to a partial sum in
a single thread lane, then atomically adds the partial to the output scalar.
This demonstrates the classic Triton reduction idiom (`tl.sum` over the axis),
and the atomic path that turns "block partials" into a global result.  The
alternative is a second kernel over the partials; atomics are fine here because
there are only n/BLOCK partials.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import Op

BLOCK = 1024


@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    partial = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, partial)


def triton_sum(x: torch.Tensor) -> torch.Tensor:
    # Accumulate in fp32: fp16 atomic_add has no native hardware support and
    # degrades to a CAS loop, and fp16 partial sums lose precision quickly.
    out = torch.zeros((), device=x.device, dtype=torch.float32)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    sum_kernel[grid](x, out, n, BLOCK=BLOCK)
    return out


def build():
    def inputs(device, dtype):
        # Reduction is only meaningful (and exact) in fp32; fp16 accumulation
        # in a single scalar loses precision and fp16 atomics are emulated.
        n = 1 << 22
        x = torch.randn(n, device=device, dtype=torch.float32)
        return (x,)

    return Op(
        name="reduction_sum",
        triton=triton_sum,
        reference=lambda x: x.sum(),
        inputs=inputs,
        note="two-level: block partial via tl.sum + atomic_add (fp32 accumulator)",
    )
