"""Vector add: the smallest Triton kernel.

The pedagogical point here is not "add two arrays faster than PyTorch" (you
won't beat a one-liner), but to establish the Triton programming model:
``program_id`` gives the block index, ``tl.arange`` builds the per-element
index vector, and ``mask`` handles the tail when ``n`` is not a multiple of
``BLOCK``.  This is the template every later kernel builds on.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import Op

BLOCK = 1024


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


def build():
    def inputs(device, dtype):
        n = 1 << 22
        x = torch.randn(n, device=device, dtype=dtype)
        y = torch.randn(n, device=device, dtype=dtype)
        return x, y

    return Op(
        name="vector_add",
        triton=triton_add,
        reference=lambda x, y: x + y,
        inputs=inputs,
        note="block size %d; tail handled by mask" % BLOCK,
    )
