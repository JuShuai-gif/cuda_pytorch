"""Softmax (per row of a 2D tensor).

One program per row.  The numerically-stable form subtracts the row max before
``exp``.  For the benchmark the row length fits in a single block, so there is
no need for the online multi-pass loop; that is introduced with flash-style
operators later.  The three steps (max, exp-and-sum, divide) map directly to
``tl.max``, ``tl.exp``, ``tl.sum``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import Op

BLOCK = 2048


@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = x_ptr + row * n_cols
    x = tl.load(base + cols, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    tl.store(out_ptr + row * n_cols + cols, e / s, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK=BLOCK)
    return out


def build():
    def inputs(device, dtype):
        n_rows, n_cols = 4096, 2048
        x = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        return (x,)

    return Op(
        name="softmax",
        triton=triton_softmax,
        reference=lambda x: torch.softmax(x, dim=-1),
        inputs=inputs,
        note="one program per row; BLOCK=%d == n_cols" % BLOCK,
    )
