"""RMSNorm (per row, fused).

y = x * rsqrt(mean(x^2) + eps) * w

RMSNorm drops the mean-centering of LayerNorm and keeps only the root-mean-
square normalization, which is what modern LLMs (LLaMA family) use.  This is
one of the first "fused custom op" wins on real models: without it, an eager
implementation would materialize ``x^2``, a mean, a reciprocal-sqrt, and a
mul as four separate kernels with four round-trips through global memory.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import Op

BLOCK = 1024


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = x_ptr + row * n_cols
    x = tl.load(base + cols, mask=mask, other=0.0)
    ms = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * n_cols + cols, x * rstd * w, mask=mask)


def triton_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    rmsnorm_kernel[(n_rows,)](x, w, out, n_cols, eps, BLOCK=BLOCK)
    return out


def build():
    def inputs(device, dtype):
        n_rows, n_cols = 4096, 1024
        x = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        w = torch.randn(n_cols, device=device, dtype=dtype)
        return x, w

    eps = 1e-5

    def reference(x, w, eps):
        rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * rstd * w

    return Op(
        name="rmsnorm",
        triton=triton_rmsnorm,
        reference=reference,
        inputs=inputs,
        kwargs={"eps": eps},
        note="fused rsqrt(mean(x^2)); BLOCK=%d" % BLOCK,
    )
