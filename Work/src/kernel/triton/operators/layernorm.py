"""LayerNorm (per row, fused into a single kernel).

The whole operation - mean, variance, normalize, scale, shift - happens in one
Triton program per row with a single load/store of ``x``.  This is the fusion
argument in miniature: PyTorch's eager ``F.layer_norm`` is already fused by
ATen, so the benchmark is a same-fused comparison rather than a win; the value
here is showing *how* the fused math is expressed in Triton.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import Op

BLOCK = 1024


@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = x_ptr + row * n_cols
    x = tl.load(base + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * n_cols + cols, xc * rstd * w + b, mask=mask)


def triton_layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    layernorm_kernel[(n_rows,)](x, w, b, out, n_cols, eps, BLOCK=BLOCK)
    return out


def build():
    def inputs(device, dtype):
        n_rows, n_cols = 4096, 1024
        x = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        w = torch.randn(n_cols, device=device, dtype=dtype)
        b = torch.randn(n_cols, device=device, dtype=dtype)
        return x, w, b

    eps = 1e-5

    def reference(x, w, b, eps):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, eps=eps)

    return Op(
        name="layernorm",
        triton=triton_layernorm,
        reference=reference,
        inputs=inputs,
        kwargs={"eps": eps},
        note="fused mean/var/normalize/scale/shift; BLOCK=%d" % BLOCK,
    )
