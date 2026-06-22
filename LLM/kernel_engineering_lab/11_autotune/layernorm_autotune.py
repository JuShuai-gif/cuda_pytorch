"""
LayerNorm and RMSNorm kernels with @triton.autotune.

Industrial context: Normalization layers are pervasive in transformers.
Autotuning block size, num_warps, and num_stages for these reduction
kernels is critical because the optimal configuration varies with
hidden dimension size and batch size.

Config constraint: BLOCK_SIZE must be >= num_warps * 32 for correct
warp-level reductions.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

EPS = 1e-5


# ======================================================================
# Autotuned LayerNorm
# ======================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
        for bs in [64, 128, 256, 512, 1024, 2048]
        for w in [2, 4, 8, 16]
        if bs >= w * 32
    ],
    key=["N"],
)
@triton.jit
def autotuned_layernorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N: int,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned LayerNorm kernel.

    Each program handles one row. w_ptr and b_ptr are optional
    learnable affine parameters (weight and bias).
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x_row = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)

    mean = tl.sum(x_row, axis=0) / N
    centered = x_row - mean
    var = tl.sum(centered * centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = centered * rstd

    weight = tl.load(w_ptr + offs, mask=mask, other=1.0)
    bias = tl.load(b_ptr + offs, mask=mask, other=0.0)

    out = normalized * weight + bias
    tl.store(out_ptr + row_start + offs, out, mask=mask)


def autotuned_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Autotuned LayerNorm.

    Args:
        x: (rows, N) tensor on CUDA.
        weight: (N,) learnable weight, or None for ones.
        bias: (N,) learnable bias, or None for zeros.
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor, same shape as x.
    """
    assert x.is_cuda
    N = x.shape[-1]

    if weight is None:
        weight = torch.ones(N, device=x.device, dtype=torch.float32)
    if bias is None:
        bias = torch.zeros(N, device=x.device, dtype=torch.float32)

    out = torch.empty_like(x)

    if x.dim() == 1:
        x_2d = x.reshape(1, -1)
        out_2d = out.reshape(1, -1)
        rows = 1
    else:
        x_2d = x.reshape(-1, N)
        out_2d = out.reshape(-1, N)
        rows = x_2d.shape[0]

    grid = (rows, 1, 1)

    autotuned_layernorm_kernel[grid](
        x_2d,
        weight,
        bias,
        out_2d,
        N,
        eps,
    )

    return out


# ======================================================================
# Autotuned RMSNorm
# ======================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [64, 128, 256, 512]
        for w in [2, 4, 8]
        for s in [1, 2, 3, 4]
    ],
    key=["N", "M"],
)
@triton.jit
def autotuned_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M: int,
    N: int,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned RMSNorm kernel.

    M = number of rows, N = hidden dimension.
    w_ptr is the learnable weight vector.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x_row = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
    x_sq = x_row * x_row
    mean_sq = tl.sum(x_sq, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    normalized = x_row * rstd

    weight = tl.load(w_ptr + offs, mask=mask, other=1.0)
    out = normalized * weight

    tl.store(out_ptr + row_start + offs, out, mask=mask)


def autotuned_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Autotuned RMSNorm.

    Args:
        x: (rows, N) tensor on CUDA.
        weight: (N,) learnable weight, or None for ones.
        eps: Epsilon.

    Returns:
        Normalized tensor, same shape as x.
    """
    assert x.is_cuda
    N = x.shape[-1]

    if weight is None:
        weight = torch.ones(N, device=x.device, dtype=torch.float32)

    out = torch.empty_like(x)

    if x.dim() == 1:
        x_2d = x.reshape(1, -1)
        out_2d = out.reshape(1, -1)
        rows = 1
    else:
        x_2d = x.reshape(-1, N)
        out_2d = out.reshape(-1, N)
        rows = x_2d.shape[0]

    grid = (rows, 1, 1)

    autotuned_rmsnorm_kernel[grid](
        x_2d,
        weight,
        out_2d,
        rows,
        N,
        eps,
    )

    return out


# ======================================================================
# Demo
# ======================================================================


def layernorm_autotune_demo() -> None:
    """Run autotuned LayerNorm and RMSNorm for typical transformer dims."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return

    print("=" * 70)
    print("  AUTOTUNE: LayerNorm & RMSNorm Demo")
    print("=" * 70)

    hidden_dims = [512, 768, 1024, 2048, 4096, 8192]
    batch_sizes = [4, 16, 64]

    for N in hidden_dims:
        for B in [4, 16]:
            x = torch.randn(B, N, device="cuda", dtype=torch.float32)
            w = torch.randn(N, device="cuda", dtype=torch.float32)
            b = torch.randn(N, device="cuda", dtype=torch.float32)

            # LayerNorm
            out_ln = autotuned_layernorm(x, w, b)
            ref_ln = torch.nn.functional.layer_norm(
                x.float(), [N], weight=w.float(), bias=b.float(), eps=EPS
            )
            err_ln = (out_ln.float() - ref_ln).abs().max().item()

            cfg_ln = autotuned_layernorm_kernel.best_config
            ln_cfg = cfg_ln.kwargs if cfg_ln else {}

            # RMSNorm
            out_rn = autotuned_rmsnorm(x, w)
            rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)
            ref_rn = x.float() * rms * w.float()
            err_rn = (out_rn.float() - ref_rn).abs().max().item()

            cfg_rn = autotuned_rmsnorm_kernel.best_config
            rn_cfg = cfg_rn.kwargs if cfg_rn else {}

            print(f"\n  Shape: ({B}, {N})")
            print(
                f"    LayerNorm: BLOCK_SIZE={ln_cfg.get('BLOCK_SIZE', '?')}, "
                f"num_warps={ln_cfg.get('num_warps', '?')}, "
                f"err={err_ln:.2e}"
            )
            print(
                f"    RMSNorm:   BLOCK_SIZE={rn_cfg.get('BLOCK_SIZE', '?')}, "
                f"num_warps={rn_cfg.get('num_warps', '?')}, "
                f"num_stages={rn_cfg.get('num_stages', '?')}, "
                f"err={err_rn:.2e}"
            )

    print(f"\n  LayerNorm configs tested: {len(autotuned_layernorm_kernel.configs)}")
    print(f"  RMSNorm configs tested:   {len(autotuned_rmsnorm_kernel.configs)}")


if __name__ == "__main__":
    layernorm_autotune_demo()
