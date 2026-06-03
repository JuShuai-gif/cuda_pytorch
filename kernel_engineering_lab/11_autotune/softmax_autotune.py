"""
Online softmax kernel with @triton.autotune.

Industrial context: Softmax is used in every attention mechanism.
Autotuning the block size and warp configuration is essential because
the optimal config varies with sequence length (which determines
the reduction dimension).

The kernel uses the online safe softmax algorithm:
  1. Find max value along the row
  2. Subtract max (for numerical stability)
  3. Compute exp of shifted values
  4. Sum the exps
  5. Divide each exp by the sum
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [64, 128, 256, 512, 1024]
        for w in [2, 4, 8]
        for s in [1, 2, 3, 4]
    ],
    key=["N"],
)
@triton.jit
def autotuned_softmax_kernel(
    x_ptr,
    out_ptr,
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Online softmax kernel with autotune.

    Each program processes one row. Uses the safe softmax algorithm
    (subtract max before exp to avoid overflow).
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x_row = tl.load(x_ptr + row_start + offs, mask=mask, other=float("-inf"))
    x_max = tl.max(x_row, axis=0)
    x_safe = x_row - x_max
    x_exp = tl.exp(x_safe)
    x_sum = tl.sum(x_exp, axis=0)
    out = x_exp / x_sum
    tl.store(out_ptr + row_start + offs, out, mask=mask)


def autotuned_softmax(x: torch.Tensor) -> torch.Tensor:
    """Autotuned softmax along the last dimension.

    Args:
        x: Tensor on CUDA, any shape. Softmax is applied over dim=-1.

    Returns:
        Softmax output, same shape as x.
    """
    assert x.is_cuda
    out = torch.empty_like(x)

    if x.dim() == 1:
        x_2d = x.reshape(1, -1)
        out_2d = out.reshape(1, -1)
        rows = 1
    else:
        x_2d = x.reshape(-1, x.shape[-1])
        out_2d = out.reshape(-1, x.shape[-1])
        rows = x_2d.shape[0]

    N = x_2d.shape[-1]
    grid = (rows, 1, 1)

    autotuned_softmax_kernel[grid](x_2d, out_2d, N)

    return out


# ======================================================================
# Demo
# ======================================================================


def softmax_autotune_demo() -> None:
    """Run autotuned softmax for various sequence lengths."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return

    print("=" * 70)
    print("  AUTOTUNE: Softmax Demo")
    print("=" * 70)

    seq_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    batch_sizes = [4, 16, 64]

    for N in seq_lens:
        for B in [4, 16]:
            x = torch.randn(B, N, device="cuda", dtype=torch.float32)

            out = autotuned_softmax(x)
            ref = torch.softmax(x.float(), dim=-1)
            err = (out.float() - ref).abs().max().item()

            cfg = autotuned_softmax_kernel.best_config
            best = cfg.kwargs if cfg else {}

            print(f"  Shape: ({B}, {N})")
            print(
                f"    Best: BLOCK_SIZE={best.get('BLOCK_SIZE', '?')}, "
                f"num_warps={best.get('num_warps', '?')}, "
                f"num_stages={best.get('num_stages', '?')}, "
                f"err={err:.2e}"
            )

    print(f"\n  Total configs tested: {len(autotuned_softmax_kernel.configs)}")


if __name__ == "__main__":
    softmax_autotune_demo()
