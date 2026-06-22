"""
Fused Add + ReLU Triton kernel.

Industrial context: In feed-forward networks, the pattern is Wx + b, then
ReLU activation. Without fusion: 2 global memory reads (x, bias) + 2 writes
(intermediate, output). With fusion: 1 read + 1 write.

out = relu(x + bias)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_relu_fused_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add + relu: out[i] = max(0, x[i] + bias[i])."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    val = x + bias
    out = tl.where(val > 0.0, val, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_add_relu(
    x: torch.Tensor,
    bias: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """Fused add + ReLU: out = relu(x + bias).

    Equivalent to torch.nn.functional.relu(x + bias) but in a single
    kernel with one global memory read of x and one global memory write.

    Args:
        x: Input tensor (CUDA).
        bias: Bias tensor (CUDA, broadcast-compatible with x).
        block_size: Elements per program.

    Returns:
        relu(x + bias).
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA"
    out = torch.empty_like(x)
    n_elements = x.numel()

    # Handle broadcasting: expand bias to match x shape if needed
    bias_expanded = bias if bias.shape == x.shape else bias.expand_as(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _add_relu_fused_kernel[grid](x, bias_expanded, out, n_elements, BLOCK_SIZE=block_size)
    return out


def sequential_add_relu(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Sequential (unfused) add + ReLU: materializes intermediate tensor.

    This is the PyTorch eager-mode equivalent, which reads x and bias,
    writes an intermediate tensor, reads the intermediate tensor,
    and writes the final output.
    """
    return torch.nn.functional.relu(x + bias)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        x = torch.randn(4096, device="cuda")
        bias = torch.randn(4096, device="cuda")
        y_fused = fused_add_relu(x, bias)
        y_seq = sequential_add_relu(x, bias)
        err = (y_fused - y_seq).abs().max().item()
        print(f"Fused Add+ReLU - max error vs sequential: {err:.2e}")
