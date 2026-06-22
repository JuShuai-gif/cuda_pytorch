"""
Fused Bias + GELU Triton kernel.

Industrial context: In transformer FFN after linear layer, the pattern is:
  h = W @ x + bias
  h = GELU(h)

GELU is more complex than ReLU (requires tanh or erf computation), making
fusion more valuable since the intermediate tensor write/read is avoided.

Uses the tanh approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bias_gelu_fused_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused bias add + GELU using tanh approximation."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    val = x + bias

    # GELU tanh approximation
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = sqrt_2_over_pi * (val + coeff * val * val * val)
    out = 0.5 * val * (1.0 + tl.tanh(inner))

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bias_gelu(
    x: torch.Tensor,
    bias: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """Fused bias + GELU: out = GELU(x + bias).

    Combines bias addition and GELU activation in a single kernel,
    avoiding intermediate tensor allocation and extra memory traffic.

    Args:
        x: Input tensor (CUDA).
        bias: Bias tensor (CUDA, broadcast-compatible or same shape).
        block_size: Elements per program.

    Returns:
        GELU(x + bias).
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA"
    out = torch.empty_like(x)
    n_elements = x.numel()

    bias_expanded = bias if bias.shape == x.shape else bias.expand_as(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _bias_gelu_fused_kernel[grid](x, bias_expanded, out, n_elements, BLOCK_SIZE=block_size)
    return out


def sequential_bias_gelu(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Sequential bias + GELU: materializes intermediate (x + bias) tensor."""
    intermediate = x + bias
    return torch.nn.functional.gelu(intermediate, approximate="tanh")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        x = torch.randn(4096, device="cuda")
        bias = torch.randn(4096, device="cuda")
        y_fused = fused_bias_gelu(x, bias)
        y_seq = sequential_bias_gelu(x, bias)
        err = (y_fused - y_seq).abs().max().item()
        print(f"Fused Bias+GELU - max error vs sequential: {err:.2e}")
