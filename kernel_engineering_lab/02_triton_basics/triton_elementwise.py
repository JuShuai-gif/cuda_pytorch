"""
Triton kernels for common activation functions used in LLMs.

Implemented activations:
  - SiLU (Sigmoid Linear Unit, aka Swish): x * sigmoid(x)
    Used in SwiGLU - the activation in LLaMA, Mistral, Gemma, Qwen
  - GELU (Gaussian Error Linear Unit): x * Phi(x)
    Used in BERT, GPT-2, early Transformers
  - ReLU (Rectified Linear Unit): max(0, x)
    Basic activation, still widely used in CNNs and some attention variants

Each kernel demonstrates proper grid/block configuration, mask handling,
and program_id usage in Triton.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# SiLU (SwiGLU activation)
# ---------------------------------------------------------------------------


@triton.jit
def _silu_kernel(
    x_ptr,
    out_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU: out = x * sigmoid(x)."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # sigmoid(x) = 1 / (1 + exp(-x))
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_silu(
    x: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """Apply SiLU activation using a Triton kernel."""
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _silu_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)
    return out


# ---------------------------------------------------------------------------
# GELU (tanh approximation)
# ---------------------------------------------------------------------------


@triton.jit
def _gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """GELU (tanh approximation): out = 0.5 * x * (1 + tanh(...))."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # tanh approximation: sqrt(2/pi) * (x + 0.044715 * x^3)
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    out = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(
    x: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """Apply GELU activation using a Triton kernel."""
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)
    return out


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------


@triton.jit
def _relu_kernel(
    x_ptr,
    out_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU: out = max(0, x)."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu(
    x: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """Apply ReLU activation using a Triton kernel."""
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)
    return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        x = torch.randn(1024, device="cuda")

        for name, triton_fn, torch_fn in [
            ("SiLU", triton_silu, lambda t: torch.nn.functional.silu(t)),
            ("GELU", triton_gelu, lambda t: torch.nn.functional.gelu(t, approximate="tanh")),
            ("ReLU", triton_relu, lambda t: torch.nn.functional.relu(t)),
        ]:
            out_triton = triton_fn(x)
            out_torch = torch_fn(x)
            err = (out_triton - out_torch).abs().max().item()
            print(f"{name:>6} - max error vs PyTorch: {err:.2e}")
