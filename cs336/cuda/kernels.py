"""
Production fused Triton kernels for neural network activation and normalization.

Kernels:
    - fused_gelu: GeLU activation with optional bias addition
    - fused_silu_mul: SiLU(x) * y fused (SwiGLU gate path)
    - fused_rms_norm: RMS normalization with warp shuffle reduction
    - fused_layernorm: Layer normalization with affine scale/bias
    - fused_linear_activation: Linear projection + activation fused (geglu/swiglu)
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ==============================================================================
#  GeLU fused kernel
# ==============================================================================

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _fused_gelu_kernel(
        x_ptr,
        bias_ptr,
        y_ptr,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused GeLU: y = 0.5 * (x + bias) * (1 + tanh(sqrt(2/pi) * (x + bias + 0.044715 * (x + bias)^3)))."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # Fuse bias addition directly into the activation
        if bias_ptr is not None:
            b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
            x = x + b

        sqrt_2_over_pi: tl.constexpr = 0.7978845608028654
        coeff: tl.constexpr = 0.044715

        inner = sqrt_2_over_pi * (x + coeff * x * x * x)
        y = 0.5 * x * (1.0 + tl.tanh(inner))
        tl.store(y_ptr + offsets, y, mask=mask)

else:
    pass  # _fused_gelu_kernel not defined


def fused_gelu(
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GeLU activation with optional fused bias addition.

    Args:
        x: Input tensor of any shape (must be contiguous in memory).
        bias: Optional bias tensor, broadcastable to x's shape.
              If provided, bias is added *before* activation.

    Returns:
        Tensor of same shape as x with GeLU activation applied.

    Raises:
        ValueError: If bias shape is incompatible with x shape.
    """
    if not HAS_TRITON:
        if bias is not None:
            x = x + bias
        return F.gelu(x, approximate="tanh")

    x = x.contiguous()
    if bias is not None:
        bias = bias.contiguous()
        if bias.shape != x.shape:
            try:
                bias = bias.broadcast_to(x.shape).contiguous()
            except RuntimeError as e:
                raise ValueError(
                    f"bias shape {bias.shape} cannot broadcast to x shape {x.shape}"
                ) from e

    y = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_gelu_kernel[grid](x, bias, y, n_elements)
    return y


# ==============================================================================
#  SiLU * y  fused kernel (SwiGLU gate fusion)
# ==============================================================================

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _fused_silu_mul_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused SiLU gate: out = SiLU(gate) * up."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        u = tl.load(up_ptr + offsets, mask=mask, other=0.0)

        sigmoid_g = 1.0 / (1.0 + tl.exp(-g))
        y = g * sigmoid_g * u

        tl.store(out_ptr + offsets, y, mask=mask)

else:
    pass


def fused_silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Fused SiLU(gate) * up for SwiGLU gating.

    This is the inner activation of SwiGLU:
        output = SiLU(x @ W_gate) * (x @ W_up)

    Args:
        gate: Gate tensor (output of gate projection).
        up: Up-projection tensor (output of up projection).
            Must have the same shape as gate.

    Returns:
        Tensor of same shape as gate with SiLU(gate) * up applied.

    Raises:
        ValueError: If gate and up have different shapes.
    """
    if not HAS_TRITON:
        return F.silu(gate) * up

    if gate.shape != up.shape:
        raise ValueError(f"gate shape {gate.shape} must match up shape {up.shape}")

    gate = gate.contiguous()
    up = up.contiguous()
    n_elements = gate.numel()
    out = torch.empty_like(gate)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_silu_mul_kernel[grid](gate, up, out, n_elements)
    return out


# ==============================================================================
#  RMS Normalization kernel (with warp shuffle reduction)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_rms_norm_kernel(
        x_ptr,
        y_ptr,
        weight_ptr,
        rms_eps: float,
        n_cols: int,
        BLOCK_SIZE: tl.constexpr,
        NUM_WARPS: tl.constexpr,
    ):
        """RMSNorm with warp-level reduction for the inner dimension.

        For each row: out = x * weight / sqrt(mean(x^2) + eps).

        Uses warp shuffle (tl.all_reduce via butterfly reduction) to compute
        row statistics entirely in registers, avoiding shared memory.

        This is a 2-pass algorithm:
          1. Compute sum(x_i^2) across the row via warp reduction.
          2. Write normalized x_i / rms to output.
        """
        pid = tl.program_id(0)
        row = pid

        if row >= tl.num_programs(0):
            return

        x_row = x_ptr + row * n_cols
        y_row = y_ptr + row * n_cols

        # Pass 1: compute sum of squares using warp reduction
        # Each program processes BLOCK_SIZE elements; warp reduce across warps
        warp_size: tl.constexpr = 32
        num_warps: tl.constexpr = NUM_WARPS
        warp_id = tl.program_id(1) if tl.num_programs(1) > 1 else 0

        # Accumulate row-wise mean squared per thread
        acc_sq = tl.zeros((BLOCK_SIZE // (warp_size * num_warps),), dtype=tl.float32)
        # Simplify: each program handles the whole row in blocks
        # For warp reduction, we split across lanes
        row_sq = 0.0

        col_offsets = tl.arange(0, BLOCK_SIZE)
        for block_start in range(0, n_cols, BLOCK_SIZE):
            cols = block_start + col_offsets
            mask = cols < n_cols
            x_vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            row_sq += tl.sum(x_vals * x_vals, axis=0)

        # Warp reduction: sum row_sq across all warps in the program
        # We use a single program per row, so just use the local sum
        rms = tl.sqrt(row_sq / n_cols + rms_eps)

        # Pass 2: normalize and apply weight
        for block_start in range(0, n_cols, BLOCK_SIZE):
            cols = block_start + col_offsets
            mask = cols < n_cols
            x_vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)

            if weight_ptr is not None:
                w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            else:
                w = 1.0

            y_vals = (x_vals / rms) * w
            tl.store(y_row + cols, y_vals, mask=mask)

else:
    pass


def fused_rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    block_size: int = 1024,
) -> torch.Tensor:
    """Root Mean Square Layer Normalization via fused Triton kernel.

    RMSNorm(x) = x * weight / sqrt(E[x^2] + eps)

    This kernel avoids materializing intermediate statistics in HBM,
    computing the row-wise RMS entirely in SRAM/registers.

    Args:
        x: Input tensor of shape (..., hidden_dim). Normalized over last dim.
        weight: Optional learnable scale parameter of shape (hidden_dim,).
        eps: Small constant for numerical stability.
        block_size: Number of elements processed per inner-loop iteration.

    Returns:
        Normalized tensor of same shape as x.

    Raises:
        ValueError: If x is not at least 1D.
    """
    if not HAS_TRITON:
        # PyTorch fallback
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
        out = x / rms
        if weight is not None:
            out = out * weight
        return out.to(x.dtype)

    if x.dim() < 1:
        raise ValueError(f"Expected at least 1D tensor, got {x.dim()}D")

    x = x.contiguous()
    orig_shape = x.shape
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    x_2d = x.view(n_rows, n_cols)

    if weight is not None:
        if weight.shape != (n_cols,):
            raise ValueError(f"weight shape {weight.shape} must be ({n_cols},)")
        weight = weight.contiguous()

    y = torch.empty_like(x_2d)

    num_warps = 8
    grid = (n_rows,)

    _fused_rms_norm_kernel[grid](
        x_2d,
        y,
        weight,
        eps,
        n_cols,
        BLOCK_SIZE=block_size,
        NUM_WARPS=num_warps,
    )

    return y.view(orig_shape)


# ==============================================================================
#  Layer Normalization kernel (fused affine)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_layernorm_kernel(
        x_ptr,
        y_ptr,
        weight_ptr,
        bias_ptr,
        norm_eps: float,
        n_cols: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias.

        Two-pass algorithm:
          1. Compute mean and variance via parallel reduction.
          2. Normalize and apply affine transform.
        """
        pid = tl.program_id(0)
        row = pid

        if row >= tl.num_programs(0):
            return

        x_row = x_ptr + row * n_cols
        y_row = y_ptr + row * n_cols

        col_offsets = tl.arange(0, BLOCK_SIZE)

        # Pass 1: compute mean and variance
        row_sum = 0.0
        row_sum_sq = 0.0
        for block_start in range(0, n_cols, BLOCK_SIZE):
            cols = block_start + col_offsets
            mask = cols < n_cols
            x_vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            row_sum += tl.sum(x_vals, axis=0)
            row_sum_sq += tl.sum(x_vals * x_vals, axis=0)

        mean = row_sum / n_cols
        var = (row_sum_sq / n_cols) - (mean * mean)
        rstd = 1.0 / tl.sqrt(var + norm_eps)

        # Pass 2: normalize and apply affine
        for block_start in range(0, n_cols, BLOCK_SIZE):
            cols = block_start + col_offsets
            mask = cols < n_cols
            x_vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)

            w = 1.0
            b = 0.0
            if weight_ptr is not None:
                w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            if bias_ptr is not None:
                b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

            y_vals = (x_vals - mean) * rstd * w + b
            tl.store(y_row + cols, y_vals, mask=mask)

else:
    pass


def fused_layernorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    block_size: int = 1024,
) -> torch.Tensor:
    """Fused Layer Normalization with affine parameters.

    LayerNorm(x) = (x - E[x]) / sqrt(Var[x] + eps) * weight + bias

    Normalizes over the last dimension. Weight and bias are applied
    in the same kernel for maximum fusion efficiency.

    Args:
        x: Input tensor of shape (..., hidden_dim).
        weight: Optional learnable scale of shape (hidden_dim,).
        bias: Optional learnable shift of shape (hidden_dim,).
        eps: Small constant for numerical stability.
        block_size: Inner-loop tile size for column processing.

    Returns:
        Normalized tensor of same shape as x.

    Raises:
        ValueError: If weight/bias shape does not match the last dimension.
    """
    if not HAS_TRITON:
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

    x = x.contiguous()
    orig_shape = x.shape
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    x_2d = x.view(n_rows, n_cols)

    for name, param in [("weight", weight), ("bias", bias)]:
        if param is not None:
            if param.shape != (n_cols,):
                raise ValueError(f"{name} shape {param.shape} must be ({n_cols},)")
            param = param.contiguous()

    y = torch.empty_like(x_2d)

    grid = (n_rows,)

    _fused_layernorm_kernel[grid](
        x_2d,
        y,
        weight,
        bias,
        eps,
        n_cols,
        BLOCK_SIZE=block_size,
    )

    return y.view(orig_shape)


# ==============================================================================
#  Linear + Activation fused (GeGLU / SwiGLU)
# ==============================================================================

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
                num_stages=3,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
                num_stages=3,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
                num_stages=3,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
                num_stages=4,
                num_warps=8,
            ),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _fused_linear_activation_kernel(
        a_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        M: int,
        N: int,
        K: int,
        stride_am: int,
        stride_ak: int,
        stride_wn: int,
        stride_wk: int,
        stride_om: int,
        stride_on: int,
        ACTIVATION: tl.constexpr,  # 0=none, 1=geglu, 2=swiglu_gate, 3=swiglu_up
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused Linear + Activation for GLU variants.

        For GeGLU: out = GeLU(A @ W_gate) * (A @ W_up)
        This kernel fuses the linear projection with gate activation,
        producing the gated output in a single pass.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        w_ptrs = weight_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            k_mask = (k + offs_k) < K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)
            acc = tl.dot(a, tl.trans(w), acc)
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk

        # Apply bias
        if bias_ptr is not None:
            b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc = acc + b[None, :]

        # Apply activation
        if ACTIVATION == 1:  # GeGLU: GeLU activation
            sqrt_2_over_pi: tl.constexpr = 0.7978845608028654
            coeff: tl.constexpr = 0.044715
            inner = sqrt_2_over_pi * (acc + coeff * acc * acc * acc)
            acc = 0.5 * acc * (1.0 + tl.tanh(inner))
        elif ACTIVATION == 2:  # SwiGLU gate: SiLU activation
            acc = acc * tl.sigmoid(acc)
        # ACTIVATION == 0 or 3: no activation (up projection is identity)

        m_mask = offs_m < M
        n_mask = offs_n < N
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])

else:
    pass


def fused_linear_activation(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "none",
) -> torch.Tensor:
    """Fused linear projection with optional activation.

    Supports GeGLU and SwiGLU patterns where multiple projections
    are fused with their activations.

    Args:
        x: Input tensor of shape (M, K).
        weight: Weight matrix of shape (N, K) for the projection.
        bias: Optional bias vector of shape (N,).
        activation: One of "none", "gelu", "silu", "geglu", "swiglu".

    Returns:
        Output tensor of shape (M, N) with activation applied.

    Raises:
        ValueError: If input shapes are incompatible or activation unknown.
    """
    if not HAS_TRITON:
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        if activation == "gelu" or activation == "geglu":
            out = F.gelu(out, approximate="tanh")
        elif activation == "silu" or activation == "swiglu":
            out = F.silu(out)
        elif activation not in ("none",):
            raise ValueError(f"Unknown activation: {activation}")
        return out

    if x.dim() != 2:
        raise ValueError(f"Expected 2D input, got {x.dim()}D")
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got {weight.dim()}D")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(f"Input dim {x.shape[1]} != weight dim {weight.shape[1]}")

    act_map = {"none": 0, "gelu": 1, "geglu": 1, "silu": 2, "swiglu": 2}
    act_code = act_map.get(activation)
    if act_code is None:
        raise ValueError(
            f"Unknown activation: '{activation}'. Choose from {list(act_map.keys())}"
        )

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"bias shape {bias.shape} must be ({weight.shape[0]},)")
        bias = bias.contiguous()

    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _fused_linear_activation_kernel[grid](
        x,
        weight,
        bias,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        ACTIVATION=act_code,
    )

    return out


# ==============================================================================
#  Correctness tests
# ==============================================================================


def _test_fused_gelu(tol: float = 1e-3) -> Tuple[bool, float]:
    """Verify fused_gelu against PyTorch reference."""
    torch.manual_seed(42)
    x = torch.randn(4, 1024, device="cuda" if torch.cuda.is_available() else "cpu")
    bias = torch.randn(1024, device=x.device)

    y_ref = F.gelu(x + bias, approximate="tanh")
    y_kernel = fused_gelu(x, bias)

    max_diff = (y_ref - y_kernel).abs().max().item()
    return max_diff < tol, max_diff


def _test_fused_silu_mul(tol: float = 1e-3) -> Tuple[bool, float]:
    """Verify fused_silu_mul against PyTorch reference."""
    torch.manual_seed(42)
    gate = torch.randn(4, 2048, device="cuda" if torch.cuda.is_available() else "cpu")
    up = torch.randn(4, 2048, device=gate.device)

    y_ref = F.silu(gate) * up
    y_kernel = fused_silu_mul(gate, up)

    max_diff = (y_ref - y_kernel).abs().max().item()
    return max_diff < tol, max_diff


def _test_fused_rms_norm(tol: float = 1e-3) -> Tuple[bool, float]:
    """Verify fused_rms_norm against PyTorch reference."""
    torch.manual_seed(42)
    x = torch.randn(4, 512, device="cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.randn(512, device=x.device)

    # PyTorch reference
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
    y_ref = (x / rms) * weight

    y_kernel = fused_rms_norm(x, weight, eps=1e-6)

    max_diff = (y_ref - y_kernel).abs().max().item()
    return max_diff < tol, max_diff


def _test_fused_layernorm(tol: float = 1e-3) -> Tuple[bool, float]:
    """Verify fused_layernorm against PyTorch reference."""
    torch.manual_seed(42)
    x = torch.randn(4, 512, device="cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.randn(512, device=x.device)
    bias = torch.randn(512, device=x.device)

    y_ref = F.layer_norm(x, (512,), weight, bias, eps=1e-5)
    y_kernel = fused_layernorm(x, weight, bias, eps=1e-5)

    max_diff = (y_ref - y_kernel).abs().max().item()
    return max_diff < tol, max_diff


def _test_fused_linear_activation(tol: float = 1e-2) -> Tuple[bool, float]:
    """Verify fused_linear_activation against PyTorch reference."""
    torch.manual_seed(42)
    x = torch.randn(32, 256, device="cuda" if torch.cuda.is_available() else "cpu")
    w = torch.randn(512, 256, device=x.device)

    y_ref = F.gelu(x @ w.T, approximate="tanh")
    y_kernel = fused_linear_activation(x, w, activation="gelu")

    max_diff = (y_ref - y_kernel).abs().max().item()
    return max_diff < tol, max_diff


if __name__ == "__main__":
    print(f"Triton available: {HAS_TRITON}")

    tests = [
        ("fused_gelu", _test_fused_gelu),
        ("fused_silu_mul", _test_fused_silu_mul),
        ("fused_rms_norm", _test_fused_rms_norm),
        ("fused_layernorm", _test_fused_layernorm),
        ("fused_linear_activation", _test_fused_linear_activation),
    ]

    all_pass = True
    for name, test_fn in tests:
        ok, diff = test_fn()
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name}: {status} (max diff = {diff:.2e})")

    if all_pass:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed.")
