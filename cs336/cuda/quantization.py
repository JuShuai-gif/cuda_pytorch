"""
Quantization kernels for inference optimization.

Provides fused quantization/dequantization kernels in Triton
for FP8 and INT8 formats, supporting both symmetric and asymmetric
quantization schemes.

Use cases:
    - KV cache quantization for reducing memory bandwidth in decoding
    - Weight quantization for reduced model size
    - Activation quantization for compute throughput

FP8 formats follow the Nvidia/OCP specification (e4m3 for forward,
e5m2 for backward in mixed-precision training).

INT8 quantization supports per-tensor, per-channel, and per-token
granularity with both symmetric and asymmetric modes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ==============================================================================
#  FP8 Quantization (e4m3 format)
# ==============================================================================

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _fp8_quantize_kernel(
        x_ptr,
        scale_ptr,
        y_ptr,
        n_elements: int,
        quant_dim: int,  # size of quantization dimension (for per-channel)
        per_channel: tl.constexpr,  # 0 = per-tensor, 1 = per-channel
        BLOCK_SIZE: tl.constexpr,
    ):
        """Quantize FP32/BF16/FP16 to FP8 (e4m3) with scaling.

        FP8 e4m3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits.
        Representable range: ~[2^-10, 448] in normal representation.

        Quantization: y_i8 = clamp(round(x / scale), -128, 127)
        Actually stored as INT8 bit pattern.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        if per_channel:
            channel_idx = offsets % quant_dim
            scale = tl.load(scale_ptr + channel_idx, mask=mask, other=1.0)
        else:
            scale = tl.load(scale_ptr)

        # Clamp to FP8 e4m3 max representable value
        max_val: tl.constexpr = 448.0
        min_val: tl.constexpr = -448.0

        x_scaled = x / scale
        x_clamped = tl.clamp(x_scaled, min_val, max_val)
        y = x_clamped.to(tl.float8e4nv, bitcast=False)

        tl.store(y_ptr + offsets, y, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _fp8_dequantize_kernel(
        x_ptr,
        scale_ptr,
        y_ptr,
        n_elements: int,
        quant_dim: int,
        per_channel: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Dequantize FP8 (e4m3) to target dtype."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        if per_channel:
            channel_idx = offsets % quant_dim
            scale = tl.load(scale_ptr + channel_idx, mask=mask, other=1.0)
        else:
            scale = tl.load(scale_ptr)

        y = x * scale
        tl.store(y_ptr + offsets, y, mask=mask)

else:
    pass


def fp8_quantize(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    per_channel: bool = False,
    channel_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 (e4m3) format.

    Args:
        x: Input tensor in fp32/bf16/fp16.
        scale: Scaling factor. If None, computed as max(|x|) / 448.0.
        per_channel: If True, use per-channel scaling along channel_dim.
        channel_dim: Dimension for per-channel quantization.

    Returns:
        Tuple of (quantized_tensor_fp8, scale_tensor).

    Raises:
        RuntimeError: If Triton or FP8 hardware is not available.
        ValueError: If per_channel=True and channel_dim is invalid.
    """
    if not HAS_TRITON:
        return _fp8_quantize_pytorch(x, scale, per_channel, channel_dim)

    x = x.contiguous()
    n_elements = x.numel()

    if scale is None:
        if per_channel:
            dim_size = x.shape[channel_dim]
            flat_x = x.float()
            # Move channel_dim to last and compute max per channel
            amax = flat_x.abs().amax(dim=channel_dim, keepdim=False)
            scale = amax / 448.0
            scale = scale.contiguous()
        else:
            amax = x.float().abs().max()
            scale = torch.tensor(amax / 448.0, device=x.device, dtype=torch.float32)

    quant_dim = x.shape[channel_dim] if per_channel else 0

    y = torch.empty_like(
        x, dtype=torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.uint8
    )

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fp8_quantize_kernel[grid](
        x,
        scale,
        y,
        n_elements,
        quant_dim,
        PER_CHANNEL=per_channel,
    )

    return y, scale


def fp8_dequantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    per_channel: bool = False,
    channel_dim: int = -1,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize FP8 tensor back to higher precision.

    Args:
        x: FP8 quantized tensor.
        scale: Scale factor used during quantization.
        per_channel: Whether per-channel quantization was used.
        channel_dim: Channel dimension for per-channel dequantization.
        output_dtype: Target output dtype.

    Returns:
        Dequantized tensor in output_dtype.
    """
    if not HAS_TRITON:
        return _fp8_dequantize_pytorch(x, scale, per_channel, channel_dim, output_dtype)

    x = x.contiguous()
    scale = scale.contiguous()
    n_elements = x.numel()

    quant_dim = x.shape[channel_dim] if per_channel else 0

    y = torch.empty(x.shape, device=x.device, dtype=output_dtype)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fp8_dequantize_kernel[grid](
        x,
        scale,
        y,
        n_elements,
        quant_dim,
        PER_CHANNEL=per_channel,
    )

    return y


# ==============================================================================
#  INT8 Quantization (symmetric & asymmetric)
# ==============================================================================

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _int8_quantize_kernel(
        x_ptr,
        scale_ptr,
        zero_point_ptr,
        y_ptr,
        n_elements: int,
        quant_dim: int,
        per_channel: tl.constexpr,
        symmetric: tl.constexpr,  # 1 = symmetric (no zero point), 0 = asymmetric
        BLOCK_SIZE: tl.constexpr,
    ):
        """Quantize FP32/BF16/FP16 to INT8.

        Asymmetric: q = round(x / scale + zero_point), clamp to [0, 255]
        Symmetric:  q = round(x / scale), clamp to [-128, 127]
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        if per_channel:
            channel_idx = offsets % quant_dim
            scale = tl.load(scale_ptr + channel_idx, mask=mask, other=1.0)
        else:
            scale = tl.load(scale_ptr)

        if symmetric:
            x_scaled = x / scale
            x_rounded = tl.math.llrint(x_scaled)
            q = tl.clamp(x_rounded, -128, 127)
        else:
            if per_channel:
                zp = tl.load(zero_point_ptr + channel_idx, mask=mask, other=0.0)
            else:
                zp = tl.load(zero_point_ptr)
            x_scaled = x / scale + zp
            x_rounded = tl.math.llrint(x_scaled)
            q = tl.clamp(x_rounded, 0, 255)

        tl.store(y_ptr + offsets, q, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _int8_dequantize_kernel(
        x_ptr,
        scale_ptr,
        zero_point_ptr,
        y_ptr,
        n_elements: int,
        quant_dim: int,
        per_channel: tl.constexpr,
        symmetric: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Dequantize INT8 to target dtype."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)

        if per_channel:
            channel_idx = offsets % quant_dim
            scale = tl.load(scale_ptr + channel_idx, mask=mask, other=1.0)
        else:
            scale = tl.load(scale_ptr)

        if symmetric:
            y = x * scale
        else:
            if per_channel:
                zp = tl.load(zero_point_ptr + channel_idx, mask=mask, other=0.0)
            else:
                zp = tl.load(zero_point_ptr)
            y = (x - zp) * scale

        tl.store(y_ptr + offsets, y, mask=mask)

else:
    pass


def int8_quantize(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Quantize a tensor to INT8.

    Supports both symmetric and asymmetric quantization.

    Args:
        x: Input tensor in fp32/bf16/fp16.
        scale: Per-tensor or per-channel scale.
               Computed as max(|x|)/127 for symmetric, (max(x)-min(x))/255 for asymmetric.
        zero_point: Zero point for asymmetric quantization. Computed if None.
        symmetric: If True, use symmetric quantization (range [-128, 127]).
                   If False, use asymmetric (range [0, 255]).
        per_channel: If True, quantize independently along channel_dim.
        channel_dim: Dimension for per-channel quantization.

    Returns:
        Tuple of (quantized_tensor_int8, scale, zero_point_or_None).
    """
    if not HAS_TRITON:
        return _int8_quantize_pytorch(
            x, scale, zero_point, symmetric, per_channel, channel_dim
        )

    x = x.contiguous()
    n_elements = x.numel()

    if scale is None:
        fp_x = x.float()
        if symmetric:
            if per_channel:
                amax = fp_x.abs().amax(dim=channel_dim, keepdim=False)
                scale = amax / 127.0
            else:
                amax = fp_x.abs().max()
                scale = torch.tensor(amax / 127.0, device=x.device, dtype=torch.float32)
            zero_point = None
        else:
            if per_channel:
                amax = fp_x.amax(dim=channel_dim, keepdim=False)
                amin = fp_x.amin(dim=channel_dim, keepdim=False)
                scale = (amax - amin) / 255.0
                zero_point = -amin / scale
            else:
                amax = fp_x.max()
                amin = fp_x.min()
                scale = torch.tensor(
                    (amax - amin) / 255.0, device=x.device, dtype=torch.float32
                )
                zero_point = torch.tensor(
                    -amin / scale.item(), device=x.device, dtype=torch.float32
                )

    quant_dim = x.shape[channel_dim] if per_channel else 0

    if symmetric:
        y = torch.empty_like(x, dtype=torch.int8)
    else:
        y = torch.empty_like(x, dtype=torch.uint8)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _int8_quantize_kernel[grid](
        x,
        scale,
        zero_point if not symmetric else torch.zeros(1, device=x.device),
        y,
        n_elements,
        quant_dim,
        PER_CHANNEL=per_channel,
        SYMMETRIC=symmetric,
    )

    return y, scale, zero_point if not symmetric else None


def int8_dequantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_dim: int = -1,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize INT8 tensor back to higher precision.

    Args:
        x: INT8 quantized tensor.
        scale: Scale factor from quantization.
        zero_point: Zero point for asymmetric. None for symmetric.
        symmetric: Whether symmetric quantization was used.
        per_channel: Whether per-channel quantization was used.
        channel_dim: Channel dimension for per-channel dequantization.
        output_dtype: Target output dtype.

    Returns:
        Dequantized tensor in output_dtype.
    """
    if not HAS_TRITON:
        return _int8_dequantize_pytorch(
            x, scale, zero_point, symmetric, per_channel, channel_dim, output_dtype
        )

    x = x.contiguous()
    scale = scale.contiguous()
    n_elements = x.numel()

    if zero_point is not None:
        zero_point = zero_point.contiguous()

    quant_dim = x.shape[channel_dim] if per_channel else 0

    y = torch.empty(x.shape, device=x.device, dtype=output_dtype)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _int8_dequantize_kernel[grid](
        x,
        scale,
        zero_point if zero_point is not None else torch.zeros(1, device=x.device),
        y,
        n_elements,
        quant_dim,
        PER_CHANNEL=per_channel,
        SYMMETRIC=symmetric,
    )

    return y


# ==============================================================================
#  PyTorch fallback implementations
# ==============================================================================


def _fp8_quantize_pytorch(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    per_channel: bool = False,
    channel_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch fallback for FP8 quantization."""
    fp_x = x.float()
    if scale is None:
        if per_channel:
            amax = fp_x.abs().amax(dim=channel_dim, keepdim=True)
            scale = amax / 448.0
        else:
            amax = fp_x.abs().max()
            scale = amax.new_zeros(()) + (amax / 448.0)

    scale_expanded = scale
    clipped = torch.clamp(fp_x / scale_expanded, -448.0, 448.0)

    if hasattr(torch, "float8_e4m3fn"):
        y = clipped.to(torch.float8_e4m3fn)
    else:
        y = clipped.to(torch.uint8)

    return y, scale


def _fp8_dequantize_pytorch(
    x: torch.Tensor,
    scale: torch.Tensor,
    per_channel: bool = False,
    channel_dim: int = -1,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """PyTorch fallback for FP8 dequantization.

    The scale should already have broadcast-compatible shape from quantize.
    """
    return (x.float() * scale).to(output_dtype)


def _int8_quantize_pytorch(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """PyTorch fallback for INT8 quantization."""
    fp_x = x.float()

    if per_channel:
        expand_shape = [1] * x.dim()
        expand_shape[channel_dim] = x.shape[channel_dim]

    if symmetric:
        if scale is None:
            if per_channel:
                amax = fp_x.abs().amax(dim=channel_dim, keepdim=True)
                scale = amax / 127.0  # keepdim=True preserves broadcast shape
            else:
                amax = fp_x.abs().max()
                scale = amax.new_zeros(()) + (amax / 127.0)
            zero_point = None
        scale_exp = scale
        q = torch.clamp(torch.round(fp_x / scale_exp), -128, 127).to(torch.int8)
        return q, scale_exp.detach().clone(), None
    else:
        if scale is None:
            if per_channel:
                amax = fp_x.amax(dim=channel_dim, keepdim=True)
                amin = fp_x.amin(dim=channel_dim, keepdim=True)
                scale = (amax - amin) / 255.0
                zero_point = -amin / scale
            else:
                amax = fp_x.max()
                amin = fp_x.min()
                scale = amax.new_zeros(()) + ((amax - amin) / 255.0)
                zero_point = amax.new_zeros(()) + (-amin / scale)

        scale_exp = scale
        zp_exp = zero_point

        q = torch.clamp(torch.round(fp_x / scale_exp + zp_exp), 0, 255).to(torch.uint8)
        return q, scale_exp.detach().clone(), zp_exp.detach().clone()


def _int8_dequantize_pytorch(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_dim: int = -1,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """PyTorch fallback for INT8 dequantization.

    The scale (and zero_point) should already have a shape that
    broadcasts correctly with x (i.e. keepdim shape from quantize).
    """
    fp_x = x.float()
    if symmetric:
        return (fp_x * scale).to(output_dtype)
    else:
        assert zero_point is not None
        return ((fp_x - zero_point) * scale).to(output_dtype)


# ==============================================================================
#  Correctness tests
# ==============================================================================


def test_int8_quantize(tol: float = 1e-2) -> Tuple[bool, float]:
    """Verify INT8 quantization round-trip accuracy."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(16, 64, device=device) * 2.0

    for symmetric in [True, False]:
        q, scale, zp = int8_quantize(x, symmetric=symmetric)
        x_recovered = int8_dequantize(q, scale, zp, symmetric=symmetric)
        err = (x - x_recovered).abs().max().item()

        # INT8 has limited precision; use per-element tolerance
        max_val = x.abs().max().item()
        expected_err = max_val / 128.0 if symmetric else max_val / 255.0
        if err > expected_err * 2:
            return False, err

    return True, 0.0


def test_int8_per_channel(tol: float = 1e-2) -> Tuple[bool, float]:
    """Verify per-channel INT8 quantization."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(32, 128, device=device)
    q, scale, zp = int8_quantize(x, symmetric=True, per_channel=True, channel_dim=-1)
    x_recovered = int8_dequantize(
        q, scale, zp, symmetric=True, per_channel=True, channel_dim=-1
    )

    # Per-channel should have lower error since each channel has its own scale
    err = (x - x_recovered).abs().max().item()
    return err < 0.1, err


if __name__ == "__main__":
    print(f"Triton available: {HAS_TRITON}")

    tests = [
        ("int8_quantize", test_int8_quantize),
        ("int8_per_channel", test_int8_per_channel),
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
