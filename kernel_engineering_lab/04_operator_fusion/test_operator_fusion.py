"""
Tests for 04_operator_fusion kernels.

Verifies correctness of fused kernels against PyTorch sequential equivalents
under various shapes, dtypes, and edge cases.

Run: pytest 04_operator_fusion/test_operator_fusion.py -v
"""

import pytest
import torch

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from kernel_add_relu import fused_add_relu, sequential_add_relu
from kernel_bias_gelu import fused_bias_gelu, sequential_bias_gelu
from kernel_residual_layernorm import fused_residual_layernorm, sequential_residual_layernorm
from kernel_rmsnorm import triton_rmsnorm, torch_rmsnorm

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

requires_triton = pytest.mark.skipif(
    not TRITON_AVAILABLE,
    reason="Triton not installed",
)


def _check_dtype(dtype: torch.dtype) -> None:
    """Skip test if GPU doesn't support the dtype."""
    if dtype == torch.float16 and torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("float16 requires compute capability >= 7.0")
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bfloat16 requires compute capability >= 8.0")


# ---------------------------------------------------------------------------
# Add + ReLU
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestAddReLU:
    """Test fused add+relu kernel."""

    TRANSFORMER_DIMS = [512, 768, 1024, 2048, 4096, 8192]
    DTYPES = [torch.float32, torch.float16, torch.bfloat16]

    def _run_test(self, shape, dtype=torch.float32):
        _check_dtype(dtype)
        x = torch.randn(shape, device="cuda", dtype=dtype)
        bias = torch.randn(shape, device="cuda", dtype=dtype)
        y_f = fused_add_relu(x, bias)
        y_s = sequential_add_relu(x, bias)
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        assert torch.allclose(y_f, y_s, atol=atol, rtol=rtol), (
            f"Max error: {(y_f - y_s).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("dim", TRANSFORMER_DIMS)
    def test_1d_transformer_dims_fp32(self, dim):
        self._run_test((dim,), torch.float32)

    @pytest.mark.parametrize("dim", [512, 768, 1024])
    def test_1d_fp16(self, dim):
        self._run_test((dim,), torch.float16)

    @pytest.mark.parametrize("dim", [512, 768, 1024])
    def test_1d_bf16(self, dim):
        self._run_test((dim,), torch.bfloat16)

    def test_2d_single_batch(self):
        self._run_test((1, 1024))

    def test_2d_batch_4(self):
        self._run_test((4, 1024))

    def test_2d_batch_32(self):
        self._run_test((32, 768))

    def test_3d_tensor(self):
        self._run_test((2, 8, 512))

    def test_non_power_of_two(self):
        self._run_test((999,))

    def test_large_tensor(self):
        self._run_test((10_000_000,))

    def test_small_tensor(self):
        self._run_test((1,))

    def test_broadcast_bias(self):
        x = torch.randn(4, 1024, device="cuda")
        bias = torch.randn(1024, device="cuda")
        y_f = fused_add_relu(x, bias)
        y_s = sequential_add_relu(x, bias)
        assert torch.allclose(y_f, y_s, atol=1e-5)


# ---------------------------------------------------------------------------
# Bias + GELU
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestBiasGELU:
    """Test fused bias+gelu kernel."""

    TRANSFORMER_DIMS = [512, 768, 1024, 2048, 4096]
    DTYPES = [torch.float32, torch.float16, torch.bfloat16]

    def _run_test(self, shape, dtype=torch.float32):
        _check_dtype(dtype)
        x = torch.randn(shape, device="cuda", dtype=dtype)
        bias = torch.randn(shape, device="cuda", dtype=dtype)
        y_f = fused_bias_gelu(x, bias)
        y_s = sequential_bias_gelu(x, bias)
        rtol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        assert torch.allclose(y_f.float(), y_s.float(), rtol=rtol, atol=atol), (
            f"Max error ({shape}, {dtype}): {(y_f.float() - y_s.float()).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("dim", TRANSFORMER_DIMS)
    def test_1d_transformer_dims_fp32(self, dim):
        self._run_test((dim,), torch.float32)

    @pytest.mark.parametrize("dim", [512, 1024])
    def test_1d_fp16(self, dim):
        self._run_test((dim,), torch.float16)

    @pytest.mark.parametrize("dim", [512, 1024])
    def test_1d_bf16(self, dim):
        self._run_test((dim,), torch.bfloat16)

    def test_2d_batch(self):
        self._run_test((2, 768))

    def test_2d_large_batch(self):
        self._run_test((4096, 512))

    def test_3d_tensor(self):
        self._run_test((4, 2, 512))

    def test_non_power_of_two(self):
        self._run_test((1000,))

    def test_multiple_block_sizes(self):
        for bs in [256, 512, 1024]:
            x = torch.randn(2048, device="cuda")
            bias = torch.randn(2048, device="cuda")
            y_f = fused_bias_gelu(x, bias, block_size=bs)
            y_s = sequential_bias_gelu(x, bias)
            assert torch.allclose(y_f, y_s, atol=1e-5), f"Failed for block_size={bs}"


# ---------------------------------------------------------------------------
# Residual + LayerNorm
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestResidualLayerNorm:
    """Test fused residual+layernorm kernel."""

    def _run_test(self, rows, cols, dtype=torch.float32):
        _check_dtype(dtype)
        x = torch.randn(rows, cols, device="cuda", dtype=dtype)
        residual = torch.randn(rows, cols, device="cuda", dtype=dtype)
        y_f = fused_residual_layernorm(x, residual, block_size=cols)
        y_s = sequential_residual_layernorm(x, residual)
        rtol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        assert torch.allclose(y_f.float(), y_s.float(), rtol=rtol, atol=atol), (
            f"Max error: {(y_f.float() - y_s.float()).abs().max().item():.2e}"
        )

    def test_small(self):
        self._run_test(2, 128)

    def test_medium(self):
        self._run_test(4, 1024)

    def test_large(self):
        self._run_test(8, 4096)

    def test_batch_1(self):
        self._run_test(1, 768)

    def test_batch_32(self):
        self._run_test(32, 512)

    def test_transformer_hidden_768(self):
        self._run_test(4, 768)

    def test_transformer_hidden_1024(self):
        self._run_test(4, 1024)

    def test_transformer_hidden_2048(self):
        self._run_test(2, 2048)

    def test_transformer_hidden_4096(self):
        self._run_test(2, 4096)

    def test_fp16(self):
        self._run_test(4, 1024, torch.float16)

    def test_bf16(self):
        self._run_test(4, 1024, torch.bfloat16)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestRMSNorm:
    """Test rmsnorm kernel."""

    def _run_test(self, rows, cols, dtype=torch.float32):
        _check_dtype(dtype)
        x = torch.randn(rows, cols, device="cuda", dtype=dtype)
        weight = torch.randn(cols, device="cuda", dtype=torch.float32)
        y_triton = triton_rmsnorm(x, weight, block_size=cols)
        y_torch = torch_rmsnorm(x, weight)
        rtol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        atol = 1e-2
        assert torch.allclose(y_triton.float(), y_torch.float(), rtol=rtol, atol=atol), (
            f"Max error: {(y_triton.float() - y_torch.float()).abs().max().item():.2e}"
        )

    def test_small(self):
        self._run_test(2, 128)

    def test_medium(self):
        self._run_test(4, 1024)

    def test_large(self):
        self._run_test(8, 4096)

    def test_batch_1(self):
        self._run_test(1, 768)

    def test_batch_32(self):
        self._run_test(32, 512)

    def test_llama_hidden_4096(self):
        self._run_test(4, 4096)

    def test_mistral_hidden_4096(self):
        self._run_test(4, 4096)

    def test_gemma_hidden_2560(self):
        self._run_test(4, 2560)

    def test_fp16(self):
        self._run_test(4, 1024, torch.float16)

    def test_bf16(self):
        self._run_test(4, 1024, torch.bfloat16)

    def test_weight_all_ones(self):
        x = torch.randn(2, 1024, device="cuda")
        weight = torch.ones(1024, device="cuda")
        y_triton = triton_rmsnorm(x, weight, block_size=1024)
        y_torch = torch_rmsnorm(x, weight)
        assert torch.allclose(y_triton, y_torch, atol=1e-3)

    def test_weight_varied(self):
        x = torch.randn(4, 512, device="cuda")
        weight = torch.randn(512, device="cuda")
        y_triton = triton_rmsnorm(x, weight, block_size=512)
        y_torch = torch_rmsnorm(x, weight)
        assert torch.allclose(y_triton.float(), y_torch.float(), atol=1e-2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestEdgeCases:
    """Test edge cases for all fusion kernels."""

    def test_add_relu_single_element(self):
        x = torch.tensor([-1.0], device="cuda")
        bias = torch.tensor([0.5], device="cuda")
        y = fused_add_relu(x, bias)
        assert y.item() == 0.0, f"Expected 0.0 (relu(-0.5)), got {y.item()}"

    def test_add_relu_all_positive(self):
        x = torch.ones(1000, device="cuda")
        bias = torch.ones(1000, device="cuda")
        y = fused_add_relu(x, bias)
        assert torch.allclose(y, torch.ones(1000, device="cuda") + 1.0, atol=1e-5)

    def test_add_relu_all_negative(self):
        x = -torch.ones(1000, device="cuda")
        bias = -torch.ones(1000, device="cuda")
        y = fused_add_relu(x, bias)
        assert torch.all(y == 0.0)

    def test_bias_gelu_edge_values(self):
        x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0], device="cuda")
        bias = torch.zeros_like(x)
        y = fused_bias_gelu(x, bias)
        # GELU is close to 0 for large negative values, identity for large positive
        assert y[0] < 0.01, f"GELU(-10) should be near 0, got {y[0]}"
        assert y[-1] > 9.0, f"GELU(10) should be near 10, got {y[-1]}"
        assert abs(y[2] - 0.0) < 0.1, f"GELU(0) should be near 0, got {y[2]}"
