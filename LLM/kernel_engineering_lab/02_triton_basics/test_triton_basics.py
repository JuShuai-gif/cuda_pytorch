"""
Tests for 02_triton_basics Triton kernels.

Verifies correctness of Triton vector_add, activation functions (SiLU, GELU, ReLU),
and basic GEMM against PyTorch reference implementations.

Run: pytest 02_triton_basics/test_triton_basics.py -v
"""

import pytest
import torch

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    import triton.language as tl  # noqa: F401
except ImportError:
    pass

from triton_vector_add import triton_vector_add
from triton_elementwise import triton_gelu, triton_relu, triton_silu
from triton_gemm_basic import triton_gemm


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

requires_triton = pytest.mark.skipif(
    not TRITON_AVAILABLE,
    reason="Triton not installed",
)


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestVectorAdd:
    """Test Triton vector_add against torch.add."""

    def _run_add_test(self, shape, dtype=torch.float32, block_size=1024):
        if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("bfloat16 requires compute capability >= 8.0")
        if dtype == torch.float16 and torch.cuda.get_device_capability()[0] < 7:
            pytest.skip("float16 requires compute capability >= 7.0")

        a = torch.randn(shape, device="cuda", dtype=dtype)
        b = torch.randn(shape, device="cuda", dtype=dtype)

        result = triton_vector_add(a, b, block_size=block_size)
        expected = a + b

        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        assert torch.allclose(result, expected, atol=atol, rtol=rtol), (
            f"Max error: {(result - expected).abs().max().item():.2e}"
        )

    def test_fp32_power_of_two(self):
        self._run_add_test(1024, torch.float32)

    def test_fp32_non_power_of_two(self):
        self._run_add_test(999, torch.float32)

    def test_fp32_large(self):
        self._run_add_test(1_000_000, torch.float32)

    def test_fp16(self):
        self._run_add_test(4096, torch.float16)

    def test_bf16(self):
        self._run_add_test(4096, torch.bfloat16)

    def test_different_block_sizes(self):
        for bs in [128, 256, 512, 1024]:
            self._run_add_test(4000, torch.float32, block_size=bs)

    def test_2d_tensor(self):
        a = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        b = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        result = triton_vector_add(a, b)
        expected = a + b
        assert torch.allclose(result, expected, atol=1e-5)

    def test_3d_tensor(self):
        a = torch.randn(2, 3, 100, device="cuda", dtype=torch.float32)
        b = torch.randn(2, 3, 100, device="cuda", dtype=torch.float32)
        result = triton_vector_add(a, b)
        expected = a + b
        assert torch.allclose(result, expected, atol=1e-5)


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestActivations:
    """Test Triton activation functions against PyTorch equivalents."""

    SHAPES = [
        (1024,),
        (999,),
        (10_000,),
        (4, 256),
        (2, 3, 128),
    ]

    DTYPES = [torch.float32, torch.float16]

    def _run_activation_test(self, triton_fn, torch_fn, name, shape, dtype):
        if dtype == torch.float16 and torch.cuda.get_device_capability()[0] < 7:
            pytest.skip("float16 requires compute capability >= 7.0")

        x = torch.randn(shape, device="cuda", dtype=dtype)
        result = triton_fn(x)
        # PyTorch activations work in fp32 internally, upcast for comparison
        expected = torch_fn(x.float())

        rtol = 5e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-2 if dtype == torch.float16 else 1e-5
        assert torch.allclose(result.float(), expected.float(), rtol=rtol, atol=atol), (
            f"{name} ({dtype}) [{shape}]: max error = "
            f"{(result.float() - expected.float()).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("shape", SHAPES)
    def test_silu(self, shape):
        self._run_activation_test(
            triton_silu, torch.nn.functional.silu, "SiLU", shape, torch.float32
        )

    @pytest.mark.parametrize("shape", SHAPES)
    def test_gelu(self, shape):
        gelu_fn = lambda t: torch.nn.functional.gelu(t, approximate="tanh")
        self._run_activation_test(triton_gelu, gelu_fn, "GELU", shape, torch.float32)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_relu(self, shape):
        self._run_activation_test(
            triton_relu, torch.nn.functional.relu, "ReLU", shape, torch.float32
        )

    def test_silu_fp16(self):
        self._run_activation_test(
            triton_silu, torch.nn.functional.silu, "SiLU", (4096,), torch.float16
        )

    def test_gelu_fp16(self):
        gelu_fn = lambda t: torch.nn.functional.gelu(t, approximate="tanh")
        self._run_activation_test(triton_gelu, gelu_fn, "GELU", (4096,), torch.float16)

    def test_relu_fp16(self):
        self._run_activation_test(
            triton_relu, torch.nn.functional.relu, "ReLU", (4096,), torch.float16
        )


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestGEMM:
    """Test Triton basic GEMM against torch.matmul."""

    def _run_gemm_test(self, M, N, K):
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        c_triton = triton_gemm(a, b)
        c_torch = torch.matmul(a, b)

        assert torch.allclose(c_triton, c_torch, atol=1e-3, rtol=1e-3), (
            f"GEMM {M}x{K}x{N}: max error = {(c_triton - c_torch).abs().max().item():.2e}"
        )

    def test_small_square(self):
        self._run_gemm_test(64, 64, 64)

    def test_power_of_two(self):
        self._run_gemm_test(256, 256, 256)

    def test_rectangular(self):
        self._run_gemm_test(128, 256, 64)

    def test_non_power_of_two(self):
        self._run_gemm_test(100, 100, 75)

    def test_large(self):
        self._run_gemm_test(512, 512, 512)
