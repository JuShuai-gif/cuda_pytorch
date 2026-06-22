"""
Tests for 05_matmul_tiling kernels.

Verifies correctness across naive, tiled, optimized, and batched matmul
implementations against torch.matmul.

Run: pytest 05_matmul_tiling/test_matmul_tiling.py -v
"""

import pytest
import torch

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from naive_matmul import naive_matmul
from tiled_matmul import tiled_matmul
from triton_matmul_optimized import optimized_matmul_preset
from batched_matmul import batched_matmul

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

requires_triton = pytest.mark.skipif(
    not TRITON_AVAILABLE,
    reason="Triton not installed",
)


def _check_dtype(dtype: torch.dtype) -> None:
    if dtype == torch.float16 and torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("float16 requires compute capability >= 7.0")
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bfloat16 requires compute capability >= 8.0")


# ---------------------------------------------------------------------------
# Naive matmul tests
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestNaiveMatmul:
    """Test naive matmul (no shared memory tiling)."""

    def _run_test(self, M, N, K, dtype=torch.float32):
        _check_dtype(dtype)
        a = torch.randn(M, K, device="cuda", dtype=dtype)
        b = torch.randn(K, N, device="cuda", dtype=dtype)
        c = naive_matmul(a, b)
        expected = torch.matmul(a, b)
        rtol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        assert torch.allclose(c.float(), expected.float(), rtol=rtol, atol=atol), (
            f"[{M}x{K}x{N} {dtype}] Max error: {(c.float() - expected.float()).abs().max().item():.2e}"
        )

    def test_small_square(self):
        self._run_test(128, 128, 128)

    def test_medium_square(self):
        self._run_test(512, 512, 512)

    def test_rectangular(self):
        self._run_test(256, 512, 128)

    def test_non_power_of_two(self):
        self._run_test(100, 100, 75)

    def test_non_square_1(self):
        self._run_test(1024, 768, 512)

    def test_non_square_2(self):
        self._run_test(768, 2048, 512)

    def test_fp16(self):
        self._run_test(256, 256, 256, torch.float16)

    def test_bf16(self):
        self._run_test(256, 256, 256, torch.bfloat16)

    def test_small_k(self):
        self._run_test(1024, 1024, 64)

    def test_large(self):
        self._run_test(2048, 2048, 1024)


# ---------------------------------------------------------------------------
# Tiled matmul tests
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestTiledMatmul:
    """Test tiled matmul with shared memory."""

    def _run_test(self, M, N, K, dtype=torch.float32):
        _check_dtype(dtype)
        a = torch.randn(M, K, device="cuda", dtype=dtype)
        b = torch.randn(K, N, device="cuda", dtype=dtype)
        c = tiled_matmul(a, b)
        expected = torch.matmul(a, b)
        rtol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        assert torch.allclose(c.float(), expected.float(), rtol=rtol, atol=atol), (
            f"[{M}x{K}x{N} {dtype}] Max error: {(c.float() - expected.float()).abs().max().item():.2e}"
        )

    def test_small(self):
        self._run_test(128, 128, 128)

    def test_medium(self):
        self._run_test(1024, 1024, 1024)

    def test_large(self):
        self._run_test(4096, 4096, 4096)

    def test_rectangular(self):
        self._run_test(512, 256, 1024)

    def test_non_square_1(self):
        self._run_test(1024, 768, 512)

    def test_non_square_2(self):
        self._run_test(768, 2048, 512)

    def test_non_power_of_two(self):
        self._run_test(100, 100, 75)

    def test_fp16(self):
        self._run_test(256, 256, 256, torch.float16)

    def test_bf16(self):
        self._run_test(256, 256, 256, torch.bfloat16)

    def test_various_block_sizes(self):
        a = torch.randn(256, 512, device="cuda", dtype=torch.float32)
        b = torch.randn(512, 256, device="cuda", dtype=torch.float32)
        expected = torch.matmul(a, b)
        for bm in (32, 64, 128):
            for bn in (32, 64, 128):
                for bk in (16, 32):
                    c = tiled_matmul(a, b, block_m=bm, block_n=bn, block_k=bk)
                    err = (c - expected).abs().max().item()
                    assert err < 1e-2, f"block_{bm}x{bn}x{bk}: max error = {err:.2e}"


# ---------------------------------------------------------------------------
# Optimized matmul tests
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestOptimizedMatmul:
    """Test optimized matmul with warp configs."""

    def _run_test(self, M, N, K, preset="medium", dtype=torch.float32):
        _check_dtype(dtype)
        a = torch.randn(M, K, device="cuda", dtype=dtype)
        b = torch.randn(K, N, device="cuda", dtype=dtype)
        c = optimized_matmul_preset(a, b, preset)
        expected = torch.matmul(a, b)
        rtol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-2
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        assert torch.allclose(c.float(), expected.float(), rtol=rtol, atol=atol), (
            f"[{M}x{K}x{N} {preset} {dtype}] Max error: "
            f"{(c.float() - expected.float()).abs().max().item():.2e}"
        )

    def test_small_preset(self):
        self._run_test(128, 128, 128, "small")

    def test_medium_preset(self):
        self._run_test(512, 512, 256, "medium")

    def test_large_preset(self):
        self._run_test(1024, 1024, 512, "large")

    def test_rectangular_medium(self):
        self._run_test(256, 512, 128, "medium")

    def test_rectangular_large(self):
        self._run_test(1024, 768, 512, "large")

    def test_non_power_of_two(self):
        self._run_test(100, 100, 75, "medium")

    def test_fp16_small(self):
        self._run_test(256, 256, 256, "small", torch.float16)

    def test_fp16_large(self):
        self._run_test(1024, 1024, 512, "large", torch.float16)

    def test_bf16(self):
        self._run_test(256, 256, 256, "medium", torch.bfloat16)

    def test_llama_qkv_4096_128(self):
        self._run_test(4096, 128, 4096, "medium")

    def test_llama_ffn_up_4096_11008(self):
        self._run_test(4096, 11008, 4096, "large")

    def test_llama_ffn_down_11008_4096(self):
        self._run_test(11008, 4096, 4096, "large")


# ---------------------------------------------------------------------------
# Batched matmul tests
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestBatchedMatmul:
    """Test batched matmul."""

    def _run_test(self, B, M, N, K, dtype=torch.float32):
        _check_dtype(dtype)
        a = torch.randn(B, M, K, device="cuda", dtype=dtype)
        b = torch.randn(B, K, N, device="cuda", dtype=dtype)
        c = batched_matmul(a, b)
        expected = torch.bmm(a, b)
        rtol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        assert torch.allclose(c.float(), expected.float(), rtol=rtol, atol=atol), (
            f"[{B}x{M}x{K}x{N} {dtype}] Max error: "
            f"{(c.float() - expected.float()).abs().max().item():.2e}"
        )

    def test_batch_1_small(self):
        self._run_test(1, 128, 128, 128)

    def test_batch_4_medium(self):
        self._run_test(4, 256, 256, 256)

    def test_batch_32_small(self):
        self._run_test(32, 64, 64, 64)

    def test_batch_8_large(self):
        self._run_test(8, 512, 512, 512)

    def test_batch_rectangular(self):
        self._run_test(4, 512, 256, 768)

    def test_fp16(self):
        self._run_test(4, 256, 256, 256, torch.float16)

    def test_bf16(self):
        self._run_test(4, 256, 256, 256, torch.bfloat16)

    def test_llama_heads_b1_d64(self):
        self._run_test(32, 1, 64, 64)

    def test_llama_heads_b128_d64(self):
        self._run_test(32, 128, 64, 64)

    def test_llama_heads_b1024_d128(self):
        self._run_test(32, 1024, 128, 128)


# ---------------------------------------------------------------------------
# Cross-kernel correctness: naive == tiled == optimized == torch
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestCrossKernelConsistency:
    """Verify all matmul implementations produce consistent results."""

    def test_all_kernels_match(self):
        M, N, K = 256, 256, 128
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        c_naive = naive_matmul(a, b)
        c_tiled = tiled_matmul(a, b)
        c_opt = optimized_matmul_preset(a, b, "medium")
        c_torch = torch.matmul(a, b)

        # Check consistency between kernels
        assert torch.allclose(c_tiled, c_opt, atol=1e-3, rtol=1e-3), (
            f"Tiled vs optimized mismatch: {(c_tiled - c_opt).abs().max().item():.2e}"
        )
        assert torch.allclose(c_naive, c_tiled, atol=1e-3, rtol=1e-3), (
            f"Naive vs tiled mismatch: {(c_naive - c_tiled).abs().max().item():.2e}"
        )

    def test_all_kernels_rectangular(self):
        M, N, K = 512, 256, 384
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        c_naive = naive_matmul(a, b)
        c_tiled = tiled_matmul(a, b)
        c_opt = optimized_matmul_preset(a, b, "medium")
        c_torch = torch.matmul(a, b)

        for name, result in [
            ("naive", c_naive),
            ("tiled", c_tiled),
            ("opt", c_opt),
        ]:
            assert torch.allclose(result, c_torch, atol=1e-3, rtol=1e-3), (
                f"{name} vs torch: max error = {(result - c_torch).abs().max().item():.2e}"
            )
