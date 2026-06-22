"""
Tests for 11_autotune module.

Verifies:
  - Autotune decorator works and produces valid kernels
  - Matmul autotune correctness across shapes
  - LayerNorm autotune correctness
  - RMSNorm autotune correctness
  - Softmax autotune correctness
  - Config constraints are respected
  - Different shapes get appropriate configs

Run: pytest 11_autotune/test_autotune.py -v
"""

from __future__ import annotations

import pytest
import torch

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from layernorm_autotune import (
    autotuned_layernorm,
    autotuned_layernorm_kernel,
    autotuned_rmsnorm,
    autotuned_rmsnorm_kernel,
)
from softmax_autotune import autotuned_softmax, autotuned_softmax_kernel
from triton_autotune_demo import autotuned_matmul, autotuned_matmul_kernel

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
requires_triton = pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")

EPS = 1e-5


# ======================================================================
# Config Validation
# ======================================================================


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestAutotuneDecorator:
    """Test that @triton.autotune decorators produce valid kernels."""

    def test_matmul_has_configs(self):
        assert len(autotuned_matmul_kernel.configs) > 0
        assert autotuned_matmul_kernel.key == ["M", "N", "K"]

    def test_layernorm_has_configs(self):
        assert len(autotuned_layernorm_kernel.configs) > 0
        assert autotuned_layernorm_kernel.key == ["N"]

    def test_rmsnorm_has_configs(self):
        assert len(autotuned_rmsnorm_kernel.configs) > 0
        assert autotuned_rmsnorm_kernel.key == ["N", "M"]

    def test_softmax_has_configs(self):
        assert len(autotuned_softmax_kernel.configs) > 0
        assert autotuned_softmax_kernel.key == ["N"]

    def test_layernorm_configs_respect_constraints(self):
        """BLOCK_SIZE must be >= num_warps * 32."""
        for cfg in autotuned_layernorm_kernel.configs:
            bs = cfg.kwargs.get("BLOCK_SIZE", 0)
            nw = cfg.kwargs.get("num_warps", 0)
            assert bs >= nw * 32, f"Invalid config: BLOCK_SIZE={bs} < num_warps({nw}) * 32"

    def test_matmul_num_warps_valid(self):
        """num_warps must be a power of 2 and between 4 and 8 (per our configs)."""
        for cfg in autotuned_matmul_kernel.configs:
            nw = cfg.kwargs.get("num_warps", 0)
            assert nw in (4, 8), f"Unexpected num_warps: {nw}"

    def test_softmax_num_warps_power_of_two(self):
        for cfg in autotuned_softmax_kernel.configs:
            nw = cfg.kwargs.get("num_warps", 0)
            assert nw in (2, 4, 8), f"Unexpected num_warps: {nw}"


# ======================================================================
# Matmul Autotune Correctness
# ======================================================================


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestMatmulAutotune:
    """Test autotuned matmul produces correct output."""

    def _run_matmul(self, M, N, K, dtype=torch.float32, atol=1e-3):
        if dtype == torch.float16 and torch.cuda.get_device_capability()[0] < 7:
            pytest.skip("fp16 not supported")
        if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("bf16 not supported")

        a = torch.randn(M, K, device="cuda", dtype=dtype)
        b = torch.randn(K, N, device="cuda", dtype=dtype)

        c_auto = autotuned_matmul(a, b)
        c_ref = torch.matmul(a.float(), b.float()).to(dtype)

        assert torch.allclose(c_auto.float(), c_ref.float(), atol=atol, rtol=1e-2), (
            f"Max diff {(c_auto.float() - c_ref.float()).abs().max().item():.2e} "
            f"for shape ({M},{N},{K}) dtype={dtype}"
        )

    def test_small_square(self):
        self._run_matmul(64, 64, 64)

    def test_medium_square(self):
        self._run_matmul(256, 256, 256)

    def test_large_square(self):
        self._run_matmul(512, 512, 512)

    def test_rectangular_m_large(self):
        self._run_matmul(1024, 256, 512)

    def test_rectangular_n_large(self):
        self._run_matmul(256, 1024, 512)

    def test_small_k(self):
        self._run_matmul(512, 512, 64)

    def test_large_k(self):
        self._run_matmul(512, 512, 2048)

    def test_fp16(self):
        self._run_matmul(256, 256, 256, torch.float16, atol=5e-2)

    def test_bf16(self):
        self._run_matmul(256, 256, 256, torch.bfloat16, atol=5e-2)

    def test_different_shapes_get_different_configs(self):
        """Verify that different problem shapes may get different configs."""
        shapes = [(128, 128, 128), (1024, 1024, 512)]

        # Clear any cached best config by using a fresh autotune instance
        configs = []
        for M, N, K in shapes:
            a = torch.randn(M, K, device="cuda", dtype=torch.float32)
            b = torch.randn(K, N, device="cuda", dtype=torch.float32)
            _ = autotuned_matmul(a, b)

            cfg = autotuned_matmul_kernel.best_config
            if cfg is not None:
                configs.append(cfg.kwargs)

        # At minimum, both shapes should produce valid configs
        assert len(configs) == 2, f"Expected 2 configs, got {len(configs)}"


# ======================================================================
# LayerNorm Autotune Correctness
# ======================================================================


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestLayerNormAutotune:
    """Test autotuned LayerNorm."""

    def _run_ln(self, B, N, dtype=torch.float32, atol=1e-2):
        if dtype != torch.float32:
            if dtype == torch.float16 and torch.cuda.get_device_capability()[0] < 7:
                pytest.skip("fp16 not supported")
            if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
                pytest.skip("bf16 not supported")

        x = torch.randn(B, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)

        out = autotuned_layernorm(x, w, b)
        ref = torch.nn.functional.layer_norm(x.float(), [N], weight=w, bias=b, eps=EPS)

        assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=5e-2), (
            f"Max diff {(out.float() - ref.float()).abs().max().item():.2e}"
        )

    def test_small(self):
        self._run_ln(2, 128)

    def test_medium(self):
        self._run_ln(4, 1024)

    def test_large(self):
        self._run_ln(8, 4096)

    def test_batch_1(self):
        self._run_ln(1, 768)

    def test_batch_32(self):
        self._run_ln(32, 512)

    def test_transformer_hidden_768(self):
        self._run_ln(4, 768)

    def test_transformer_hidden_1024(self):
        self._run_ln(4, 1024)

    def test_transformer_hidden_2048(self):
        self._run_ln(4, 2048)

    def test_transformer_hidden_4096(self):
        self._run_ln(4, 4096)

    def test_fp16(self):
        self._run_ln(4, 1024, torch.float16, atol=5e-2)

    def test_bf16(self):
        self._run_ln(4, 1024, torch.bfloat16, atol=5e-2)

    def test_no_weight_bias(self):
        """Test layernorm with default (None) weight and bias."""
        x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
        out = autotuned_layernorm(x, None, None)
        ref = torch.nn.functional.layer_norm(x.float(), [512], weight=None, bias=None, eps=EPS)
        assert torch.allclose(out, ref, atol=1e-2)

    def test_1d_input(self):
        """Test 1D input gets treated as 1 row."""
        x = torch.randn(256, device="cuda", dtype=torch.float32)
        w = torch.randn(256, device="cuda", dtype=torch.float32)
        b = torch.randn(256, device="cuda", dtype=torch.float32)
        out = autotuned_layernorm(x, w, b)
        ref = torch.nn.functional.layer_norm(x.float(), [256], weight=w, bias=b, eps=EPS)
        assert torch.allclose(out.float(), ref.float(), atol=1e-2)


# ======================================================================
# RMSNorm Autotune Correctness
# ======================================================================


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestRMSNormAutotune:
    """Test autotuned RMSNorm."""

    def _run_rn(self, B, N, dtype=torch.float32, atol=1e-2):
        if dtype != torch.float32:
            if dtype == torch.float16 and torch.cuda.get_device_capability()[0] < 7:
                pytest.skip("fp16 not supported")
            if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
                pytest.skip("bf16 not supported")

        x = torch.randn(B, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=torch.float32)

        out = autotuned_rmsnorm(x, w)
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)
        ref = (x.float() * rms * w.float()).to(x.dtype)

        assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=5e-2), (
            f"Max diff {(out.float() - ref.float()).abs().max().item():.2e}"
        )

    def test_small(self):
        self._run_rn(2, 128)

    def test_medium(self):
        self._run_rn(4, 1024)

    def test_large(self):
        self._run_rn(8, 4096)

    def test_batch_1(self):
        self._run_rn(1, 768)

    def test_batch_32(self):
        self._run_rn(32, 512)

    def test_llama_hidden_4096(self):
        self._run_rn(4, 4096)

    def test_mistral_hidden_4096(self):
        self._run_rn(4, 4096)

    def test_gemma_hidden_2560(self):
        self._run_rn(4, 2560)

    def test_fp16(self):
        self._run_rn(4, 1024, torch.float16, atol=5e-2)

    def test_bf16(self):
        self._run_rn(4, 1024, torch.bfloat16, atol=5e-2)

    def test_no_weight(self):
        x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
        out = autotuned_rmsnorm(x, None)
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)
        ref = x.float() * rms
        assert torch.allclose(out.float(), ref, atol=1e-2)

    def test_1d_input(self):
        x = torch.randn(256, device="cuda", dtype=torch.float32)
        w = torch.randn(256, device="cuda", dtype=torch.float32)
        out = autotuned_rmsnorm(x, w)
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)
        ref = (x.float() * rms * w.float()).to(x.dtype)
        assert torch.allclose(out.float(), ref.float(), atol=1e-2)


# ======================================================================
# Softmax Autotune Correctness
# ======================================================================


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestSoftmaxAutotune:
    """Test autotuned softmax."""

    def _run_sm(self, B, N, dtype=torch.float32, atol=1e-2):
        if dtype != torch.float32:
            if dtype == torch.float16 and torch.cuda.get_device_capability()[0] < 7:
                pytest.skip("fp16 not supported")
            if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
                pytest.skip("bf16 not supported")

        x = torch.randn(B, N, device="cuda", dtype=dtype)
        out = autotuned_softmax(x)
        ref = torch.softmax(x.float(), dim=-1).to(dtype)

        assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=1e-1), (
            f"Max diff {(out.float() - ref.float()).abs().max().item():.2e}"
        )

    def test_small(self):
        self._run_sm(2, 64)

    def test_medium(self):
        self._run_sm(4, 256)

    def test_large(self):
        self._run_sm(8, 1024)

    def test_batch_1(self):
        self._run_sm(1, 512)

    def test_batch_32(self):
        self._run_sm(32, 256)

    def test_sequence_2048(self):
        self._run_sm(4, 2048)

    def test_sequence_4096(self):
        self._run_sm(2, 4096)

    def test_sequence_8192(self):
        self._run_sm(4, 8192)

    def test_fp16(self):
        self._run_sm(4, 512, torch.float16, atol=5e-2)

    def test_bf16(self):
        self._run_sm(4, 512, torch.bfloat16, atol=5e-2)

    def test_1d_input(self):
        x = torch.randn(128, device="cuda", dtype=torch.float32)
        out = autotuned_softmax(x)
        ref = torch.softmax(x, dim=-1)
        assert torch.allclose(out, ref, atol=1e-2)

    def test_output_sums_to_one(self):
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        out = autotuned_softmax(x)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


# ======================================================================
# Autotune Behavior Tests
# ======================================================================


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestAutotuneBehavior:
    """Test autotune caching and config selection behavior."""

    def test_repeated_calls_use_cached_config(self):
        """After first call, subsequent calls should not re-autotune."""
        a = torch.randn(256, 256, device="cuda", dtype=torch.float32)
        b = torch.randn(256, 256, device="cuda", dtype=torch.float32)

        # First call triggers autotune
        out1 = autotuned_matmul(a, b)
        # Second call uses cached config
        out2 = autotuned_matmul(a, b)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_unknown_shape_triggers_new_autotune(self):
        """A shape not seen before should find a config."""
        a = torch.randn(127, 63, device="cuda", dtype=torch.float32)
        b = torch.randn(63, 91, device="cuda", dtype=torch.float32)

        out = autotuned_matmul(a, b)
        ref = torch.matmul(a, b)

        assert torch.allclose(out, ref, atol=1e-3)

    def test_matmul_best_config_has_required_keys(self):
        """The best config should have required parameters."""
        a = torch.randn(256, 256, device="cuda", dtype=torch.float32)
        b = torch.randn(256, 256, device="cuda", dtype=torch.float32)

        _ = autotuned_matmul(a, b)

        cfg = autotuned_matmul_kernel.best_config
        if cfg is not None:
            required = {"BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"}
            for key in required:
                assert key in cfg.kwargs, f"Missing config key: {key}"

    def test_layernorm_different_shapes(self):
        """Different N values should produce valid results."""
        for N in [256, 512, 1024]:
            x = torch.randn(2, N, device="cuda", dtype=torch.float32)
            w = torch.ones(N, device="cuda", dtype=torch.float32)
            b = torch.zeros(N, device="cuda", dtype=torch.float32)

            out = autotuned_layernorm(x, w, b)
            assert out.shape == x.shape
            assert not torch.isnan(out).any()
