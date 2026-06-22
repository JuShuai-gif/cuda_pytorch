"""
Tests for 06_attention_flash_like kernels.

Verifies correctness of naive, tiled, prefill, and decode attention
implementations against torch.nn.functional.scaled_dot_product_attention.

Run: pytest 06_attention_flash_like/test_attention.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from flash_attention_kv_cache import attention_decode, attention_prefill
from naive_attention import naive_attention_torch, naive_attention_triton
from tiled_attention import _scaled_dot_product_attention_ref, tiled_attention

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

requires_triton = pytest.mark.skipif(
    not TRITON_AVAILABLE,
    reason="Triton not installed",
)


_BF16_CAPABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
_FP16_CAPABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7


def _check_dtype(dtype: torch.dtype) -> None:
    if dtype == torch.float16 and not _FP16_CAPABLE:
        pytest.skip("float16 requires compute capability >= 7.0")
    if dtype == torch.bfloat16 and not _BF16_CAPABLE:
        pytest.skip("bfloat16 requires compute capability >= 8.0")


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference using manual implementation (for cross-checking torch SDP)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        Q_len, KV_len = q.shape[-2], k.shape[-2]
        diag = KV_len - Q_len + 1 if KV_len > Q_len else 1
        m = torch.triu(torch.ones(Q_len, KV_len, device=s.device, dtype=torch.bool), diagonal=diag)
        s = s.masked_fill(m, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


# ---------------------------------------------------------------------------
# Naive attention tests
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
class TestNaiveAttention:
    """Test naive attention (both PyTorch and Triton implementations)."""

    DTYPES = [torch.float32]
    # Add fp16/bf16 if supported
    if _FP16_CAPABLE:
        DTYPES.append(torch.float16)
    if _BF16_CAPABLE:
        DTYPES.append(torch.bfloat16)

    SHAPES = [
        (1, 1, 64, 64, 64),
        (1, 4, 128, 128, 64),
        (2, 8, 256, 256, 64),
        (1, 4, 512, 512, 128),
        (2, 1, 128, 256, 64),
        (1, 8, 256, 512, 128),
    ]

    @pytest.mark.parametrize("B,H,Q_len,KV_len,D", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_naive_torch_vs_reference(self, B, H, Q_len, KV_len, D, dtype):
        _check_dtype(dtype)
        q = torch.randn(B, H, Q_len, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=dtype)

        out = naive_attention_torch(q, k, v)
        ref = _reference_attention(q, k, v)
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        rtol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol), (
            f"shape=({B},{H},{Q_len},{KV_len},{D}) dtype={dtype}: "
            f"max err={(out.float() - ref.float()).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("B,H,Q_len,KV_len,D", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_naive_triton_vs_reference(self, B, H, Q_len, KV_len, D, dtype):
        _check_dtype(dtype)
        q = torch.randn(B, H, Q_len, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=dtype)

        out = naive_attention_triton(q, k, v)
        ref = _reference_attention(q, k, v)
        atol = 1e-2
        rtol = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 5e-2
        assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol), (
            f"shape=({B},{H},{Q_len},{KV_len},{D}) dtype={dtype}: "
            f"max err={(out.float() - ref.float()).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_naive_torch_causal(self, dtype):
        _check_dtype(dtype)
        B, H, Q_len, D = 1, 2, 64, 64
        q = torch.randn(B, H, Q_len, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, Q_len, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, Q_len, D, device="cuda", dtype=dtype)

        out = naive_attention_torch(q, k, v, causal_mask=True)
        ref = _reference_attention(q, k, v, causal=True)
        rtol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=rtol)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_naive_triton_causal(self, dtype):
        _check_dtype(dtype)
        B, H, Q_len, D = 1, 2, 64, 64
        q = torch.randn(B, H, Q_len, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, Q_len, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, Q_len, D, device="cuda", dtype=dtype)

        out = naive_attention_triton(q, k, v, causal_mask=True)
        ref = _reference_attention(q, k, v, causal=True)
        rtol = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 5e-2
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=rtol)


# ---------------------------------------------------------------------------
# Tiled attention tests
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
class TestTiledAttention:
    """Test tiled attention with online softmax."""

    DTYPES = [torch.float32]
    if _FP16_CAPABLE:
        DTYPES.append(torch.float16)
    if _BF16_CAPABLE:
        DTYPES.append(torch.bfloat16)

    SHAPES = [
        (1, 1, 64, 64, 64),
        (1, 4, 128, 128, 64),
        (2, 8, 256, 256, 64),
        (1, 4, 512, 512, 128),
        (2, 1, 128, 256, 64),
        (1, 8, 256, 512, 128),
    ]

    @pytest.mark.parametrize("B,H,Q_len,KV_len,D", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_tiled_vs_torch_sdpa(self, B, H, Q_len, KV_len, D, dtype):
        _check_dtype(dtype)
        q = torch.randn(B, H, Q_len, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=dtype)

        out = tiled_attention(q, k, v)
        ref = _scaled_dot_product_attention_ref(q, k, v, causal_mask=False)
        atol = 1e-2
        rtol = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 5e-2
        assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol), (
            f"shape=({B},{H},{Q_len},{KV_len},{D}) dtype={dtype}: "
            f"max err={(out.float() - ref.float()).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_tiled_causal(self, dtype):
        _check_dtype(dtype)
        B, H, L, D = 1, 4, 128, 64
        q = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, L, D, device="cuda", dtype=dtype)

        out = tiled_attention(q, k, v, causal_mask=True)
        ref = _scaled_dot_product_attention_ref(q, k, v, causal_mask=True)
        rtol = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 5e-2
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=rtol)

    def test_tiled_block_size_variations(self):
        B, H, L, D = 1, 2, 128, 64
        q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        ref = _scaled_dot_product_attention_ref(q, k, v)

        for bm in (32, 64):
            for bn in (32, 64):
                out = tiled_attention(q, k, v, block_m=bm, block_n=bn, block_d=64)
                err = (out.float() - ref.float()).abs().max().item()
                assert err < 5e-2, f"block_{bm}x{bn}x64: max error = {err:.2e}"

    def test_tiled_decode_pattern(self):
        """Test decode pattern: Q_len=1, KV_len >> Q_len."""
        B, H, KV_len, D = 2, 4, 256, 64
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)

        out = tiled_attention(q, k, v)
        ref = _scaled_dot_product_attention_ref(q, k, v)
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Prefill attention tests
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
class TestPrefillAttention:
    """Test prefill attention kernel."""

    DTYPES = [torch.float32]
    if _FP16_CAPABLE:
        DTYPES.append(torch.float16)
    if _BF16_CAPABLE:
        DTYPES.append(torch.bfloat16)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_prefill_vs_torch(self, dtype):
        _check_dtype(dtype)
        B, H, L, D = 2, 4, 128, 64
        q = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, L, D, device="cuda", dtype=dtype)

        out = attention_prefill(q, k, v, causal_mask=False)
        ref = _scaled_dot_product_attention_ref(q, k, v, causal_mask=False)
        rtol = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 5e-2
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=rtol)

    def test_prefill_causal(self):
        B, H, L, D = 1, 4, 128, 64
        q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)

        out = attention_prefill(q, k, v, causal_mask=True)
        ref = _scaled_dot_product_attention_ref(q, k, v, causal_mask=True)
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=5e-2)

    def test_prefill_long_sequence(self):
        B, H, L, D = 1, 2, 512, 64
        q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)

        out = attention_prefill(q, k, v)
        ref = _scaled_dot_product_attention_ref(q, k, v)
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=5e-2)

    def test_prefill_cross_attention(self):
        """Test KV_len != Q_len (cross-attention or KV cache scenario)."""
        B, H, Q_len, KV_len, D = 1, 2, 64, 128, 64
        q = torch.randn(B, H, Q_len, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)

        out = attention_prefill(q, k, v)
        ref = _scaled_dot_product_attention_ref(q, k, v)
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Decode attention tests
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
class TestDecodeAttention:
    """Test decode attention kernel (single Q token)."""

    DTYPES = [torch.float32]
    if _FP16_CAPABLE:
        DTYPES.append(torch.float16)
    if _BF16_CAPABLE:
        DTYPES.append(torch.bfloat16)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_decode_vs_torch(self, dtype):
        _check_dtype(dtype)
        B, H, KV_len, D = 2, 4, 128, 64
        q = torch.randn(B, H, 1, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=dtype)

        out = attention_decode(q, k, v)
        ref = _scaled_dot_product_attention_ref(q, k, v)
        rtol = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 5e-2
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=rtol)

    def test_decode_long_cache(self):
        """Test decode with long KV cache."""
        B, H, KV_len, D = 1, 2, 1024, 64
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)

        out = attention_decode(q, k, v)
        ref = _scaled_dot_product_attention_ref(q, k, v)
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Numerical stability tests
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
class TestNumericalStability:
    """Test numerical stability with extreme values and long sequences."""

    def test_large_seq_len(self):
        """Test with larger sequence length to exercise online softmax."""
        B, H, L, D = 1, 2, 256, 64
        q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)

        out_tiled = tiled_attention(q, k, v)
        out_prefill = attention_prefill(q, k, v)
        ref = _scaled_dot_product_attention_ref(q, k, v)

        assert torch.allclose(out_tiled.float(), ref.float(), atol=1e-2, rtol=5e-2)
        assert torch.allclose(out_prefill.float(), ref.float(), atol=1e-2, rtol=5e-2)

    def test_extreme_values(self):
        """Test with very large and negative query values."""
        B, H, L, D = 1, 2, 64, 64
        q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32) * 10.0
        k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32) * 10.0
        v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)

        out_tiled = tiled_attention(q, k, v)
        ref = _scaled_dot_product_attention_ref(q, k, v)
        assert torch.allclose(out_tiled.float(), ref.float(), atol=1e-2, rtol=5e-2)

    def test_online_softmax_consistency(self):
        """Verify online softmax matches full softmax."""
        B, H, L, D = 1, 2, 64, 64
        q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)

        out_naive = naive_attention_torch(q, k, v)
        out_tiled = tiled_attention(q, k, v)

        assert torch.allclose(out_naive.float(), out_tiled.float(), atol=1e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
class TestCrossImplementationConsistency:
    """Verify all implementations produce consistent results."""

    def test_naive_tiled_prefill_match(self):
        B, H, L, D = 1, 2, 128, 64
        q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)

        naive = naive_attention_torch(q, k, v)
        naive_t = naive_attention_triton(q, k, v)
        tiled = tiled_attention(q, k, v)
        prefill = attention_prefill(q, k, v)
        ref = _scaled_dot_product_attention_ref(q, k, v)

        for name, result in [
            ("naive_torch", naive),
            ("naive_triton", naive_t),
            ("tiled", tiled),
            ("prefill", prefill),
        ]:
            assert torch.allclose(result.float(), ref.float(), atol=1e-2, rtol=5e-2), (
                f"{name}: max error = {(result.float() - ref.float()).abs().max().item():.2e}"
            )
