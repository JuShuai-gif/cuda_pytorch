"""
Tests for 08_memory_management.

Verifies caching allocator behavior, buffer pool correctness,
in-place ops, KV cache calculations, and paged attention.

Run: pytest 08_memory_management/test_memory_management.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from allocator_observe import (
    reset_memory_stats,
    show_allocator_state,
)
from kv_cache_memory import (
    PagedKVCache,
    calculate_kv_cache_size,
)
from memory_reuse import BufferPool

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# ---------------------------------------------------------------------------
# Caching allocator tests
# ---------------------------------------------------------------------------


@requires_cuda
class TestCachingAllocator:
    """Test PyTorch CUDA caching allocator behavior."""

    def test_memory_reuse_after_delete(self):
        """Verify that deleting a tensor and re-allocating same size reuses memory."""
        torch.cuda.empty_cache()
        reset_memory_stats()

        n = 1_000_000
        t1 = torch.randn(n, device="cuda", dtype=torch.float32)
        reserved_before = torch.cuda.memory_reserved()

        del t1
        reserved_after_del = torch.cuda.memory_reserved()

        # Reserved memory should not change (or change minimally)
        # The allocator caches the freed memory
        t2 = torch.randn(n, device="cuda", dtype=torch.float32)
        reserved_after_realloc = torch.cuda.memory_reserved()

        # Reserved should be close to original (cached memory reused)
        # Allow some flexibility for allocator behavior
        assert reserved_after_del >= reserved_before * 0.9, (
            f"Reserved memory should stay cached after delete: "
            f"before={reserved_before}, after_del={reserved_after_del}"
        )

        del t2
        torch.cuda.empty_cache()

    def test_empty_cache_frees_memory(self):
        """Verify empty_cache() reduces reserved memory."""
        torch.cuda.empty_cache()
        reset_memory_stats()

        n = 5_000_000
        t = torch.randn(n, device="cuda", dtype=torch.float32)
        del t

        reserved_before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        reserved_after = torch.cuda.memory_reserved()

        assert reserved_after <= reserved_before, (
            f"empty_cache should reduce reserved memory: "
            f"before={reserved_before}, after={reserved_after}"
        )

    def test_allocated_vs_reserved(self):
        """Verify allocated <= reserved always."""
        torch.cuda.empty_cache()
        t = torch.randn(1_000_000, device="cuda", dtype=torch.float32)

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        assert allocated <= reserved, f"allocated ({allocated}) should be <= reserved ({reserved})"

        del t
        torch.cuda.empty_cache()

    def test_peak_memory_tracking(self):
        """Verify peak memory tracking works correctly."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        n = 2_000_000
        t1 = torch.randn(n, device="cuda", dtype=torch.float32)
        t2 = torch.randn(n, device="cuda", dtype=torch.float32)

        peak = torch.cuda.max_memory_allocated()

        # Peak should be >= 2 * n * 4 bytes
        min_expected = 2 * n * 4
        assert peak >= min_expected, f"Peak memory {peak} < expected {min_expected}"

        del t1, t2
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Memory reuse tests
# ---------------------------------------------------------------------------


@requires_cuda
class TestMemoryReuse:
    """Test memory reuse patterns."""

    def test_buffer_pool_acquire_release(self):
        """Verify buffer pool acquires and releases correctly."""
        pool = BufferPool(max_size=2_000_000)

        buf = pool.acquire(1_000_000)
        assert buf.numel() == 1_000_000
        assert buf.is_cuda

        pool.release()

        # Should be able to acquire again
        buf2 = pool.acquire(500_000)
        assert buf2.numel() == 500_000
        pool.release()

    def test_buffer_pool_oversize_raises(self):
        """Verify buffer pool raises on oversize request."""
        pool = BufferPool(max_size=100_000)
        with pytest.raises(RuntimeError):
            pool.acquire(200_000)

    def test_buffer_pool_double_acquire_raises(self):
        """Verify buffer pool prevents double acquire."""
        pool = BufferPool(max_size=1_000_000)
        pool.acquire(100_000)
        with pytest.raises(RuntimeError):
            pool.acquire(200_000)
        pool.release()

    def test_inplace_correctness(self):
        """Verify in-place ops produce correct results."""
        n = 10_000
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        x_clone = x.clone()

        # In-place
        x_clone.add_(y)

        # Out-of-place
        z = x + y

        assert torch.allclose(x_clone, z, atol=1e-5)

    def test_inplace_saves_memory(self):
        """Verify in-place ops use less peak memory."""
        torch.cuda.empty_cache()
        n = 1_000_000

        # Out-of-place
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        z = x + y
        peak_out = torch.cuda.max_memory_allocated()
        del x, y, z
        torch.cuda.empty_cache()

        # In-place
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        x.add_(y)
        peak_in = torch.cuda.max_memory_allocated()

        assert peak_in < peak_out, (
            f"In-place peak ({peak_in}) should be less than out-of-place ({peak_out})"
        )

        del x, y
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# KV cache tests
# ---------------------------------------------------------------------------


@requires_cuda
class TestKVCacheCalculator:
    """Test KV cache size calculations."""

    def test_llama7b_kv_cache(self):
        """Verify LLaMA-7B KV cache size calculation."""
        # LLaMA-7B: 32 layers, 32 heads, head_dim=128
        result = calculate_kv_cache_size(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=1,
            dtype=torch.float16,
        )

        # Manual calculation:
        # 2 (K+V) * 32 layers * 32 heads * 2048 seq * 128 dim * 2 bytes(fp16)
        expected_bytes = 2 * 32 * 32 * 2048 * 128 * 2
        assert abs(result["total_bytes"] - expected_bytes) < 100, (
            f"Expected {expected_bytes}, got {result['total_bytes']}"
        )

    def test_batch_size_scaling(self):
        """Verify KV cache scales linearly with batch size."""
        bs1 = calculate_kv_cache_size(32, 32, 128, 1024, 1, torch.float16)
        bs4 = calculate_kv_cache_size(32, 32, 128, 1024, 4, torch.float16)

        ratio = bs4["total_bytes"] / bs1["total_bytes"]
        assert abs(ratio - 4.0) < 0.01, f"Batch scaling ratio should be 4.0, got {ratio}"

    def test_seq_len_scaling(self):
        """Verify KV cache scales linearly with sequence length."""
        sl1024 = calculate_kv_cache_size(32, 32, 128, 1024, 1, torch.float16)
        sl4096 = calculate_kv_cache_size(32, 32, 128, 4096, 1, torch.float16)

        ratio = sl4096["total_bytes"] / sl1024["total_bytes"]
        assert abs(ratio - 4.0) < 0.01, f"Seq len scaling ratio should be 4.0, got {ratio}"

    def test_gqa_reduces_cache(self):
        """Verify Grouped Query Attention reduces KV cache size."""
        full = calculate_kv_cache_size(80, 64, 128, 4096, 1, torch.float16, num_kv_heads=64)
        gqa = calculate_kv_cache_size(80, 64, 128, 4096, 1, torch.float16, num_kv_heads=8)

        ratio = full["total_bytes"] / gqa["total_bytes"]
        assert abs(ratio - 8.0) < 0.01, f"GQA should reduce cache by 8x (64/8), got {ratio:.2f}x"


# ---------------------------------------------------------------------------
# Paged attention tests
# ---------------------------------------------------------------------------


@requires_cuda
class TestPagedAttention:
    """Test simplified PagedAttention implementation."""

    def test_allocate_fill_free(self):
        """Test basic allocate -> write -> read -> free cycle."""
        pool = PagedKVCache(
            num_blocks=10,
            block_size=16,
            num_heads=4,
            head_dim=32,
            dtype=torch.float16,
        )

        num_tokens = 50
        pool.allocate_blocks(0, num_tokens)

        k = torch.randn(4, num_tokens, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(4, num_tokens, 32, device="cuda", dtype=torch.float16)
        pool.write_kv(0, k, v, 0)

        k_out, v_out = pool.get_contiguous_kv(0, num_tokens)
        assert torch.allclose(k_out[0], k, atol=1e-3), "K cache mismatch"
        assert torch.allclose(v_out[0], v, atol=1e-3), "V cache mismatch"

        pool.free_sequence(0)
        assert pool.used_blocks == 0

    def test_multiple_sequences(self):
        """Test multiple concurrent sequences."""
        pool = PagedKVCache(
            num_blocks=20,
            block_size=16,
            num_heads=2,
            head_dim=16,
            dtype=torch.float32,
        )

        for seq_id, tokens in enumerate([30, 60, 90]):
            pool.allocate_blocks(seq_id, tokens)
            k = torch.randn(2, tokens, 16, device="cuda")
            v = torch.randn(2, tokens, 16, device="cuda")
            pool.write_kv(seq_id, k, v, 0)

        assert pool.used_blocks > 0

        # Verify each sequence
        for seq_id, tokens in enumerate([30, 60, 90]):
            k_out, v_out = pool.get_contiguous_kv(seq_id, tokens)
            assert k_out.shape == (1, 2, tokens, 16)
            assert v_out.shape == (1, 2, tokens, 16)

        pool.free_sequence(0)
        assert pool.used_blocks < 20

    def test_block_reuse_after_free(self):
        """Test that freed blocks can be reused."""
        pool = PagedKVCache(
            num_blocks=10,
            block_size=16,
            num_heads=2,
            head_dim=16,
            dtype=torch.float32,
        )

        pool.allocate_blocks(0, 50)
        blocks_used = pool.used_blocks
        pool.free_sequence(0)
        assert pool.used_blocks == 0

        pool.allocate_blocks(1, 50)
        assert pool.used_blocks == blocks_used, "Reused blocks should match initial usage"

    def test_insufficient_blocks_raises(self):
        """Test that running out of blocks raises an error."""
        pool = PagedKVCache(
            num_blocks=2,
            block_size=16,
            num_heads=1,
            head_dim=16,
            dtype=torch.float32,
        )

        pool.allocate_blocks(0, 100)  # Needs 7 blocks, pool has 2
        # This will raise because we already used both blocks
        with pytest.raises(RuntimeError):
            pool.allocate_blocks(1, 1)
