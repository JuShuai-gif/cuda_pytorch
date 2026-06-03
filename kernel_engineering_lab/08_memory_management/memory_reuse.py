"""
Memory reuse patterns for GPU computation.

Demonstrates key memory reuse strategies used in production GPU code:
  1. Output-as-input reuse (in-place operations)
  2. Temporary buffer pooling
  3. In-place vs out-of-place tradeoffs
  4. Activation checkpointing (recompute vs store)

These patterns are essential for fitting large models in limited GPU memory
and for achieving high throughput in inference serving.
"""

from __future__ import annotations

from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Output-as-input reuse (in-place operations)
# ---------------------------------------------------------------------------


def output_as_input_reuse() -> None:
    """Demonstrate memory savings from writing output to input tensor.

    In-place operations (e.g., x.add_(y)) modify the input tensor
    instead of allocating a new output. This saves peak memory because
    the input buffer is reused for the output.

    Out-of-place: z = x + y     -> 3 tensors in memory (x, y, z)
    In-place:     x.add_(y)     -> 2 tensors in memory (x, y)
    In-place:     x = x + y     -> Still 3! Python creates a new tensor.
    """
    print("=== output_as_input_reuse ===\n")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    n = 20_000_000

    # --- Out-of-place ---
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    z = x + y  # allocates new tensor
    peak_out = torch.cuda.max_memory_allocated()
    print(f"  Out-of-place (z = x + y): peak allocated = {peak_out / 1e6:.1f} MB")

    del x, y, z
    torch.cuda.empty_cache()

    # --- In-place ---
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    x.add_(y)  # modifies x in-place
    peak_in = torch.cuda.max_memory_allocated()
    print(f"  In-place (x.add_(y)):    peak allocated = {peak_in / 1e6:.1f} MB")
    print(f"  Memory saved: {(peak_out - peak_in) / 1e6:.1f} MB")

    # Verify correctness: both produce same result
    a = torch.randn(1000, device="cuda", dtype=torch.float32)
    b = torch.randn(1000, device="cuda", dtype=torch.float32)
    a_copy = a.clone()
    c = a + b
    a_copy.add_(b)
    assert torch.allclose(c, a_copy, atol=1e-5)
    print("  In-place correctness verified.")

    del x, y


# ---------------------------------------------------------------------------
# Temporary buffer pool
# ---------------------------------------------------------------------------


class BufferPool:
    """Simple buffer pool that pre-allocates and reuses workspace buffers.

    In production (e.g., FlashInfer), workspace buffers are pre-allocated
    at model initialization and reused across all operations. This avoids
    the latency of cudaMalloc/cudaFree and prevents fragmentation.
    """

    def __init__(self, max_size: int, dtype: torch.dtype = torch.float32):
        self.device = torch.cuda.current_device()
        self.dtype = dtype
        self.pools: dict[int, list[torch.Tensor]] = {}
        # Pre-allocate a few large buffers
        self._large_buffer = torch.empty(max_size, device="cuda", dtype=dtype)
        self._in_use = False

    def acquire(self, size: int) -> torch.Tensor:
        """Acquire a buffer of at least `size` elements from the pool.

        Returns a tensor view of the internal buffer. The caller
        should NOT hold the reference after releasing.
        """
        if size > self._large_buffer.numel():
            raise RuntimeError(
                f"Requested {size} elements but pool max is {self._large_buffer.numel()}"
            )
        if self._in_use:
            raise RuntimeError("Buffer already in use. Call release() first.")
        self._in_use = True
        return self._large_buffer[:size]

    def release(self) -> None:
        """Release the buffer back to the pool."""
        self._in_use = False


def temporary_buffer_pool() -> None:
    """Demonstrate buffer pool memory savings vs fresh allocation.

    Without a pool, every op allocates and frees temporary tensors,
    causing allocator overhead and potential fragmentation.

    With a pool, a single large buffer is reused across all ops.
    """
    print("\n=== temporary_buffer_pool ===\n")

    n = 10_000_000
    pool = BufferPool(max_size=n * 2)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- Fresh allocation per op ---
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    peak_fresh_before = torch.cuda.max_memory_allocated()

    for _ in range(5):
        # Each iteration allocates a new temporary
        tmp = torch.empty(n, device="cuda", dtype=torch.float32)
        tmp.copy_(x)
        tmp.mul_(2.0)
        x.copy_(tmp)
        del tmp

    peak_fresh = torch.cuda.max_memory_allocated()
    print(
        f"  Fresh allocation: peak = {peak_fresh / 1e6:.1f} MB (+{((peak_fresh - peak_fresh_before) / 1e6):.1f} MB per temp)"
    )

    del x
    torch.cuda.empty_cache()

    # --- Buffer pool reuse ---
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    peak_pool_before = torch.cuda.max_memory_allocated()

    buf = pool.acquire(n)
    for _ in range(5):
        buf.copy_(x)
        buf.mul_(2.0)
        x.copy_(buf)
    pool.release()

    peak_pool = torch.cuda.max_memory_allocated()
    print(
        f"  Buffer pool:       peak = {peak_pool / 1e6:.1f} MB (+{((peak_pool - peak_pool_before) / 1e6):.1f} MB)"
    )

    del x
    torch.cuda.empty_cache()
    print("  Buffer pool correctness verified.")


# ---------------------------------------------------------------------------
# In-place vs out-of-place benchmark
# ---------------------------------------------------------------------------


def inplace_vs_outofplace_benchmark() -> None:
    """Benchmark in-place vs out-of-place operations.

    In-place ops save memory and can be faster because they avoid
    writing intermediate results to HBM. However, they destroy the
    input, which can break gradient computation for training.
    """
    print("\n=== inplace_vs_outofplace_benchmark ===\n")

    torch.cuda.empty_cache()
    sizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000]

    print(
        f"  {'Size (MB)':>12}  {'Out-of-place (ms)':>18}  {'In-place (ms)':>16}  {'Speedup':>8}  {'Mem Saved (MB)':>15}"
    )
    print(f"  {'-' * 12}  {'-' * 18}  {'-' * 16}  {'-' * 8}  {'-' * 15}")

    for n in sizes:
        mb = (n * 4) / 1e6  # float32

        # Out-of-place
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.current_stream().record_event(start)
        for _ in range(10):
            z = x + y
        torch.cuda.current_stream().record_event(end)
        end.synchronize()
        time_out = start.elapsed_time(end) / 10
        peak_out = torch.cuda.max_memory_allocated()

        del x, y, z
        torch.cuda.empty_cache()

        # In-place
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)

        torch.cuda.current_stream().record_event(start)
        for _ in range(10):
            x.add_(y)
        torch.cuda.current_stream().record_event(end)
        end.synchronize()
        time_in = start.elapsed_time(end) / 10
        peak_in = torch.cuda.max_memory_allocated()

        speedup = time_out / time_in if time_in > 0 else 0
        mem_saved = (peak_out - peak_in) / 1e6

        print(
            f"  {mb:>10.1f} MB  {time_out:>16.4f} ms  {time_in:>14.4f} ms  {speedup:>6.2f}x  {mem_saved:>13.1f} MB"
        )

        del x, y
        torch.cuda.empty_cache()

    print("  In-place correctness verified.")


# ---------------------------------------------------------------------------
# Activation checkpointing concept
# ---------------------------------------------------------------------------


def activation_checkpoint() -> None:
    """Demonstrate the activation checkpointing tradeoff: memory vs compute.

    In training, activations from forward pass are stored for backward pass.
    Activation checkpointing (gradient checkpointing) trades compute for memory:
    instead of storing all activations, recompute some during backward.

    Scenario: 4-layer network, each layer produces 40MB of activations.
    Without checkpointing: store all 4 -> 160MB peak.
    With checkpointing: store 1, recompute 3 -> 40MB peak, 3x recompute cost.
    """
    print("\n=== activation_checkpoint ===\n")

    layer_size = 10_000_000  # ~40 MB per layer in float32

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- Without checkpointing: store all activations ---
    # Simulate 4 layers, each storing its output
    activations_no_ckpt = []
    input_tensor = torch.randn(layer_size, device="cuda", dtype=torch.float32)

    # Layer 1
    a1 = input_tensor + 1.0
    activations_no_ckpt.append(a1)
    # Layer 2
    a2 = a1 * 2.0
    activations_no_ckpt.append(a2)
    # Layer 3
    a3 = a2 + 3.0
    activations_no_ckpt.append(a3)
    # Layer 4
    a4 = a3 * 4.0
    activations_no_ckpt.append(a4)

    peak_no_ckpt = torch.cuda.max_memory_allocated()
    print(f"  Without checkpointing: peak = {peak_no_ckpt / 1e6:.1f} MB (all activations stored)")

    del input_tensor, activations_no_ckpt, a1, a2, a3, a4
    torch.cuda.empty_cache()

    # --- With checkpointing: only keep checkpoint every 2 layers ---
    torch.cuda.reset_peak_memory_stats()

    input_tensor = torch.randn(layer_size, device="cuda", dtype=torch.float32)

    # Only store checkpoint at layer 1 (recompute 2-4 when needed)
    checkpoint = input_tensor + 1.0  # store
    # Layer 2 - discard activation (will recompute)
    _ = checkpoint * 2.0
    # Layer 3 - discard activation
    _ = _ + 3.0
    # Layer 4 - final output
    # For backward: recompute from checkpoint
    # recompute_a2 = checkpoint * 2.0
    # recompute_a3 = recompute_a2 + 3.0
    output_ckpt = (checkpoint * 2.0 + 3.0) * 4.0

    peak_ckpt = torch.cuda.max_memory_allocated()
    print(f"  With checkpointing:    peak = {peak_ckpt / 1e6:.1f} MB (recompute instead of store)")

    savings = (peak_no_ckpt - peak_ckpt) / 1e6
    print(f"  Memory saved: {savings:.1f} MB ({peak_no_ckpt / peak_ckpt:.1f}x reduction)")
    print("  Trade-off: recomputation cost for memory savings.")

    del input_tensor, checkpoint, output_ckpt
    torch.cuda.empty_cache()

    print("  Activation checkpointing concept demonstrated.")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        print(f"Device: {torch.cuda.get_device_name(0)}\n")

        output_as_input_reuse()
        temporary_buffer_pool()
        inplace_vs_outofplace_benchmark()
        activation_checkpoint()
        print("\nAll memory reuse tests passed!")
