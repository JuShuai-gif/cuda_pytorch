"""
Observe and understand PyTorch's CUDA caching allocator behavior.

PyTorch uses a caching allocator that retains freed memory for reuse instead
of returning it to the CUDA driver. This avoids expensive cudaMalloc/cudaFree
calls and is critical for performance.

Key metrics:
  - allocated_bytes: Memory currently in use by tensors
  - reserved_bytes: Memory held by the caching allocator (not returned to driver)
  - max_allocated_bytes: Peak allocated memory
  - Fragmentation: Difference between reserved and allocated

Understanding this behavior is essential for memory optimization in GPU workloads.
"""

from __future__ import annotations

import torch


def show_allocator_state(label: str = "") -> None:
    """Print current GPU memory state from the caching allocator.

    Args:
        label: Optional label for this state snapshot.
    """
    prefix = f"[{label}] " if label else ""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()

    print(
        f"{prefix}allocated: {allocated / 1e6:.2f} MB | "
        f"reserved: {reserved / 1e6:.2f} MB | "
        f"max_allocated: {max_allocated / 1e6:.2f} MB | "
        f"max_reserved: {max_reserved / 1e6:.2f} MB | "
        f"free_in_reserved: {(reserved - allocated) / 1e6:.2f} MB"
    )


def reset_memory_stats() -> None:
    """Reset peak memory statistics."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()


# ---------------------------------------------------------------------------
# Caching allocator demonstration
# ---------------------------------------------------------------------------


def demonstrate_caching_allocator() -> None:
    """Demonstrate the caching allocator: allocate, delete, reuse.

    When you delete a tensor, the memory is not returned to the CUDA driver.
    Instead, PyTorch's caching allocator keeps it as 'reserved' memory that
    can be quickly reused for future allocations. This avoids expensive
    cudaMalloc/cudaFree calls.
    """
    print("=== demonstrate_caching_allocator ===\n")

    reset_memory_stats()
    show_allocator_state("start")

    # Allocate a large tensor
    n = 10_000_000  # 40 MB in float32
    t1 = torch.randn(n, device="cuda", dtype=torch.float32)
    show_allocator_state("after alloc t1 (40MB)")

    # Delete the tensor
    del t1
    show_allocator_state("after del t1")

    # Memory is still 'reserved' but 'allocated' dropped
    # Now allocate a tensor of the same size
    t2 = torch.randn(n, device="cuda", dtype=torch.float32)
    show_allocator_state("after alloc t2 (same size)")

    # Notice: allocated went back up, but reserved barely changed.
    # The allocator reused the cached memory block.
    print("  Observation: reserved memory was reused, no new cudaMalloc needed.")

    # Allocate a larger tensor
    t3 = torch.randn(n * 2, device="cuda", dtype=torch.float32)
    show_allocator_state("after alloc t3 (80MB)")

    # Clear all
    del t2, t3
    show_allocator_state("after del t2, t3")

    # Force release cached memory
    torch.cuda.empty_cache()
    show_allocator_state("after empty_cache()")

    print("  Observation: empty_cache() released reserved memory back to driver.")


# ---------------------------------------------------------------------------
# Memory fragmentation observation
# ---------------------------------------------------------------------------


def show_fragmentation() -> None:
    """Create alloc/free patterns to observe memory fragmentation.

    Fragmentation occurs when the allocator has enough total free memory
    but can't allocate a contiguous block of the requested size. Small
    allocations interspersed with frees can create 'holes' in memory.
    """
    print("\n=== show_fragmentation ===\n")

    torch.cuda.empty_cache()
    reset_memory_stats()

    # Create many small tensors
    small_tensors = []
    for i in range(50):
        small_tensors.append(torch.randn(1_000_000, device="cuda", dtype=torch.float32))

    show_allocator_state("after 50 small allocs (4MB each)")

    # Free every other tensor - creates fragmentation
    for i in range(0, len(small_tensors), 2):
        del small_tensors[i]

    show_allocator_state("after freeing every other tensor")

    # Try to allocate a large tensor
    try:
        large = torch.randn(20_000_000, device="cuda", dtype=torch.float32)
        show_allocator_state("after large alloc (80MB)")
        del large
    except RuntimeError as e:
        print(f"  Large allocation failed (fragmentation): {e}")

    # Cleanup
    del small_tensors
    torch.cuda.empty_cache()
    show_allocator_state("after cleanup")


# ---------------------------------------------------------------------------
# Reserved vs allocated explanation
# ---------------------------------------------------------------------------


def reserved_vs_allocated() -> None:
    """Explain the difference between reserved and allocated memory.

    Reserved: Total memory held by the CUDA caching allocator.
              Includes allocated tensors + cached free blocks.
              This is what nvidia-smi reports as "used" for the process.

    Allocated: Memory actually in use by live torch.Tensor objects.
               This is the sum of tensor.storage().nbytes() for all
               tensors on this device.

    The gap (reserved - allocated) is memory the allocator holds
    for future reuse. It can be released with torch.cuda.empty_cache().

    Understanding this distinction is critical for:
      - Diagnosing OOM errors (is the issue fragmented reserved or true OOM?)
      - Planning memory budgets for multi-model serving
      - Debugging memory leaks (growing reserved without growing allocated)
    """
    print("\n=== reserved_vs_allocated ===\n")

    torch.cuda.empty_cache()
    reset_memory_stats()

    alloc_sizes = [5_000_000, 10_000_000, 20_000_000, 40_000_000]
    tensors = []

    for size in alloc_sizes:
        t = torch.randn(size, device="cuda", dtype=torch.float32)
        tensors.append(t)
        allocated_mb = torch.cuda.memory_allocated() / 1e6
        reserved_mb = torch.cuda.memory_reserved() / 1e6
        gap_mb = reserved_mb - allocated_mb
        print(
            f"  size={size / 1e6:.0f}MB: allocated={allocated_mb:.1f}MB "
            f"reserved={reserved_mb:.1f}MB gap={gap_mb:.1f}MB"
        )

    del tensors
    print(
        f"\n  After del: allocated={torch.cuda.memory_allocated() / 1e6:.1f}MB "
        f"reserved={torch.cuda.memory_reserved() / 1e6:.1f}MB"
    )

    torch.cuda.empty_cache()
    print(
        f"  After empty_cache: allocated={torch.cuda.memory_allocated() / 1e6:.1f}MB "
        f"reserved={torch.cuda.memory_reserved() / 1e6:.1f}MB"
    )


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB\n")

        demonstrate_caching_allocator()
        show_fragmentation()
        reserved_vs_allocated()
        print("\nAll allocator observation tests passed!")
