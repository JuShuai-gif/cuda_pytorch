"""
Async data transfer patterns using CUDA streams.

Demonstrates the critical patterns for overlapping data transfer with compute:
  - Pinned (page-locked) memory enables true async H2D/D2H transfers
  - Pageable memory blocks the host during transfer
  - Double-buffering: copy next chunk while computing current chunk

Production context: In inference servers (Triton, vLLM), overlapping
H2D of the next batch with D2H of the results from the current batch
can hide transfer latency and keep the GPU utilized.
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Simple elementwise kernel for compute phases
# ---------------------------------------------------------------------------


@triton.jit
def _compute_kernel(x_ptr, out_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
    """Simple compute kernel for pipeline demo."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Simulate some compute: multiply, square, sqrt
    val = x * x
    val = val * x
    tl.store(out_ptr + offsets, val, mask=mask)


def _launch_compute(
    x: torch.Tensor, out: torch.Tensor, stream: torch.cuda.Stream | None = None
) -> None:
    """Launch compute kernel on a specific stream."""
    n = x.numel()
    BLOCK_SIZE = 256
    grid = triton.cdiv(n, BLOCK_SIZE)
    if stream is not None:
        with torch.cuda.stream(stream):
            _compute_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    else:
        _compute_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# H2D with pinned memory (async)
# ---------------------------------------------------------------------------


def h2d_async_pinned() -> None:
    """Host-to-device copy using pinned (page-locked) memory.

    Pinned memory is locked in physical RAM, allowing the GPU DMA engine
    to directly access it without CPU involvement. This enables truly
    asynchronous H2D transfers while the CPU continues other work.

    Without pinned memory, the driver must first copy to a staging buffer,
    which is synchronous from the host thread's perspective.
    """
    print("=== h2d_async_pinned ===")

    n = 1_000_000
    # Allocate pinned memory on CPU
    host_data = torch.randn(n, dtype=torch.float32, pin_memory=True)
    device_data = torch.empty(n, device="cuda", dtype=torch.float32)

    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    stream.record_event(start)
    with torch.cuda.stream(stream):
        device_data.copy_(host_data, non_blocking=True)
    stream.record_event(end)

    # While GPU is copying, CPU can do other work
    cpu_work_result = 0
    for i in range(100_000):
        cpu_work_result += i

    end.synchronize()
    elapsed = start.elapsed_time(end)

    print(f"  Async H2D transfer time: {elapsed:.3f} ms")
    print(f"  CPU work done during transfer: {cpu_work_result}")
    assert torch.allclose(device_data.cpu(), host_data, atol=1e-5)
    print("  Transfer correct.")


# ---------------------------------------------------------------------------
# H2D with pageable memory (blocking)
# ---------------------------------------------------------------------------


def h2d_blocking_pageable() -> None:
    """Host-to-device copy using pageable (non-pinned) memory.

    Pageable memory can be swapped by the OS at any time, so the CUDA
    driver must copy data to a pinned staging buffer first. This makes
    the transfer effectively synchronous from the host perspective,
    even when using non_blocking transfers.
    """
    print("=== h2d_blocking_pageable ===")

    n = 1_000_000
    # Regular pageable memory (no pin_memory=True)
    host_data = torch.randn(n, dtype=torch.float32)  # NOT pinned
    device_data = torch.empty(n, device="cuda", dtype=torch.float32)

    stream = torch.cuda.Stream()
    start = time.perf_counter()

    with torch.cuda.stream(stream):
        device_data.copy_(host_data, non_blocking=True)
    stream.synchronize()

    elapsed = (time.perf_counter() - start) * 1000

    print(f"  Pageable H2D transfer time: {elapsed:.3f} ms")
    assert torch.allclose(device_data.cpu(), host_data, atol=1e-5)
    print("  Transfer correct.")


# ---------------------------------------------------------------------------
# D2H async copy
# ---------------------------------------------------------------------------


def d2h_async() -> None:
    """Device-to-host async copy with pinned memory.

    D2H transfers can also be asynchronous when the destination
    is pinned memory. This is useful for overlapping result read-back
    with the next inference step.
    """
    print("=== d2h_async ===")

    n = 1_000_000
    device_data = torch.randn(n, device="cuda", dtype=torch.float32)
    host_pinned = torch.empty(n, dtype=torch.float32, pin_memory=True)

    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    stream.record_event(start)
    with torch.cuda.stream(stream):
        host_pinned.copy_(device_data, non_blocking=True)
    stream.record_event(end)

    # CPU work during D2H
    intermediate = 0
    for i in range(50_000):
        intermediate += i

    end.synchronize()
    elapsed = start.elapsed_time(end)

    print(f"  Async D2H transfer time: {elapsed:.3f} ms")
    assert torch.allclose(host_pinned, device_data.cpu(), atol=1e-5)
    print("  D2H transfer correct.")


# ---------------------------------------------------------------------------
# Overlap H2D + Compute + D2H (double-buffering pipeline)
# ---------------------------------------------------------------------------


def overlap_h2d_compute_d2h() -> None:
    """Full pipeline: overlap H2D, compute, and D2H using streams.

    This is the classic double-buffering pattern used in inference servers:
      1. Copy chunk 1 from host to device on stream A
      2. Launch compute on chunk 1 on stream A
      3. Copy result from chunk 1 to host on stream A
      4. Meanwhile, copy chunk 2 to device on stream B, compute on B, etc.

    Uses 2 streams and pipelining to overlap communication and computation.
    This can hide data transfer latency entirely if compute time >= transfer time.
    """
    print("=== overlap_h2d_compute_d2h ===")

    total_elements = 10_000_000
    num_chunks = 4
    chunk_size = total_elements // num_chunks

    # Create pinned host arrays
    host_inputs = [
        torch.randn(chunk_size, dtype=torch.float32, pin_memory=True) for _ in range(num_chunks)
    ]
    host_outputs = [
        torch.empty(chunk_size, dtype=torch.float32, pin_memory=True) for _ in range(num_chunks)
    ]

    # Device buffers for double buffering
    dev_buf_a = torch.empty(chunk_size, device="cuda", dtype=torch.float32)
    dev_buf_b = torch.empty(chunk_size, device="cuda", dtype=torch.float32)
    dev_out_a = torch.empty(chunk_size, device="cuda", dtype=torch.float32)
    dev_out_b = torch.empty(chunk_size, device="cuda", dtype=torch.float32)

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.current_stream().record_event(start_event)

    for i in range(0, num_chunks, 2):
        chunk_i = i
        chunk_j = i + 1

        # Process chunk_i on stream A
        with torch.cuda.stream(stream_a):
            dev_buf_a.copy_(host_inputs[chunk_i], non_blocking=True)
            _launch_compute(dev_buf_a, dev_out_a)
            host_outputs[chunk_i].copy_(dev_out_a, non_blocking=True)

        # Process chunk_j on stream B (concurrently with chunk_i)
        if chunk_j < num_chunks:
            with torch.cuda.stream(stream_b):
                dev_buf_b.copy_(host_inputs[chunk_j], non_blocking=True)
                _launch_compute(dev_buf_b, dev_out_b)
                host_outputs[chunk_j].copy_(dev_out_b, non_blocking=True)

    # Wait for all streams
    stream_a.synchronize()
    stream_b.synchronize()

    torch.cuda.current_stream().record_event(end_event)
    end_event.synchronize()
    elapsed = start_event.elapsed_time(end_event)

    print(f"  Pipeline time ({num_chunks} chunks): {elapsed:.3f} ms")

    # Verify correctness
    for i in range(num_chunks):
        expected = host_inputs[i] ** 3  # x^3
        assert torch.allclose(host_outputs[i], expected, atol=1e-3), (
            f"Chunk {i}: max error = {(host_outputs[i] - expected).abs().max().item():.2e}"
        )

    # Compare with sequential version
    start_seq = torch.cuda.Event(enable_timing=True)
    end_seq = torch.cuda.Event(enable_timing=True)
    torch.cuda.current_stream().record_event(start_seq)

    dev_seq = torch.empty(chunk_size, device="cuda", dtype=torch.float32)
    dev_seq_out = torch.empty(chunk_size, device="cuda", dtype=torch.float32)

    for i in range(num_chunks):
        dev_seq.copy_(host_inputs[i])
        _launch_compute(dev_seq, dev_seq_out)
        host_outputs[i].copy_(dev_seq_out)

    torch.cuda.current_stream().record_event(end_seq)
    end_seq.synchronize()
    seq_time = start_seq.elapsed_time(end_seq)

    print(f"  Sequential time ({num_chunks} chunks): {seq_time:.3f} ms")
    speedup = seq_time / elapsed if elapsed > 0 else 0.0
    print(f"  Pipeline speedup: {speedup:.2f}x")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        print(f"Device: {torch.cuda.get_device_name(0)}\n")

        h2d_async_pinned()
        h2d_blocking_pageable()
        d2h_async()
        overlap_h2d_compute_d2h()
        print("\nAll async copy tests passed!")
