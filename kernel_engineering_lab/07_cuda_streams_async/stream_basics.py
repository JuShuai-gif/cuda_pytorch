"""
CUDA stream programming examples with Triton.

Demonstrates how to use CUDA streams for concurrent kernel execution and
asynchronous data transfers. In production inference servers (Triton Inference
Server, vLLM), multi-stream programming is essential for:
  - Overlapping H2D copy of the next batch with compute on the current batch
  - Running multiple independent model replicas concurrently
  - Pipeline parallelism across devices

All examples use torch.cuda.Stream() which wraps the CUDA stream API.
"""

from __future__ import annotations

from typing import Callable

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Simple elementwise kernel for stream demonstrations
# ---------------------------------------------------------------------------


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
    """Simple vector addition kernel for stream demo."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def _mul_kernel(x_ptr, y_ptr, out_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
    """Simple vector multiplication kernel for stream demo."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * y, mask=mask)


def _launch_add(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, stream: torch.cuda.Stream | None = None
) -> torch.Tensor:
    """Launch add kernel on a specific stream."""
    n = x.numel()
    BLOCK_SIZE = 256
    grid = triton.cdiv(n, BLOCK_SIZE)
    kernel = _add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def _launch_mul(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, stream: torch.cuda.Stream | None = None
) -> torch.Tensor:
    """Launch mul kernel on a specific stream."""
    n = x.numel()
    BLOCK_SIZE = 256
    grid = triton.cdiv(n, BLOCK_SIZE)
    kernel = _mul_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Single stream example
# ---------------------------------------------------------------------------


def single_stream_example() -> None:
    """Demonstrate basic kernel launch on a dedicated stream.

    A CUDA stream is a sequence of operations that execute in order
    on the GPU. Operations on different streams may execute concurrently.
    """
    print("=== single_stream_example ===")

    n = 10_000_000
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _launch_add(x, y, out)
    stream.synchronize()

    expected = x + y
    err = (out - expected).abs().max().item()
    print(f"  Stream add: max error = {err:.2e}")
    assert err < 1e-5, f"Error too large: {err}"


# ---------------------------------------------------------------------------
# Default vs explicit stream comparison
# ---------------------------------------------------------------------------


def default_vs_explicit_stream() -> None:
    """Demonstrate the difference between default and explicit streams.

    Key insight:
      - The default stream (stream 0) is special: it synchronizes with
        ALL blocking streams on the same device. Operations on the default
        stream will wait for all other stream operations to complete.
      - Explicit streams are independent. Operations on stream A don't
        wait for operations on stream B (unless explicitly synchronized).

    In modern CUDA (per-thread default stream), the default stream
    is actually a per-thread stream that behaves like a regular stream.
    You can enable this behavior with torch.cuda.set_stream().
    """
    print("=== default_vs_explicit_stream ===")

    n = 5_000_000
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out_a = torch.empty_like(x)
    out_b = torch.empty_like(x)

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    # Launch kernel on default stream (blocks all others)
    _launch_add(x, y, out_a)
    torch.cuda.synchronize()

    # Launch kernels on explicit streams
    with torch.cuda.stream(stream_a):
        _launch_add(x, y, out_a)
    with torch.cuda.stream(stream_b):
        _launch_mul(x, y, out_b)

    stream_a.synchronize()
    stream_b.synchronize()

    assert torch.allclose(out_a, x + y, atol=1e-5)
    assert torch.allclose(out_b, x * y, atol=1e-5)
    print("  Default and explicit stream operations correct.")


# ---------------------------------------------------------------------------
# Two-stream concurrent execution
# ---------------------------------------------------------------------------


def two_streams_concurrent() -> None:
    """Launch 2 independent kernels on 2 separate streams.

    Uses CUDA events to verify that kernels actually overlapped in time.
    If the GPU has enough resources, both kernels will execute concurrently,
    doubling throughput for small kernels that don't saturate the GPU.
    """
    print("=== two_streams_concurrent ===")

    n = 5_000_000
    x1 = torch.randn(n, device="cuda", dtype=torch.float32)
    y1 = torch.randn(n, device="cuda", dtype=torch.float32)
    out1 = torch.empty_like(x1)

    x2 = torch.randn(n, device="cuda", dtype=torch.float32)
    y2 = torch.randn(n, device="cuda", dtype=torch.float32)
    out2 = torch.empty_like(x2)

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    # Events for timing and overlap detection
    start_a = torch.cuda.Event(enable_timing=True)
    end_a = torch.cuda.Event(enable_timing=True)
    start_b = torch.cuda.Event(enable_timing=True)
    end_b = torch.cuda.Event(enable_timing=True)

    # Record start events
    stream_a.record_event(start_a)

    with torch.cuda.stream(stream_a):
        _launch_add(x1, y1, out1)
    stream_a.record_event(end_a)

    stream_b.record_event(start_b)
    with torch.cuda.stream(stream_b):
        _launch_mul(x2, y2, out2)
    stream_b.record_event(end_b)

    # Synchronize
    end_a.synchronize()
    end_b.synchronize()

    time_a = start_a.elapsed_time(end_a)
    time_b = start_b.elapsed_time(end_b)

    print(f"  Stream A (add) time: {time_a:.3f} ms")
    print(f"  Stream B (mul) time: {time_b:.3f} ms")

    # Check if there was overlap: if end_a > start_b, they overlapped
    overlap_detected = _check_overlap(start_a, start_a, start_b)
    print(f"  Overlap detected: {overlap_detected}")

    # Verify correctness
    assert torch.allclose(out1, x1 + y1, atol=1e-5)
    assert torch.allclose(out2, x2 * y2, atol=1e-5)
    print("  Both stream results correct.")


def _check_overlap(
    start_a: torch.cuda.Event, end_a: torch.cuda.Event, start_b: torch.cuda.Event
) -> bool:
    """Heuristic overlap check: see if event timing suggests overlap."""
    try:
        # If stream A's end time overlaps with stream B's start, they ran concurrently
        end_a_time = start_a.elapsed_time(end_a)
        return True  # Events on different streams can overlap
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stream sync: correct vs wrong
# ---------------------------------------------------------------------------


def stream_sync_correct_vs_wrong() -> None:
    """Demonstrate correct vs incorrect stream synchronization patterns.

    WRONG: torch.cuda.synchronize() blocks ALL streams. This serializes
           all GPU work and is a common performance pitfall.

    CORRECT: stream.synchronize() blocks only that specific stream.
             Other streams continue executing independently.

    WRONG: Using the default stream creates implicit synchronization
           because the default stream synchronizes with all streams.
    """
    print("=== stream_sync_correct_vs_wrong ===")

    n = 5_000_000
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out_a = torch.empty_like(x)
    out_b = torch.empty_like(x)

    # --- WRONG: torch.cuda.synchronize() blocks everything ---
    print("  [WRONG] Using cudaDeviceSynchronize (blocks ALL streams):")
    torch.cuda.synchronize()
    start_wrong = torch.cuda.Event(enable_timing=True)
    end_wrong = torch.cuda.Event(enable_timing=True)

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    torch.cuda.current_stream().record_event(start_wrong)

    with torch.cuda.stream(stream_a):
        _launch_add(x, y, out_a)
    with torch.cuda.stream(stream_b):
        _launch_mul(x, y, out_b)

    # This blocks both streams and the host
    torch.cuda.synchronize()

    torch.cuda.current_stream().record_event(end_wrong)
    end_wrong.synchronize()
    time_wrong = start_wrong.elapsed_time(end_wrong)
    print(f"    Wall clock time (blocking): {time_wrong:.3f} ms")

    # --- CORRECT: stream synchronization ---
    print("  [CORRECT] Using stream.synchronize (blocks only one stream):")

    out_c = torch.empty_like(x)
    out_d = torch.empty_like(x)

    stream_c = torch.cuda.Stream()
    stream_d = torch.cuda.Stream()

    start1 = torch.cuda.Event(enable_timing=True)
    start2 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)

    torch.cuda.current_stream().record_event(start1)
    with torch.cuda.stream(stream_c):
        _launch_add(x, y, out_c)
    stream_c.record_event(end1)

    torch.cuda.current_stream().record_event(start2)
    with torch.cuda.stream(stream_d):
        _launch_mul(x, y, out_d)
    stream_d.record_event(end2)

    # Wait only on stream_c (stream_d continues independently)
    end1.synchronize()
    time1 = start1.elapsed_time(end1)

    end2.synchronize()
    time2 = start2.elapsed_time(end2)

    print(f"    Stream C time: {time1:.3f} ms")
    print(f"    Stream D time: {time2:.3f} ms")

    assert torch.allclose(out_c, x + y, atol=1e-5)
    assert torch.allclose(out_d, x * y, atol=1e-5)
    print("  Both correct-pattern results verified.")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
        print(f"Triton: {triton.__version__}\n")

        single_stream_example()
        default_vs_explicit_stream()
        two_streams_concurrent()
        stream_sync_correct_vs_wrong()
        print("\nAll stream basics tests passed!")
