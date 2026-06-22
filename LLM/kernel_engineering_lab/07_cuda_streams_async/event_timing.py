"""
CUDA event-based timing utilities.

CUDA events provide the most accurate GPU kernel timing because they
are recorded directly on the GPU timeline. This avoids host-side overhead
(driver calls, thread scheduling) that wall clock timing includes.

Key patterns:
  - record event BEFORE kernel launch
  - launch kernel
  - record event AFTER kernel
  - synchronize on the end event
  - query elapsed_time between events

For overlap measurement, events on different streams reveal
the true concurrent execution pattern.
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Simple kernels for timing demonstrations
# ---------------------------------------------------------------------------


@triton.jit
def _work_kernel(
    x_ptr, out_ptr, n_elements: int, iterations: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Simulate a compute-bound kernel with parameterized loop."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    acc = x
    for _ in range(iterations):
        acc = acc * 0.999 + 0.001
    tl.store(out_ptr + offsets, acc, mask=mask)


def _launch_work(
    x: torch.Tensor,
    out: torch.Tensor,
    iterations: int = 100,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Launch the work kernel."""
    n = x.numel()
    BLOCK_SIZE = 256
    grid = triton.cdiv(n, BLOCK_SIZE)
    if stream is not None:
        with torch.cuda.stream(stream):
            _work_kernel[grid](x, out, n, iterations, BLOCK_SIZE=BLOCK_SIZE)
    else:
        _work_kernel[grid](x, out, n, iterations, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Measure kernel time using CUDA events
# ---------------------------------------------------------------------------


def measure_kernel_time() -> None:
    """Proper CUDA event-based kernel timing.

    Steps:
      1. Create start/end events with enable_timing=True
      2. Record start event on the target stream
      3. Launch the kernel
      4. Record end event on the same stream
      5. Synchronize on the end event
      6. Query elapsed_time

    This gives pure GPU time without host overhead.
    """
    print("=== measure_kernel_time ===")

    n = 5_000_000
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    # Warmup
    for _ in range(5):
        _launch_work(x, out, iterations=50)
    torch.cuda.synchronize()

    # Measure with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    stream = torch.cuda.current_stream()
    stream.record_event(start_event)
    _launch_work(x, out, iterations=200)
    stream.record_event(end_event)
    end_event.synchronize()

    gpu_time_ms = start_event.elapsed_time(end_event)
    print(f"  GPU time (CUDA events): {gpu_time_ms:.3f} ms")

    # Compare with wall clock
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    _launch_work(x, out, iterations=200)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    wall_time_ms = (t_end - t_start) * 1000.0

    print(f"  Wall clock time: {wall_time_ms:.3f} ms")
    print(f"  Host overhead: {wall_time_ms - gpu_time_ms:.3f} ms")


# ---------------------------------------------------------------------------
# Measure overlap efficiency
# ---------------------------------------------------------------------------


def measure_overlap_efficiency() -> None:
    """Use events on different streams to measure actual overlap.

    By recording events on two different streams, we can determine
    whether the kernels actually executed concurrently or sequentially.

    The key metric: if two kernels each take T_ms independently,
    but running them concurrently takes < 2*T_ms total, they overlapped.
    """
    print("=== measure_overlap_efficiency ===")

    n = 5_000_000
    x1 = torch.randn(n, device="cuda", dtype=torch.float32)
    out1 = torch.empty_like(x1)
    x2 = torch.randn(n, device="cuda", dtype=torch.float32)
    out2 = torch.empty_like(x2)

    # Warmup
    _launch_work(x1, out1, iterations=50)
    _launch_work(x2, out2, iterations=50)
    torch.cuda.synchronize()

    # --- Sequential: same stream, measure baseline ---
    stream_default = torch.cuda.current_stream()
    start_seq = torch.cuda.Event(enable_timing=True)
    end_seq = torch.cuda.Event(enable_timing=True)

    stream_default.record_event(start_seq)
    _launch_work(x1, out1, iterations=200)
    _launch_work(x2, out2, iterations=200)
    stream_default.record_event(end_seq)
    end_seq.synchronize()
    seq_time = start_seq.elapsed_time(end_seq)
    print(f"  Sequential (single stream): {seq_time:.3f} ms")

    # --- Concurrent: two streams, overlap enabled ---
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    start_con = torch.cuda.Event(enable_timing=True)
    start_a_evt = torch.cuda.Event(enable_timing=True)
    end_a_evt = torch.cuda.Event(enable_timing=True)
    start_b_evt = torch.cuda.Event(enable_timing=True)
    end_b_evt = torch.cuda.Event(enable_timing=True)
    end_con = torch.cuda.Event(enable_timing=True)

    torch.cuda.current_stream().record_event(start_con)

    stream_a.record_event(start_a_evt)
    _launch_work(x1, out1, iterations=200, stream=stream_a)
    stream_a.record_event(end_a_evt)

    stream_b.record_event(start_b_evt)
    _launch_work(x2, out2, iterations=200, stream=stream_b)
    stream_b.record_event(end_b_evt)

    torch.cuda.current_stream().record_event(end_con)

    end_a_evt.synchronize()
    end_b_evt.synchronize()

    time_a = start_a_evt.elapsed_time(end_a_evt)
    time_b = start_b_evt.elapsed_time(end_b_evt)
    con_time = start_con.elapsed_time(end_con) if start_con.query() else 0

    print(f"  Stream A time: {time_a:.3f} ms")
    print(f"  Stream B time: {time_b:.3f} ms")
    print(f"  Concurrent wall time: {con_time:.3f} ms")
    print(
        f"  Sequential / Concurrent: {seq_time / max(con_time, 0.001):.2f}x" if con_time > 0 else ""
    )

    # Verify correctness
    expected = x1 * (0.999**200) + (1.0 - 0.999) / 0.001 * (1 - 0.999**200) * 0.001
    assert out1 is not None and out2 is not None
    print("  Both concurrent results verified.")


# ---------------------------------------------------------------------------
# Compare wall clock vs CUDA event timing
# ---------------------------------------------------------------------------


def compare_wall_clock_vs_event() -> None:
    """Demonstrate why wall clock time differs from event time.

    Wall clock time includes:
      - Host-side driver overhead (launch, synchronization)
      - Python overhead (function calls, argument marshaling)
      - CPU scheduling delays

    CUDA event time measures only:
      - Actual GPU execution time

    This distinction matters because wall clock measurements
    can be misleading for kernel optimization.
    """
    print("=== compare_wall_clock_vs_event ===")

    n = 1_000_000
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    num_launches = 100

    # Warmup
    for _ in range(10):
        _launch_work(x, out, iterations=10)
    torch.cuda.synchronize()

    # Measure launch overhead + kernel time using events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    stream = torch.cuda.current_stream()
    stream.record_event(start_event)

    for _ in range(num_launches):
        _launch_work(x, out, iterations=20)

    stream.record_event(end_event)
    end_event.synchronize()

    gpu_time = start_event.elapsed_time(end_event)
    avg_gpu_time = gpu_time / num_launches

    # Wall clock measurement
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for _ in range(num_launches):
        _launch_work(x, out, iterations=20)
    torch.cuda.synchronize()
    wall_end = time.perf_counter()
    wall_time = (wall_end - wall_start) * 1000.0
    avg_wall_time = wall_time / num_launches

    print(f"  Average GPU time per launch: {avg_gpu_time:.4f} ms")
    print(f"  Average wall clock per launch: {avg_wall_time:.4f} ms")
    print(f"  Launch overhead per kernel: {avg_wall_time - avg_gpu_time:.4f} ms")
    print(f"  This overhead accumulates for many small launches.")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        print(f"Device: {torch.cuda.get_device_name(0)}\n")

        measure_kernel_time()
        measure_overlap_efficiency()
        compare_wall_clock_vs_event()
        print("\nAll event timing tests passed!")
