"""
GPU profiling utilities for CUDA/Triton kernel analysis.

Tools:
    - CUDAEventTimer: Precise kernel timing using CUDA events
    - MemoryBandwidthCalculator: Bandwidth utilization analysis
    - OccupancyCalculator: SM occupancy analysis
    - KernelProfiler: Unified profiling wrapper for Triton/PyTorch ops
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ==============================================================================
#  GPU specification data
# ==============================================================================


@dataclass
class GPUCapabilities:
    """Hardware capabilities of a GPU model."""

    name: str
    max_threads_per_sm: int
    max_warps_per_sm: int
    max_blocks_per_sm: int
    max_registers_per_sm: int
    max_shared_memory_per_sm_bytes: int
    max_threads_per_block: int
    warp_size: int = 32
    register_allocation_granularity: int = 256


GPU_CAPABILITIES: Dict[str, GPUCapabilities] = {
    "A100": GPUCapabilities(
        name="NVIDIA A100 (Ampere)",
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=168960,  # 165 KiB
        max_threads_per_block=1024,
    ),
    "H100": GPUCapabilities(
        name="NVIDIA H100 (Hopper)",
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=233472,  # 228 KiB
        max_threads_per_block=1024,
    ),
    "RTX4090": GPUCapabilities(
        name="NVIDIA RTX 4090 (Ada Lovelace)",
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        max_blocks_per_sm=24,
        max_registers_per_sm=65536,
        max_shared_memory_per_sm_bytes=102400,  # 100 KiB
        max_threads_per_block=1024,
    ),
}


# ==============================================================================
#  CUDA event timing
# ==============================================================================


@dataclass
class TimingResult:
    """Result of a kernel timing measurement.

    Attributes:
        name: Name of the timed operation.
        mean_ms: Mean execution time in milliseconds.
        std_ms: Standard deviation in milliseconds.
        min_ms: Minimum execution time in milliseconds.
        max_ms: Maximum execution time in milliseconds.
        samples: Number of timing samples collected.
    """

    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    samples: int


class CUDAEventTimer:
    """Precise GPU kernel timing using CUDA events.

    Uses cuda.Event for GPU-side synchronization, which is more
    accurate than CPU-side timers for kernel measurements.

    Example:
        >>> timer = CUDAEventTimer()
        >>> with timer.record("my_kernel"):
        ...     some_triton_kernel(x, y)
        >>> result = timer.get_result("my_kernel")
    """

    def __init__(self, device: torch.device | str = "cuda"):
        self._device = torch.device(device)
        self._results: Dict[str, List[float]] = {}
        self._start_events: Dict[str, torch.cuda.Event] = {}
        self._end_events: Dict[str, torch.cuda.Event] = {}

    @contextmanager
    def record(self, name: str):
        """Context manager that records CUDA event times.

        Args:
            name: Identifier for this timing measurement.
        """
        if not torch.cuda.is_available():
            start = time.perf_counter()
            yield
            elapsed = (time.perf_counter() - start) * 1e3
            self._results.setdefault(name, []).append(elapsed)
            return

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        yield
        end_event.record()
        torch.cuda.synchronize()

        elapsed = start_event.elapsed_time(end_event)  # ms
        self._results.setdefault(name, []).append(elapsed)

    def get_result(
        self,
        name: str,
        warmup: int = 0,
    ) -> Optional[TimingResult]:
        """Get timing statistics for a recorded operation.

        Args:
            name: Operation identifier.
            warmup: Number of initial samples to discard.

        Returns:
            TimingResult with mean/std/min/max, or None if no data.
        """
        if name not in self._results:
            return None

        times = self._results[name][warmup:]
        if not times:
            return None

        mean = sum(times) / len(times)
        std = (
            (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
            if len(times) > 1
            else 0.0
        )
        return TimingResult(
            name=name,
            mean_ms=mean,
            std_ms=std,
            min_ms=min(times),
            max_ms=max(times),
            samples=len(times),
        )

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._results.clear()
        self._start_events.clear()
        self._end_events.clear()


def benchmark_kernel(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 5,
    repeat: int = 20,
    name: str = "kernel",
    **kwargs: Any,
) -> TimingResult:
    """Benchmark a kernel function with warmup and multiple iterations.

    Uses CUDA events when available for precise GPU-side timing.

    Args:
        fn: The kernel function to benchmark.
        *args: Positional arguments for the function.
        warmup: Number of warmup iterations (not measured).
        repeat: Number of timing iterations.
        name: Label for the result.
        **kwargs: Keyword arguments for the function.

    Returns:
        TimingResult with timing statistics.
    """
    timer = CUDAEventTimer()

    for _ in range(warmup):
        fn(*args, **kwargs)

    for _ in range(repeat):
        with timer.record(name):
            fn(*args, **kwargs)

    result = timer.get_result(name)
    if result is None:
        return TimingResult(
            name=name, mean_ms=-1, std_ms=-1, min_ms=-1, max_ms=-1, samples=0
        )
    return result


# ==============================================================================
#  Memory bandwidth utilization calculator
# ==============================================================================


@dataclass
class BandwidthReport:
    """Memory bandwidth utilization report.

    Attributes:
        total_bytes: Total bytes transferred (read + write).
        time_ms: Kernel execution time in milliseconds.
        achieved_bandwidth_gbs: Achieved bandwidth in GB/s.
        peak_bandwidth_gbs: Peak theoretical bandwidth in GB/s.
        utilization_pct: Bandwidth utilization percentage.
    """

    total_bytes: float
    time_ms: float
    achieved_bandwidth_gbs: float
    peak_bandwidth_gbs: float
    utilization_pct: float


def compute_memory_bandwidth(
    total_bytes: float,
    time_ms: float,
    peak_bandwidth_gbs: float = 3350.0,
) -> BandwidthReport:
    """Calculate memory bandwidth utilization from kernel timing.

    Args:
        total_bytes: Total bytes moved (reads + writes) by the kernel.
        time_ms: Kernel execution time in milliseconds.
        peak_bandwidth_gbs: Peak theoretical HBM bandwidth in GB/s.

    Returns:
        BandwidthReport with utilization analysis.

    Raises:
        ValueError: If time_ms is zero or negative.
    """
    if time_ms <= 0:
        raise ValueError(f"time_ms must be positive, got {time_ms}")

    achieved_gbs = total_bytes / (time_ms * 1e6)
    utilization = achieved_gbs / peak_bandwidth_gbs * 100.0

    return BandwidthReport(
        total_bytes=total_bytes,
        time_ms=time_ms,
        achieved_bandwidth_gbs=achieved_gbs,
        peak_bandwidth_gbs=peak_bandwidth_gbs,
        utilization_pct=utilization,
    )


def estimate_bandwidth_utilization(
    bytes_read: int,
    bytes_write: int,
    time_ms: float,
    gpu_name: str = "H100",
) -> BandwidthReport:
    """Estimate HBM bandwidth utilization for a kernel.

    Args:
        bytes_read: Number of bytes read from HBM.
        bytes_write: Number of bytes written to HBM.
        time_ms: Kernel execution time in milliseconds.
        gpu_name: GPU model name for peak bandwidth lookup.

    Returns:
        BandwidthReport.
    """
    peak_bw = {
        "V100": 900.0,
        "A100": 2039.0,
        "H100": 3350.0,
        "B200": 8000.0,
        "RTX4090": 1008.0,
    }.get(gpu_name, 3350.0)

    return compute_memory_bandwidth(bytes_read + bytes_write, time_ms, peak_bw)


# ==============================================================================
#  Occupancy calculator
# ==============================================================================


@dataclass
class OccupancyReport:
    """SM occupancy analysis report.

    Attributes:
        thread_blocks_per_sm: Number of thread blocks that can fit on one SM.
        warps_per_sm: Number of active warps per SM.
        threads_per_sm: Number of active threads per SM.
        theoretical_occupancy: Fraction of maximum warps (0.0 to 1.0).
        limiting_factor: Which resource limits occupancy
            (registers / shared_memory / threads / blocks).
        registers_per_thread: Register usage per thread.
        shared_mem_per_block_bytes: Shared memory per block in bytes.
        threads_per_block: Threads per thread block.
    """

    thread_blocks_per_sm: int
    warps_per_sm: int
    threads_per_sm: int
    theoretical_occupancy: float
    limiting_factor: str
    registers_per_thread: int
    shared_mem_per_block_bytes: int
    threads_per_block: int


def calculate_occupancy(
    registers_per_thread: int,
    shared_mem_per_block_bytes: int,
    threads_per_block: int,
    gpu_name: str = "H100",
) -> OccupancyReport:
    """Calculate theoretical SM occupancy for a kernel configuration.

    Determines how many thread blocks can concurrently execute
    on a Streaming Multiprocessor, limited by registers, shared
    memory, threads, and hardware block limits.

    Args:
        registers_per_thread: Number of registers used per thread.
        shared_mem_per_block_bytes: Static + dynamic shared memory
                                    per thread block.
        threads_per_block: Number of threads per thread block.
        gpu_name: GPU model identifier.

    Returns:
        OccupancyReport with detailed analysis.

    Raises:
        ValueError: If threads_per_block exceeds hardware maximum.
        KeyError: If gpu_name is not recognized.
    """
    limits = GPU_CAPABILITIES[gpu_name]

    if threads_per_block > limits.max_threads_per_block:
        raise ValueError(
            f"threads_per_block ({threads_per_block}) exceeds "
            f"hardware max ({limits.max_threads_per_block})"
        )
    if threads_per_block % limits.warp_size != 0:
        raise ValueError(
            f"threads_per_block ({threads_per_block}) must be "
            f"a multiple of warp_size ({limits.warp_size})"
        )

    warps_per_block = threads_per_block // limits.warp_size

    # Register allocation (rounded up to allocation granularity)
    regs_per_block = warps_per_block * limits.warp_size * registers_per_thread
    regs_per_block = (
        (regs_per_block + limits.register_allocation_granularity - 1)
        // limits.register_allocation_granularity
    ) * limits.register_allocation_granularity

    # Compute limits
    reg_limited = (
        limits.max_registers_per_sm // regs_per_block
        if regs_per_block > 0
        else limits.max_blocks_per_sm
    )
    smem_limited = (
        limits.max_shared_memory_per_sm_bytes // shared_mem_per_block_bytes
        if shared_mem_per_block_bytes > 0
        else limits.max_blocks_per_sm
    )
    thread_limited = limits.max_threads_per_sm // threads_per_block
    block_limited = limits.max_blocks_per_sm

    actual_blocks = min(reg_limited, smem_limited, thread_limited, block_limited)
    actual_blocks = max(actual_blocks, 1)

    active_warps = actual_blocks * warps_per_block
    occupancy = active_warps / limits.max_warps_per_sm

    # Determine limiting factor
    limiting_values = {
        "registers": (reg_limited, regs_per_block),
        "shared_memory": (
            smem_limited,
            shared_mem_per_block_bytes
            if shared_mem_per_block_bytes > 0
            else float("inf"),
        ),
        "threads": (thread_limited, threads_per_block),
        "blocks": (block_limited, 1),
    }
    limiting_factor = min(limiting_values, key=lambda k: limiting_values[k][0])

    return OccupancyReport(
        thread_blocks_per_sm=actual_blocks,
        warps_per_sm=active_warps,
        threads_per_sm=actual_blocks * threads_per_block,
        theoretical_occupancy=occupancy,
        limiting_factor=limiting_factor,
        registers_per_thread=registers_per_thread,
        shared_mem_per_block_bytes=shared_mem_per_block_bytes,
        threads_per_block=threads_per_block,
    )


def find_optimal_block_size(
    registers_per_thread: int,
    shared_mem_per_block_bytes: int,
    gpu_name: str = "H100",
    block_sizes: Optional[List[int]] = None,
) -> List[Tuple[int, float, str]]:
    """Find block sizes that maximize SM occupancy.

    Args:
        registers_per_thread: Registers used per thread.
        shared_mem_per_block_bytes: Shared memory per block.
        gpu_name: GPU model.
        block_sizes: Block sizes to evaluate. Default: powers of 2 up to 1024.

    Returns:
        Sorted list of (block_size, occupancy, limiting_factor) tuples,
        best first.
    """
    if block_sizes is None:
        block_sizes = [32, 64, 128, 256, 512, 1024]

    results: List[Tuple[int, float, str]] = []
    for bs in block_sizes:
        try:
            report = calculate_occupancy(
                registers_per_thread, shared_mem_per_block_bytes, bs, gpu_name
            )
            results.append((bs, report.theoretical_occupancy, report.limiting_factor))
        except ValueError:
            pass

    results.sort(key=lambda x: (-x[1], x[0]))
    return results


# ==============================================================================
#  Kernel profiler (unified interface)
# ==============================================================================


@dataclass
class KernelProfileResult:
    """Comprehensive profile of a single kernel execution.

    Attributes:
        name: Kernel name.
        timing: Timing statistics.
        bandwidth: Memory bandwidth analysis (if applicable).
        occupancy: SM occupancy analysis (if applicable).
        tflops: Achieved TFLOPS (if applicable).
    """

    name: str
    timing: TimingResult
    bandwidth: Optional[BandwidthReport] = None
    occupancy: Optional[OccupancyReport] = None
    tflops: Optional[float] = None


class KernelProfiler:
    """Unified profiler for Triton and PyTorch kernels.

    Combines timing, bandwidth, occupancy, and FLOP analysis
    into a single workflow.

    Example:
        >>> profiler = KernelProfiler(gpu_name="H100")
        >>> result = profiler.profile(
        ...     my_kernel_fn, x, y,
        ...     name="my_fused_op",
        ...     bytes_read=2 * x.numel() * x.element_size(),
        ...     bytes_write=x.numel() * x.element_size(),
        ...     flops=x.numel() * 5,
        ...     registers=64,
        ...     shared_mem=0,
        ...     threads_per_block=256,
        ... )
    """

    def __init__(self, gpu_name: str = "H100"):
        self.gpu_name = gpu_name
        self._timer = CUDAEventTimer()

    def profile(
        self,
        fn: Callable[..., Any],
        *args: Any,
        name: str = "kernel",
        bytes_read: int = 0,
        bytes_write: int = 0,
        flops: int = 0,
        registers: int = 0,
        shared_mem: int = 0,
        threads_per_block: int = 256,
        warmup: int = 5,
        repeat: int = 20,
        **kwargs: Any,
    ) -> KernelProfileResult:
        """Profile a kernel and return comprehensive metrics.

        Args:
            fn: Kernel function to profile.
            *args: Arguments to pass to the function.
            name: Label for this kernel.
            bytes_read: Bytes read from HBM per invocation.
            bytes_write: Bytes written to HBM per invocation.
            flops: Floating point operations per invocation.
            registers: Registers per thread (0 to skip occupancy).
            shared_mem: Shared memory per block in bytes.
            threads_per_block: Threads per block.
            warmup: Number of warmup iterations.
            repeat: Number of timing iterations.
            **kwargs: Additional keyword arguments for fn.

        Returns:
            KernelProfileResult with all computed metrics.
        """
        timing = benchmark_kernel(
            fn, *args, warmup=warmup, repeat=repeat, name=name, **kwargs
        )

        result = KernelProfileResult(name=name, timing=timing)

        # Bandwidth analysis
        if timing.mean_ms > 0 and (bytes_read > 0 or bytes_write > 0):
            result.bandwidth = estimate_bandwidth_utilization(
                bytes_read, bytes_write, timing.mean_ms, self.gpu_name
            )

        # FLOP analysis
        if timing.mean_ms > 0 and flops > 0:
            result.tflops = flops / (timing.mean_ms * 1e6 * 1e12)

        # Occupancy analysis
        if registers > 0:
            try:
                result.occupancy = calculate_occupancy(
                    registers, shared_mem, threads_per_block, self.gpu_name
                )
            except (ValueError, KeyError):
                pass

        return result

    def summary(self, result: KernelProfileResult) -> str:
        """Format a profile result as a human-readable summary string.

        Args:
            result: Profile result from profile().

        Returns:
            Multi-line summary string.
        """
        lines = [
            f"Kernel: {result.name}",
            f"  Time:        {result.timing.mean_ms:.3f} ms +/- {result.timing.std_ms:.3f} ms",
        ]

        if result.bandwidth is not None:
            lines.append(
                f"  Bandwidth:   {result.bandwidth.achieved_bandwidth_gbs:.1f} GB/s "
                f"({result.bandwidth.utilization_pct:.1f}% of peak)"
            )

        if result.tflops is not None:
            lines.append(f"  TFLOPS:      {result.tflops:.1f}")

        if result.occupancy is not None:
            lines.append(
                f"  Occupancy:   {result.occupancy.theoretical_occupancy:.1%} "
                f"(limited by {result.occupancy.limiting_factor})"
            )

        return "\n".join(lines)


# ==============================================================================
#  Demonstration
# ==============================================================================

if __name__ == "__main__":
    print("=== Occupancy Calculator ===\n")

    for gpu_name in ["A100", "H100", "RTX4090"]:
        report = calculate_occupancy(64, 48 * 1024, 256, gpu_name)
        print(f"{gpu_name}:")
        print(f"  Blocks/SM:  {report.thread_blocks_per_sm}")
        print(f"  Warps/SM:   {report.warps_per_sm}")
        print(f"  Occupancy:  {report.theoretical_occupancy:.1%}")
        print(f"  Limited by: {report.limiting_factor}")
        print()

    print("=== Optimal Block Size Scan (H100, 64 regs, 48 KiB smem) ===\n")
    for bs, occ, limiter in find_optimal_block_size(64, 48 * 1024, "H100"):
        print(f"  block_size={bs:4d}  occupancy={occ:.1%}  limited_by={limiter}")

    print("\nAll checks passed.")
