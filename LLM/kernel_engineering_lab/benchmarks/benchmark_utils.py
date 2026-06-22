"""
GPU Kernel Benchmarking Utilities.

Provides timing infrastructure using CUDA events for precise GPU kernel
measurement, comparison, and report generation.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from tabulate import tabulate


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    warmup_steps: int = 10
    measure_steps: int = 50
    repeat: int = 3


@dataclass
class BenchmarkResult:
    """Result of a single kernel benchmark."""

    name: str
    p50_ms: float
    p90_ms: float
    p99_ms: float
    throughput: float
    bandwidth_gb_s: float
    gflops: float
    device: str = ""


def _get_device_name() -> str:
    """Return the name of the current CUDA device."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def benchmark_kernel(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    name: str = "kernel",
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkResult:
    """
    Benchmark a GPU kernel using CUDA events for precise timing.

    Args:
        fn: The kernel function to benchmark. Must accept (*args, **kwargs).
        args: Positional arguments passed to fn.
        kwargs: Keyword arguments passed to fn.
        name: Human-readable name for this benchmark.
        config: Benchmark configuration (warmup, measure steps, repeats).

    Returns:
        BenchmarkResult with p50/p90/p99 latency, throughput, and bandwidth estimates.
    """
    if kwargs is None:
        kwargs = {}
    if config is None:
        config = BenchmarkConfig()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot benchmark GPU kernels.")

    device_name = _get_device_name()

    # Warmup
    for _ in range(config.warmup_steps):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Measurement repeats
    all_times_ms: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(config.repeat):
        times_ms: list[float] = []
        stream = torch.cuda.current_stream()
        for _ in range(config.measure_steps):
            stream.record_event(start_event)
            fn(*args, **kwargs)
            stream.record_event(end_event)
            end_event.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            times_ms.append(elapsed_ms)
        all_times_ms.extend(times_ms)

    sorted_times = sorted(all_times_ms)
    n = len(sorted_times)
    p50 = sorted_times[int(n * 0.50)]
    p90 = sorted_times[int(n * 0.90)]
    p99 = sorted_times[int(n * 0.99)]

    # Throughput in operations per second
    mean_time_s = statistics.mean(sorted_times) / 1000.0
    throughput = 1.0 / mean_time_s if mean_time_s > 0 else 0.0

    # Estimate memory bandwidth: attempt to infer bytes read/written from args
    bandwidth_gb_s = _estimate_bandwidth(fn, args, kwargs, mean_time_s)

    # Estimate FLOPS
    gflops = _estimate_gflops(fn, args, kwargs, mean_time_s)

    return BenchmarkResult(
        name=name,
        p50_ms=p50,
        p90_ms=p90,
        p99_ms=p99,
        throughput=throughput,
        bandwidth_gb_s=bandwidth_gb_s,
        gflops=gflops,
        device=device_name,
    )


def benchmark_torch(
    fn: Callable,
    *args: Any,
    name: str = "torch_op",
    config: Optional[BenchmarkConfig] = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """
    Benchmark a PyTorch function with CUDA event timing.

    Args:
        fn: The PyTorch function to benchmark.
        *args: Positional arguments passed to fn.
        name: Human-readable name for this benchmark.
        config: Benchmark configuration.
        **kwargs: Keyword arguments passed to fn.

    Returns:
        BenchmarkResult with timing and throughput statistics.
    """
    return benchmark_kernel(
        fn=fn,
        args=args,
        kwargs=kwargs,
        name=name,
        config=config,
    )


def _estimate_bandwidth(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    elapsed_s: float,
) -> float:
    """
    Estimate memory bandwidth (GB/s) by counting bytes in tensor arguments.

    This is a heuristic: count the total bytes of all torch.Tensor arguments
    multiplied by 2 (read + write for elementwise ops).
    """
    total_bytes = 0
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            total_bytes += arg.numel() * arg.element_size()
    for val in kwargs.values():
        if isinstance(val, torch.Tensor) and val.is_cuda:
            total_bytes += val.numel() * val.element_size()

    # Assume read+write for elementwise operations
    total_bytes *= 2

    if elapsed_s > 0:
        return (total_bytes / elapsed_s) / 1e9
    return 0.0


def _estimate_gflops(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    elapsed_s: float,
) -> float:
    """
    Estimate GFLOPS by counting FLOPs on tensor arguments.

    This is a minimal heuristic: assumes one FLOP per element of output tensors.
    Override for more accurate analysis of specific kernels.
    """
    total_elements = 0
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            total_elements += arg.numel()
    for val in kwargs.values():
        if isinstance(val, torch.Tensor) and val.is_cuda:
            total_elements += val.numel()

    if elapsed_s > 0:
        return (total_elements / elapsed_s) / 1e9
    return 0.0


def compare_kernels(benchmark_results: list[BenchmarkResult]) -> None:
    """
    Pretty print a comparison table of kernel benchmark results.

    Args:
        benchmark_results: List of BenchmarkResult objects from multiple benchmarks.
    """
    if not benchmark_results:
        print("No benchmark results to compare.")
        return

    headers = [
        "Kernel",
        "p50 (ms)",
        "p90 (ms)",
        "p99 (ms)",
        "Throughput (op/s)",
        "Bandwidth (GB/s)",
        "GFLOPS",
        "Device",
    ]
    rows = []
    for r in benchmark_results:
        rows.append(
            [
                r.name,
                f"{r.p50_ms:.4f}",
                f"{r.p90_ms:.4f}",
                f"{r.p99_ms:.4f}",
                f"{r.throughput:.2e}",
                f"{r.bandwidth_gb_s:.2f}",
                f"{r.gflops:.2f}",
                r.device,
            ]
        )

    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="right"))


def generate_report(
    results: list[BenchmarkResult],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a benchmark report in CSV and Markdown formats.

    Args:
        results: List of BenchmarkResult objects.
        output_path: Optional base path for output files (without extension).
                     If provided, saves to {output_path}.csv and {output_path}.md.

    Returns:
        A Markdown-formatted string of the report.
    """
    if not results:
        return "# Benchmark Report\n\nNo results.\n"

    # Build Markdown
    lines = [
        "# GPU Kernel Benchmark Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device: {results[0].device if results else 'unknown'}",
        "",
        "| Kernel | p50 (ms) | p90 (ms) | p99 (ms) | Throughput (op/s) | Bandwidth (GB/s) | GFLOPS |",
        "|--------|----------|----------|----------|--------------------|-------------------|--------|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.p50_ms:.4f} | {r.p90_ms:.4f} | {r.p99_ms:.4f} "
            f"| {r.throughput:.2e} | {r.bandwidth_gb_s:.2f} | {r.gflops:.2f} |"
        )

    report_md = "\n".join(lines)

    if output_path:
        base = Path(output_path)
        # Write CSV
        csv_path = base.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "name",
                    "p50_ms",
                    "p90_ms",
                    "p99_ms",
                    "throughput",
                    "bandwidth_gb_s",
                    "gflops",
                    "device",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.name,
                        r.p50_ms,
                        r.p90_ms,
                        r.p99_ms,
                        r.throughput,
                        r.bandwidth_gb_s,
                        r.gflops,
                        r.device,
                    ]
                )
        print(f"CSV report saved to: {csv_path}")

        # Write Markdown
        md_path = base.with_suffix(".md")
        with open(md_path, "w") as f:
            f.write(report_md)
        print(f"Markdown report saved to: {md_path}")

    return report_md


def results_to_dict(results: list[BenchmarkResult]) -> list[dict]:
    """Serialize benchmark results to a list of dictionaries."""
    return [dataclasses.asdict(r) for r in results]


def dict_to_results(data: list[dict]) -> list[BenchmarkResult]:
    """Deserialize benchmark results from a list of dictionaries."""
    return [BenchmarkResult(**d) for d in data]


def save_results(results: list[BenchmarkResult], path: str) -> None:
    """Save benchmark results to a JSON file."""
    with open(path, "w") as f:
        json.dump(results_to_dict(results), f, indent=2)
    print(f"Results saved to: {path}")


def load_results(path: str) -> list[BenchmarkResult]:
    """Load benchmark results from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return dict_to_results(data)
