"""Minimal benchmark with correctness, synchronized timing, and Roofline estimates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .common import (
    DTYPES,
    benchmark_callable,
    dump_json,
    environment_metadata,
    resolve_device,
    resolve_dtype,
    seed_everything,
)
from .workloads import WORKLOAD_NAMES, compare_outputs, prepare_workload


def _roofline(
    *,
    flops: int,
    bytes_lower_bound: int,
    milliseconds: float,
    peak_tflops: float | None,
    peak_bandwidth_gbs: float | None,
) -> dict[str, Any]:
    seconds = milliseconds / 1_000.0
    achieved_tflops = flops / seconds / 1e12 if flops else 0.0
    achieved_gbs = bytes_lower_bound / seconds / 1e9
    intensity = flops / bytes_lower_bound if bytes_lower_bound else None
    result: dict[str, Any] = {
        "arithmetic_intensity_flop_per_byte": intensity,
        "achieved_tflops_from_estimate": achieved_tflops,
        "achieved_gbs_from_byte_lower_bound": achieved_gbs,
        "hardware_roof_tflops": None,
        "roofline_efficiency": None,
        "limitation": "FLOPs and bytes are analytical estimates, not hardware-counter measurements.",
    }
    if peak_tflops is not None and peak_bandwidth_gbs is not None and intensity is not None:
        bandwidth_roof_tflops = intensity * peak_bandwidth_gbs / 1_000.0
        roof = min(peak_tflops, bandwidth_roof_tflops)
        result.update(
            {
                "hardware_roof_tflops": roof,
                "bandwidth_roof_tflops": bandwidth_roof_tflops,
                "roofline_efficiency": achieved_tflops / roof if roof > 0 else None,
            }
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)
    names = list(WORKLOAD_NAMES) if args.workload == "all" else [args.workload]
    variants = ("baseline", "optimized") if args.variant == "both" else (args.variant,)
    result_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for name in names:
        prepared = prepare_workload(
            name,
            device=device,
            dtype=dtype,
            vector_elements=args.vector_elements,
            inner_iterations=args.inner_iterations,
            matrix_size=args.matrix_size,
            cpu_delay_ms=args.cpu_delay_ms,
        )
        with torch.no_grad():
            baseline_output = prepared.baseline()
            optimized_output = prepared.optimized()
        correctness = compare_outputs(
            baseline_output, optimized_output, rtol=args.rtol, atol=args.atol
        )
        if not correctness["passed"] and not args.allow_correctness_failure:
            raise RuntimeError(f"Correctness failed for {name}: {correctness}")

        rows_by_variant: dict[str, dict[str, Any]] = {}
        for variant in variants:
            function = getattr(prepared, variant)
            allocated_before = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                allocated_before = torch.cuda.memory_allocated(device)
            timing = benchmark_callable(
                function,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            peak_allocated = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            )
            wall_ms = timing.synchronized_wall.median
            cost = prepared.costs[variant]
            row = {
                "workload": name,
                "variant": variant,
                "expected_bottleneck": prepared.expected_bottleneck,
                "workload_config": prepared.config,
                "correctness_against_pair": correctness,
                "timing": timing,
                "throughput_calls_per_second": 1_000.0 / wall_ms,
                "effective_elements_per_second": (
                    cost.effective_elements_per_call * 1_000.0 / wall_ms
                ),
                "analytical_cost": cost,
                "roofline": _roofline(
                    flops=cost.flops_per_call,
                    bytes_lower_bound=cost.bytes_per_call_lower_bound,
                    milliseconds=(
                        timing.cuda_event.median if timing.cuda_event is not None else wall_ms
                    ),
                    peak_tflops=args.peak_tflops,
                    peak_bandwidth_gbs=args.peak_bandwidth_gbs,
                ),
                "memory": {
                    "allocated_before_bytes": allocated_before,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_increment_bytes": (
                        peak_allocated - allocated_before
                        if peak_allocated is not None and allocated_before is not None
                        else None
                    ),
                    "scope": "PyTorch live tensor allocator only; CUDA context and non-PyTorch allocations excluded.",
                },
                "trace_only_metrics": {
                    "gpu_active_time_ms": None,
                    "gpu_idle_bubble_ms": None,
                    "kernel_count": None,
                    "kernel_duration_distribution": None,
                    "reason": "Use profile_workloads.py/Nsight; this benchmark does not infer trace metrics.",
                },
            }
            rows_by_variant[variant] = row
            result_rows.append(row)

        if "baseline" in rows_by_variant and "optimized" in rows_by_variant:
            base_ms = rows_by_variant["baseline"]["timing"].synchronized_wall.median
            opt_ms = rows_by_variant["optimized"]["timing"].synchronized_wall.median
            comparisons.append(
                {
                    "workload": name,
                    "correctness": correctness,
                    "wall_time_speedup_baseline_over_optimized": base_ms / opt_ms,
                    "baseline_wall_ms": base_ms,
                    "optimized_wall_ms": opt_ms,
                    "interpretation": "Measured on this run; the optimized candidate is not assumed faster.",
                }
            )

    return {
        "schema_version": 1,
        "experiment": "gpu_execution_and_memory_microbenchmarks",
        "environment": environment_metadata(device),
        "config": vars(args),
        "results": result_rows,
        "comparisons": comparisons,
        "limitations": [
            "No GPU utilization, kernel count, or bubble value is fabricated from wall timing.",
            "Microbenchmarks are shape/dtype/hardware specific and are not training MFU.",
            "CPU fallback validates behavior but cannot validate CUDA asynchrony or GPU bottleneck labels.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("all", *WORKLOAD_NAMES), default="all")
    parser.add_argument("--variant", choices=("both", "baseline", "optimized"), default="both")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--vector-elements", type=int, default=262_144)
    parser.add_argument("--inner-iterations", type=int, default=16)
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--cpu-delay-ms", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--allow-correctness-failure", action="store_true")
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--peak-bandwidth-gbs", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run(args)
    print(dump_json(payload, args.output))


if __name__ == "__main__":
    main()

