#!/usr/bin/env python3
"""Benchmark baseline/optimized profiling workloads without inventing results."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

try:
    from .workloads import (
        CASES,
        VARIANTS,
        WorkloadConfig,
        estimated_executed_flops,
        make_inputs,
        resolve_device,
        run_workload,
    )
except ImportError:  # Direct execution: python benchmark.py
    from workloads import (  # type: ignore
        CASES,
        VARIANTS,
        WorkloadConfig,
        estimated_executed_flops,
        make_inputs,
        resolve_device,
        run_workload,
    )


def percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute percentile of an empty sample")
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    case: str,
    variant: str,
    config: WorkloadConfig,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> Dict[str, Any]:
    inputs = make_inputs(case, config, device)
    output = None
    with torch.inference_mode():
        for _ in range(warmup):
            output = run_workload(case, variant, inputs, config)
        synchronize(device)

        wall_ms: List[float] = []
        gpu_ms: List[float] = []
        for _ in range(iterations):
            start_event = end_event = None
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            start_ns = time.perf_counter_ns()
            output = run_workload(case, variant, inputs, config)
            if end_event is not None:
                end_event.record()
            synchronize(device)
            wall_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
            if start_event is not None and end_event is not None:
                gpu_ms.append(float(start_event.elapsed_time(end_event)))

    # Force a real read after timing so lazy mistakes cannot hide behind an
    # unused result.  PyTorch eager CUDA already executes eagerly; this is a
    # correctness guard, not part of measured latency.
    if output is None or not bool(torch.isfinite(output).all().item()):
        raise RuntimeError(f"{case}/{variant} produced a non-finite output")

    stats: Dict[str, Any] = {
        "case": case,
        "variant": variant,
        "samples": iterations,
        "warmup": warmup,
        "wall_ms": {
            "mean": statistics.fmean(wall_ms),
            "stddev": statistics.stdev(wall_ms) if len(wall_ms) > 1 else 0.0,
            "p50": percentile(wall_ms, 0.50),
            "p90": percentile(wall_ms, 0.90),
            "p99": percentile(wall_ms, 0.99),
            "median": statistics.median(wall_ms),
            "p95": percentile(wall_ms, 0.95),
            "min": min(wall_ms),
            "max": max(wall_ms),
            "raw": wall_ms,
        },
        "estimated_executed_flops_per_invocation": estimated_executed_flops(case, variant, config),
        "estimate_warning": "source-level arithmetic estimate; not a measured hardware counter and not MFU",
    }
    if gpu_ms:
        stats["gpu_event_ms"] = {
            "mean": statistics.fmean(gpu_ms),
            "stddev": statistics.stdev(gpu_ms) if len(gpu_ms) > 1 else 0.0,
            "p50": percentile(gpu_ms, 0.50),
            "p90": percentile(gpu_ms, 0.90),
            "p99": percentile(gpu_ms, 0.99),
            "median": statistics.median(gpu_ms),
            "p95": percentile(gpu_ms, 0.95),
            "min": min(gpu_ms),
            "max": max(gpu_ms),
            "raw": gpu_ms,
        }
    else:
        stats["gpu_event_ms"] = None
    return stats


def make_run_dir(root: Path, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for suffix in range(100):
        candidate = root / f"{label}_{stamp}_{os.getpid()}_{suffix:02d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a unique run directory under {root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--numel", type=int, default=262_144)
    parser.add_argument("--matrix-size", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--cpu-gap-ms", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "benchmarks",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("--warmup must be >= 0 and --iterations must be > 0")
    config = WorkloadConfig(
        numel=args.numel,
        matrix_size=args.matrix_size,
        repeats=args.repeats,
        cpu_gap_ms=args.cpu_gap_ms,
    )
    config.validate()
    device = resolve_device(args.device)
    cases = CASES if args.case == "all" else (args.case,)
    variants = VARIANTS if args.variant == "both" else (args.variant,)
    run_dir = make_run_dir(args.output_root.resolve(), "ab")

    results: List[Dict[str, Any]] = []
    for case in cases:
        for variant in variants:
            result = measure(case, variant, config, device, args.warmup, args.iterations)
            results.append(result)
            wall = result["wall_ms"]
            gpu = result["gpu_event_ms"]
            gpu_text = "unavailable" if gpu is None else f"{gpu['median']:.4f} ms"
            print(
                f"{case:>7}/{variant:<9} wall median={wall['median']:.4f} ms "
                f"p95={wall['p95']:.4f} ms gpu-event median={gpu_text}"
            )

    comparisons: List[Dict[str, Any]] = []
    if args.variant == "both":
        for case in cases:
            baseline = next(r for r in results if r["case"] == case and r["variant"] == "baseline")
            optimized = next(r for r in results if r["case"] == case and r["variant"] == "optimized")
            base_ms = baseline["wall_ms"]["median"]
            opt_ms = optimized["wall_ms"]["median"]
            comparisons.append(
                {
                    "case": case,
                    "wall_median_speedup": base_ms / opt_ms if opt_ms else None,
                    "baseline_wall_median_ms": base_ms,
                    "optimized_wall_median_ms": opt_ms,
                }
            )

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "config": {
            "numel": config.numel,
            "matrix_size": config.matrix_size,
            "repeats": config.repeats,
            "cpu_gap_ms": config.cpu_gap_ms,
        },
        "results": results,
        "comparisons": comparisons,
        "interpretation_guardrail": (
            "These are measurements of synthetic signatures on this machine. "
            "Each invocation is synchronized to measure isolated latency; that "
            "synchronization intentionally perturbs launch/overlap behavior. Use "
            "profile_target.py for a representative asynchronous timeline. Do not "
            "generalize speedups to a training job without an end-to-end A/B run."
        ),
    }
    result_path = run_dir / "results.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote immutable run artifact: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
