"""Reproducible single-process training-step benchmark with JSON output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any

from .baseline import build_baseline
from .metrics import (
    linear_forward_flops,
    summarize_latencies,
    training_flop_estimate,
    utilization_from_step,
)
from .optimized import build_compiled
from .workload import (
    WorkloadConfig,
    make_input,
    parameter_count,
    require_torch,
    resolve_device,
    resolve_dtype,
    synchronize,
    train_step,
)


def collect_environment(device: str) -> dict[str, Any]:
    torch = require_torch()
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count_visible": torch.cuda.device_count(),
        "selected_device": device,
        "selected_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "NCCL_DEBUG",
                "TORCH_LOGS",
            )
            if key in os.environ
        },
    }
    if device == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata["gpu"] = {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        }
    else:
        metadata["gpu"] = None
    return metadata


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _build_model(variant: str, config: WorkloadConfig, args: argparse.Namespace, dtype: Any) -> Any:
    if variant == "eager":
        return build_baseline(config, device=args.resolved_device, dtype=dtype)
    return build_compiled(
        config,
        device=args.resolved_device,
        dtype=dtype,
        mode=args.compile_mode,
    )


def benchmark_variant(
    variant: str,
    config: WorkloadConfig,
    args: argparse.Namespace,
    dtype: Any,
) -> dict[str, Any]:
    torch = require_torch()
    model = _build_model(variant, config, args, dtype)
    model.train()
    inputs = make_input(config, device=args.resolved_device, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    # Compilation, allocator growth, autotuning, and lazy library initialization are
    # intentionally outside the measured region.
    for _ in range(args.warmup):
        train_step(model, optimizer, inputs)
    synchronize(args.resolved_device)

    if args.resolved_device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step_times_s: list[float] = []
    last_loss = None
    for _ in range(args.iterations):
        # Synchronizing both boundaries makes this a complete isolated step latency.
        # It intentionally does not measure cross-step host/device pipelining.
        synchronize(args.resolved_device)
        started = time.perf_counter()
        last_loss = train_step(model, optimizer, inputs)
        synchronize(args.resolved_device)
        step_times_s.append(time.perf_counter() - started)

    latency = summarize_latencies(step_times_s)
    mean_step_s = latency.mean_ms / 1_000.0
    samples_per_second = config.batch_size / mean_step_s
    tokens_per_second = config.tokens_per_step / mean_step_s

    forward_flops = linear_forward_flops(
        config.tokens_per_step,
        (
            (config.hidden_size, config.mlp_size),
            (config.mlp_size, config.hidden_size),
        ),
    )
    flop_estimate = training_flop_estimate(
        forward_flops,
        backward_to_forward_ratio=2.0,
        recompute_forward_fraction=0.0,
        convention=(
            "FMA=2; two Linear matmuls only; backward=2*forward; no recompute; "
            "GELU, loss, bias, and SGD update excluded"
        ),
    )
    peak = args.peak_tflops * 1.0e12 if args.peak_tflops is not None else None
    utilization = utilization_from_step(
        flop_estimate,
        mean_step_s,
        peak_flops_per_device_per_second=peak,
        device_count=1,
    )

    memory: dict[str, int] | None = None
    if args.resolved_device == "cuda":
        memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "end_allocated_bytes": torch.cuda.memory_allocated(),
            "end_reserved_bytes": torch.cuda.memory_reserved(),
        }

    return {
        "variant": variant,
        "latency": latency.to_dict(),
        "raw_step_times_ms": [value * 1_000.0 for value in step_times_s],
        "throughput": {
            "samples_per_second_at_mean_step_time": samples_per_second,
            "tokens_per_second_at_mean_step_time": tokens_per_second,
            "global_samples_per_step": config.batch_size,
            "global_tokens_per_step": config.tokens_per_step,
        },
        "flops_per_step": flop_estimate.to_dict(),
        "utilization": utilization.to_dict(),
        "memory": memory,
        "parameter_count": parameter_count(model),
        "last_loss": float(last_loss.item()) if last_loss is not None else None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("eager", "compiled", "both"), default="eager")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--sequence-length", type=_positive_int, default=128)
    parser.add_argument("--hidden-size", type=_positive_int, default=256)
    parser.add_argument("--mlp-size", type=_positive_int, default=1024)
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--iterations", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help=(
            "verified dense peak TFLOP/s for the selected dtype/mode and one device; "
            "no default is inferred from GPU name"
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args(argv)
    if args.peak_tflops is not None and args.peak_tflops <= 0.0:
        parser.error("--peak-tflops must be > 0")
    args.resolved_device = resolve_device(args.device)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dtype = resolve_dtype(args.dtype, args.resolved_device)
    config = WorkloadConfig(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        mlp_size=args.mlp_size,
        seed=args.seed,
    )
    variants = ("eager", "compiled") if args.variant == "both" else (args.variant,)
    results = [benchmark_variant(variant, config, args, dtype) for variant in variants]
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "tiny_token_mlp_training_step",
        "measurement_boundary": (
            "optimizer.zero_grad through optimizer.step; host timer with CUDA "
            "synchronize immediately before and after every measured step"
        ),
        "environment": collect_environment(args.resolved_device),
        "configuration": {
            "variant": args.variant,
            "device": args.resolved_device,
            "dtype": str(dtype),
            "batch_size": config.batch_size,
            "sequence_length": config.sequence_length,
            "hidden_size": config.hidden_size,
            "mlp_size": config.mlp_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "compile_mode": args.compile_mode,
            "caller_declared_peak_tflops_per_device": args.peak_tflops,
        },
        "results": results,
        "warnings": [
            "p90/p99 are descriptive sample quantiles; use many more iterations for tail SLO claims",
            "the FLOP numerator excludes GELU, loss, bias, and optimizer kernels",
        ],
    }
    if args.peak_tflops is None:
        report["warnings"].append(
            "MFU/HFU are null because no verified matching --peak-tflops was supplied"
        )
    if len(results) == 2:
        eager, compiled = results
        report["comparison"] = {
            "compiled_speedup_from_mean_step_time": (
                eager["latency"]["mean_ms"] / compiled["latency"]["mean_ms"]
            ),
            "compiled_speedup_from_p50_step_time": (
                eager["latency"]["p50_ms"] / compiled["latency"]["p50_ms"]
            ),
            "interpretation": "greater than 1 is faster; this field makes no speedup claim",
        }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
