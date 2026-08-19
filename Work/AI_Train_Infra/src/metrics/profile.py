"""PyTorch Profiler or NVTX entry point for the metrics workload."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .baseline import build_baseline
from .optimized import build_compiled
from .workload import (
    WorkloadConfig,
    make_input,
    require_torch,
    resolve_device,
    resolve_dtype,
    synchronize,
    train_step,
)


def _build(args: argparse.Namespace, config: WorkloadConfig, dtype: Any) -> Any:
    if args.variant == "eager":
        return build_baseline(config, device=args.resolved_device, dtype=dtype)
    return build_compiled(
        config,
        device=args.resolved_device,
        dtype=dtype,
        mode=args.compile_mode,
    )


def _nvtx_context(torch: Any, device: str, message: str) -> Any:
    return torch.cuda.nvtx.range(message) if device == "cuda" else nullcontext()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "nvtx"), default="torch")
    parser.add_argument("--variant", choices=("eager", "compiled"), default="eager")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--trace", type=Path, default=Path("/tmp/ai_infra_metrics_trace.json"))
    parser.add_argument("--with-stack", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.steps <= 0:
        parser.error("--warmup must be >= 0 and --steps must be > 0")
    args.resolved_device = resolve_device(args.device)

    torch = require_torch()
    dtype = resolve_dtype(args.dtype, args.resolved_device)
    config = WorkloadConfig()
    model = _build(args, config, dtype)
    inputs = make_input(config, device=args.resolved_device, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    for _ in range(args.warmup):
        train_step(model, optimizer, inputs)
    synchronize(args.resolved_device)

    if args.backend == "nvtx":
        # Do not synchronize between steps: Nsight Systems should reveal natural
        # launch behavior and stream dependencies. The outer sync closes capture.
        with _nvtx_context(torch, args.resolved_device, "metrics_measured_region"):
            for _ in range(args.steps):
                with _nvtx_context(torch, args.resolved_device, "train_step"):
                    train_step(model, optimizer, inputs)
        synchronize(args.resolved_device)
        print("NVTX workload complete; inspect metrics_measured_region/train_step ranges")
        return 0

    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.resolved_device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    if args.trace.exists():
        raise SystemExit(f"refusing to overwrite existing trace: {args.trace}")
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=args.with_stack,
    ) as profiler:
        for _ in range(args.steps):
            with torch.profiler.record_function("train_step"):
                train_step(model, optimizer, inputs)
            profiler.step()
    synchronize(args.resolved_device)
    profiler.export_chrome_trace(str(args.trace))
    sort_key = "self_cuda_time_total" if args.resolved_device == "cuda" else "self_cpu_time_total"
    print(profiler.key_averages().table(sort_by=sort_key, row_limit=20))
    print(f"Chrome trace: {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
