#!/usr/bin/env python3
"""Capture a scheduled PyTorch Profiler trace for one synthetic signature."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch

try:
    from .workloads import CASES, VARIANTS, WorkloadConfig, make_inputs, resolve_device, run_workload
except ImportError:
    from workloads import CASES, VARIANTS, WorkloadConfig, make_inputs, resolve_device, run_workload  # type: ignore


def make_run_dir(root: Path, case: str, variant: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for suffix in range(100):
        candidate = root / f"{case}_{variant}_{stamp}_{os.getpid()}_{suffix:02d}"
        try:
            candidate.mkdir()
            (candidate / "traces").mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate a unique run under {root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="launch")
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--numel", type=int, default=262_144)
    parser.add_argument("--matrix-size", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--cpu-gap-ms", type=float, default=2.0)
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "torch_profiler",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("wait", "warmup"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name} must be >= 0")
    if args.active <= 0 or args.repeat <= 0:
        raise SystemExit("--active and --repeat must be > 0")

    config = WorkloadConfig(args.numel, args.matrix_size, args.repeats, args.cpu_gap_ms)
    config.validate()
    device = resolve_device(args.device)
    inputs = make_inputs(args.case, config, device)
    run_dir = make_run_dir(args.output_root.resolve(), args.case, args.variant)
    trace_dir = run_dir / "traces"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    total_steps = (args.wait + args.warmup + args.active) * args.repeat
    schedule = torch.profiler.schedule(
        wait=args.wait,
        warmup=args.warmup,
        active=args.active,
        repeat=args.repeat,
    )
    metadata = {
        "case": args.case,
        "variant": args.variant,
        "device": str(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "schedule": {"wait": args.wait, "warmup": args.warmup, "active": args.active, "repeat": args.repeat},
        "warning": "Profiler overhead is non-zero; use benchmark.py for latency claims.",
    }
    with (run_dir / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    output = None
    with torch.inference_mode(), torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as prof:
        for step in range(total_steps):
            with torch.profiler.record_function(f"step/{step}"):
                output = run_workload(args.case, args.variant, inputs, config, emit_nvtx=False)
            prof.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if output is None or not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("profiled workload produced a non-finite result")

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = prof.key_averages(group_by_input_shape=args.record_shapes).table(
        sort_by=sort_key,
        row_limit=30,
    )
    with (run_dir / "key_averages.txt").open("x", encoding="utf-8") as handle:
        handle.write(table)
        handle.write("\n")
    print(table)
    print(f"wrote immutable profiler run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
