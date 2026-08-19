"""NVTX and PyTorch Profiler entry point for the paired workloads."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import torch

from .common import DTYPES, resolve_device, resolve_dtype, seed_everything, synchronize
from .workloads import WORKLOAD_NAMES, prepare_workload


@contextmanager
def step_range(label: str, device: torch.device) -> Iterator[None]:
    with torch.autograd.profiler.record_function(label):
        if device.type == "cuda":
            torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.nvtx.range_pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOAD_NAMES, default="launch")
    parser.add_argument("--variant", choices=("baseline", "optimized", "both"), default="both")
    parser.add_argument("--profiler", choices=("torch", "external"), default="torch")
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/gpu_basics_traces"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--vector-elements", type=int, default=262_144)
    parser.add_argument("--inner-iterations", type=int, default=16)
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--cpu-delay-ms", type=float, default=1.0)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be >= 1 and warmup must be >= 0")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)
    prepared = prepare_workload(
        args.workload,
        device=device,
        dtype=dtype,
        vector_elements=args.vector_elements,
        inner_iterations=args.inner_iterations,
        matrix_size=args.matrix_size,
        cpu_delay_ms=args.cpu_delay_ms,
    )
    variants = ("baseline", "optimized") if args.variant == "both" else (args.variant,)
    for _ in range(args.warmup):
        for variant in variants:
            getattr(prepared, variant)()
    synchronize(device)

    def run_steps(profiler: torch.profiler.profile | None = None) -> None:
        for step in range(args.steps):
            for variant in variants:
                # '/' is an Nsight Compute NVTX stack delimiter; keep range
                # names directly usable by --nvtx-include <name>/.
                label = f"gpu_basics_step_{args.workload}_{variant}"
                with step_range(label, device):
                    getattr(prepared, variant)()
            if profiler is not None:
                profiler.step()

    if args.profiler == "external":
        run_steps()
        synchronize(device)
        print("Completed NVTX ranges. Inspect the external Nsight report; no trace was fabricated.")
        return

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = args.trace_dir / f"gpu_basics_{args.workload}_{stamp}.json"
    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as profiler:
        run_steps(profiler)
        synchronize(device)
    profiler.export_chrome_trace(str(trace_path))
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(profiler.key_averages().table(sort_by=sort_key, row_limit=20))
    print(f"Chrome trace: {trace_path}")


if __name__ == "__main__":
    main()
