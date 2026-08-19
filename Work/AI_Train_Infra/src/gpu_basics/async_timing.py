"""Contrast CUDA enqueue time with synchronized wall and CUDA-event time."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from .common import dump_json, environment_metadata, resolve_device, seed_everything, synchronize


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    seed_everything(args.seed)
    a = torch.randn(args.matrix_size, args.matrix_size, device=device)
    b = torch.randn(args.matrix_size, args.matrix_size, device=device)

    def workload() -> torch.Tensor:
        output = a
        for _ in range(args.launches):
            output = torch.mm(output, b)
        return output

    for _ in range(args.warmup):
        workload()
    synchronize(device)

    if device.type == "cuda":
        start_host = perf_counter()
        workload()
        enqueue_end = perf_counter()
        synchronize(device)
        completion_end = perf_counter()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        synchronize(device)
        start_event.record()
        workload()
        end_event.record()
        end_event.synchronize()
        event_ms = start_event.elapsed_time(end_event)

        synchronize(device)
        wall_start = perf_counter()
        workload()
        synchronize(device)
        synchronized_wall_ms = (perf_counter() - wall_start) * 1_000.0
        measurements: dict[str, Any] = {
            "host_enqueue_ms_not_kernel_latency": (enqueue_end - start_host) * 1_000.0,
            "same_run_enqueue_to_completion_ms": (completion_end - start_host) * 1_000.0,
            "separate_run_cuda_event_ms": event_ms,
            "separate_run_synchronized_wall_ms": synchronized_wall_ms,
            "warning": "The three values include different runs; warm clocks and variance still matter.",
        }
    else:
        wall_start = perf_counter()
        workload()
        synchronized_wall_ms = (perf_counter() - wall_start) * 1_000.0
        measurements = {
            "cpu_wall_ms": synchronized_wall_ms,
            "host_enqueue_ms_not_kernel_latency": None,
            "cuda_event_ms": None,
            "warning": "CPU execution is synchronous here, so it cannot demonstrate CUDA enqueue semantics.",
        }
    return {
        "schema_version": 1,
        "experiment": "cuda_async_timing",
        "environment": environment_metadata(device),
        "config": vars(args),
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--launches", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(dump_json(run(args), args.output))


if __name__ == "__main__":
    main()

