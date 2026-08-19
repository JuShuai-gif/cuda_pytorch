"""Nsight Systems target for the cuda_async experiments.

Adds NVTX ranges around the H2D copy and the stream work so the nsys timeline
shows, per region, where the CPU is busy and where the GPU is busy.  NVTX names
use underscores (not ``/``) because ``/`` is range-stack syntax in NCU filters.
"""

from __future__ import annotations

import argparse

import torch

from common.env import resolve_device
from kernel.cuda_async.workloads import benchmark_streams


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--mat-size", type=int, default=512)
    p.add_argument("--work-per-stream", type=int, default=8)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--steps", type=int, default=5)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)

    if device.type != "cuda":
        raise RuntimeError("nsys target requires CUDA")

    # Single stream: sequential matmuls.
    a = torch.randn(args.mat_size, args.mat_size, device=device)
    b = torch.randn(args.mat_size, args.mat_size, device=device)
    for step in range(args.steps):
        torch.cuda.nvtx.range_push(f"single_stream_step_{step}")
        for _ in range(args.n_streams * args.work_per_stream):
            torch.mm(a, b)
        torch.cuda.nvtx.range_pop()

    # Multiple streams: chunked independent matmuls.
    streams = [torch.cuda.Stream() for _ in range(args.n_streams)]
    chunks = [
        (
            torch.randn(args.mat_size, args.mat_size, device=device),
            torch.randn(args.mat_size, args.mat_size, device=device),
        )
        for _ in range(args.n_streams)
    ]
    for step in range(args.steps):
        torch.cuda.nvtx.range_push(f"multi_stream_step_{step}")
        for s, (ca, cb) in zip(streams, chunks):
            with torch.cuda.stream(s):
                for _ in range(args.work_per_stream):
                    torch.mm(ca, cb)
        for s in streams:
            s.synchronize()
        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
