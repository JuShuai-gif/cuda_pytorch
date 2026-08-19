#!/usr/bin/env python3
"""Low-overhead NVTX target for Nsight Systems and Nsight Compute."""

from __future__ import annotations

import argparse
from typing import Iterable

import torch

try:
    from .workloads import CASES, VARIANTS, WorkloadConfig, make_inputs, nvtx_range, resolve_device, run_workload
except ImportError:
    from workloads import CASES, VARIANTS, WorkloadConfig, make_inputs, nvtx_range, resolve_device, run_workload  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="launch")
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--numel", type=int, default=262_144)
    parser.add_argument("--matrix-size", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--cpu-gap-ms", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.steps <= 0:
        raise SystemExit("--warmup must be >= 0 and --steps must be > 0")
    config = WorkloadConfig(args.numel, args.matrix_size, args.repeats, args.cpu_gap_ms)
    config.validate()
    device = resolve_device(args.device)
    inputs = make_inputs(args.case, config, device)

    output = None
    with torch.inference_mode():
        for _ in range(args.warmup):
            output = run_workload(args.case, args.variant, inputs, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Avoid '/' in push/pop range names: Nsight Compute uses it as part of
        # the NVTX filter stack grammar (the trailing '/' selects push/pop).
        with nvtx_range("profile_steady_state", True, device):
            for step in range(args.steps):
                with nvtx_range(f"profile_step_{step}", True, device):
                    output = run_workload(args.case, args.variant, inputs, config, emit_nvtx=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if output is None or not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("profile target produced a non-finite output")
    print(
        f"completed case={args.case} variant={args.variant} device={device} "
        f"warmup={args.warmup} steps={args.steps} checksum={float(output.float().sum().item()):.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
