#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kernel-level profiling for VLA policies.

Two layers of granularity:

  1. torch.profiler  - in-process, gives per-operator / per-CUDA-kernel time,
                       FLOPs and memory, and can export a Chrome/Perfetto trace
                       that opens in chrome://tracing or Nsight Systems.
  2. ncu / nsys      - out-of-process NVIDIA tools for true SM-level metrics
                       (occupancy, memory throughput, warp stalls). This module
                       builds the correct command line and can launch it.

torch 2.x renamed CUDA timing fields to the generic ``device`` namespace, so
this module reads ``self_device_time_total`` with a CPU fallback.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class KernelStat:
    name: str
    device_us: float  # self device (CUDA) time, microseconds
    cpu_us: float  # self CPU time, microseconds
    count: int
    flops: float


@dataclass
class KernelProfile:
    kernels: list[KernelStat]
    total_device_us: float
    total_cpu_us: float
    device: str
    trace_path: str | None = None
    extra: dict = field(default_factory=dict)


def _as_inputs(dummy_input):
    return (
        tuple(dummy_input) if isinstance(dummy_input, (tuple, list)) else (dummy_input,)
    )


def _self_device_us(evt) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(evt, attr):
            return float(getattr(evt, attr))
    return 0.0


def profile_kernels(
    model: nn.Module,
    dummy_input,
    device: str = "cuda",
    steps: int = 20,
    warmup: int = 5,
    export_trace: str | None = None,
    top: int = 15,
) -> KernelProfile:
    """Run torch.profiler and return the hottest kernels."""
    from torch.profiler import ProfilerActivity, profile

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    inputs = _as_inputs(dummy_input)
    inputs = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs)
    model = model.to(device).eval()

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    with profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(steps):
                model(*inputs)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()

    on_gpu = device.startswith("cuda")
    rows: list[KernelStat] = []
    for evt in prof.key_averages():
        rows.append(
            KernelStat(
                name=evt.key,
                device_us=_self_device_us(evt),
                cpu_us=float(getattr(evt, "self_cpu_time_total", 0.0)),
                count=int(evt.count),
                flops=float(getattr(evt, "flops", 0) or 0),
            )
        )

    key = (lambda r: r.device_us) if on_gpu else (lambda r: r.cpu_us)
    rows.sort(key=key, reverse=True)

    total_dev = sum(r.device_us for r in rows)
    total_cpu = sum(r.cpu_us for r in rows)

    trace_path = None
    if export_trace:
        prof.export_chrome_trace(export_trace)
        trace_path = export_trace

    return KernelProfile(
        kernels=rows[:top],
        total_device_us=total_dev,
        total_cpu_us=total_cpu,
        device=device,
        trace_path=trace_path,
        extra={"steps": steps},
    )


def render_kernel_table(prof: KernelProfile) -> str:
    on_gpu = prof.device.startswith("cuda")
    total = prof.total_device_us if on_gpu else prof.total_cpu_us
    total = total or 1.0

    lines = [
        "[Kernel Breakdown] (torch.profiler, top by "
        + ("CUDA" if on_gpu else "CPU")
        + " self-time)"
    ]
    lines.append(f"{'kernel':<38}{'self(ms)':>10}{'%':>7}{'calls':>7}{'GFLOPs':>9}")
    lines.append("-" * 71)
    for k in prof.kernels:
        t_us = k.device_us if on_gpu else k.cpu_us
        pct = t_us / total * 100.0
        gflops = k.flops / 1e9 if k.flops else 0.0
        name = k.name if len(k.name) <= 37 else k.name[:34] + "..."
        lines.append(
            f"{name:<38}{t_us / 1000:>10.3f}{pct:>6.1f}%{k.count:>7}{gflops:>9.2f}"
        )
    if prof.trace_path:
        lines.append("")
        lines.append(f"Chrome/Perfetto trace -> {prof.trace_path}")
        lines.append("  open in chrome://tracing or `nsys-ui`")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Out-of-process NVIDIA tools (Nsight Compute / Systems)
# --------------------------------------------------------------------------- #
def has_ncu() -> bool:
    return shutil.which("ncu") is not None


def has_nsys() -> bool:
    return shutil.which("nsys") is not None


def build_ncu_command(
    output: str = "vla_ncu",
    target_args: list[str] | None = None,
    section: str = "full",
) -> list[str]:
    """Build an Nsight Compute command that re-launches this profiler.

    Example::

        ncu --set full --target-processes all -o vla_ncu -f \\
            python -m vla_profiler.main --kernels --no-measure
    """
    target_args = target_args or [
        sys.executable,
        "-m",
        "vla_profiler.main",
        "--kernels",
        "--no-measure",
    ]
    return [
        "ncu",
        "--set",
        section,
        "--target-processes",
        "all",
        "-o",
        output,
        "-f",
        *target_args,
    ]


def build_nsys_command(
    output: str = "vla_nsys",
    target_args: list[str] | None = None,
) -> list[str]:
    target_args = target_args or [
        sys.executable,
        "-m",
        "vla_profiler.main",
        "--kernels",
        "--no-measure",
    ]
    return [
        "nsys",
        "profile",
        "--trace",
        "cuda,nvtx,osrt",
        "-o",
        output,
        "--force-overwrite",
        "true",
        *target_args,
    ]


def run_external(cmd: list[str], timeout: int = 600) -> int:
    """Launch an external profiler command; returns its exit code."""
    logger.info("Launching: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, timeout=timeout, check=False)
        return proc.returncode
    except FileNotFoundError:
        logger.error("Tool not found: %s", cmd[0])
        return 127
    except subprocess.TimeoutExpired:
        logger.error("External profiler timed out after %ss", timeout)
        return 124
