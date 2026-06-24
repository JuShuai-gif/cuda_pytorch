#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latency modelling for VLA policies: theoretical roofline vs measured.

Theoretical latency is the ideal lower bound assuming the GPU runs at peak
FLOP throughput. Measured latency is what the hardware actually delivers.
``efficiency = theoretical / measured`` is the single most useful number for
deciding whether a model is compute-bound (high efficiency) or stalled on
memory / kernel-launch (low efficiency).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Peak compute presets in TFLOPs (dense, no sparsity).
GPU_PRESETS_TFLOPS = {
    "a100_fp16": 312.0,
    "a100_fp32": 19.5,
    "a100_tf32": 156.0,
    "h100_fp16": 989.0,
    "h100_fp8": 1979.0,
    "rtx4090_fp16": 165.0,
    "jetson_orin_int8": 275.0,
    "jetson_nano_fp16": 0.47,
}


@dataclass
class LatencyStats:
    theoretical_ms: float
    measured_ms: float | None
    efficiency: float | None  # theoretical / measured, in [0, 1]
    p50_ms: float | None = None
    p99_ms: float | None = None
    throughput_sps: float | None = None
    device: str = "cpu"
    gpu_tflops: float = 0.0


def theoretical_latency_ms(macs: float, gpu_tflops: float) -> float:
    """Ideal latency: 2*MACs flops divided by peak throughput."""
    flops = macs * 2.0
    peak = gpu_tflops * 1e12
    return flops / peak * 1e3


def measure_latency(
    model: nn.Module,
    dummy_input,
    device: str = "cuda",
    warmup: int = 50,
    repeat: int = 200,
) -> dict[str, float]:
    """Device-accurate latency. Uses CUDA events on GPU, perf_counter on CPU."""
    import time

    import numpy as np

    inputs = dummy_input if isinstance(dummy_input, (tuple, list)) else (dummy_input,)
    inputs = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs)
    model = model.to(device).eval()

    timings: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(repeat):
                starter.record()
                model(*inputs)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        else:
            for _ in range(repeat):
                t0 = time.perf_counter()
                model(*inputs)
                timings.append((time.perf_counter() - t0) * 1e3)

    arr = np.asarray(timings)
    mean = float(arr.mean())
    return {
        "mean_ms": mean,
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "throughput_sps": 1000.0 / mean if mean > 0 else 0.0,
    }


def estimate_latency(
    macs: float,
    gpu_tflops: float,
    model: nn.Module | None = None,
    dummy_input=None,
    device: str = "cuda",
    measure: bool = True,
    warmup: int = 50,
    repeat: int = 200,
) -> LatencyStats:
    theo = theoretical_latency_ms(macs, gpu_tflops)

    measured = p50 = p99 = thr = None
    if measure and model is not None and dummy_input is not None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        m = measure_latency(model, dummy_input, device, warmup, repeat)
        measured, p50, p99, thr = (
            m["mean_ms"],
            m["p50_ms"],
            m["p99_ms"],
            m["throughput_sps"],
        )

    eff = (theo / measured) if (measured and measured > 0) else None
    return LatencyStats(
        theoretical_ms=theo,
        measured_ms=measured,
        efficiency=eff,
        p50_ms=p50,
        p99_ms=p99,
        throughput_sps=thr,
        device=device,
        gpu_tflops=gpu_tflops,
    )
