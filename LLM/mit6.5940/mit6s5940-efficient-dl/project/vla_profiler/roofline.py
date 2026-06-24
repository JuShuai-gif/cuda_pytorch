#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Roofline analysis (MIT 6.5940 core) for VLA policies.

Given the compute (MACs) and the bytes that must be moved, the roofline model
tells you whether a kernel is limited by the GPU's peak FLOP rate
(compute-bound) or by its memory bandwidth (memory-bound).

    arithmetic_intensity (AI) = FLOPs / bytes_moved      [FLOP/byte]
    ridge_point               = peak_flops / bandwidth   [FLOP/byte]
    attainable_flops          = min(peak_flops, bandwidth * AI)

A model whose AI sits left of the ridge point is memory-bound: reducing FLOPs
buys nothing, you must cut bytes moved (fusion, smaller activations, lower
precision weights).
"""

from __future__ import annotations

from dataclasses import dataclass

# Peak HBM / DRAM bandwidth in GB/s.
GPU_BANDWIDTH_GBPS = {
    "a100": 1555.0,
    "a100_80g": 2039.0,
    "h100": 3350.0,
    "rtx4090": 1008.0,
    "jetson_orin": 204.0,
    "jetson_nano": 25.6,
}


@dataclass
class RooflineStats:
    arithmetic_intensity: float  # FLOP / byte
    ridge_point: float  # FLOP / byte
    peak_tflops: float
    bandwidth_gbps: float
    attainable_tflops: float
    bytes_moved: float
    compute_bound_ratio: float  # ~1.0 means compute bound
    memory_bound_ratio: float  # ~1.0 means memory bound
    regime: str  # "compute-bound" | "memory-bound"


def roofline(
    macs: float,
    bytes_moved: float,
    peak_tflops: float,
    bandwidth_gbps: float,
) -> RooflineStats:
    flops = macs * 2.0
    bw = bandwidth_gbps * 1e9
    peak = peak_tflops * 1e12

    ai = flops / bytes_moved if bytes_moved > 0 else 0.0
    ridge = peak / bw if bw > 0 else float("inf")

    attainable = min(peak, bw * ai)
    compute_ratio = min(ai / ridge, 1.0) if ridge > 0 else 1.0
    memory_ratio = 1.0 - compute_ratio
    regime = "compute-bound" if ai >= ridge else "memory-bound"

    return RooflineStats(
        arithmetic_intensity=ai,
        ridge_point=ridge,
        peak_tflops=peak_tflops,
        bandwidth_gbps=bandwidth_gbps,
        attainable_tflops=attainable / 1e12,
        bytes_moved=bytes_moved,
        compute_bound_ratio=compute_ratio,
        memory_bound_ratio=memory_ratio,
        regime=regime,
    )


def estimate_bytes_moved(
    param_bytes: float,
    activation_bytes: float,
    weight_reuse: float = 1.0,
) -> float:
    """Rough bytes-moved estimate = weights (read once / reuse) + activations.

    ``weight_reuse`` < 1 models weights staying resident in cache across the
    batch; for batch=1 inference set it to 1.0 (each weight read once).
    """
    return param_bytes * weight_reuse + activation_bytes
