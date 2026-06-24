#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLA-specific cost models.

A VLA policy differs from a plain CNN/Transformer in four ways that dominate
real robot latency. Each is modelled here:

    1. Action chunking      - one inference emits an H-step action chunk; the
                              effective per-control-step cost depends on the
                              re-planning strategy.
    2. KV-cache bandwidth   - the fusion transformer's attention is usually
                              memory-bandwidth bound, not compute bound.
    3. Multi-camera input   - the vision encoder cost scales with the number
                              of camera streams and their resolution.
    4. ROS / actuation      - measured compute latency is only part of the
                              end-to-end control loop budget.
"""

from __future__ import annotations

from dataclasses import dataclass

_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}


# --------------------------------------------------------------------------- #
# 1. Action chunking
# --------------------------------------------------------------------------- #
@dataclass
class ChunkRolloutStats:
    chunk_steps: int
    strategy: str
    encode_macs: float  # vision+language+fusion, paid once per chunk
    action_macs_per_step: float
    macs_per_chunk: float
    macs_per_control_step: float
    amortization: float  # naive(full replan) / chunked, >1 is the win


def chunk_rollout_cost(
    encode_macs: float,
    action_macs: float,
    chunk_steps: int,
    strategy: str = "chunk_once",
) -> ChunkRolloutStats:
    """Cost of producing / executing one action chunk.

    strategy="chunk_once": encode once, action head emits all H steps. This is
        how SmolVLA / pi0 run in practice -> cheap per control step.
    strategy="replan_every_step": full forward every control step (worst case,
        e.g. closed-loop MPC) -> H full forwards.
    """
    full_forward = encode_macs + action_macs
    if strategy == "replan_every_step":
        macs_per_chunk = full_forward * chunk_steps
    else:  # chunk_once
        macs_per_chunk = encode_macs + action_macs * chunk_steps

    per_step = macs_per_chunk / max(chunk_steps, 1)
    naive = full_forward * chunk_steps
    amortization = naive / macs_per_chunk if macs_per_chunk > 0 else 1.0

    return ChunkRolloutStats(
        chunk_steps=chunk_steps,
        strategy=strategy,
        encode_macs=encode_macs,
        action_macs_per_step=action_macs,
        macs_per_chunk=macs_per_chunk,
        macs_per_control_step=per_step,
        amortization=amortization,
    )


# --------------------------------------------------------------------------- #
# 2. KV-cache bandwidth
# --------------------------------------------------------------------------- #
@dataclass
class KVCacheStats:
    bytes_total: float
    bytes_mb: float
    read_time_ms: float  # time to stream the KV cache once
    is_bandwidth_bound: bool


def kv_cache_memory(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    dtype: str = "fp16",
    bandwidth_gbps: float = 1555.0,
) -> KVCacheStats:
    """KV cache size and the time to stream it once at peak bandwidth.

    Attention decode is memory-bound when the time to *read* the KV cache
    exceeds the time to *compute* the attention scores; for typical VLA fusion
    sizes this is almost always the case, hence the default True heuristic when
    the cache is non-trivial.
    """
    b = _DTYPE_BYTES.get(dtype, 2)
    bytes_total = 2.0 * batch * num_layers * num_heads * head_dim * seq_len * b
    read_time_ms = bytes_total / (bandwidth_gbps * 1e9) * 1e3
    return KVCacheStats(
        bytes_total=bytes_total,
        bytes_mb=bytes_total / (1024**2),
        read_time_ms=read_time_ms,
        is_bandwidth_bound=bytes_total > 1e6,
    )


# --------------------------------------------------------------------------- #
# 3. Multi-camera input
# --------------------------------------------------------------------------- #
@dataclass
class MultiCameraStats:
    num_cameras: int
    base_resolution: int
    resolution: int
    vision_macs_single: float
    vision_macs_total: float
    resolution_scale: float


def multi_camera_cost(
    vision_macs_single: float,
    num_cameras: int,
    base_resolution: int = 224,
    resolution: int = 224,
) -> MultiCameraStats:
    """Vision encoder cost scales linearly with cameras and ~quadratically
    with input resolution (token count grows as (res/base)^2)."""
    res_scale = (resolution / base_resolution) ** 2 if base_resolution else 1.0
    total = vision_macs_single * num_cameras * res_scale
    return MultiCameraStats(
        num_cameras=num_cameras,
        base_resolution=base_resolution,
        resolution=resolution,
        vision_macs_single=vision_macs_single,
        vision_macs_total=total,
        resolution_scale=res_scale,
    )


# --------------------------------------------------------------------------- #
# 4. ROS / actuation coupling
# --------------------------------------------------------------------------- #
@dataclass
class ROSLatencyStats:
    compute_ms: float
    sensor_ms: float
    actuation_ms: float
    end_to_end_ms: float
    control_hz_required: float
    control_hz_achievable: float
    meets_realtime: bool
    chunk_covers_latency: bool


def ros_latency_coupling(
    compute_ms: float,
    control_hz_required: float,
    chunk_steps: int,
    sensor_ms: float = 5.0,
    actuation_ms: float = 3.0,
) -> ROSLatencyStats:
    """End-to-end control-loop feasibility.

    Compute latency is only one term; sensor acquisition / ROS transport and
    actuation add fixed overhead. Action chunking hides inference latency as
    long as one chunk (H steps) takes longer to *execute* than to *compute*.
    """
    e2e = compute_ms + sensor_ms + actuation_ms
    achievable = 1000.0 / e2e if e2e > 0 else 0.0
    period_ms = 1000.0 / control_hz_required if control_hz_required > 0 else 0.0
    chunk_exec_ms = period_ms * chunk_steps
    return ROSLatencyStats(
        compute_ms=compute_ms,
        sensor_ms=sensor_ms,
        actuation_ms=actuation_ms,
        end_to_end_ms=e2e,
        control_hz_required=control_hz_required,
        control_hz_achievable=achievable,
        meets_realtime=achievable >= control_hz_required,
        chunk_covers_latency=chunk_exec_ms >= e2e,
    )
