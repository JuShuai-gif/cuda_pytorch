#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLAProfiler - the main orchestrator.

Runs the full analysis pipeline on a Vision-Language-Action policy and returns
a single ``ProfileResult`` that the report module renders. Designed to work on
any nn.Module whose submodules can be classified by ``module_splitter`` (real
SmolVLA / pi0 checkpoints or the bundled synthetic model).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from . import latency_estimator as le
from . import macs_analyzer as ma
from . import model_analyzer as moa
from . import module_splitter as ms
from . import roofline as rf
from . import vla_extensions as vx

logger = logging.getLogger(__name__)


@dataclass
class ProfilerConfig:
    # Hardware
    gpu_name: str = "A100"
    precision: str = "fp16"
    gpu_tflops: float = 312.0  # A100 FP16
    bandwidth_gbps: float = 2039.0  # A100 80GB HBM2e
    device: str = "cuda"
    # Measurement
    measure_latency: bool = True
    warmup: int = 30
    repeat: int = 100
    macs_backend: str = "auto"
    # VLA specifics
    chunk_steps: int = 50
    control_hz: float = 30.0
    num_cameras: int = 1
    resolution: int = 224
    base_resolution: int = 224
    # KV cache (optional; skipped if any is None)
    kv_layers: int | None = None
    kv_heads: int | None = None
    kv_head_dim: int | None = None
    kv_seq_len: int | None = None
    split_config: ms.SplitConfig | None = None


@dataclass
class ProfileResult:
    params: moa.ParamStats
    macs: ma.MacsStats
    latency: le.LatencyStats
    roofline: rf.RooflineStats
    activation_bytes: float
    weight_bytes: float
    config: ProfilerConfig
    chunk: vx.ChunkRolloutStats | None = None
    kv_cache: vx.KVCacheStats | None = None
    multi_camera: vx.MultiCameraStats | None = None
    ros: vx.ROSLatencyStats | None = None
    bottlenecks: list[str] = field(default_factory=list)


_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}


def _estimate_activation_bytes(
    model: nn.Module,
    inputs: tuple,
    dtype_bytes: int,
) -> float:
    """Sum of forward output tensor bytes across all leaf modules (proxy)."""
    total = 0.0
    handles = []

    def hook(module, inp, out):
        nonlocal total
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for t in outs:
            if isinstance(t, torch.Tensor):
                total += t.numel() * dtype_bytes

    for module in model.modules():
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(*inputs)
    if was_training:
        model.train()
    for h in handles:
        h.remove()
    return total


class VLAProfiler:
    """One-call profiler for VLA policies.

    Example
    -------
    >>> profiler = VLAProfiler(model, ProfilerConfig(gpu_name="A100"))
    >>> result = profiler.run((images, lang_tokens))
    """

    def __init__(self, model: nn.Module, config: ProfilerConfig | None = None):
        self.model = model
        self.config = config or ProfilerConfig()

    def run(self, dummy_input: Any) -> ProfileResult:
        cfg = self.config
        inputs = (
            tuple(dummy_input)
            if isinstance(dummy_input, (tuple, list))
            else (dummy_input,)
        )

        # 1. Parameters
        params = moa.count_params(self.model, cfg.split_config)

        # 2. MACs
        macs = ma.compute_macs(self.model, inputs, cfg.macs_backend, cfg.split_config)

        # 3. Latency (theoretical + measured)
        device = cfg.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU latency.")
            device = "cpu"
        latency = le.estimate_latency(
            macs.total_macs,
            cfg.gpu_tflops,
            model=self.model,
            dummy_input=inputs,
            device=device,
            measure=cfg.measure_latency,
            warmup=cfg.warmup,
            repeat=cfg.repeat,
        )

        # 4. Roofline (needs bytes moved)
        dtype_bytes = _DTYPE_BYTES.get(cfg.precision, 2)
        weight_bytes = params.size_mb * (1024**2)
        try:
            act_bytes = _estimate_activation_bytes(
                self.model.to("cpu"),
                tuple(
                    t.to("cpu") if isinstance(t, torch.Tensor) else t for t in inputs
                ),
                dtype_bytes,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Activation byte estimate failed: %s", exc)
            act_bytes = weight_bytes  # conservative fallback
        bytes_moved = rf.estimate_bytes_moved(weight_bytes, act_bytes)
        roof = rf.roofline(
            macs.total_macs, bytes_moved, cfg.gpu_tflops, cfg.bandwidth_gbps
        )

        # 5. VLA-specific extensions
        encode_macs = (
            macs.per_category["vision"]
            + macs.per_category["language"]
            + macs.per_category["fusion"]
        )
        action_macs = macs.per_category["action"]
        chunk = vx.chunk_rollout_cost(
            encode_macs, action_macs, cfg.chunk_steps, "chunk_once"
        )

        kv = None
        if all(
            v is not None
            for v in (cfg.kv_layers, cfg.kv_heads, cfg.kv_head_dim, cfg.kv_seq_len)
        ):
            kv = vx.kv_cache_memory(
                cfg.kv_layers,
                cfg.kv_heads,
                cfg.kv_head_dim,
                cfg.kv_seq_len,
                dtype=cfg.precision,
                bandwidth_gbps=cfg.bandwidth_gbps,
            )

        cams = vx.multi_camera_cost(
            macs.per_category["vision"],
            cfg.num_cameras,
            cfg.base_resolution,
            cfg.resolution,
        )

        ros = None
        if latency.measured_ms is not None:
            ros = vx.ros_latency_coupling(
                latency.measured_ms, cfg.control_hz, cfg.chunk_steps
            )

        bottlenecks = self._analyze_bottlenecks(macs, latency, roof, kv)

        return ProfileResult(
            params=params,
            macs=macs,
            latency=latency,
            roofline=roof,
            activation_bytes=act_bytes,
            weight_bytes=weight_bytes,
            config=cfg,
            chunk=chunk,
            kv_cache=kv,
            multi_camera=cams,
            ros=ros,
            bottlenecks=bottlenecks,
        )

    @staticmethod
    def _analyze_bottlenecks(
        macs: ma.MacsStats,
        latency: le.LatencyStats,
        roof: rf.RooflineStats,
        kv: vx.KVCacheStats | None,
    ) -> list[str]:
        notes: list[str] = []

        if roof.regime == "memory-bound":
            if kv is not None and kv.is_bandwidth_bound:
                notes.append("Primary: Attention KV memory bandwidth")
            else:
                notes.append("Primary: Activation/weight memory bandwidth")
        else:
            notes.append("Primary: Compute (Tensor-Core) throughput")

        # Secondary: dominant category by MACs.
        frac = macs.category_fraction
        dominant = max(frac, key=frac.get)
        sec = {
            "vision": "Vision backbone resolution cost",
            "language": "Language encoder sequence length",
            "fusion": "Fusion transformer depth / attention",
            "action": "Action head / chunk decoding",
        }[dominant]
        notes.append(f"Secondary: {sec} ({frac[dominant] * 100:.0f}% of MACs)")

        if latency.efficiency is not None and latency.efficiency < 0.10:
            notes.append(
                "Note: efficiency <10% -> kernel-launch / small-batch / "
                "memory-latency bound (typical for batch=1 robot inference)"
            )
        return notes
