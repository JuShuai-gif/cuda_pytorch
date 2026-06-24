#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLA Profiler - an industrial profiler for Vision-Language-Action policies.

A Torch-profiler / Nsight style analyzer specialized for VLA policies
(SmolVLA / pi0.5), action chunking, ROS latency constraints and ~705M models.

Quick start
-----------
>>> from vla_profiler import VLAProfiler, ProfilerConfig
>>> from vla_profiler.models.synthetic_vla import build_synthetic_vla
>>> model = build_synthetic_vla("705M")
>>> result = VLAProfiler(model, ProfilerConfig()).run(model.dummy_inputs())
>>> from vla_profiler.report import render_text
>>> print(render_text(result))
"""

from __future__ import annotations

from .latency_estimator import GPU_PRESETS_TFLOPS, LatencyStats, estimate_latency
from .kernel_profiler import (
    KernelProfile,
    build_ncu_command,
    build_nsys_command,
    profile_kernels,
    render_kernel_table,
)
from .macs_analyzer import MacsStats, compute_macs
from .model_analyzer import ParamStats, count_params
from .module_splitter import CATEGORIES, SplitConfig, classify
from .plot import save_roofline_plot
from .profiler import ProfilerConfig, ProfileResult, VLAProfiler
from .report import render_markdown, render_text, save_markdown
from .roofline import GPU_BANDWIDTH_GBPS, RooflineStats, roofline

__all__ = [
    "VLAProfiler",
    "ProfilerConfig",
    "ProfileResult",
    "compute_macs",
    "MacsStats",
    "count_params",
    "ParamStats",
    "estimate_latency",
    "LatencyStats",
    "roofline",
    "RooflineStats",
    "classify",
    "SplitConfig",
    "CATEGORIES",
    "render_text",
    "render_markdown",
    "save_markdown",
    "save_roofline_plot",
    "profile_kernels",
    "render_kernel_table",
    "KernelProfile",
    "build_ncu_command",
    "build_nsys_command",
    "GPU_PRESETS_TFLOPS",
    "GPU_BANDWIDTH_GBPS",
]
