"""
MCUNet / TinyML Memory Budget Simulation (Lecture 10)
=====================================================
This module simulates the memory constraints of deploying tiny neural
networks on microcontroller units (MCUs).  It implements a TinyCNN
builder that analytically checks SRAM and Flash budgets *before*
instantiating the model, and generates a formatted memory budget report.

Key concepts demonstrated:
  - SRAM budget  (activation memory / runtime memory, typically ~256 KB)
  - Flash budget (parameter storage, typically ~1 MB on-chip)
  - Analytical MAC / parameter / activation-memory computation
  - Budget-aware architecture construction that refuses invalid configs

All computation is CPU-only; standard-library dependencies only (torch, numpy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BYTES_PER_FLOAT32 = 4  # float32 element size in bytes
KB = 1024
MB = 1024 * KB

# Default MCU budgets (can be overridden via constructor arguments)
DEFAULT_SRAM_BUDGET = 256 * KB  # 256 KB  – typical Cortex-M4/M7 SRAM
DEFAULT_FLASH_BUDGET = 1 * MB  # 1 MB    – typical on-chip Flash


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """Analytical statistics for a single layer in the network."""

    name: str
    layer_type: str
    spatial_in: str  # e.g. "1x28x28"
    spatial_out: str  # e.g. "8x26x26"
    params: int  # number of trainable parameters
    param_bytes: int  # Flash footprint of parameters
    macs: int  # multiply-accumulate operations
    activation_elements: int  # output tensor element count
    activation_bytes: int  # SRAM footprint of output activation


@dataclass
class MCUMemoryReport:
    """Aggregated memory report for a TinyCNN candidate."""

    model_name: str
    layers: List[LayerStats] = field(default_factory=list)
    total_params: int = 0
    total_param_bytes: int = 0
    total_macs: int = 0
    peak_activation_bytes: int = 0
    peak_activation_kb: float = 0.0
    sram_budget_bytes: int = DEFAULT_SRAM_BUDGET
    flash_budget_bytes: int = DEFAULT_FLASH_BUDGET
    sram_ok: bool = True
    flash_ok: bool = True
    passed: bool = True
    rejection_reason: str = ""


# ---------------------------------------------------------------------------
# Analytical helper functions
# ---------------------------------------------------------------------------


def _conv_output_size(
    h_in: int,
    w_in: int,
    kernel_size: int | Tuple[int, int],
    stride: int | Tuple[int, int] = 1,
    padding: int | Tuple[int, int] = 0,
    dilation: int | Tuple[int, int] = 1,
) -> Tuple[int, int]:
    """Compute spatial output dimensions for a Conv2d / Pool2d layer."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    h_out = math.floor(
        (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    w_out = math.floor(
        (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )
    return h_out, w_out


def _compute_conv2d_macs(
    in_c: int,
    out_c: int,
    k_h: int,
    k_w: int,
    out_h: int,
    out_w: int,
    groups: int = 1,
) -> int:
    """Compute MACs for a Conv2d layer (without bias-add overhead).

    MACs = out_c * (in_c / groups) * k_h * k_w * out_h * out_w
    """
    return out_c * (in_c // groups) * k_h * k_w * out_h * out_w


def _compute_conv2d_params(
    in_c: int,
    out_c: int,
    k_h: int,
    k_w: int,
    bias: bool = True,
    groups: int = 1,
) -> int:
    """Number of trainable parameters for a Conv2d layer."""
    params = out_c * (in_c // groups) * k_h * k_w
    if bias:
        params += out_c
    return params


def _compute_linear_macs(in_features: int, out_features: int) -> int:
    return in_features * out_features


def _compute_linear_params(
    in_features: int, out_features: int, bias: bool = True
) -> int:
    params = in_features * out_features
    if bias:
        params += out_features
    return params


# ---------------------------------------------------------------------------
# TinyCNN builder
# ---------------------------------------------------------------------------


class TinyCNN(nn.Module):
    """A memory-budget-aware tiny CNN for MCU deployment simulation.

    Do not instantiate directly; use ``build_tiny_cnn()`` which validates
    budgets and returns the model together with an MCU memory report.
    """

    def __init__(self, layers: nn.ModuleList, report: MCUMemoryReport):
        super().__init__()
        self.features = layers
        self.memory_report = report

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            x = layer(x)
        return x


def build_tiny_cnn(
    model_name: str,
    layer_configs: List[dict],
    input_shape: Tuple[int, int, int],  # (C, H, W)
    sram_budget: int = DEFAULT_SRAM_BUDGET,
    flash_budget: int = DEFAULT_FLASH_BUDGET,
    byte_width: int = BYTES_PER_FLOAT32,
) -> Tuple[Optional[TinyCNN], MCUMemoryReport]:
    """Validate budgets and (if budgets pass) build a TinyCNN.

    Parameters
    ----------
    model_name : str
        Human-readable name for the report.
    layer_configs : list of dict
        Ordered layer descriptions.  Each dict must contain a ``"type"`` key.
        Supported types and their extra keys:

          - ``"conv"`` : ``out_channels``, ``kernel_size``, ``stride`` (default 1),
            ``padding`` (default 0), ``bias`` (default True), ``groups`` (default 1).
            ``in_channels`` is inferred from the current spatial tracker.
          - ``"maxpool"`` / ``"avgpool"`` : ``kernel_size``, ``stride``,
            ``padding`` (default 0).
          - ``"relu"`` : no extra keys (in-place activation, no spatial change).
          - ``"flatten"`` : no extra keys.
          - ``"fc"`` : ``out_features``, ``bias`` (default True).
            ``in_features`` is inferred from the flattened dimension.

    input_shape : (C, H, W)
        Shape of a *single sample* (batch dimension excluded).
    sram_budget : int
        SRAM budget in bytes (default 256 KB).
    flash_budget : int
        Flash budget in bytes (default 1 MB).
    byte_width : int
        Bytes per element (default 4 for float32).

    Returns
    -------
    model : TinyCNN or None
        The constructed model, or ``None`` if budgets are exceeded.
    report : MCUMemoryReport
        Detailed analytical memory report.
    """
    report = MCUMemoryReport(
        model_name=model_name,
        sram_budget_bytes=sram_budget,
        flash_budget_bytes=flash_budget,
    )

    c, h, w = input_shape
    # Track total flattened features for FC layers that follow flatten
    flattened_dim: Optional[int] = None
    flattened: bool = False
    total_params = 0
    total_macs = 0
    peak_act_bytes = 0

    layers: List[LayerStats] = []
    modules: List[nn.Module] = []

    # Input activation footprint
    input_act_bytes = c * h * w * byte_width
    peak_act_bytes = max(peak_act_bytes, input_act_bytes)

    for idx, cfg in enumerate(layer_configs):
        lt = cfg["type"]
        layer_name = f"{lt}_{idx}"

        if lt == "conv":
            out_c = cfg["out_channels"]
            k = cfg.get("kernel_size", 3)
            s = cfg.get("stride", 1)
            p = cfg.get("padding", 0)
            bias = cfg.get("bias", True)
            groups = cfg.get("groups", 1)

            if isinstance(k, int):
                k_h, k_w = k, k
            else:
                k_h, k_w = k

            h_out, w_out = _conv_output_size(h, w, k, s, p)

            params = _compute_conv2d_params(c, out_c, k_h, k_w, bias, groups)
            macs = _compute_conv2d_macs(c, out_c, k_h, k_w, h_out, w_out, groups)
            act_elems = out_c * h_out * w_out
            act_bytes = act_elems * byte_width

            modules.append(
                nn.Conv2d(
                    c,
                    out_c,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    bias=bias,
                    groups=groups,
                )
            )

            spatial_in_str = f"{c}x{h}x{w}"
            spatial_out_str = f"{out_c}x{h_out}x{w_out}"

            # Update spatial tracker
            c, h, w = out_c, h_out, w_out

        elif lt in ("maxpool", "avgpool"):
            k = cfg.get("kernel_size", 2)
            s = cfg.get("stride", k)
            p = cfg.get("padding", 0)

            h_out, w_out = _conv_output_size(h, w, k, s, p)
            params = 0
            macs = 0  # pooling MACs are negligible; we ignore them
            act_elems = c * h_out * w_out
            act_bytes = act_elems * byte_width

            pool_cls = nn.MaxPool2d if lt == "maxpool" else nn.AvgPool2d
            modules.append(pool_cls(kernel_size=k, stride=s, padding=p))

            spatial_in_str = f"{c}x{h}x{w}"
            spatial_out_str = f"{c}x{h_out}x{w_out}"

            # Update spatial tracker (channels unchanged)
            h, w = h_out, w_out

        elif lt == "relu":
            modules.append(nn.ReLU(inplace=True))
            params = 0
            macs = 0
            act_elems = c * h * w
            act_bytes = act_elems * byte_width

            spatial_in_str = f"{c}x{h}x{w}"
            spatial_out_str = spatial_in_str

        elif lt == "flatten":
            flattened_dim = c * h * w
            flattened = True
            modules.append(nn.Flatten())
            params = 0
            macs = 0
            act_elems = flattened_dim
            act_bytes = act_elems * byte_width

            spatial_in_str = f"{c}x{h}x{w}"
            spatial_out_str = str(flattened_dim)

        elif lt == "fc":
            if not flattened:
                # Auto-flatten
                flattened_dim = c * h * w
                flattened = True

            out_f = cfg["out_features"]
            bias = cfg.get("bias", True)

            params = _compute_linear_params(flattened_dim, out_f, bias)
            macs = _compute_linear_macs(flattened_dim, out_f)
            act_elems = out_f
            act_bytes = act_elems * byte_width

            modules.append(nn.Linear(flattened_dim, out_f, bias=bias))

            spatial_in_str = str(flattened_dim)
            spatial_out_str = str(out_f)

            # After an FC layer we are no longer in spatial mode
            flattened_dim = out_f
            c, h, w = out_f, 1, 1  # dummy spatial tracker

        else:
            raise ValueError(f"Unsupported layer type: {lt}")

        param_bytes = params * byte_width
        total_params += params
        total_macs += macs
        peak_act_bytes = max(peak_act_bytes, act_bytes)

        # ----- budget checks ------------------------------------------------
        layer_fail_reason = ""
        if act_bytes > sram_budget:
            layer_fail_reason = (
                f"Layer '{layer_name}' output activation ({act_bytes:,} bytes) "
                f"exceeds SRAM budget ({sram_budget:,} bytes)"
            )
        if param_bytes > flash_budget:
            layer_fail_reason += (
                f"{' | ' if layer_fail_reason else ''}"
                f"Layer '{layer_name}' parameters ({param_bytes:,} bytes) "
                f"exceed Flash budget ({flash_budget:,} bytes)"
            )

        if layer_fail_reason:
            report.rejection_reason = layer_fail_reason
            report.passed = False
            report.sram_ok = False if "SRAM" in layer_fail_reason else report.sram_ok
            report.flash_ok = False if "Flash" in layer_fail_reason else report.flash_ok
            # Still record the layer stats so the report is informative
            layers.append(
                LayerStats(
                    name=layer_name,
                    layer_type=lt,
                    spatial_in=spatial_in_str,
                    spatial_out=spatial_out_str,
                    params=params,
                    param_bytes=param_bytes,
                    macs=macs,
                    activation_elements=act_elems,
                    activation_bytes=act_bytes,
                )
            )
            break

        layers.append(
            LayerStats(
                name=layer_name,
                layer_type=lt,
                spatial_in=spatial_in_str,
                spatial_out=spatial_out_str,
                params=params,
                param_bytes=param_bytes,
                macs=macs,
                activation_elements=act_elems,
                activation_bytes=act_bytes,
            )
        )

    # ----- aggregate checks ------------------------------------------------
    total_param_bytes = total_params * byte_width

    if report.passed and total_param_bytes > flash_budget:
        report.passed = False
        report.flash_ok = False
        report.rejection_reason = (
            f"Total parameter storage ({total_param_bytes:,} bytes) "
            f"exceeds Flash budget ({flash_budget:,} bytes)"
        )

    if report.passed and peak_act_bytes > sram_budget:
        report.passed = False
        report.sram_ok = False
        report.rejection_reason = (
            f"Peak activation memory ({peak_act_bytes:,} bytes) "
            f"exceeds SRAM budget ({sram_budget:,} bytes)"
        )

    # ----- populate report -------------------------------------------------
    report.layers = layers
    report.total_params = total_params
    report.total_param_bytes = total_param_bytes
    report.total_macs = total_macs
    report.peak_activation_bytes = peak_act_bytes
    report.peak_activation_kb = peak_act_bytes / KB

    if not report.passed:
        return None, report

    model = TinyCNN(nn.ModuleList(modules), report)
    return model, report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report_table(report: MCUMemoryReport) -> str:
    """Produce a formatted MCU memory budget report as a multi-line string."""
    sep = "-" * 100
    lines: List[str] = [
        sep,
        f"  MCU Memory Budget Report  |  Model: {report.model_name}",
        sep,
        f"  SRAM Budget : {report.sram_budget_bytes:>10,} bytes  ({report.sram_budget_bytes / KB:8.1f} KB)  |  "
        f"Peak Activation : {report.peak_activation_bytes:>8,} bytes  ({report.peak_activation_kb:8.1f} KB)",
        f"  Flash Budget: {report.flash_budget_bytes:>10,} bytes  ({report.flash_budget_bytes / KB:8.1f} KB)  |  "
        f"Total Parameters: {report.total_param_bytes:>8,} bytes  ({report.total_param_bytes / KB:8.1f} KB)",
        sep,
        f"  {'RESULT':>8s}: {'PASS' if report.passed else 'FAIL'}",
    ]
    if not report.passed:
        lines.append(f"  Reason: {report.rejection_reason}")
    lines.append(sep)

    # Layer detail header
    lines.append(
        f"  {'Layer':<16s} {'Type':<8s} {'Spatial In':>12s} {'Spatial Out':>12s} "
        f"{'Params':>8s} {'Param(B)':>10s} {'MACs':>12s} {'Act(B)':>10s}"
    )
    lines.append("  " + "-" * 94)

    for ls in report.layers:
        lines.append(
            f"  {ls.name:<16s} {ls.layer_type:<8s} {ls.spatial_in:>12s} {ls.spatial_out:>12s} "
            f"{ls.params:>8,d} {ls.param_bytes:>10,d} {ls.macs:>12,d} {ls.activation_bytes:>10,d}"
        )

    lines.append("  " + "-" * 94)
    lines.append(
        f"  {'TOTAL':<16s} {'':8s} {'':>12s} {'':>12s} "
        f"{report.total_params:>8,d} {report.total_param_bytes:>10,d} "
        f"{report.total_macs:>12,d} {report.peak_activation_bytes:>10,d} (peak)"
    )
    lines.append(sep)

    # Budget utilization summary
    sram_pct = (report.peak_activation_bytes / report.sram_budget_bytes) * 100
    flash_pct = (report.total_param_bytes / report.flash_budget_bytes) * 100
    lines.append(f"  SRAM  utilization: {sram_pct:5.1f}%")
    lines.append(f"  Flash utilization: {flash_pct:5.1f}%")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick integrity check (sanity test)
# ---------------------------------------------------------------------------


def _run_sanity_check() -> None:
    """Verify that a known-tiny architecture produces correct analytical stats."""
    configs: List[dict] = [
        {"type": "conv", "out_channels": 2, "kernel_size": 3},
        {"type": "relu"},
        {"type": "flatten"},
        {"type": "fc", "out_features": 5},
    ]
    _, report = build_tiny_cnn(
        "SanityCheck",
        configs,
        input_shape=(1, 8, 8),
        sram_budget=1 * MB,
        flash_budget=10 * MB,
    )
    model = TinyCNN.__new__(TinyCNN)  # bypass __init__ for manual param count
    # Manually compute expected values
    with torch.no_grad():
        conv = nn.Conv2d(1, 2, 3)
        fc = nn.Linear(2 * 6 * 6, 5)
    expected_params = sum(
        p.numel() for p in list(conv.parameters()) + list(fc.parameters())
    )
    expected_conv_macs = 2 * 1 * 3 * 3 * 6 * 6  # 648
    expected_fc_macs = 72 * 5  # 360
    expected_macs = expected_conv_macs + expected_fc_macs
    expected_peak_act = max(
        1 * 8 * 8 * 4,  # input: 256
        2 * 6 * 6 * 4,  # conv output: 288
        2 * 6 * 6 * 4,  # relu (same): 288
        72 * 4,  # flatten: 288
        5 * 4,  # fc: 20
    )

    assert report.total_params == expected_params, (
        f"Params: {report.total_params} != {expected_params}"
    )
    assert report.total_macs == expected_macs, (
        f"MACs: {report.total_macs} != {expected_macs}"
    )
    assert report.peak_activation_bytes == expected_peak_act, (
        f"Peak act: {report.peak_activation_bytes} != {expected_peak_act}"
    )
    assert report.passed, "Sanity check model should pass budgets"
    print("[sanity] All assertions passed.")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate the TinyCNN budget simulator with valid and invalid architectures."""
    print("=" * 100)
    print("  MCUNet / TinyML Memory Budget Simulator  |  Lecture 10")
    print("=" * 100)
    print()

    # ------------------------------------------------------------------
    # 1. Sanity check
    # ------------------------------------------------------------------
    print("--- Sanity Check ---")
    _run_sanity_check()
    print()

    # ------------------------------------------------------------------
    # 2. Valid architecture: TinyNet (fits in 256 KB SRAM / 1 MB Flash)
    # ------------------------------------------------------------------
    tiny_net_configs: List[dict] = [
        # Input: 1x28x28  (MNIST-style grayscale)
        {
            "type": "conv",
            "out_channels": 8,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
        },
        {"type": "relu"},
        {"type": "maxpool", "kernel_size": 2, "stride": 2},
        {
            "type": "conv",
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
        },
        {"type": "relu"},
        {"type": "maxpool", "kernel_size": 2, "stride": 2},
        {"type": "flatten"},
        {"type": "fc", "out_features": 10},
    ]

    model_valid, report_valid = build_tiny_cnn(
        "TinyNet (valid)",
        tiny_net_configs,
        input_shape=(1, 28, 28),
    )

    print(format_report_table(report_valid))
    print()

    if model_valid is not None:
        # Quick forward-pass sanity check on CPU
        with torch.no_grad():
            dummy = torch.randn(1, 1, 28, 28)
            out = model_valid(dummy)
        print(f"  Forward pass OK.  Output shape: {tuple(out.shape)}")
        print(
            f"  Model size on disk (state_dict): "
            f"{sum(p.numel() for p in model_valid.parameters()) * BYTES_PER_FLOAT32:,} bytes"
        )
    print()
    print()

    # ------------------------------------------------------------------
    # 3. Invalid architecture: WideNet (exceeds SRAM budget)
    # ------------------------------------------------------------------
    wide_net_configs: List[dict] = [
        # Input: 1x28x28
        {
            "type": "conv",
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
        },
        {"type": "relu"},
        {
            "type": "conv",
            "out_channels": 128,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
        },
        # ^^^ 128x24x24 = 73,728 elements = 294,912 bytes > 256 KB → fails SRAM
        {"type": "relu"},
        {"type": "flatten"},
        {"type": "fc", "out_features": 10},
    ]

    model_wide, report_wide = build_tiny_cnn(
        "WideNet (invalid - SRAM)",
        wide_net_configs,
        input_shape=(1, 28, 28),
    )

    print(format_report_table(report_wide))
    print()

    assert model_wide is None, "WideNet should have been rejected"
    print("  (correctly refused to build WideNet – SRAM budget exceeded)")
    print()

    # ------------------------------------------------------------------
    # 4. Invalid architecture: FatFC (exceeds Flash budget via huge FC)
    # ------------------------------------------------------------------
    # Use a very large FC layer to blow the Flash budget.
    fat_fc_configs: List[dict] = [
        {"type": "conv", "out_channels": 4, "kernel_size": 3},
        {"type": "relu"},
        {"type": "flatten"},
        # 4x26x26 = 2704 → FC(2704, 1024)  2704*1024 ≈ 2.77 M params ≈ 11 MB → blows Flash
        {"type": "fc", "out_features": 1024},
    ]

    model_fat, report_fat = build_tiny_cnn(
        "FatFC (invalid - Flash)",
        fat_fc_configs,
        input_shape=(1, 28, 28),
    )

    print(format_report_table(report_fat))
    print()

    assert model_fat is None, "FatFC should have been rejected"
    print("  (correctly refused to build FatFC – Flash budget exceeded)")
    print()
    print("=" * 100)
    print("  Demo complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
