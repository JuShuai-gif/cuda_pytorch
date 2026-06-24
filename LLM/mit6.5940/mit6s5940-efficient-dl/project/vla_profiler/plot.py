#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Roofline + module-breakdown plotting for VLA profiles.

Uses the non-interactive Agg backend so it runs headless on a server / over
SSH. ``save_roofline_plot`` writes a single figure with two panels:

    left  - the classic roofline (log-log) with the model's operating point,
            the memory-bandwidth roof, the compute roof and the ridge point.
    right - per-module params% vs MACs% grouped bars.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .profiler import ProfileResult  # noqa: E402


def save_roofline_plot(result: ProfileResult, path: str, dpi: int = 130) -> str:
    roof = result.roofline
    cfg = result.config
    peak = cfg.gpu_tflops
    bw_flops = cfg.bandwidth_gbps * 1e9  # bytes/s

    fig, (ax_r, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

    # --- left: roofline ---
    ai = np.logspace(-1, 4, 400)  # 0.1 .. 1e4 FLOP/byte
    mem_roof = bw_flops * ai / 1e12  # TFLOPs
    attainable = np.minimum(mem_roof, peak)
    ax_r.loglog(ai, attainable, color="#222", lw=2.2, label="Roofline")
    ax_r.axhline(
        peak, color="#888", ls="--", lw=1, label=f"Compute roof {peak:.0f} TFLOPs"
    )
    ax_r.axvline(
        roof.ridge_point,
        color="#1f77b4",
        ls=":",
        lw=1.2,
        label=f"Ridge {roof.ridge_point:.0f} F/B",
    )

    color = "#d62728" if roof.regime == "memory-bound" else "#2ca02c"
    ax_r.scatter(
        [roof.arithmetic_intensity],
        [roof.attainable_tflops],
        s=130,
        color=color,
        zorder=5,
        edgecolors="k",
        label=f"Model ({roof.regime})",
    )
    ax_r.annotate(
        f"AI={roof.arithmetic_intensity:.1f}\n{roof.attainable_tflops:.0f} TFLOPs",
        (roof.arithmetic_intensity, roof.attainable_tflops),
        textcoords="offset points",
        xytext=(8, -28),
        fontsize=9,
    )
    ax_r.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax_r.set_ylabel("Attainable Performance (TFLOPs)")
    ax_r.set_title(f"Roofline - {cfg.gpu_name} {cfg.precision}")
    ax_r.grid(True, which="both", ls=":", alpha=0.4)
    ax_r.legend(fontsize=8, loc="lower right")

    # --- right: module breakdown ---
    cats = ["vision", "language", "fusion", "action"]
    labels = ["Vision", "Language", "Fusion", "Action"]
    pf = [result.params.category_fraction[c] * 100 for c in cats]
    mf = [result.macs.category_fraction[c] * 100 for c in cats]
    x = np.arange(len(cats))
    w = 0.38
    ax_b.bar(x - w / 2, pf, w, label="Params %", color="#4c72b0")
    ax_b.bar(x + w / 2, mf, w, label="MACs %", color="#dd8452")
    for i, (p, m) in enumerate(zip(pf, mf)):
        ax_b.text(i - w / 2, p + 1, f"{p:.0f}", ha="center", fontsize=8)
        ax_b.text(i + w / 2, m + 1, f"{m:.0f}", ha="center", fontsize=8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Percentage (%)")
    ax_b.set_title("Module Breakdown: Params vs MACs")
    ax_b.legend(fontsize=9)
    ax_b.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle(
        f"VLA Profile  |  {result.params.total / 1e6:.0f}M params  |  "
        f"{result.macs.total_macs / 1e9:.1f}G MACs  |  "
        f"efficiency "
        + (
            f"{result.latency.efficiency * 100:.1f}%"
            if result.latency.efficiency is not None
            else "n/a"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
