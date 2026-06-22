#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool Comparison: Multi-Library FLOPs & Metrics Comparison

Demonstrates how different FLOPs counting libraries produce different
results on the same model, and how to compare them.

Usage:
    python src/model_compression/tool_comparison.py

Output:
    reports/tool_comparison_report.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model_compression.models import SmallCNN, TransformerAttentionBlock
from src.model_compression.metrics import (
    estimate_flops_all,
    measure_parameters,
    measure_model_size_disk,
    measure_inference_latency,
    measure_latency_benchmark,
    measure_memory_usage,
    measure_throughput,
    torchinfo_summary,
    get_gpu_info_pynvml,
    measure_gpu_power_during_inference,
    MetricsLogger,
    _has_fvcore,
    _has_thop,
    _has_torchprofile,
    _has_calflops,
    _has_torchinfo,
    _has_pynvml,
    _has_wandb,
    _has_psutil,
)

REPORTS_DIR = _PROJECT_ROOT / "reports"


def generate_report(model: nn.Module, model_name: str, device: torch.device) -> str:
    """Generate a comprehensive comparison report for one model."""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append(f"# Tool Comparison Report: {model_name}")
    lines.append("")
    lines.append(f"**Generated**: {now}  |  **Device**: {device}  |  **PyTorch**: {torch.__version__}")
    lines.append("")

    # ---- Library availability ----
    lines.append("## Library Availability")
    lines.append("")
    lines.append("| Library | Installed | Purpose |")
    lines.append("|---------|-----------|--------|")
    libs = [
        ("fvcore", _has_fvcore(), "FLOPs analysis (Meta)"),
        ("thop", _has_thop(), "FLOPs counter (lightweight)"),
        ("torchprofile", _has_torchprofile(), "MACs profiler"),
        ("calflops", _has_calflops(), "HF-aware FLOPs counter"),
        ("torchinfo", _has_torchinfo(), "Model summary"),
        ("psutil", _has_psutil(), "CPU memory measurement"),
        ("pynvml", _has_pynvml(), "GPU monitoring (nvidia-smi API)"),
        ("wandb", _has_wandb(), "Experiment tracking"),
    ]
    for name, installed, purpose in libs:
        status = "Yes" if installed else "No"
        lines.append(f"| {name} | {status} | {purpose} |")
    lines.append("")

    # ---- Model parameters ----
    lines.append("## Model Parameters")
    lines.append("")
    params = measure_parameters(model)
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in params.items():
        if isinstance(v, (int, float)):
            lines.append(f"| {k} | {v} |")
    lines.append("")

    # ---- torchinfo ----
    if _has_torchinfo():
        lines.append("## torchinfo Summary")
        lines.append("")
        result = torchinfo_summary(model, (3, 32, 32) if "CNN" in model_name else (64, 128))
        if result:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in result.items():
                if isinstance(v, (int, float)):
                    lines.append(f"| {k} | {v} |")
            lines.append("")
        else:
            lines.append("> torchinfo failed to generate summary.")
            lines.append("")
    else:
        lines.append("> torchinfo not installed. `pip install torchinfo`")
        lines.append("")

    # ---- FLOPs via ALL libraries ----
    lines.append("## FLOPs Comparison (All Libraries)")
    lines.append("")
    if "CNN" in model_name:
        input_shape = (3, 32, 32)
    else:
        input_shape = (64, 128)  # seq_len, hidden_size

    flops_all = estimate_flops_all(model, input_shape)

    # Summary table
    lines.append("| Method | Total FLOPs | MFLOPs | GFLOPs |")
    lines.append("|--------|------------|--------|--------|")
    flops_values: list[float] = []
    for method_name in ["manual", "fvcore", "thop", "torchprofile", "calflops"]:
        r = flops_all.get(method_name)
        if r is None:
            lines.append(f"| {method_name} | N/A (not installed) | - | - |")
        else:
            total = r["total_flops"]
            mflops = r.get("total_mflops", total / 1e6)
            gflops = r.get("total_gflops", total / 1e9)
            lines.append(f"| {method_name} | {total:,} | {mflops} | {gflops} |")
            flops_values.append(total)
    lines.append("")

    # Summary statistics
    if len(flops_values) >= 2:
        import numpy as np
        arr = np.array(flops_values)
        lines.append("### Cross-Library FLOPs Statistics")
        lines.append("")
        lines.append(f"- **Num methods**: {len(flops_values)}")
        lines.append(f"- **Mean**: {arr.mean():,.0f} FLOPs ({arr.mean()/1e6:.3f} MFLOPs)")
        lines.append(f"- **Std**: {arr.std():,.0f} FLOPs")
        lines.append(f"- **Min**: {arr.min():,} FLOPs")
        lines.append(f"- **Max**: {arr.max():,} FLOPs")
        spread = (arr.max() - arr.min()) / arr.mean() * 100 if arr.mean() > 0 else 0
        lines.append(f"- **Spread**: {spread:.2f}% (差异 < 10% 属正常范围)")
        lines.append("")

    # ---- Latency measurement (two methods) ----
    lines.append("## Latency Measurement (Two Methods)")
    lines.append("")

    def _input_fn():
        if "CNN" in model_name:
            return torch.randn(1, 3, 32, 32, device=device)
        else:
            return torch.randn(1, 64, 128, device=device)

    lat_perf = measure_inference_latency(model, _input_fn, warmup_runs=10, measure_runs=50, device=device)
    lat_bench = measure_latency_benchmark(model, _input_fn, num_runs=100, device=device)

    lines.append("| Method | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |")
    lines.append("|--------|-----------|-------------|----------|----------|")
    lines.append(f"| time.perf_counter | {lat_perf['mean_ms']} | {lat_perf['median_ms']} | {lat_perf['p95_ms']} | {lat_perf['p99_ms']} |")
    lines.append(f"| torch.utils.benchmark | {lat_bench.get('mean_ms', 'N/A')} | {lat_bench.get('median_ms', 'N/A')} | {lat_bench.get('p95_ms', 'N/A')} | {lat_bench.get('p99_ms', 'N/A')} |")
    lines.append("")

    # ---- Memory ----
    lines.append("## Memory Usage")
    lines.append("")
    mem = measure_memory_usage(model, _input_fn, device)
    lines.append("| Metric | Value | Method |")
    lines.append("|--------|-------|--------|")
    for k, v in mem.items():
        if "method" in k:
            continue
        method_key = k + "_method"
        method = mem.get(method_key, "N/A")
        if isinstance(v, (int, float)):
            lines.append(f"| {k} | {v} | {method} |")
    lines.append("")

    # ---- GPU monitoring ----
    if device.type == "cuda":
        lines.append("## GPU Monitoring (pynvml)")
        lines.append("")
        gpu_info = get_gpu_info_pynvml(0)
        if gpu_info:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in gpu_info.items():
                lines.append(f"| {k} | {v} |")

            # Power measurement
            power = measure_gpu_power_during_inference(model, _input_fn, duration_seconds=3.0, device=device)
            if power:
                lines.append(f"| avg_power_w | {power['avg_power_w']} |")
                lines.append(f"| peak_power_w | {power['peak_power_w']} |")
            lines.append("")
        else:
            lines.append("> pynvml not available or GPU monitoring failed.")
            lines.append("")

    # ---- Throughput ----
    lines.append("## Throughput (batch_size=8)")
    lines.append("")

    def _batch_input_fn():
        if "CNN" in model_name:
            return torch.randn(8, 3, 32, 32, device=device)
        else:
            return torch.randn(8, 64, 128, device=device)

    tp = measure_throughput(model, _batch_input_fn, batch_size=8, num_batches=30, device=device)
    lines.append(f"- Samples/second: {tp['samples_per_second']}")
    lines.append(f"- ms/sample: {tp['ms_per_sample']}")
    lines.append(f"- ms/batch: {tp['ms_per_batch']}")
    lines.append("")

    # ---- Disk size ----
    lines.append("## Disk Size")
    lines.append("")
    size = measure_model_size_disk(model)
    lines.append(f"- Saved state_dict: {size['disk_size_mb']} MB ({size['disk_size_kb']} KB)")
    lines.append(f"- FP32 theoretical: {size['fp32_theoretical_mb']} MB")
    lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by tool_comparison.py at {now}*")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ---- Model 1: SmallCNN ----
    print("=" * 60)
    print("Benchmarking SmallCNN...")
    print("=" * 60)
    cnn = SmallCNN().to(device)
    cnn.eval()
    report_cnn = generate_report(cnn, "SmallCNN", device)

    # ---- Model 2: Transformer ----
    print("=" * 60)
    print("Benchmarking TransformerAttentionBlock...")
    print("=" * 60)
    tf = TransformerAttentionBlock(hidden_size=128).to(device)
    tf.eval()
    report_tf = generate_report(tf, "TransformerAttentionBlock", device)

    # ---- Write combined report ----
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = REPORTS_DIR / "tool_comparison_report.md"
    combined = report_cnn + "\n\n---\n\n" + report_tf
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"\nReport saved to: {report_path}")

    # ---- Test MetricsLogger ----
    print("\n" + "=" * 60)
    print("Testing MetricsLogger (TensorBoard)...")
    print("=" * 60)
    tb_dir = str(REPORTS_DIR / "tb_logs")
    logger = MetricsLogger(tb_dir=tb_dir)
    logger.log_scalar("test/latency_ms", 1.23, step=0)
    logger.log_metrics({"params_M": 0.621, "flops_M": 50.0}, step=0)
    logger.close()
    print(f"TensorBoard logs saved to: {tb_dir}")
    print("View with: tensorboard --logdir=" + tb_dir)


if __name__ == "__main__":
    main()
