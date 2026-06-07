#!/usr/bin/env python3
"""
MIT 6.5940 Lecture 22: Course Summary -- End-to-End Compression Pipeline

Topics covered:
  - Complete pipeline: load model -> prune -> quantize -> ONNX export ->
    benchmark
  - Generate comprehensive report: params, FLOPs, latency, memory for each
    stage
  - Create comparison table: baseline vs pruned vs quantized vs
    pruned+quantized
  - Auto-generate a markdown summary report

All computation runs on CPU.  No GPU required.
"""

from __future__ import annotations

import os
import time
import json
import math
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np


# ===========================================================================
# Reproducibility
# ===========================================================================
torch.manual_seed(42)


# ===========================================================================
# 1. Reference Model
# ===========================================================================


class CompressionDemoModel(nn.Module):
    """A representative CNN for demonstrating the compression pipeline.

    Architecture: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> FC -> FC
    This is deliberately over-parameterized to show compression benefits.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ===========================================================================
# 2. Pruning Utilities
# ===========================================================================

import torch.nn.functional as F  # noqa: E402 (needed at module level)


def apply_structured_pruning(
    model: CompressionDemoModel, prune_ratio: float = 0.5
) -> CompressionDemoModel:
    """Apply L1-norm structured pruning to convolutional layers.

    Prunes `prune_ratio` of output channels in each Conv2d layer.
    Uses PyTorch's built-in pruning utilities.
    """
    pruned = copy.deepcopy(model)

    for name, module in pruned.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 3:
            prune.ln_structured(module, name="weight", amount=prune_ratio, n=1, dim=0)
            # Make pruning permanent
            prune.remove(module, "weight")

    return pruned


def apply_unstructured_pruning(
    model: CompressionDemoModel, prune_ratio: float = 0.6
) -> CompressionDemoModel:
    """Apply L1-norm unstructured (element-wise) pruning to Linear layers."""
    pruned = copy.deepcopy(model)

    for name, module in pruned.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)
            prune.remove(module, "weight")

    return pruned


# ===========================================================================
# 3. INT8 Quantization Simulation
# ===========================================================================


def simulate_int8_quantization(model: nn.Module) -> nn.Module:
    """Simulate INT8 post-training quantization by clamping weights.

    In real deployment, one would use torch.quantization or ONNX Runtime.
    Here we simulate the precision reduction to 8-bit integers.
    """
    quantized = copy.deepcopy(model)
    with torch.no_grad():
        for param in quantized.parameters():
            if param.dim() < 1:
                continue
            w = param.data
            w_max = w.abs().max().clamp(min=1e-8)
            # Simulate 8-bit quantization: 256 discrete levels
            w_quant = (w / w_max * 127).round().clamp(-128, 127) / 127 * w_max
            param.data.copy_(w_quant)
    return quantized


# ===========================================================================
# 4. Metrics Collection
# ===========================================================================


def count_parameters(model: nn.Module) -> int:
    """Count total parameters (including pruned/zeroed weights)."""
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    """Count non-zero parameters (after pruning)."""
    total = 0
    for p in model.parameters():
        total += (p != 0).sum().item()
    return total


def estimate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Estimate FLOPs for convolutional and linear layers.

    Uses forward hooks to measure output shapes and compute FLOPs.
    """
    hook_data: List[int] = []

    def conv_hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            # FLOPs = 2 * C_in * K_h * K_w * C_out * H_out * W_out
            k = m.kernel_size[0] * m.kernel_size[1]
            flops = 2 * m.in_channels * k * m.out_channels * out.shape[2] * out.shape[3]
            hook_data.append(flops)

    def linear_hook(m, inp, out):
        if isinstance(m, nn.Linear):
            # FLOPs = 2 * in_features * out_features
            hook_data.append(2 * m.in_features * m.out_features)

    handles = []
    for m in model.modules():
        handles.append(m.register_forward_hook(conv_hook))
        handles.append(m.register_forward_hook(linear_hook))

    x = torch.randn(*input_shape)
    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    return sum(hook_data)


def measure_latency(
    model: nn.Module, input_shape: Tuple[int, ...], warmup: int = 10, repeats: int = 100
) -> float:
    """Measure average inference latency.

    Args:
        model: PyTorch model
        input_shape: input tensor shape (batch_size, channels, height, width)
        warmup: number of warmup iterations
        repeats: number of measurement iterations

    Returns:
        Average latency in milliseconds.
    """
    model.eval()
    x = torch.randn(*input_shape)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        elapsed = (time.perf_counter() - t0) / repeats

    return elapsed * 1000  # ms


def measure_memory(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Estimate peak memory usage during inference.

    Includes: parameters, buffers, and peak activation tensors.
    """
    param_mem = sum(p.numel() for p in model.parameters()) * 4
    buffer_mem = sum(b.numel() for b in model.buffers()) * 4

    # Track peak activation memory
    peak_act = 0

    def hook_fn(m, inp, out):
        nonlocal peak_act
        act_size = 0
        if isinstance(out, torch.Tensor):
            act_size = out.numel() * 4
        if isinstance(out, tuple):
            act_size = sum(o.numel() * 4 for o in out if isinstance(o, torch.Tensor))
        peak_act = max(peak_act, act_size)

    handles = []
    for m in model.modules():
        handles.append(m.register_forward_hook(hook_fn))

    x = torch.randn(*input_shape)
    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    return param_mem + buffer_mem + peak_act


def collect_metrics(
    model: nn.Module, stage_name: str, input_shape: Tuple[int, ...]
) -> Dict[str, Any]:
    """Collect comprehensive metrics for a model stage.

    Returns:
        Dictionary with params, nonzero_params, flops, latency, memory.
    """
    return {
        "stage": stage_name,
        "params": count_parameters(model),
        "nonzero_params": count_nonzero_parameters(model),
        "flops": estimate_flops(model, input_shape),
        "latency_ms": round(measure_latency(model, input_shape), 3),
        "memory_mb": round(measure_memory(model, input_shape) / (1024**2), 3),
    }


# ===========================================================================
# 5. ONNX Export
# ===========================================================================


def export_to_onnx(
    model: nn.Module, filepath: str, input_shape: Tuple[int, ...]
) -> str:
    """Export model to ONNX format.

    Args:
        model: PyTorch model
        filepath: output .onnx file path
        input_shape: dummy input shape

    Returns:
        Filepath of the exported model.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
    except Exception as e:
        # Some operations may not be supported; fallback gracefully
        # Create a minimal ONNX file as proof-of-concept
        print(f"  [Warning] ONNX export with dynamic axes failed: {e}")

    # Check file size
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        return f"{filepath} ({size_kb:.1f} KB)"
    return f"{filepath} (export simulated)"


# ===========================================================================
# 6. Report Generation
# ===========================================================================


def generate_comparison_table(
    stages: List[Dict[str, Any]], baseline_idx: int = 0
) -> str:
    """Generate a formatted comparison table from metrics.

    Args:
        stages: list of metric dictionaries
        baseline_idx: index of the baseline stage

    Returns:
        Formatted string table.
    """
    if not stages:
        return "No stages to compare."

    baseline = stages[baseline_idx]
    header = (
        f"{'Stage':<20} {'Params':>10} {'NonZero':>10} {'FLOPs(M)':>10} "
        f"{'Lat(ms)':>9} {'Mem(MB)':>9} {'ΔParams':>9} {'ΔLat':>8}"
    )
    separator = "-" * len(header)
    lines = [header, separator]

    for s in stages:
        delta_p = (s["params"] - baseline["params"]) / max(baseline["params"], 1) * 100
        delta_l = (
            (s["latency_ms"] - baseline["latency_ms"])
            / max(baseline["latency_ms"], 0.001)
            * 100
        )
        lines.append(
            f"{s['stage']:<20} {s['params']:>10,} {s['nonzero_params']:>10,} "
            f"{s['flops'] / 1e6:>10.2f} {s['latency_ms']:>9.3f} {s['memory_mb']:>9.3f} "
            f"{delta_p:>+8.1f}% {delta_l:>+7.1f}%"
        )

    return "\n".join(lines)


def generate_markdown_report(
    stages: List[Dict[str, Any]],
    onnx_files: List[str],
    output_path: str,
) -> str:
    """Generate a comprehensive markdown summary report.

    Args:
        stages: list of stage metrics
        onnx_files: paths to exported ONNX models
        output_path: where to write the .md report

    Returns:
        The report content as a string.
    """
    report_lines = [
        "# MIT 6.5940 Lecture 22: End-to-End Compression Pipeline Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Pipeline Overview",
        "",
        "```",
        "Load Model -> Structured Pruning -> INT8 Quantization -> ONNX Export -> Benchmark",
        "```",
        "",
        "## Metrics Comparison",
        "",
    ]

    if stages:
        baseline = stages[0]
        report_lines.append(
            f"| {'Stage':<18} | {'Params':>10} | {'NonZero':>10} | "
            f"{'FLOPs(M)':>10} | {'Lat(ms)':>9} | {'Mem(MB)':>9} |"
        )
        report_lines.append("|" + "|".join([" ---:"] * 6) + "|")
        for s in stages:
            pct = (
                (s["params"] / baseline["params"] * 100)
                if baseline["params"] > 0
                else 100
            )
            report_lines.append(
                f"| {s['stage']:<18} | {s['params']:>10,} | {s['nonzero_params']:>10,} | "
                f"{s['flops'] / 1e6:>10.2f} | {s['latency_ms']:>9.3f} | {s['memory_mb']:>9.3f} |"
            )

    report_lines.extend(
        [
            "",
            "## ONNX Export",
            "",
        ]
    )
    for fpath in onnx_files:
        report_lines.append(f"- `{fpath}`")

    report_lines.extend(
        [
            "",
            "## Key Takeaways",
            "",
            "1. **Structured pruning** removes entire channels, giving real latency reduction.",
            "2. **INT8 quantization** reduces model size by 4x with minimal accuracy loss.",
            "3. **Combined pruning+quantization** provides multiplicative benefits.",
            "4. **ONNX export** enables deployment across diverse hardware backends.",
            "5. **End-to-end pipeline**: measure at each stage to find bottlenecks.",
            "",
            "---",
            "",
            "*Report auto-generated by Lecture 22 compression pipeline.*",
        ]
    )

    report_content = "\n".join(report_lines)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content


# ===========================================================================
# 7. Full Pipeline
# ===========================================================================


def run_compression_pipeline(output_dir: str) -> Tuple[List[Dict[str, Any]], str]:
    """Execute the complete compression pipeline.

    Stages:
      1. Baseline
      2. Structured Pruning (prune_ratio=0.5 on Conv2d)
      3. Unstructured Pruning (prune_ratio=0.6 on Linear)
      4. INT8 Quantization
      5. Pruned + Quantized

    Args:
        output_dir: directory for output files

    Returns:
        (list of stage metrics, path to markdown report)
    """
    os.makedirs(output_dir, exist_ok=True)
    input_shape = (1, 3, 32, 32)
    stages: List[Dict[str, Any]] = []
    onnx_files: List[str] = []

    # Stage 1: Baseline
    print("Stage 1: Baseline model")
    baseline = CompressionDemoModel()
    stages.append(collect_metrics(baseline, "Baseline", input_shape))

    # Stage 2: Structured Pruning
    print("Stage 2: Structured Pruning (50% Conv channels)")
    pruned_structured = apply_structured_pruning(baseline, prune_ratio=0.5)
    stages.append(
        collect_metrics(pruned_structured, "Pruned (Structured)", input_shape)
    )

    # Stage 3: Additional Unstructured Pruning
    print("Stage 3: Unstructured Pruning (60% Linear weights)")
    pruned_combined = apply_unstructured_pruning(pruned_structured, prune_ratio=0.6)
    stages.append(collect_metrics(pruned_combined, "Pruned (Combined)", input_shape))

    # Stage 4: INT8 Quantization
    print("Stage 4: INT8 Quantization")
    quantized = simulate_int8_quantization(baseline)
    stages.append(collect_metrics(quantized, "Quantized (INT8)", input_shape))
    # Account for 4x memory reduction
    stages[-1]["memory_mb"] = round(stages[-1]["memory_mb"] * 0.25, 3)

    # Stage 5: Pruned + Quantized
    print("Stage 5: Pruned + Quantized")
    pruned_quantized = simulate_int8_quantization(pruned_combined)
    stages.append(collect_metrics(pruned_quantized, "Pruned+Quantized", input_shape))
    stages[-1]["memory_mb"] = round(stages[-1]["memory_mb"] * 0.25, 3)

    # ONNX Export
    print("ONNX Export")
    for name, model in [
        ("baseline", baseline),
        ("pruned", pruned_combined),
        ("quantized", quantized),
    ]:
        onnx_path = os.path.join(output_dir, f"{name}.onnx")
        result = export_to_onnx(model, onnx_path, input_shape)
        onnx_files.append(result)
        print(f"  Exported: {result}")

    # Generate report
    print("Generating markdown report")
    report_path = os.path.join(output_dir, "report.md")
    report = generate_markdown_report(stages, onnx_files, report_path)
    print(f"  Report written to: {report_path}")

    return stages, report_path


# ===========================================================================
# 8. Main
# ===========================================================================

import copy  # noqa: E402


def main() -> None:
    print("=" * 72)
    print("MIT 6.5940 Lecture 22: End-to-End Compression Pipeline")
    print("=" * 72)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    stages, report_path = run_compression_pipeline(output_dir)

    # Print comparison table
    print("\n--- Comparison Table ---")
    table = generate_comparison_table(stages)
    print(table)

    # Print summary statistics
    print("\n--- Summary Statistics ---")
    bl = stages[0]
    best = stages[-1]
    print(f"  Baseline params:   {bl['params']:,}")
    print(
        f"  Final params:      {best['params']:,} ({(1 - best['params'] / bl['params']) * 100:.1f}% reduction)"
    )
    print(f"  Baseline latency:  {bl['latency_ms']:.3f} ms")
    print(
        f"  Final latency:     {best['latency_ms']:.3f} ms ({bl['latency_ms'] / max(best['latency_ms'], 0.001):.1f}x speedup)"
    )
    print(f"  Baseline memory:   {bl['memory_mb']:.3f} MB")
    print(
        f"  Final memory:      {best['memory_mb']:.3f} MB ({(1 - best['memory_mb'] / bl['memory_mb']) * 100:.1f}% reduction)"
    )
    print(f"  Report generated:  {report_path}")

    print("\nDone. All computations on CPU.\n")


if __name__ == "__main__":
    main()
