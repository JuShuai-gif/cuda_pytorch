"""
Efficiency Metrics Deep Dive (Lecture 02)
===========================================
Benchmarks latency, throughput, parameter count, and FLOPs for three CNN
architectures on CPU:

  - CustomCNN:  a hand-crafted 6-layer convnet (~0.4M params)
  - ResNet18:   torchvision.models.resnet18   (~11.7M params)
  - MobileNetV2: torchvision.models.mobilenet_v2 (~3.5M params)

Key concepts:
  - Latency  = time to process a single sample (batch_size=1)
  - Throughput = samples per second at different batch sizes
  - MACs/FLOPs estimation via forward hooks (Conv2d only)
  - Parameter counting and model size computation

All computations are CPU-only; standard library + PyTorch + torchvision.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision import models

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SHAPE: Tuple[int, int, int] = (3, 224, 224)  # (C, H, W)
WARMUP_ITERS: int = 5
TIMED_ITERS: int = 100
BATCH_SIZES: List[int] = [1, 4, 16, 32]
BYTES_PER_FP32: int = 4
MIB: int = 1024 * 1024


# ===========================================================================
# Custom CNN
# ===========================================================================


class CustomCNN(nn.Module):
    """A hand-crafted 6-layer convolutional network for efficiency benchmarking.

    Architecture:
        Conv2d(3, 16, 3, padding=1) -> BN -> ReLU
        Conv2d(16, 32, 3, stride=2, padding=1) -> BN -> ReLU
        Conv2d(32, 64, 3, padding=1) -> BN -> ReLU
        Conv2d(64, 128, 3, stride=2, padding=1) -> BN -> ReLU
        Conv2d(128, 256, 3, padding=1) -> BN -> ReLU
        AdaptiveAvgPool2d(1) -> Flatten -> Linear(256, 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ===========================================================================
# Parameter Counting
# ===========================================================================


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params) for the given model.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        A tuple of (total_parameters, trainable_parameters).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ===========================================================================
# FLOPs Estimation (Conv2d hooks)
# ===========================================================================


def estimate_flops_conv2d(
    in_c: int,
    out_c: int,
    k: int,
    h: int,
    w: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
) -> int:
    """Estimate MACs (multiply-accumulate operations) for one Conv2d layer.

    Assumes square kernels (k x k) and symmetric stride/padding.

    Args:
        in_c:  Number of input channels.
        out_c: Number of output channels.
        k:     Kernel size (square, k x k).
        h:     Input feature-map height.
        w:     Input feature-map width.
        stride:Stride (default 1).
        padding:Padding (default 0).
        groups:Number of groups (default 1).

    Returns:
        Estimated MACs for this layer on a single input sample.
    """
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    macs = out_c * h_out * w_out * (in_c // groups) * k * k
    return macs


def estimate_total_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Estimate total Conv2d MACs for the model by tracing a forward pass.

    Only counts Conv2d layers.  BatchNorm, ReLU, pooling, and Linear layers
    are ignored because they account for a tiny fraction of total compute
    in CNN backbones.

    Args:
        model:       A PyTorch nn.Module.
        input_shape: (C, H, W) of a single input sample (no batch dim).

    Returns:
        Total estimated MACs across all Conv2d layers for one sample.
    """
    model.eval()
    total_macs = 0
    dummy = torch.randn(1, *input_shape)

    with torch.no_grad():

        def _hook(
            module: nn.Module,
            inp: Tuple[torch.Tensor, ...],
            out: torch.Tensor,
            /,
        ) -> None:
            nonlocal total_macs
            if isinstance(module, nn.Conv2d):
                x = inp[0]
                in_c = x.shape[1]
                h_in = x.shape[2]
                w_in = x.shape[3]
                total_macs += estimate_flops_conv2d(
                    in_c=in_c,
                    out_c=module.out_channels,
                    k=module.kernel_size[0],
                    h=h_in,
                    w=w_in,
                    stride=module.stride[0],
                    padding=module.padding[0],
                    groups=module.groups,
                )

        handles = []
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                handles.append(m.register_forward_hook(_hook))

        _ = model(dummy)

        for h in handles:
            h.remove()

    return total_macs


# ===========================================================================
# Model Size
# ===========================================================================


def model_size_mb(total_params: int, bytes_per_param: int = BYTES_PER_FP32) -> float:
    """Convert parameter count to model size in mebibytes.

    Args:
        total_params:   Number of parameters.
        bytes_per_param:Bytes per parameter (4 for FP32, 2 for FP16, 1 for INT8).

    Returns:
        Model size in MiB (2^20 bytes).
    """
    return total_params * bytes_per_param / MIB


# ===========================================================================
# Latency Measurement
# ===========================================================================


def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    warmup: int = WARMUP_ITERS,
    repeats: int = TIMED_ITERS,
) -> float:
    """Measure average single-sample inference latency on CPU.

    Uses batch_size=1.  Returns latency in seconds.

    Args:
        model:       A PyTorch nn.Module.
        input_shape: (C, H, W) of a single input sample (no batch dim).
        warmup:      Number of warmup iterations (not timed).
        repeats:     Number of timed iterations.

    Returns:
        Average forward-pass latency in seconds (float).
    """
    model.eval()
    dummy = torch.randn(1, *input_shape)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy)
    end = time.perf_counter()

    return (end - start) / repeats


# ===========================================================================
# Throughput Measurement
# ===========================================================================


def measure_throughput(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int,
    warmup: int = WARMUP_ITERS,
    repeats: int = TIMED_ITERS,
) -> float:
    """Measure throughput (samples/second) at a given batch size on CPU.

    Args:
        model:       A PyTorch nn.Module.
        input_shape: (C, H, W) of a single input sample (no batch dim).
        batch_size:  Number of samples per forward pass.
        warmup:      Number of warmup iterations.
        repeats:     Number of timed iterations.

    Returns:
        Throughput in samples per second.
    """
    model.eval()
    shape = (batch_size, *input_shape)
    dummy = torch.randn(*shape)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy)
    end = time.perf_counter()

    total_time = end - start
    total_samples = batch_size * repeats
    return total_samples / total_time


# ===========================================================================
# Model Builder
# ===========================================================================


def build_models() -> List[Tuple[str, nn.Module]]:
    """Build and return all three benchmark models.

    Returns:
        List of (model_name, model_instance) tuples.
    """
    models_list: List[Tuple[str, nn.Module]] = []

    # CustomCNN
    custom = CustomCNN(num_classes=10)
    models_list.append(("CustomCNN", custom))

    # ResNet18
    rn18 = models.resnet18(weights=None)
    models_list.append(("ResNet18", rn18))

    # MobileNetV2
    mbv2 = models.mobilenet_v2(weights=None)
    models_list.append(("MobileNetV2", mbv2))

    return models_list


# ===========================================================================
# Report Generation
# ===========================================================================


def generate_markdown_report(
    results: List[Dict[str, object]],
    batch_sizes: List[int],
) -> str:
    """Generate a formatted Markdown report table from benchmark results.

    Args:
        results:    List of dicts, each containing model_name, total_params,
                    trainable_params, total_macs, total_flops, size_mb,
                    latency_ms, and throughput keys (throughput is a dict
                    mapping batch_size -> samples/sec).
        batch_sizes: List of batch sizes used for throughput measurement.

    Returns:
        A multi-line Markdown string suitable for printing or saving.
    """
    lines: List[str] = []
    sep = "=" * 80

    lines.append("")
    lines.append(sep)
    lines.append("  LECTURE 02: Efficiency Metrics Deep Dive  --  Benchmark Report")
    lines.append(sep)
    lines.append("")

    # ----  Model Overview  ------------------------------------------------
    lines.append("## Model Overview")
    lines.append("")
    lines.append(
        f"| {'Model':<14s} | {'Params':>12s} | {'Trainable':>12s} | "
        f"{'MACs':>14s} | {'FLOPs':>14s} | {'Size (MiB)':>11s} |"
    )
    lines.append(
        f"| {'-' * 14:<14s} | {'-' * 12:>12s} | {'-' * 12:>12s} | "
        f"{'-' * 14:>14s} | {'-' * 14:>14s} | {'-' * 11:>11s} |"
    )

    for r in results:
        lines.append(
            f"| {str(r['model_name']):<14s} "
            f"| {int(r['total_params']):>12,d} "
            f"| {int(r['trainable_params']):>12,d} "
            f"| {int(r['total_macs']):>14,d} "
            f"| {int(r['total_flops']):>14,d} "
            f"| {float(r['size_mb']):>10.2f}  |"
        )

    lines.append("")

    # ----  Latency  -------------------------------------------------------
    lines.append("## Latency (batch_size=1, CPU)")
    lines.append("")
    lines.append(f"| {'Model':<14s} | {'Latency (ms)':>14s} |")
    lines.append(f"| {'-' * 14:<14s} | {'-' * 14:>14s} |")
    for r in results:
        lines.append(
            f"| {str(r['model_name']):<14s} | {float(r['latency_ms']):>13.2f}  |"
        )
    lines.append("")

    # ----  Throughput  ----------------------------------------------------
    lines.append("## Throughput (samples/sec) vs Batch Size (CPU)")
    lines.append("")
    header = (
        f"| {'Model':<14s} | "
        + " | ".join(f"b={bs:>2d}".ljust(14) for bs in batch_sizes)
        + " |"
    )
    lines.append(header)
    sep_row = f"| {'-' * 14:<14s} | " + " | ".join("-" * 14 for _ in batch_sizes) + " |"
    lines.append(sep_row)
    for r in results:
        tp = r["throughput"]  # type: Dict[int, float]
        cells = " | ".join(f"{tp[bs]:>13.1f} " for bs in batch_sizes)
        lines.append(f"| {str(r['model_name']):<14s} | {cells} |")
    lines.append("")

    # ----  Efficiency Ratios  -----------------------------------------------
    lines.append("## Efficiency Ratios (batch_size=1)")
    lines.append("")
    lines.append(
        f"| {'Model':<14s} | {'MACs/Param':>12s} | "
        f"{'MACs/ms':>10s} | {'Params/MiB':>11s} |"
    )
    lines.append(
        f"| {'-' * 14:<14s} | {'-' * 12:>12s} | {'-' * 10:>10s} | {'-' * 11:>11s} |"
    )
    for r in results:
        macs_param = int(r["total_macs"]) / max(int(r["total_params"]), 1)
        macs_ms = int(r["total_macs"]) / max(float(r["latency_ms"]), 0.001)
        params_mib = int(r["total_params"]) / max(float(r["size_mb"]), 0.001)
        lines.append(
            f"| {str(r['model_name']):<14s} "
            f"| {macs_param:>12.1f} "
            f"| {macs_ms:>10.3e} "
            f"| {params_mib:>10.1f}  |"
        )
    lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    """Run the full efficiency benchmark suite and print a Markdown report."""
    print("=" * 80)
    print("  LECTURE 02: Efficiency Metrics Deep Dive")
    print("=" * 80)
    print()

    results: List[Dict[str, object]] = []

    for model_name, model in build_models():
        print(f"[{model_name}] Running benchmarks ...")
        model.eval()

        # ---- 1. Parameter Counting ----------------------------------------
        total_params, trainable_params = count_parameters(model)
        print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

        # ---- 2. FLOPs Estimation ------------------------------------------
        total_macs = estimate_total_flops(model, INPUT_SHAPE)
        total_flops = total_macs * 2
        print(f"  Conv2d MACs: {total_macs:,}  (FLOPs: {total_flops:,})")

        # ---- 3. Model Size ------------------------------------------------
        size_mb = model_size_mb(total_params)
        print(f"  Model size (FP32): {size_mb:.2f} MiB")

        # ---- 4. Latency (batch=1) -----------------------------------------
        print("  Measuring latency (batch=1) ...")
        latency_s = measure_latency(model, INPUT_SHAPE)
        latency_ms = latency_s * 1000.0
        print(f"  Latency: {latency_ms:.2f} ms")

        # ---- 5. Throughput (multiple batch sizes) -------------------------
        throughput: Dict[int, float] = {}
        for bs in BATCH_SIZES:
            tp = measure_throughput(model, INPUT_SHAPE, batch_size=bs)
            throughput[bs] = tp
            print(f"  Throughput (batch={bs:>2d}): {tp:>10.1f} samples/s")

        results.append(
            {
                "model_name": model_name,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "total_macs": total_macs,
                "total_flops": total_flops,
                "size_mb": size_mb,
                "latency_ms": latency_ms,
                "throughput": throughput,
            }
        )
        print()

    # ---- 6. Generate and print Markdown report ----------------------------
    report = generate_markdown_report(results, BATCH_SIZES)
    print(report)

    print("Benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
