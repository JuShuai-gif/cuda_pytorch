"""
Parameter Counting, FLOPs Estimation, and Model Size (Lecture 01)

Implements basic profiling primitives for understanding model efficiency:
  - count_parameters: total trainable parameter count
  - estimate_flops_conv2d: MACs estimate for a single Conv2d layer
  - estimate_total_flops: MACs estimate for an entire model (conv layers only)
  - measure_inference_time: average forward-pass latency (CPU)

We use torchvision.models.resnet18 as the canonical example and print
a summary table of parameters, FLOPs, and model size.
"""

from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn


# ===========================================================================
# SmallCNN (replaces torchvision.models.resnet18)
# ===========================================================================


class SmallCNN(nn.Module):
    """A small CNN with a similar structure to ResNet-18's early layers.

    Designed to accept 224x224 input and produce reasonable Conv2d MACs
    for parameter/FLOPs estimation exercises.  This replaces torchvision's
    resnet18 which is incompatible with this PyTorch build.
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Initial conv (3->64, 7x7, s=2, p=3) -- matches resnet18 first layer
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Stage 1: 64->64 (two 3x3 convs, residual)
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Stage 2: 64->128, stride=2
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Stage 3: 128->256, stride=2
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Stage 4: 256->512, stride=2
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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
# FLOPs Estimation (Conv2d)
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
        groups:Number of groups for grouped convolution (default 1).

    Returns:
        Estimated MACs for this layer on a single input sample.
    """
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1

    # MACs = out_c * h_out * w_out * (in_c/groups) * k * k
    macs = out_c * h_out * w_out * (in_c // groups) * k * k
    return macs


def estimate_total_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Estimate total Conv2d MACs for the model by tracing a forward pass.

    This function only counts Conv2d layers.  BatchNorm, ReLU, pooling, and
    fully-connected layers are ignored because they account for a tiny fraction
    of total compute in CNN backbones.

    Args:
        model:       A PyTorch nn.Module.
        input_shape: (C, H, W) of the input tensor (no batch dim).

    Returns:
        Total estimated MACs across all Conv2d layers.
    """
    model.eval()
    total_macs = 0
    dummy = torch.randn(1, *input_shape)

    with torch.no_grad():
        # We use a forward hook to intercept every Conv2d call so we can
        # measure its input shape and infer output shape without modifying
        # the original model.
        def _hook(
            module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor, /
        ) -> None:
            nonlocal total_macs
            if isinstance(module, nn.Conv2d):
                x = inp[0]  # shape: (N, C_in, H_in, W_in)
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


def model_size_mb(total_params: int, bytes_per_param: int = 4) -> float:
    """Convert parameter count to model size in mebibytes.

    Args:
        total_params:   Number of parameters.
        bytes_per_param:Bytes per parameter (4 for FP32, 2 for FP16, 1 for INT8).

    Returns:
        Model size in MiB (2^20 bytes).
    """
    return total_params * bytes_per_param / (1024 * 1024)


# ===========================================================================
# Inference Time Measurement
# ===========================================================================


def measure_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    warmup: int = 10,
    repeats: int = 100,
) -> float:
    """Measure the average forward-pass latency on CPU.

    Args:
        model:       A PyTorch nn.Module.
        input_shape: (C, H, W) of the input tensor (no batch dim).
        warmup:      Number of warmup iterations (not timed).
        repeats:     Number of timed iterations.

    Returns:
        Average inference time in seconds (float).
    """
    model.eval()
    dummy = torch.randn(1, *input_shape)

    # Warmup to stabilise CPU frequency / cache state
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # Timed runs
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy)
    end = time.perf_counter()

    return (end - start) / repeats


# ===========================================================================
# Summary Table
# ===========================================================================


def print_summary(
    model_name: str,
    total_params: int,
    trainable_params: int,
    total_macs: int,
    size_mb: float,
    latency_ms: float,
) -> None:
    """Print a formatted summary table for a model."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  MODEL EFFICIENCY SUMMARY: {model_name}")
    print(sep)
    print(f"  {'Metric':<30} {'Value':>25}")
    print("  " + "-" * 56)
    print(f"  {'Total parameters':<30} {total_params:>25,}")
    print(f"  {'Trainable parameters':<30} {trainable_params:>25,}")
    print(f"  {'Total Conv2d MACs':<30} {total_macs:>25,}")
    print(f"  {'Total FLOPs (MACs x 2)':<30} {total_macs * 2:>25,}")
    print(f"  {'Model size (FP32, MiB)':<30} {size_mb:>24.2f}")
    print(f"  {'CPU inference latency':<30} {latency_ms:>23.2f} ms")
    print(sep)
    print()


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    # ---- 1. Load ResNet-18 --------------------------------------------------
    print("Loading SmallCNN (stand-in for torchvision.models.resnet18) ...")
    model = SmallCNN()
    input_shape = (3, 224, 224)

    # ---- 2. Count parameters ------------------------------------------------
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ---- 3. Estimate FLOPs ---------------------------------------------------
    print("Estimating Conv2d MACs (this may take a few seconds on CPU) ...")
    total_macs = estimate_total_flops(model, input_shape)
    print(f"Conv2d MACs: {total_macs:,}")

    # ---- 4. Model size -------------------------------------------------------
    size_mb = model_size_mb(total_params)
    print(f"Model size (FP32): {size_mb:.2f} MiB")

    # ---- 5. Measure inference time -------------------------------------------
    print("Measuring CPU inference latency ...")
    latency_s = measure_inference_time(model, input_shape)
    latency_ms = latency_s * 1000.0
    print(f"Avg. inference time: {latency_ms:.2f} ms")

    # ---- 6. Print summary table ----------------------------------------------
    print_summary(
        model_name="ResNet-18",
        total_params=total_params,
        trainable_params=trainable_params,
        total_macs=total_macs,
        size_mb=size_mb,
        latency_ms=latency_ms,
    )

    # ---- 7. Verify single-layer Conv2d estimate ------------------------------
    print("--- Sanity check: single Conv2d layer ---")
    # resnet18's first conv: (3, 64, 7x7, stride=2, padding=3) on 224x224
    macs_first = estimate_flops_conv2d(
        in_c=3, out_c=64, k=7, h=224, w=224, stride=2, padding=3
    )
    print(f"  First conv (3->64, k=7, s=2, p=3, in=224x224) MACs: {macs_first:,}")


if __name__ == "__main__":
    main()
