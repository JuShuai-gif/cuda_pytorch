"""
Parameter Counting, FLOPs Estimation, and Model Size (Lecture 01)
第 01 讲：参数统计、FLOPs 估算与模型大小

Implements basic profiling primitives for understanding model efficiency:
实现基础的模型效率分析原语，用于理解模型的效率指标：
  - count_parameters: total trainable parameter count
    count_parameters: 统计可训练参数总数
  - estimate_flops_conv2d: MACs estimate for a single Conv2d layer
    estimate_flops_conv2d: 估算单个 Conv2d 层的 MACs（乘加操作数）
  - estimate_total_flops: MACs estimate for an entire model (conv layers only)
    estimate_total_flops: 估算整个模型的 MACs（仅统计卷积层）
  - measure_inference_time: average forward-pass latency (CPU)
    measure_inference_time: 平均前向传播延迟（CPU）

We use torchvision.models.resnet18 as the canonical example and print
a summary table of parameters, FLOPs, and model size.
以 torchvision.models.resnet18 作为标准示例，打印包含参数量、FLOPs 和模型大小的汇总表。
"""

from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn


# ===========================================================================
# SmallCNN（替代 torchvision.models.resnet18）
# SmallCNN (replaces torchvision.models.resnet18)
# ===========================================================================


class SmallCNN(nn.Module):
    """A small CNN with a similar structure to ResNet-18's early layers.
    一个小型 CNN，其结构与 ResNet-18 的前几层相似。

    Designed to accept 224x224 input and produce reasonable Conv2d MACs
    for parameter/FLOPs estimation exercises.  This replaces torchvision's
    resnet18 which is incompatible with this PyTorch build.
    设计用于接受 224x224 输入并产生合理的 Conv2d MACs，供参数/FLOPs 估算练习使用。
    此模型替代了 torchvision 的 resnet18，因为后者与此 PyTorch 构建版本不兼容。
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Initial conv (3->64, 7x7, s=2, p=3) -- matches resnet18 first layer
            # 初始卷积层 (3->64, 7x7, s=2, p=3) -- 与 resnet18 第一层一致
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Stage 1: 64->64 (two 3x3 convs, residual)
            # 阶段 1：64->64（两个 3x3 卷积，残差块）
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Stage 2: 64->128, stride=2
            # 阶段 2：64->128，步长=2
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Stage 3: 128->256, stride=2
            # 阶段 3：128->256，步长=2
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Stage 4: 256->512, stride=2
            # 阶段 4：256->512，步长=2
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
# 参数统计
# Parameter Counting
# ===========================================================================


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params) for the given model.
    返回给定模型的 (参数总数, 可训练参数数)。

    Args:
        model: A PyTorch nn.Module.
               model: PyTorch nn.Module 实例。

    Returns:
        A tuple of (total_parameters, trainable_parameters).
        返回 (参数总数, 可训练参数数) 的元组。
    """
    # 统计所有参数的元素总数
    total = sum(p.numel() for p in model.parameters())
    # 统计需要梯度的（可训练）参数的元素总数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ===========================================================================
# FLOPs 估算（Conv2d）
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
    估算单个 Conv2d 层的 MACs（乘加操作数）。

    Assumes square kernels (k x k) and symmetric stride/padding.
    假设使用方形卷积核 (k x k) 且步长/padding 对称。

    Args:
        in_c:  Number of input channels.
               in_c: 输入通道数。
        out_c: Number of output channels.
               out_c: 输出通道数。
        k:     Kernel size (square, k x k).
               k: 卷积核尺寸（方形，k x k）。
        h:     Input feature-map height.
               h: 输入特征图高度。
        w:     Input feature-map width.
               w: 输入特征图宽度。
        stride:Stride (default 1).
               stride: 步长（默认 1）。
        padding:Padding (default 0).
                padding: 填充（默认 0）。
        groups:Number of groups for grouped convolution (default 1).
               groups: 分组卷积的组数（默认 1）。

    Returns:
        Estimated MACs for this layer on a single input sample.
        返回该层在单个输入样本上的估算 MACs。
    """
    # 计算输出特征图尺寸
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1

    # MACs = out_c * h_out * w_out * (in_c/groups) * k * k
    # MACs = 输出通道 × 输出高度 × 输出宽度 × (输入通道/组数) × 核高 × 核宽
    macs = out_c * h_out * w_out * (in_c // groups) * k * k
    return macs


def estimate_total_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Estimate total Conv2d MACs for the model by tracing a forward pass.
    通过跟踪一次前向传播来估算模型中所有 Conv2d 层的 MACs 总和。

    This function only counts Conv2d layers.  BatchNorm, ReLU, pooling, and
    fully-connected layers are ignored because they account for a tiny fraction
    of total compute in CNN backbones.
    此函数仅统计 Conv2d 层。BatchNorm、ReLU、池化层和全连接层被忽略，
    因为它们在 CNN 主干网络中仅占总计算量的极小部分。

    Args:
        model:       A PyTorch nn.Module.
                     model: PyTorch nn.Module 实例。
        input_shape: (C, H, W) of the input tensor (no batch dim).
                     input_shape: 输入张量的 (C, H, W)，不含批次维度。

    Returns:
        Total estimated MACs across all Conv2d layers.
        返回所有 Conv2d 层的估算 MACs 总和。
    """
    model.eval()
    total_macs = 0
    # 创建一个虚拟输入张量用于前向跟踪
    dummy = torch.randn(1, *input_shape)

    with torch.no_grad():
        # We use a forward hook to intercept every Conv2d call so we can
        # measure its input shape and infer output shape without modifying
        # the original model.
        # 使用前向钩子拦截每个 Conv2d 调用，从而在不修改原始模型的情况下
        # 测量其输入形状并推断输出形状。
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
        # 为所有 Conv2d 模块注册前向钩子
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                handles.append(m.register_forward_hook(_hook))

        # 执行一次前向传播以触发钩子
        _ = model(dummy)

        # 移除所有钩子，避免内存泄漏
        for h in handles:
            h.remove()

    return total_macs


# ===========================================================================
# 模型大小
# Model Size
# ===========================================================================


def model_size_mb(total_params: int, bytes_per_param: int = 4) -> float:
    """Convert parameter count to model size in mebibytes.
    将参数数量转换为以 MiB（2^20 字节）为单位的模型大小。

    Args:
        total_params:   Number of parameters.
                       total_params: 参数总数。
        bytes_per_param:Bytes per parameter (4 for FP32, 2 for FP16, 1 for INT8).
                        bytes_per_param: 每个参数的字节数（FP32=4, FP16=2, INT8=1）。

    Returns:
        Model size in MiB (2^20 bytes).
        返回以 MiB（2^20 字节）为单位的模型大小。
    """
    return total_params * bytes_per_param / (1024 * 1024)


# ===========================================================================
# 推理时间测量
# Inference Time Measurement
# ===========================================================================


def measure_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    warmup: int = 10,
    repeats: int = 100,
) -> float:
    """Measure the average forward-pass latency on CPU.
    测量 CPU 上的平均前向传播延迟。

    Args:
        model:       A PyTorch nn.Module.
                     model: PyTorch nn.Module 实例。
        input_shape: (C, H, W) of the input tensor (no batch dim).
                     input_shape: 输入张量的 (C, H, W)，不含批次维度。
        warmup:      Number of warmup iterations (not timed).
                     warmup: 预热迭代次数（不计时）。
        repeats:     Number of timed iterations.
                     repeats: 计时的迭代次数。

    Returns:
        Average inference time in seconds (float).
        返回平均推理时间（秒）。
    """
    model.eval()
    dummy = torch.randn(1, *input_shape)

    # Warmup to stabilise CPU frequency / cache state
    # 预热阶段：稳定 CPU 频率和缓存状态
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # Timed runs
    # 计时运行阶段
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy)
    end = time.perf_counter()

    return (end - start) / repeats


# ===========================================================================
# 汇总表格
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
    """Print a formatted summary table for a model.
    打印模型的格式化汇总表格。
    """
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
# 主函数
# Main
# ===========================================================================


def main() -> None:
    # ---- 1. Load ResNet-18 --------------------------------------------------
    # ---- 1. 加载 ResNet-18 --------------------------------------------------
    print("Loading SmallCNN (stand-in for torchvision.models.resnet18) ...")
    model = SmallCNN()
    input_shape = (3, 224, 224)

    # ---- 2. Count parameters ------------------------------------------------
    # ---- 2. 统计参数 --------------------------------------------------------
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ---- 3. Estimate FLOPs ---------------------------------------------------
    # ---- 3. 估算 FLOPs -------------------------------------------------------
    print("Estimating Conv2d MACs (this may take a few seconds on CPU) ...")
    total_macs = estimate_total_flops(model, input_shape)
    print(f"Conv2d MACs: {total_macs:,}")

    # ---- 4. Model size -------------------------------------------------------
    # ---- 4. 模型大小 ---------------------------------------------------------
    size_mb = model_size_mb(total_params)
    print(f"Model size (FP32): {size_mb:.2f} MiB")

    # ---- 5. Measure inference time -------------------------------------------
    # ---- 5. 测量推理时间 -----------------------------------------------------
    print("Measuring CPU inference latency ...")
    latency_s = measure_inference_time(model, input_shape)
    latency_ms = latency_s * 1000.0
    print(f"Avg. inference time: {latency_ms:.2f} ms")

    # ---- 6. Print summary table ----------------------------------------------
    # ---- 6. 打印汇总表格 ----------------------------------------------------
    print_summary(
        model_name="ResNet-18",
        total_params=total_params,
        trainable_params=trainable_params,
        total_macs=total_macs,
        size_mb=size_mb,
        latency_ms=latency_ms,
    )

    # ---- 7. Verify single-layer Conv2d estimate ------------------------------
    # ---- 7. 验证单层 Conv2d 估算 --------------------------------------------
    print("--- Sanity check: single Conv2d layer ---")
    # resnet18's first conv: (3, 64, 7x7, stride=2, padding=3) on 224x224
    # resnet18 的第一个卷积层：(3, 64, 7x7, stride=2, padding=3) 输入 224x224
    macs_first = estimate_flops_conv2d(
        in_c=3, out_c=64, k=7, h=224, w=224, stride=2, padding=3
    )
    print(f"  First conv (3->64, k=7, s=2, p=3, in=224x224) MACs: {macs_first:,}")


if __name__ == "__main__":
    main()
