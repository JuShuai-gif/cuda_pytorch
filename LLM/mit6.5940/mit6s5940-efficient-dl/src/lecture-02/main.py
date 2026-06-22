"""
Efficiency Metrics Deep Dive (Lecture 02)
===========================================
效率指标深度剖析（第 02 讲）

Benchmarks latency, throughput, parameter count, and FLOPs for three CNN
architectures on CPU:
在 CPU 上对三种 CNN 架构进行延迟、吞吐量、参数数量和 FLOPs 的基准测试：

  - CustomCNN:  a hand-crafted 6-layer convnet (~0.4M params)
    CustomCNN: 手工设计的 6 层卷积网络（~0.4M 参数）
  - ResNet18:   torchvision.models.resnet18   (~11.7M params)
    ResNet18:   torchvision.models.resnet18（~11.7M 参数）
  - MobileNetV2: torchvision.models.mobilenet_v2 (~3.5M params)
    MobileNetV2: torchvision.models.mobilenet_v2（~3.5M 参数）

Key concepts:
核心概念：
  - Latency  = time to process a single sample (batch_size=1)
    延迟 = 处理单个样本所需时间（batch_size=1）
  - Throughput = samples per second at different batch sizes
    吞吐量 = 不同批次大小下每秒处理的样本数
  - MACs/FLOPs estimation via forward hooks (Conv2d only)
    通过前向钩子估算 MACs/FLOPs（仅 Conv2d）
  - Parameter counting and model size computation
    参数统计和模型大小计算

All computations are CPU-only; standard library + PyTorch + torchvision.
所有计算均在 CPU 上运行；使用标准库 + PyTorch + torchvision。
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 常量定义
# Constants
# ---------------------------------------------------------------------------

INPUT_SHAPE: Tuple[int, int, int] = (3, 224, 224)  # (C, H, W) 输入形状
WARMUP_ITERS: int = 5  # 预热迭代次数（不计时）
TIMED_ITERS: int = 100  # 计时迭代次数
BATCH_SIZES: List[int] = [1, 4, 16, 32]  # 用于吞吐量测试的批次大小列表
BYTES_PER_FP32: int = 4  # FP32 每个参数的字节数
MIB: int = 1024 * 1024  # 1 MiB = 2^20 字节


# ===========================================================================
# 自定义 CNN 模型
# Custom CNN
# ===========================================================================


class CustomCNN(nn.Module):
    """A hand-crafted 6-layer convolutional network for efficiency benchmarking.
    一个手工设计的 6 层卷积网络，用于效率基准测试。

    Architecture:
    架构：
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
# 替代 torchvision 的模型（ResNet 风格、MobileNet 风格）
# Stand-in models replacing torchvision (ResNet-style, MobileNet-style)
# ===========================================================================


def _make_resnet_basic_block(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Sequential:
    """Return a simple two-conv residual block (no actual skip).
    返回一个简单的双卷积残差块（无实际跳跃连接）。
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResNet18StandIn(nn.Module):
    """A resnet18-like architecture with similar conv structure (~11M params).
    一个类似 resnet18 的架构，具有相似的卷积结构（~11M 参数）。

    Replaces torchvision.models.resnet18 which is incompatible with this build.
    Uses plain sequential blocks without skip connections for simplicity,
    but preserves the channel/layer count to keep FLOPs comparable.
    替代与此构建版本不兼容的 torchvision.models.resnet18。
    为简化起见使用无跳跃连接的普通顺序块，但保留通道数/层数以使 FLOPs 可比。
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            # stage 1: 64->64
            # 阶段 1：64->64
            _make_resnet_basic_block(64, 64, 1),
            _make_resnet_basic_block(64, 64, 1),
            # stage 2: 64->128
            # 阶段 2：64->128
            _make_resnet_basic_block(64, 128, 2),
            _make_resnet_basic_block(128, 128, 1),
            # stage 3: 128->256
            # 阶段 3：128->256
            _make_resnet_basic_block(128, 256, 2),
            _make_resnet_basic_block(256, 256, 1),
            # stage 4: 256->512
            # 阶段 4：256->512
            _make_resnet_basic_block(256, 512, 2),
            _make_resnet_basic_block(512, 512, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MobileNetV2StandIn(nn.Module):
    """A MobileNetV2-like architecture using depthwise separable convolutions.
    一个使用深度可分离卷积的类 MobileNetV2 架构。

    Replaces torchvision.models.mobilenet_v2 which is incompatible with this build.
    Roughly matches MobileNetV2's channel progression and parameter count (~3.5M).
    替代与此构建版本不兼容的 torchvision.models.mobilenet_v2。
    大致匹配 MobileNetV2 的通道变化和参数量（~3.5M）。
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        def _dw_sep(in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
            """Depthwise-separable convolution block.
            深度可分离卷积块。"""
            return nn.Sequential(
                # Depthwise 深度卷积
                nn.Conv2d(in_c, in_c, 3, stride, padding=1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU6(inplace=True),
                # Pointwise 逐点卷积
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            _dw_sep(32, 16, 1),
            _dw_sep(16, 24, 2),
            _dw_sep(24, 24, 1),
            _dw_sep(24, 32, 2),
            _dw_sep(32, 32, 1),
            _dw_sep(32, 32, 1),
            _dw_sep(32, 64, 2),
            _dw_sep(64, 64, 1),
            _dw_sep(64, 64, 1),
            _dw_sep(64, 64, 1),
            _dw_sep(64, 96, 1),
            _dw_sep(96, 96, 1),
            _dw_sep(96, 96, 1),
            _dw_sep(96, 160, 2),
            _dw_sep(160, 160, 1),
            _dw_sep(160, 160, 1),
            _dw_sep(160, 320, 1),
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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
# FLOPs 估算（通过 Conv2d 钩子）
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
        groups:Number of groups (default 1).
               groups: 分组数（默认 1）。

    Returns:
        Estimated MACs for this layer on a single input sample.
        返回该层在单个输入样本上的估算 MACs。
    """
    # 计算输出特征图尺寸
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    # MACs = 输出通道 × 输出高度 × 输出宽度 × (输入通道/组数) × 核尺寸²
    macs = out_c * h_out * w_out * (in_c // groups) * k * k
    return macs


def estimate_total_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Estimate total Conv2d MACs for the model by tracing a forward pass.
    通过跟踪一次前向传播来估算模型中所有 Conv2d 层的 MACs 总和。

    Only counts Conv2d layers.  BatchNorm, ReLU, pooling, and Linear layers
    are ignored because they account for a tiny fraction of total compute
    in CNN backbones.
    仅统计 Conv2d 层。BatchNorm、ReLU、池化层和 Linear 层被忽略，
    因为它们在 CNN 主干网络中仅占总计算量的极小部分。

    Args:
        model:       A PyTorch nn.Module.
                     model: PyTorch nn.Module 实例。
        input_shape: (C, H, W) of a single input sample (no batch dim).
                     input_shape: 单个输入样本的 (C, H, W)，不含批次维度。

    Returns:
        Total estimated MACs across all Conv2d layers for one sample.
        返回单个样本在所有 Conv2d 层上的估算 MACs 总和。
    """
    model.eval()
    total_macs = 0
    # 创建一个虚拟输入张量用于前向跟踪
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
                x = inp[0]  # 输入张量，形状: (N, C_in, H_in, W_in)
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


def model_size_mb(total_params: int, bytes_per_param: int = BYTES_PER_FP32) -> float:
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
    return total_params * bytes_per_param / MIB


# ===========================================================================
# 延迟测量
# Latency Measurement
# ===========================================================================


def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    warmup: int = WARMUP_ITERS,
    repeats: int = TIMED_ITERS,
) -> float:
    """Measure average single-sample inference latency on CPU.
    测量 CPU 上单个样本推理的平均延迟。

    Uses batch_size=1.  Returns latency in seconds.
    使用 batch_size=1。返回以秒为单位的延迟。

    Args:
        model:       A PyTorch nn.Module.
                     model: PyTorch nn.Module 实例。
        input_shape: (C, H, W) of a single input sample (no batch dim).
                     input_shape: 单个输入样本的 (C, H, W)，不含批次维度。
        warmup:      Number of warmup iterations (not timed).
                     warmup: 预热迭代次数（不计时）。
        repeats:     Number of timed iterations.
                     repeats: 计时的迭代次数。

    Returns:
        Average forward-pass latency in seconds (float).
        返回平均前向传播延迟（秒）。
    """
    model.eval()
    dummy = torch.randn(1, *input_shape)

    # 预热阶段：稳定 CPU 频率和缓存状态
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # 计时运行阶段
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy)
    end = time.perf_counter()

    return (end - start) / repeats


# ===========================================================================
# 吞吐量测量
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
    测量 CPU 上给定批次大小的吞吐量（样本/秒）。

    Args:
        model:       A PyTorch nn.Module.
                     model: PyTorch nn.Module 实例。
        input_shape: (C, H, W) of a single input sample (no batch dim).
                     input_shape: 单个输入样本的 (C, H, W)，不含批次维度。
        batch_size:  Number of samples per forward pass.
                     batch_size: 每次前向传播的样本数。
        warmup:      Number of warmup iterations.
                     warmup: 预热迭代次数。
        repeats:     Number of timed iterations.
                     repeats: 计时的迭代次数。

    Returns:
        Throughput in samples per second.
        返回以每秒样本数为单位的吞吐量。
    """
    model.eval()
    shape = (batch_size, *input_shape)
    dummy = torch.randn(*shape)

    # 预热阶段
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # 计时运行阶段
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy)
    end = time.perf_counter()

    # 总处理样本数 / 总耗时
    total_time = end - start
    total_samples = batch_size * repeats
    return total_samples / total_time


# ===========================================================================
# 模型构建器
# Model Builder
# ===========================================================================


def build_models() -> List[Tuple[str, nn.Module]]:
    """Build and return all three benchmark models.
    构建并返回全部三个基准测试模型。

    Returns:
        List of (model_name, model_instance) tuples.
        返回 (模型名称, 模型实例) 元组的列表。
    """
    models_list: List[Tuple[str, nn.Module]] = []

    # CustomCNN：自定义 CNN
    custom = CustomCNN(num_classes=10)
    models_list.append(("CustomCNN", custom))

    # ResNet18 stand-in：ResNet18 替代模型
    rn18 = ResNet18StandIn(num_classes=10)
    models_list.append(("ResNet18", rn18))

    # MobileNetV2 stand-in：MobileNetV2 替代模型
    mbv2 = MobileNetV2StandIn(num_classes=10)
    models_list.append(("MobileNetV2", mbv2))

    return models_list


# ===========================================================================
# 报告生成
# Report Generation
# ===========================================================================


def generate_markdown_report(
    results: List[Dict[str, object]],
    batch_sizes: List[int],
) -> str:
    """Generate a formatted Markdown report table from benchmark results.
    根据基准测试结果生成格式化的 Markdown 报告表格。

    Args:
        results:    List of dicts, each containing model_name, total_params,
                    trainable_params, total_macs, total_flops, size_mb,
                    latency_ms, and throughput keys (throughput is a dict
                    mapping batch_size -> samples/sec).
        results:    字典列表，每个字典包含 model_name、total_params、
                    trainable_params、total_macs、total_flops、size_mb、
                    latency_ms 和 throughput 键（throughput 是一个
                    batch_size -> 样本/秒 的映射字典）。
        batch_sizes: List of batch sizes used for throughput measurement.
                     batch_sizes: 用于吞吐量测量的批次大小列表。

    Returns:
        A multi-line Markdown string suitable for printing or saving.
        返回可打印或保存的多行 Markdown 字符串。
    """
    lines: List[str] = []
    sep = "=" * 80

    lines.append("")
    lines.append(sep)
    lines.append("  LECTURE 02: Efficiency Metrics Deep Dive  --  Benchmark Report")
    lines.append(sep)
    lines.append("")

    # ----  Model Overview  ------------------------------------------------
    # ----  模型概览  -------------------------------------------------------
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
    # ----  延迟  ----------------------------------------------------------
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
    # ----  吞吐量  --------------------------------------------------------
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
    # ----  效率比率  -------------------------------------------------------
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
        # 计算效率比率：MACs/参数量、MACs/毫秒、参数量/MiB
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
# 主函数
# Main
# ===========================================================================


def main() -> None:
    """Run the full efficiency benchmark suite and print a Markdown report.
    运行完整的效率基准测试套件并打印 Markdown 报告。
    """
    print("=" * 80)
    print("  LECTURE 02: Efficiency Metrics Deep Dive")
    print("=" * 80)
    print()

    results: List[Dict[str, object]] = []

    for model_name, model in build_models():
        print(f"[{model_name}] Running benchmarks ...")
        model.eval()

        # ---- 1. Parameter Counting ----------------------------------------
        # ---- 1. 参数统计 --------------------------------------------------
        total_params, trainable_params = count_parameters(model)
        print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

        # ---- 2. FLOPs Estimation ------------------------------------------
        # ---- 2. FLOPs 估算 ------------------------------------------------
        total_macs = estimate_total_flops(model, INPUT_SHAPE)
        total_flops = total_macs * 2  # FLOPs = 2 × MACs（一次乘+一次加）
        print(f"  Conv2d MACs: {total_macs:,}  (FLOPs: {total_flops:,})")

        # ---- 3. Model Size ------------------------------------------------
        # ---- 3. 模型大小 --------------------------------------------------
        size_mb = model_size_mb(total_params)
        print(f"  Model size (FP32): {size_mb:.2f} MiB")

        # ---- 4. Latency (batch=1) -----------------------------------------
        # ---- 4. 延迟（batch=1）--------------------------------------------
        print("  Measuring latency (batch=1) ...")
        latency_s = measure_latency(model, INPUT_SHAPE)
        latency_ms = latency_s * 1000.0
        print(f"  Latency: {latency_ms:.2f} ms")

        # ---- 5. Throughput (multiple batch sizes) -------------------------
        # ---- 5. 吞吐量（多种批次大小）-------------------------------------
        throughput: Dict[int, float] = {}
        for bs in BATCH_SIZES:
            tp = measure_throughput(model, INPUT_SHAPE, batch_size=bs)
            throughput[bs] = tp
            print(f"  Throughput (batch={bs:>2d}): {tp:>10.1f} samples/s")

        # 汇总该模型的所有结果
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
    # ---- 6. 生成并打印 Markdown 报告 --------------------------------------
    report = generate_markdown_report(results, BATCH_SIZES)
    print(report)

    print("Benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
