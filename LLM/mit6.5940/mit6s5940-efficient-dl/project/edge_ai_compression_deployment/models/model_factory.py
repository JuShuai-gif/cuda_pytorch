#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工厂模块 —— 定义边缘 AI 适用的模型架构。

本模块提供以下模型：
- ResNet-18 / ResNet-34：经典残差网络的小型版本
- MobileNetV2：专为移动端设计的轻量级网络
- TinyCNN：针对 MCU 级设备（<512KB SRAM）的极简 CNN

所有模型均面向 CIFAR-10 数据集（32×32×3 输入，10 类输出）。
纯 CPU 环境下可运行，不依赖 CUDA。

参考论文：
- Deep Compression (Han et al., 2016)
- MCUNet (Lin et al., 2020)
"""

from __future__ import annotations

import logging
from typing import Any, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ============================================================
# 模块名称到类名的映射（用于从配置文件实例化模型）
# ============================================================
MODEL_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """装饰器：将模型类注册到 MODEL_REGISTRY 中。"""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def create_model(
    architecture: str,
    num_classes: int = 10,
    **kwargs: Any,
) -> nn.Module:
    """
    根据架构名称创建模型实例。

    参数：
        architecture: 模型架构名称（resnet18, resnet34, mobilenetv2, tinycnn）
        num_classes: 分类类别数
        **kwargs: 传递给模型构造函数的额外参数

    返回：
        模型实例

    异常：
        ValueError: 当 architecture 不在注册表中时抛出
    """
    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"未知的模型架构: '{architecture}'。"
            f"支持的架构: {list(MODEL_REGISTRY.keys())}"
        )
    model_cls = MODEL_REGISTRY[architecture]
    logger.info("创建模型: %s (类别数: %d)", architecture, num_classes)
    return model_cls(num_classes=num_classes, **kwargs)


# ============================================================
# 通用模块：BasicBlock（ResNet 的基本构建块）
# ============================================================


class BasicBlock(nn.Module):
    """ResNet 的基础残差块，包含两个 3×3 卷积层。"""

    expansion: int = 1  # BasicBlock 不扩展通道数

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        """
        参数：
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 第一个卷积层的步长
            downsample: 下采样模块（当尺寸/通道不匹配时使用）
        """
        super().__init__()
        # 第一个 3×3 卷积（可能改变空间尺寸）
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个 3×3 卷积（保持尺寸不变）
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：F(x) = Conv→BN→ReLU→Conv→BN + x。"""
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out


# ============================================================
# ResNet 系列模型（适用于 32×32 CIFAR 输入）
# ============================================================


class ResNet(nn.Module):
    """通用的 ResNet 实现，通过配置 block 参数支持不同深度。"""

    def __init__(
        self,
        block: Type[BasicBlock],
        layers: list[int],
        num_classes: int = 10,
        input_channels: int = 3,
    ) -> None:
        """
        参数：
            block: 残差块类型（BasicBlock）
            layers: 各 stage 的 block 数量，如 [2, 2, 2, 2] 对应 ResNet-18
            num_classes: 分类类别数
            input_channels: 输入通道数
        """
        super().__init__()
        self.in_channels = 64

        # 初始卷积层（CIFAR-10 版本：kernel=3, stride=1，不做大 stride 下采样）
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 四个残差 stage
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化 + 全连接分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """
        构建一个残差 stage。

        参数：
            block: 残差块类型
            out_channels: 该 stage 的输出通道数
            blocks: block 数量
            stride: 第一个 block 的步长

        返回：
            包含多个 block 的 Sequential 容器
        """
        downsample = None
        # 当输入/输出通道不一致或 stride != 1 时需要下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers_list: list[nn.Module] = []
        # 第一个 block 可能带下采样
        layers_list.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        # 剩余 blocks
        for _ in range(1, blocks):
            layers_list.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers_list)

    def _initialize_weights(self) -> None:
        """使用 Kaiming 正态分布初始化卷积权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：提取特征 → 全局池化 → 分类。"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@register_model("resnet18")
def create_resnet18(num_classes: int = 10, **kwargs: Any) -> ResNet:
    """创建 ResNet-18 模型（~11M 参数）。"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


@register_model("resnet34")
def create_resnet34(num_classes: int = 10, **kwargs: Any) -> ResNet:
    """创建 ResNet-34 模型（~21M 参数）。"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


# ============================================================
# MobileNetV2 模型 —— 专为移动端设计
# ============================================================
# MobileNetV2 的核心思想：
# 1. Depthwise Separable Convolution（深度可分离卷积）
# 2. Inverted Residual（倒残差结构：先升维→深度卷积→降维）
# 3. Linear Bottleneck（最后一个 1×1 不带 ReLU）


class InvertedResidual(nn.Module):
    """MobileNetV2 的倒残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        """
        参数：
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 深度卷积步长
            expand_ratio: 扩展比（中间层通道数 = in_channels * expand_ratio）
        """
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []

        # 扩展层（1×1 卷积升维，仅当 expand_ratio > 1 时添加）
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        # 深度卷积（3×3，每组一个通道）
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        )

        # 投影层（1×1 卷积降维，线性瓶颈：不带 ReLU）
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：残差连接 + 倒残差卷积。"""
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


@register_model("mobilenetv2")
class MobileNetV2(nn.Module):
    """
    MobileNetV2 模型 —— 适用于边缘设备的轻量级网络。

    参数量约 3.5M（针对 CIFAR-10 32×32 输入）。
    参考：MobileNetV2: Inverted Residuals and Linear Bottlenecks
    """

    # 每层的配置: (expand_ratio, out_channels, num_blocks, stride)
    _CONFIG = [
        # expand_ratio, out_channels, num_blocks, stride
        (1, 16, 1, 1),  # Stage 1
        (6, 24, 2, 1),  # Stage 2（第一个 block stride=1）
        (6, 32, 3, 2),  # Stage 3
        (6, 64, 4, 2),  # Stage 4
        (6, 96, 3, 1),  # Stage 5
        (6, 160, 3, 2),  # Stage 6
        (6, 320, 1, 1),  # Stage 7
    ]

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        input_channels: int = 3,
    ) -> None:
        """
        参数：
            num_classes: 分类类别数
            width_mult: 宽度乘子（控制模型大小，0.5/0.75/1.0）
            input_channels: 输入通道数
        """
        super().__init__()

        # 初始卷积层
        init_channels = self._make_divisible(32 * width_mult, 8)
        self.features: list[nn.Module] = [
            nn.Conv2d(
                input_channels, init_channels, 3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU6(inplace=True),
        ]

        # 构建倒残差块
        in_channels = init_channels
        for t, c, n, s in self._CONFIG:
            out_channels = self._make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1  # 仅每个 stage 的第一个 block 下采样
                self.features.append(
                    InvertedResidual(in_channels, out_channels, stride, t)
                )
                in_channels = out_channels

        # 最后的 1×1 卷积（扩展特征维度）
        last_channels = self._make_divisible(1280 * width_mult, 8)
        self.features.extend(
            [
                nn.Conv2d(in_channels, last_channels, 1, bias=False),
                nn.BatchNorm2d(last_channels),
                nn.ReLU6(inplace=True),
            ]
        )

        self.features = nn.Sequential(*self.features)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channels, num_classes),
        )

        # 初始化权重
        self._initialize_weights()

    @staticmethod
    def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
        """确保通道数是 divisor 的整数倍。"""
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # 确保向下取整不会减少超过 10%
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _initialize_weights(self) -> None:
        """初始化权重：卷积用 Kaiming Normal，BN 用常数。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# TinyCNN —— 极低资源 CNN（MCU 级别，<512KB SRAM）
# ============================================================
# 设计目标：在 <512KB SRAM、<1MB Flash 的 MCU 上运行
# 参考 MCUNet 的极小模型设计思路


@register_model("tinycnn")
class TinyCNN(nn.Module):
    """
    极简 CNN 模型 —— 面向微控制器（MCU）的低资源设备。

    参数量约 0.5M，内存占用 <500KB。
    设计原则：
    - 使用深度可分离卷积减少参数量
    - 最多 4 个卷积层，控制特征图大小
    - 使用 1×1 卷积代替全连接层
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        base_channels: int = 8,
    ) -> None:
        """
        参数：
            num_classes: 分类类别数
            input_channels: 输入通道数（RGB=3）
            base_channels: 基础通道数（控制模型大小）
        """
        super().__init__()

        # 极简卷积网络：深度可分离卷积 × 3 + 点卷积分类头
        self.conv1 = nn.Conv2d(
            input_channels,
            base_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_channels * 2)

        # 深度可分离卷积 Block 1: 16→32
        self.dwconv1 = nn.Conv2d(
            base_channels * 2,
            base_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=base_channels * 2,
            bias=False,
        )
        self.pwconv1 = nn.Conv2d(
            base_channels * 2, base_channels * 4, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(base_channels * 4)

        # 深度可分离卷积 Block 2: 32→64（下采样）
        self.dwconv2 = nn.Conv2d(
            base_channels * 4,
            base_channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=base_channels * 4,
            bias=False,
        )
        self.pwconv2 = nn.Conv2d(
            base_channels * 4, base_channels * 8, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(base_channels * 8)

        # 全局平均池化后的 1×1 卷积（代替全连接层）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(
            base_channels * 8, num_classes, kernel_size=1, bias=True
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化所有卷积层权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        # Block 0
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # Block 1：深度可分离 + 点卷积
        x = F.relu(self.bn2(self.pwconv1(self.dwconv1(x))), inplace=True)
        # Block 2：深度可分离 + 点卷积（下采样）
        x = F.relu(self.bn3(self.pwconv2(self.dwconv2(x))), inplace=True)
        # 分类头
        x = self.avgpool(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x


# ============================================================
# 工具函数
# ============================================================


def get_model_info(model: nn.Module) -> dict[str, Any]:
    """
    获取模型的统计信息。

    返回：
        dict: 包含参数量（总/可训练）、模型类别等
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # 假设 FP32 存储

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
        "model_class": model.__class__.__name__,
    }


def list_available_models() -> list[str]:
    """列出所有可用的模型架构。"""
    return sorted(MODEL_REGISTRY.keys())
