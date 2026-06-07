"""
在 CIFAR-10 上使用进化搜索的延迟感知 NAS (第 08 讲)
======================================================================
实现一种进化神经网络架构搜索 (NAS) 算法，同时优化准确率和推理延迟。
搜索空间与第 07 讲相同（CNN 搜索空间）。使用 **模拟延迟查找表**
提供每层延迟估计（以毫秒为单位），模拟真实硬件行为——更大的卷积核、
更多的通道数和更高的分辨率会非线性地增加计算开销。

脚本包含三个阶段:

    1. **随机搜索基线** -- 20 个随机架构，每个在 CIFAR-10 上
       进行 3 个 epoch 的代理训练。
    2. **进化搜索** -- 10 个个体的种群，经过 5 代进化，使用
       锦标赛选择、单点交叉和三种变异算子（卷积核、通道、深度）。
       多目标适应度使用非支配排序（NSGA-II 风格），
       使算法自然地探索帕累托前沿。
    3. **比较与可视化** -- 准确率-延迟散点图，叠加显示随机搜索和
       进化搜索结果，以及两种策略的汇总比较表。

核心概念:
  - 带种群/变异/交叉的进化 NAS
  - 通过模拟查找表进行延迟感知搜索
  - 帕累托前沿可视化（准确率 vs 延迟）
  - 随机搜索 vs 进化搜索对比

所有计算在 CPU 上运行；无需 GPU。
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 使用非交互式后端，以便在没有显示器的情况下保存图像
matplotlib.use("Agg")

# =============================================================================
# 常量定义
# =============================================================================

# --- 搜索空间 ----------------------------------------------------------------
KERNEL_SIZES: List[int] = [3, 5, 7]  # 允许的卷积核大小
CHANNEL_CHOICES: List[int] = [16, 32, 64, 128]  # 允许的输出通道数
DEPTHS: List[int] = [1, 2, 3, 4]  # 允许的网络深度

# --- 进化算法参数 ------------------------------------------------------------
POPULATION_SIZE: int = 10  # 每代种群中的个体数
NUM_GENERATIONS: int = 5  # 进化代数
TOURNAMENT_SIZE: int = 3  # 锦标赛选择的候选数
CROSSOVER_PROB: float = 0.7  # 交叉概率
MUTATION_PROB: float = 0.3  # 每个个体的变异概率

# --- NAS 实验参数 ------------------------------------------------------------
NUM_RANDOM_SAMPLES: int = 20  # 随机搜索基线大小
NAS_EPOCHS: int = 3  # 每个架构的代理训练 epoch 数
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.01

# --- CIFAR-10 数据参数 -------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
TRAIN_SUBSET: int = 5000  # 训练子集，用于快速代理评估
VAL_SUBSET: int = 2000  # 固定验证子集

# --- 可复现性 ---------------------------------------------------------------
SEED: int = 42

# --- 设备与输出 --------------------------------------------------------------
DEVICE = torch.device("cpu")
OUTPUT_PLOT: str = "nas_accuracy_vs_latency.png"


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class ArchSpec:
    """单个 CNN 架构的规格说明。

    属性:
        depth: 卷积层数 (1--4)。
        kernel_sizes: 每层的卷积核大小，长度等于 depth。
        out_channels: 每层的输出通道数，长度等于 depth。
    """

    depth: int
    kernel_sizes: List[int]
    out_channels: List[int]

    def __post_init__(self) -> None:
        """初始化后验证：确保 kernel_sizes 和 out_channels 的长度与 depth 一致。"""
        if len(self.kernel_sizes) != self.depth:
            raise ValueError(
                f"kernel_sizes 长度 {len(self.kernel_sizes)} 与 depth {self.depth} 不匹配"
            )
        if len(self.out_channels) != self.depth:
            raise ValueError(
                f"out_channels 长度 {len(self.out_channels)} 与 depth {self.depth} 不匹配"
            )


@dataclass
class EvalResult:
    """单个架构的评估结果。

    属性:
        arch: 架构规格说明。
        accuracy: 验证集 top-1 准确率，取值范围 (0, 1)。
        latency_ms: 估计的总推理延迟（毫秒）。
        train_time_s: 训练实际耗时（秒）。
        source: 绘图标签："random" 或 "evolutionary"。
    """

    arch: ArchSpec
    accuracy: float
    latency_ms: float
    train_time_s: float
    source: str = "random"


# =============================================================================
# 模拟延迟查找表
# =============================================================================
#
# 在真实的硬件感知 NAS（例如 ProxylessNAS、MNasNet、FBNet）中，延迟
# 在目标设备上测量并存储在按 (kernel_size, in_channels, out_channels, height, width)
# 索引的查找表中。这里我们使用一个参数化模型来模拟这一点，该模型捕捉了关键趋势：
#
#   延迟 ~ (kernel_size^2 * in_c * out_c * H_out * W_out) / peak_ops
#
# 再加上对小张量的小惩罚（启动开销效应）和对非常大的层的非线性上限
# （内存带宽受限行为）。
#
# 该表采用惰性填充，只有搜索期间实际查询的条目才会被计算。


class LatencyLookupTable:
    """用于硬件感知 NAS 的模拟延迟查找表。

    将每层延迟（毫秒）建模为卷积核大小、输入/输出通道数和
    空间分辨率的函数。结果会被缓存，因此对相同键的重复查询会立即返回。

    属性:
        peak_ops_per_ms: 每秒百万次操作的峰值吞吐量。
        overhead_ms:     每个 Conv2d 层的固定启动开销（毫秒）。
        cache:           内部字典，映射 (k, in_c, out_c, h, w) -> latency_ms。
    """

    def __init__(
        self,
        peak_ops_per_ms: float = 1e5,
        overhead_ms: float = 0.02,
    ) -> None:
        """初始化延迟查找表。

        参数:
            peak_ops_per_ms: 峰值吞吐量（每秒百万次操作）。
            overhead_ms:     每个 Conv2d 层的固定启动开销（毫秒）。
        """
        self.peak_ops_per_ms = peak_ops_per_ms
        self.overhead_ms = overhead_ms
        self._cache: Dict[Tuple[int, int, int, int, int], float] = {}

    def query(
        self,
        kernel: int,
        in_c: int,
        out_c: int,
        h: int,
        w: int,
        stride: int = 1,
        padding: int = 0,
    ) -> float:
        """返回一个 Conv2d 层的模拟延迟（毫秒）。

        参数:
            kernel:  方形卷积核大小。
            in_c:    输入通道数。
            out_c:   输出通道数。
            h:       输入空间高度。
            w:       输入空间宽度。
            stride:  步长（默认 1）。
            padding: 填充（默认 0）。

        返回:
            模拟延迟（毫秒）。
        """
        key = (kernel, in_c, out_c, h, w)
        # 如果缓存命中，直接返回
        if key in self._cache:
            return self._cache[key]

        # 计算输出空间尺寸
        h_out = (h + 2 * padding - kernel) // stride + 1
        w_out = (w + 2 * padding - kernel) // stride + 1

        # 该层的总 MACs
        macs = out_c * h_out * w_out * in_c * kernel * kernel

        # 基础延迟：计算受限部分
        latency_compute = macs / self.peak_ops_per_ms

        # 对非常大的层施加内存带宽惩罚
        elements = out_c * h_out * w_out
        if elements > 100_000:
            latency_compute *= 1.3  # 因内存带宽限制 +30% 惩罚

        # 启动开销 + 计算开销
        latency_ms = self.overhead_ms + latency_compute

        # 小型非线性缩放以模拟硬件流水线效应
        if kernel >= 5:
            latency_ms *= 1.15  # 在真实硬件上大卷积核的额外开销

        self._cache[key] = latency_ms  # 缓存结果
        return latency_ms

    def estimate_model_latency(
        self,
        spec: ArchSpec,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
    ) -> float:
        """估算整个架构的总推理延迟。

        模拟通过 VGG 风格骨干网络的前向传播（每层 Conv2d -> MaxPool2d(2)），
        并将每层 Conv2d 的延迟求和。

        参数:
            spec:        架构规格说明。
            input_shape: 输入图像的 (C, H, W)。

        返回:
            总模拟延迟（毫秒）。
        """
        in_c, h, w = input_shape
        total_ms = 0.0

        # 逐层累加延迟，并模拟 MaxPool2d(2) 后的空间尺寸减半
        for i in range(spec.depth):
            out_c = spec.out_channels[i]
            k = spec.kernel_sizes[i]
            total_ms += self.query(k, in_c, out_c, h, w, stride=1, padding=k // 2)
            # MaxPool2d(2) 后：空间尺寸减半，通道数不变
            h //= 2
            w //= 2
            in_c = out_c

        return total_ms


# =============================================================================
# NAS CNN 构建器（与第 07 讲共用）
# =============================================================================


class NasCNN(nn.Module):
    """根据 ArchSpec 构建的 VGG 风格 CNN 网络。

    每一层由以下模块组成:
        Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d(2)

    在卷积骨干网络之后，通过 AdaptiveAvgPool2d(1) 进行特征降维，
    最后使用一个 Linear 层进行分类。

    参数:
        spec:        架构规格说明（深度、卷积核、通道数）。
        in_channels: 输入图像的通道数（CIFAR-10 为 3）。
        num_classes: 输出类别数（CIFAR-10 为 10）。
    """

    def __init__(
        self,
        spec: ArchSpec,
        in_channels: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_ch = in_channels

        # 根据 spec 逐层构建 Conv2d -> BN -> ReLU -> MaxPool 块
        for i in range(spec.depth):
            out_ch = spec.out_channels[i]
            k = spec.kernel_sizes[i]
            layers.append(
                nn.Conv2d(in_ch, out_ch, k, padding=k // 2)
            )  # 卷积层，保持空间尺寸
            layers.append(nn.BatchNorm2d(out_ch))  # 批归一化
            layers.append(nn.ReLU(inplace=True))  # 激活函数
            layers.append(nn.MaxPool2d(2))  # 2倍下采样
            in_ch = out_ch  # 更新输入通道数供下一层使用

        self.backbone = nn.Sequential(*layers)  # 卷积骨干网络
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.flatten = nn.Flatten(1)  # 展平操作
        self.classifier = nn.Linear(in_ch, num_classes)  # 分类头

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：骨干网络 -> 全局平均池化 -> 展平 -> 分类器。"""
        x = self.backbone(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# =============================================================================
# CIFAR-10 数据加载
# =============================================================================


def get_cifar10_subset(
    num_train: int = TRAIN_SUBSET,
    num_val: int = VAL_SUBSET,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader]:
    """加载 CIFAR-10 并创建固定的训练和验证子集。

    使用较小的子集可以保持 NAS 搜索在 CPU 上的速度，同时仍然
    为架构排序提供有意义的准确率信号。

    参数:
        num_train: 训练样本数量。
        num_val:   验证样本数量。
        seed:      随机种子，用于保证子集选择的确定性。

    返回:
        (train_loader, val_loader) 元组。
    """
    # 训练集数据增强：随机裁剪 + 随机水平翻转 + 归一化
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    # 验证集仅做归一化
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    # 下载并加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val
    )

    # 固定验证子集（确定性选择，保证公平比较）
    rng = np.random.RandomState(seed)
    val_indices = rng.choice(
        len(val_dataset), size=min(num_val, len(val_dataset)), replace=False
    )
    val_subset = Subset(val_dataset, val_indices)

    # 训练子集（同样确定性）
    train_indices = rng.choice(
        len(train_dataset), size=min(num_train, len(train_dataset)), replace=False
    )
    train_subset = Subset(train_dataset, train_indices)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


# =============================================================================
# 训练与评估工具函数
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """训练模型一个 epoch。

    参数:
        model:     已放置在正确设备上的 PyTorch nn.Module。
        loader:    产生 (images, labels) 批次的 DataLoader。
        optimizer: 优化器实例。
        criterion: 损失函数。

    返回:
        该 epoch 的平均训练损失。
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item() * images.size(0)  # 累加加权损失
        total_samples += images.size(0)

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """评估 top-1 准确率。

    参数:
        model:  已放置在正确设备上的 PyTorch nn.Module。
        loader: 产生 (images, labels) 批次的 DataLoader。

    返回:
        准确率，取值范围 [0.0, 1.0]。
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)  # 取预测得分最高的类别
        correct += (preds == labels).sum().item()  # 统计正确预测数
        total += labels.size(0)

    return correct / max(total, 1)


def train_and_evaluate(
    spec: ArchSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = NAS_EPOCHS,
    lr: float = LEARNING_RATE,
) -> Tuple[float, float]:
    """构建、训练并评估单个架构。

    参数:
        spec:         架构规格说明。
        train_loader: 训练 DataLoader。
        val_loader:   验证 DataLoader。
        epochs:       训练 epoch 数。
        lr:           学习率。

    返回:
        (验证准确率, 训练实际耗时_秒) 元组。
    """
    model = NasCNN(spec, in_channels=3, num_classes=10).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )  # 余弦退火学习率
    criterion = nn.CrossEntropyLoss()

    t_start = time.time()
    for _epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()  # 更新学习率

    acc = evaluate_accuracy(model, val_loader)
    elapsed = time.time() - t_start

    return acc, elapsed


# =============================================================================
# 搜索空间：随机采样器
# =============================================================================


def random_sample_architecture(
    kernel_choices: Sequence[int] = KERNEL_SIZES,
    channel_choices: Sequence[int] = CHANNEL_CHOICES,
    depth_choices: Sequence[int] = DEPTHS,
    rng: random.Random | None = None,
) -> ArchSpec:
    """从搜索空间中随机采样一个架构。

    参数:
        kernel_choices:  允许的卷积核大小。
        channel_choices: 允许的输出通道数。
        depth_choices:   允许的网络深度。
        rng:             可选带种子的 random.Random 实例。

    返回:
        一个随机采样的 ArchSpec。
    """
    if rng is None:
        rng = random.Random()

    depth = rng.choice(list(depth_choices))  # 随机选择深度
    kernel_sizes = [
        rng.choice(list(kernel_choices)) for _ in range(depth)
    ]  # 每层随机选择卷积核大小
    out_channels = [
        rng.choice(list(channel_choices)) for _ in range(depth)
    ]  # 每层随机选择输出通道数

    return ArchSpec(depth=depth, kernel_sizes=kernel_sizes, out_channels=out_channels)


# =============================================================================
# 进化算子：变异
# =============================================================================


def mutate_kernel(
    spec: ArchSpec,
    kernel_choices: Sequence[int] = KERNEL_SIZES,
    rng: random.Random | None = None,
) -> ArchSpec:
    """变异随机选择的一层的卷积核大小。

    参数:
        spec:           原始架构。
        kernel_choices: 允许的卷积核大小。
        rng:            可选带种子的 Random。

    返回:
        新 ArchSpec，其中一个卷积核发生了变异。
    """
    if rng is None:
        rng = random.Random()

    new_kernels = list(spec.kernel_sizes)
    idx = rng.randint(0, spec.depth - 1)  # 随机选择要变异的位置
    old_k = new_kernels[idx]
    # 选择一个不同于当前值的卷积核大小
    choices = [k for k in kernel_choices if k != old_k]
    if not choices:
        choices = list(kernel_choices)
    new_kernels[idx] = rng.choice(choices)

    return ArchSpec(
        depth=spec.depth,
        kernel_sizes=new_kernels,
        out_channels=list(spec.out_channels),
    )


def mutate_channels(
    spec: ArchSpec,
    channel_choices: Sequence[int] = CHANNEL_CHOICES,
    rng: random.Random | None = None,
) -> ArchSpec:
    """变异随机选择的一层的通道数。

    参数:
        spec:            原始架构。
        channel_choices: 允许的输出通道数。
        rng:             可选带种子的 Random。

    返回:
        新 ArchSpec，其中一个通道数发生了变异。
    """
    if rng is None:
        rng = random.Random()

    new_channels = list(spec.out_channels)
    idx = rng.randint(0, spec.depth - 1)  # 随机选择要变异的位置
    old_ch = new_channels[idx]
    # 选择一个不同于当前值的通道数
    choices_ch = [c for c in channel_choices if c != old_ch]
    if not choices_ch:
        choices_ch = list(channel_choices)
    new_channels[idx] = rng.choice(choices_ch)

    return ArchSpec(
        depth=spec.depth,
        kernel_sizes=list(spec.kernel_sizes),
        out_channels=new_channels,
    )


def mutate_depth(
    spec: ArchSpec,
    depth_choices: Sequence[int] = DEPTHS,
    kernel_choices: Sequence[int] = KERNEL_SIZES,
    channel_choices: Sequence[int] = CHANNEL_CHOICES,
    rng: random.Random | None = None,
) -> ArchSpec:
    """变异架构的深度（添加或删除一层）。

    - 如果深度已经是最小值，强制添加。
    - 如果深度已经是最大值，强制删除。
    - 否则随机选择添加或删除。

    添加时，新层的卷积核/通道从现有层中随机继承。
    删除时，随机丢弃一层。

    参数:
        spec:            原始架构。
        depth_choices:   允许的深度范围。
        kernel_choices:  允许的卷积核大小（用于新层）。
        channel_choices: 允许的通道数（用于新层）。
        rng:             可选带种子的 Random。

    返回:
        深度变化 +/- 1 的新 ArchSpec。
    """
    if rng is None:
        rng = random.Random()

    current_depth = spec.depth
    can_add = current_depth < max(depth_choices)  # 是否还能增加深度
    can_remove = current_depth > min(depth_choices)  # 是否还能减少深度

    # 根据边界条件决定是添加还是删除
    if can_add and can_remove:
        add_layer = rng.random() < 0.5  # 50% 概率添加
    elif can_add:
        add_layer = True
    else:
        add_layer = False  # 必须删除

    if add_layer:
        # 在随机位置插入新层
        insert_pos = rng.randint(0, current_depth)
        new_kernel = rng.choice(list(kernel_choices))
        new_channel = rng.choice(list(channel_choices))

        new_kernels = list(spec.kernel_sizes)
        new_channels = list(spec.out_channels)
        new_kernels.insert(insert_pos, new_kernel)
        new_channels.insert(insert_pos, new_channel)

        return ArchSpec(
            depth=current_depth + 1,
            kernel_sizes=new_kernels,
            out_channels=new_channels,
        )
    else:
        # 删除随机位置的一层
        remove_pos = rng.randint(0, current_depth - 1)
        new_kernels = list(spec.kernel_sizes)
        new_channels = list(spec.out_channels)
        new_kernels.pop(remove_pos)
        new_channels.pop(remove_pos)

        return ArchSpec(
            depth=current_depth - 1,
            kernel_sizes=new_kernels,
            out_channels=new_channels,
        )


def mutate(
    spec: ArchSpec,
    kernel_choices: Sequence[int] = KERNEL_SIZES,
    channel_choices: Sequence[int] = CHANNEL_CHOICES,
    depth_choices: Sequence[int] = DEPTHS,
    rng: random.Random | None = None,
) -> ArchSpec:
    """对架构应用一种随机变异算子。

    在卷积核变异、通道变异和深度变异之间均匀随机选择。

    参数:
        spec:            原始架构。
        kernel_choices:  允许的卷积核大小。
        channel_choices: 允许的输出通道数。
        depth_choices:   允许的深度范围。
        rng:             可选带种子的 Random。

    返回:
        变异后的 ArchSpec。
    """
    if rng is None:
        rng = random.Random()

    op = rng.choice(["kernel", "channel", "depth"])  # 随机选择变异类型
    if op == "kernel":
        return mutate_kernel(spec, kernel_choices, rng)
    elif op == "channel":
        return mutate_channels(spec, channel_choices, rng)
    else:
        return mutate_depth(spec, depth_choices, kernel_choices, channel_choices, rng)


# =============================================================================
# 进化算子：交叉
# =============================================================================


def crossover(
    parent1: ArchSpec,
    parent2: ArchSpec,
    rng: random.Random | None = None,
) -> Tuple[ArchSpec, ArchSpec]:
    """对两个父架构的层列表进行单点交叉。

    两个父代必须具有相同的深度，交叉才有意义。
    如果深度不同，则较长的父代被截断为较短的长度，
    并附加一个随机层，使得子代与较长父代具有相同的深度。

    参数:
        parent1: 第一个父代 ArchSpec。
        parent2: 第二个父代 ArchSpec。
        rng:     可选带种子的 Random。

    返回:
        两个子代 ArchSpec 的元组 (child1, child2)。
    """
    if rng is None:
        rng = random.Random()

    d1, d2 = parent1.depth, parent2.depth
    min_depth = min(d1, d2)

    # 深度为 1 时交叉没有意义；直接返回克隆
    if min_depth < 2:
        return (
            ArchSpec(
                depth=d1,
                kernel_sizes=list(parent1.kernel_sizes),
                out_channels=list(parent1.out_channels),
            ),
            ArchSpec(
                depth=d2,
                kernel_sizes=list(parent2.kernel_sizes),
                out_channels=list(parent2.out_channels),
            ),
        )

    # 对齐到相同深度以进行交叉
    k1 = list(parent1.kernel_sizes[:min_depth])
    k2 = list(parent2.kernel_sizes[:min_depth])
    ch1 = list(parent1.out_channels[:min_depth])
    ch2 = list(parent2.out_channels[:min_depth])

    # 随机选择交叉点 (1..min_depth-1)
    point = rng.randint(1, min_depth - 1)

    # 交换尾部
    child1_k = k1[:point] + k2[point:]
    child1_ch = ch1[:point] + ch2[point:]
    child2_k = k2[:point] + k1[point:]
    child2_ch = ch2[:point] + ch1[point:]

    # 如果父代深度不同，保留较长父代的形状
    # 通过追加原始父代的额外层来实现
    if d1 > min_depth:
        child1_k.extend(parent1.kernel_sizes[min_depth:])
        child2_k.extend(parent1.kernel_sizes[min_depth:])
        child1_ch.extend(parent1.out_channels[min_depth:])
        child2_ch.extend(parent1.out_channels[min_depth:])
    elif d2 > min_depth:
        child1_k.extend(parent2.kernel_sizes[min_depth:])
        child2_k.extend(parent2.kernel_sizes[min_depth:])
        child1_ch.extend(parent2.out_channels[min_depth:])
        child2_ch.extend(parent2.out_channels[min_depth:])

    target_depth = len(child1_k)
    return (
        ArchSpec(depth=target_depth, kernel_sizes=child1_k, out_channels=child1_ch),
        ArchSpec(depth=target_depth, kernel_sizes=child2_k, out_channels=child2_ch),
    )


# =============================================================================
# 进化算子：选择
# =============================================================================


def non_dominated_sorting(
    results: List[EvalResult],
) -> List[List[int]]:
    """基于准确率和延迟的 NSGA-II 非支配排序。

    返回一个前沿列表，其中第一个前沿包含所有非支配个体的索引，
    第二个前沿包含仅被第一个前沿支配的个体的索引，依此类推。

    我们最大化准确率并最小化延迟。

    参数:
        results: 种群的 EvalResult 列表。

    返回:
        前沿列表；每个前沿是一个指向 ``results`` 的索引列表。
    """
    n = len(results)
    # dominates: S[i] = i 支配的索引集合
    dominates: List[List[int]] = [[] for _ in range(n)]
    # dominated_by_count[i] = 支配 i 的个体数量
    dominated_by_count: List[int] = [0] * n

    # 两两比较，确定支配关系
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i 支配 j，当 i 的准确率 >= j 且延迟 <= j，且至少一个严格占优
            better_acc = results[i].accuracy >= results[j].accuracy
            better_lat = results[i].latency_ms <= results[j].latency_ms
            strictly_better = (
                results[i].accuracy > results[j].accuracy
                or results[i].latency_ms < results[j].latency_ms
            )
            if better_acc and better_lat and strictly_better:
                dominates[i].append(j)
                dominated_by_count[j] += 1

    fronts: List[List[int]] = []
    # 第一个前沿：dominated_by_count == 0 的个体（不被任何其他个体支配）
    current_front = [i for i in range(n) if dominated_by_count[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front: List[int] = []
        for i in current_front:
            for j in dominates[i]:
                dominated_by_count[j] -= 1
                # 如果 j 不再被任何未处理的前沿支配，则加入下一个前沿
                if dominated_by_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def tournament_select(
    population: List[ArchSpec],
    pop_results: List[EvalResult],
    fronts: List[List[int]],
    tournament_size: int = TOURNAMENT_SIZE,
    rng: random.Random | None = None,
) -> ArchSpec:
    """基于帕累托等级（来自非支配排序）的锦标赛选择。

    从种群中随机选择 ``tournament_size`` 个候选个体。
    返回帕累托等级最好（最低）的一个，等级相同时优先选择准确率更高的。

    参数:
        population:       当前种群（ArchSpec 列表）。
        pop_results:      每个个体的对应 EvalResult。
        fronts:           非支配排序前沿列表（索引列表的列表）。
        tournament_size:  每个锦标赛中的个体数。
        rng:              可选带种子的 Random。

    返回:
        被选中的 ArchSpec。
    """
    if rng is None:
        rng = random.Random()

    # 预计算每个个体的等级
    rank_of: Dict[int, int] = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            rank_of[idx] = rank

    n = len(population)
    # 随机选择 tournament_size 个候选
    candidates = [rng.randint(0, n - 1) for _ in range(tournament_size)]

    best_idx = candidates[0]
    best_rank = rank_of.get(best_idx, 999999)
    best_acc = pop_results[best_idx].accuracy

    # 遍历其余候选，找出等级最低（准确率最高打破平局）的个体
    for idx in candidates[1:]:
        r = rank_of.get(idx, 999999)
        acc = pop_results[idx].accuracy
        if r < best_rank or (r == best_rank and acc > best_acc):
            best_idx = idx
            best_rank = r
            best_acc = acc

    return copy.deepcopy(population[best_idx])


# =============================================================================
# 进化算法主循环
# =============================================================================


def run_evolutionary_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    latency_table: LatencyLookupTable,
    population_size: int = POPULATION_SIZE,
    generations: int = NUM_GENERATIONS,
    rng: random.Random | None = None,
) -> List[EvalResult]:
    """运行完整的进化 NAS 搜索。

    1. 初始化 ``population_size`` 个架构的随机种群。
    2. 评估每个个体的适应度（准确率 + 延迟）。
    3. 每一代:
       a. 对当前种群执行非支配排序。
       b. 通过锦标赛选择、交叉、变异创建子代。
       c. 评估子代。
       d. 合并父代 + 子代，使用帕累托等级 + 准确率打破平局，
          选择前 ``population_size`` 个个体。

    参数:
        train_loader:    训练 DataLoader。
        val_loader:      验证 DataLoader。
        latency_table:   延迟查找表。
        population_size: 种群大小。
        generations:     进化代数。
        rng:             可选带种子的 Random。

    返回:
        最终种群的 EvalResult 列表。
    """
    if rng is None:
        rng = random.Random()

    # --- 1. 初始化随机种群 ---------------------------------------------------
    population: List[ArchSpec] = []
    for _ in range(population_size):
        population.append(random_sample_architecture(rng=rng))

    pop_results: List[EvalResult] = []
    print(f"\n  Evaluating initial population ({population_size} architectures) ...")
    for i, spec in enumerate(population):
        acc, train_time = train_and_evaluate(spec, train_loader, val_loader)
        lat = latency_table.estimate_model_latency(spec)  # 估算模型延迟
        pop_results.append(
            EvalResult(
                arch=spec,
                accuracy=acc,
                latency_ms=lat,
                train_time_s=train_time,
                source="evolutionary",
            )
        )
        print(
            f"    init [{i + 1:>2d}/{population_size}]  "
            f"{arch_summary(spec):<35}  "
            f"acc={acc * 100:.2f}%  lat={lat:.3f}ms"
        )

    # --- 2. 进化循环 ---------------------------------------------------------
    for gen in range(generations):
        print(
            f"\n  --- Generation {gen + 1}/{generations} "
            f"(population {population_size}) ---"
        )

        # 对当前种群进行非支配排序
        fronts = non_dominated_sorting(pop_results)
        print(
            f"    Pareto fronts: {len(fronts)}  (front 0: {len(fronts[0])} individuals)"
        )

        # 创建子代
        offspring: List[ArchSpec] = []
        while len(offspring) < population_size:
            # 锦标赛选择：从种群中选择两个父代
            p1 = tournament_select(population, pop_results, fronts, rng=rng)
            p2 = tournament_select(population, pop_results, fronts, rng=rng)

            # 交叉：以一定概率对两个父代进行单点交叉
            if rng.random() < CROSSOVER_PROB and p1.depth >= 2 and p2.depth >= 2:
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1 = copy.deepcopy(p1)
                c2 = copy.deepcopy(p2)

            # 变异：以一定概率对子代应用随机变异
            if rng.random() < MUTATION_PROB:
                c1 = mutate(c1, rng=rng)
            if rng.random() < MUTATION_PROB:
                c2 = mutate(c2, rng=rng)

            offspring.append(c1)
            if len(offspring) < population_size:
                offspring.append(c2)

        # 评估子代
        offspring_results: List[EvalResult] = []
        print(f"    Evaluating offspring ({len(offspring)} architectures) ...")
        for i, spec in enumerate(offspring):
            acc, train_time = train_and_evaluate(spec, train_loader, val_loader)
            lat = latency_table.estimate_model_latency(spec)  # 估算模型延迟
            offspring_results.append(
                EvalResult(
                    arch=spec,
                    accuracy=acc,
                    latency_ms=lat,
                    train_time_s=train_time,
                    source="evolutionary",
                )
            )

        # --- 环境选择：保留最优的 population_size 个个体 -----------------------
        combined_pop = population + offspring
        combined_results = pop_results + offspring_results

        # 对所有合并个体进行排序
        combined_fronts = non_dominated_sorting(combined_results)

        # 按帕累托等级选择前 population_size 个（等级相同时按准确率打破平局）
        rank_of: Dict[int, int] = {}
        for rank, front in enumerate(combined_fronts):
            for idx in front:
                rank_of[idx] = rank

        # 按 (rank, -accuracy) 排序——最优的排在前面
        ranked_indices = sorted(
            range(len(combined_results)),
            key=lambda i: (rank_of.get(i, 999999), -combined_results[i].accuracy),
        )

        # 保留最优个体
        kept_indices = ranked_indices[:population_size]
        population = [combined_pop[i] for i in kept_indices]
        pop_results = [combined_results[i] for i in kept_indices]

        # 打印该代的汇总信息
        gen_accs = [r.accuracy * 100 for r in pop_results]
        gen_lats = [r.latency_ms for r in pop_results]
        print(
            f"    Gen {gen + 1} summary: "
            f"acc mean={np.mean(gen_accs):.2f}%  "
            f"best={np.max(gen_accs):.2f}%  "
            f"lat mean={np.mean(gen_lats):.3f}ms  "
            f"min={np.min(gen_lats):.3f}ms"
        )

    return pop_results


# =============================================================================
# 绘图：帕累托前沿（准确率 vs 延迟）
# =============================================================================


def plot_pareto_frontier(
    random_results: List[EvalResult],
    evo_results: List[EvalResult],
    save_path: str = OUTPUT_PLOT,
) -> None:
    """绘制包含随机搜索和进化搜索结果的准确率 vs 延迟散点图。

    随机搜索结果用蓝色圆圈表示；进化搜索结果用红色三角形表示。
    合并后的帕累托前沿用连接线和填充标记高亮显示。

    参数:
        random_results: 随机搜索的 EvalResult 列表。
        evo_results:    进化搜索的 EvalResult 列表。
        save_path:      输出 PNG 文件的保存路径。
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- 随机搜索点 ----------------------------------------------------------
    rand_acc = [r.accuracy * 100 for r in random_results]
    rand_lat = [r.latency_ms for r in random_results]
    ax.scatter(
        rand_lat,
        rand_acc,
        c="steelblue",
        marker="o",
        s=70,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
        label=f"Random Search (n={len(random_results)})",
        zorder=3,
    )

    # --- 进化搜索点 -----------------------------------------------------------
    evo_acc = [r.accuracy * 100 for r in evo_results]
    evo_lat = [r.latency_ms for r in evo_results]
    ax.scatter(
        evo_lat,
        evo_acc,
        c="firebrick",
        marker="^",
        s=90,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
        label=f"Evolutionary Search (n={len(evo_results)})",
        zorder=4,
    )

    # --- 帕累托前沿（合并后）-------------------------------------------------
    all_results = random_results + evo_results
    pareto = compute_pareto_frontier(all_results)  # 计算合并后的帕累托前沿
    pareto_acc = [r.accuracy * 100 for r in pareto]
    pareto_lat = [r.latency_ms for r in pareto]

    # 按延迟排序以绘制连接线
    pareto_sorted = sorted(pareto, key=lambda r: r.latency_ms)
    sorted_acc = [r.accuracy * 100 for r in pareto_sorted]
    sorted_lat = [r.latency_ms for r in pareto_sorted]

    ax.plot(
        sorted_lat,
        sorted_acc,
        "o-",
        color="darkorange",
        linewidth=2.0,
        markersize=8,
        markerfacecolor="gold",
        markeredgecolor="black",
        markeredgewidth=0.8,
        label=f"Pareto Frontier ({len(pareto)} candidates)",
        zorder=5,
    )

    ax.set_xlabel("Inference Latency (ms)", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title(
        "Latency-Aware NAS: Accuracy vs Latency Pareto Frontier",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPareto frontier plot saved to: {save_path}")


def compute_pareto_frontier(results: List[EvalResult]) -> List[EvalResult]:
    """识别准确率 vs 延迟的帕累托前沿（非支配集合）。

    我们最大化准确率并最小化延迟。架构 A 支配 B，当且仅当
    A 的准确率 >= B 且延迟 <= B，且至少一个为严格不等。

    参数:
        results: EvalResult 列表。

    返回:
        非支配的 EvalResult 列表。
    """
    pareto: List[EvalResult] = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            # 如果 other 在准确率上不差于 r 且在延迟上不多于 r
            if other.accuracy >= r.accuracy and other.latency_ms <= r.latency_ms:
                # 至少有一个严格占优
                if other.accuracy > r.accuracy or other.latency_ms < r.latency_ms:
                    dominated = True
                    break
        if not dominated:
            pareto.append(r)
    return pareto


# =============================================================================
# 工具函数
# =============================================================================


def arch_summary(spec: ArchSpec) -> str:
    """返回描述架构的紧凑单行字符串。

    参数:
        spec: 架构规格说明。

    返回:
        类似 "D3_C[32,64,128]_K[5,3,7]" 的字符串。
    """
    ch_str = ",".join(str(c) for c in spec.out_channels)
    k_str = ",".join(str(k) for k in spec.kernel_sizes)
    return f"D{spec.depth}_C[{ch_str}]_K[{k_str}]"


def format_latency(ms: float) -> str:
    """将延迟值格式化为带适当单位的字符串。

    参数:
        ms: 延迟（毫秒）。

    返回:
        格式化的字符串，如 "2.345ms" 或 "0.123ms"。
    """
    if ms < 0.01:
        return f"{ms * 1000:.2f}us"
    return f"{ms:.3f}ms"


# =============================================================================
# 主流程
# =============================================================================


def main() -> None:
    """运行完整的进化 NAS 流程。"""
    # 设置随机种子以保证可复现性
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    print("=" * 72)
    print("  LECTURE 08: Evolutionary Search with Latency-Aware NAS")
    print("=" * 72)

    # ---- 1. 搜索空间与延迟表 -----------------------------------------------
    print(f"\n[1] Search space definition:")
    print(f"  Kernel sizes  : {KERNEL_SIZES}")
    print(f"  Channels      : {CHANNEL_CHOICES}")
    print(f"  Depths        : {DEPTHS}")
    # 计算搜索空间中所有可能的架构组合总数
    total_configs = sum(
        len(CHANNEL_CHOICES) ** d * len(KERNEL_SIZES) ** d for d in DEPTHS
    )
    print(f"  Total possible architectures: {total_configs}")
    print(f"  Random search samples        : {NUM_RANDOM_SAMPLES}")
    print(f"  Evolution: pop={POPULATION_SIZE}, gens={NUM_GENERATIONS}")
    print(f"  Training epochs per arch     : {NAS_EPOCHS}")

    latency_table = LatencyLookupTable()
    print(
        f"\n  Latency lookup table initialised ({len(latency_table._cache)} entries in cache)"
    )

    # ---- 2. 加载 CIFAR-10 数据 ---------------------------------------------
    print(
        f"\n[2] Loading CIFAR-10 (train subset={TRAIN_SUBSET}, val subset={VAL_SUBSET}) ..."
    )
    train_loader, val_loader = get_cifar10_subset()
    print(f"  Train batches: {len(train_loader)},  Val batches: {len(val_loader)}")

    # ---- 3. 随机搜索基线 ---------------------------------------------------
    print(
        f"\n[3] Running RANDOM SEARCH baseline ({NUM_RANDOM_SAMPLES} architectures) ..."
    )
    print(
        f"     {'#':<4} {'Architecture':<35} {'Accuracy':>8} {'Latency':>10} {'Time':>8}"
    )
    print(f"     {'---':<4} {'---':<35} {'---':>8} {'---':>10} {'---':>8}")

    random_results: List[EvalResult] = []
    for i in range(NUM_RANDOM_SAMPLES):
        spec = random_sample_architecture(rng=rng)
        acc, train_time = train_and_evaluate(spec, train_loader, val_loader)
        lat = latency_table.estimate_model_latency(spec)  # 估算模型延迟
        result = EvalResult(
            arch=spec,
            accuracy=acc,
            latency_ms=lat,
            train_time_s=train_time,
            source="random",
        )
        random_results.append(result)

        print(
            f"     {i + 1:>3d}  {arch_summary(spec):<35} "
            f"{acc * 100:>7.2f}% {format_latency(lat):>9}  {train_time:>6.1f}s"
        )

    # 随机搜索汇总
    rand_accs = [r.accuracy * 100 for r in random_results]
    rand_lats = [r.latency_ms for r in random_results]
    rand_pareto = compute_pareto_frontier(random_results)
    print(f"\n  Random search summary:")
    print(
        f"    Accuracy: mean={np.mean(rand_accs):.2f}%, "
        f"min={np.min(rand_accs):.2f}%, max={np.max(rand_accs):.2f}%"
    )
    print(
        f"    Latency:  mean={np.mean(rand_lats):.4f}ms, "
        f"min={np.min(rand_lats):.4f}ms, max={np.max(rand_lats):.4f}ms"
    )
    print(f"    Pareto frontier:  {len(rand_pareto)} candidates")

    # ---- 4. 进化搜索 -------------------------------------------------------
    print(f"\n[4] Running EVOLUTIONARY SEARCH ...")
    t_evo_start = time.time()
    evo_results = run_evolutionary_search(
        train_loader,
        val_loader,
        latency_table,
        population_size=POPULATION_SIZE,
        generations=NUM_GENERATIONS,
        rng=rng,
    )
    t_evo_elapsed = time.time() - t_evo_start

    # 进化搜索汇总
    evo_accs = [r.accuracy * 100 for r in evo_results]
    evo_lats = [r.latency_ms for r in evo_results]
    evo_pareto = compute_pareto_frontier(evo_results)
    print(f"\n  Evolutionary search summary:")
    print(f"    Generations: {NUM_GENERATIONS}, population: {POPULATION_SIZE}")
    print(f"    Wall time: {t_evo_elapsed:.1f}s ({t_evo_elapsed / 60:.1f} min)")
    print(
        f"    Accuracy: mean={np.mean(evo_accs):.2f}%, "
        f"min={np.min(evo_accs):.2f}%, max={np.max(evo_accs):.2f}%"
    )
    print(
        f"    Latency:  mean={np.mean(evo_lats):.4f}ms, "
        f"min={np.min(evo_lats):.4f}ms, max={np.max(evo_lats):.4f}ms"
    )
    print(f"    Pareto frontier:  {len(evo_pareto)} candidates")
    print(f"    Latency lookup table: {len(latency_table._cache)} entries cached")

    # ---- 5. 比较：随机搜索 vs 进化搜索 -------------------------------------
    print(f"\n[5] COMPARISON: Random Search vs Evolutionary Search")
    print(f"  {'=' * 60}")
    print(f"  {'Metric':<25} {'Random Search':>16} {'Evolutionary':>16}")
    print(f"  {'-' * 59}")
    print(f"  {'Evaluations':<25} {len(random_results):>16d} {len(evo_results):>16d}")
    print(
        f"  {'Best Accuracy':<25} {np.max(rand_accs):>15.2f}% "
        f"{np.max(evo_accs):>15.2f}%"
    )
    print(
        f"  {'Mean Accuracy':<25} {np.mean(rand_accs):>15.2f}% "
        f"{np.mean(evo_accs):>15.2f}%"
    )
    print(
        f"  {'Min Latency':<25} {format_latency(np.min(rand_lats)):>15}  "
        f"{format_latency(np.min(evo_lats)):>15}"
    )
    print(
        f"  {'Mean Latency':<25} {format_latency(np.mean(rand_lats)):>15}  "
        f"{format_latency(np.mean(evo_lats)):>15}"
    )
    print(
        f"  {'Pareto Frontier Size':<25} {len(rand_pareto):>16d} {len(evo_pareto):>16d}"
    )

    # ---- 6. 绘制帕累托前沿 --------------------------------------------------
    print(f"\n[6] Plotting accuracy vs latency Pareto frontier ...")
    plot_pareto_frontier(random_results, evo_results, save_path=OUTPUT_PLOT)

    # ---- 7. 结束 ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(
        f"  Search space:    kernel={KERNEL_SIZES}, ch={CHANNEL_CHOICES}, "
        f"depth={DEPTHS} ({total_configs} configs)"
    )
    print(f"  Random search:   {NUM_RANDOM_SAMPLES} samples")
    print(
        f"  Evolution:       pop={POPULATION_SIZE} x gen={NUM_GENERATIONS} "
        f"(tournament={TOURNAMENT_SIZE})"
    )
    print(
        f"  Training:        {NAS_EPOCHS} proxy epochs on {TRAIN_SUBSET} CIFAR-10 samples"
    )
    print(
        f"  Best accuracy:   random={np.max(rand_accs):.2f}%  "
        f"evolutionary={np.max(evo_accs):.2f}%"
    )
    print(
        f"  Best latency:    random={format_latency(np.min(rand_lats))}  "
        f"evolutionary={format_latency(np.min(evo_lats))}"
    )
    print(
        f"  Pareto frontier: random={len(rand_pareto)}  evolutionary={len(evo_pareto)}"
    )
    print(f"  Latency entries: {len(latency_table._cache)} cached")
    print(f"  Plot saved to:   {OUTPUT_PLOT}")
    print("=" * 72)

    print("\nLecture 08 complete.")


if __name__ == "__main__":
    main()
