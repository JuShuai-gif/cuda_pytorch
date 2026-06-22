"""
在 CIFAR-10 上的简化 NAS 随机搜索 (第 07 讲)
=======================================================
对由以下参数定义的 CNN 搜索空间实现随机搜索神经网络架构搜索 (NAS):

    - 卷积核大小: [3, 5, 7]
    - 输出通道数: [16, 32, 64, 128]
    - 网络深度:   [1, 2, 3, 4]

对于每个随机采样的架构，我们在 CIFAR-10 上进行少量 epoch 的训练，
评估验证集准确率，并估算 MACs（乘加操作数）。
最终生成的 准确率 vs MACs 散点图揭示了模型计算开销与预测性能之间的权衡关系。

核心概念:
  - NAS 搜索空间定义
  - 随机架构采样
  - 代理任务训练（短时间训练以快速评估）
  - 通过前向钩子 (forward hooks) 估算 MACs
  - 准确率 vs 效率的帕累托前沿

所有计算在 CPU 上运行；无需 GPU。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

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

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 搜索空间
KERNEL_SIZES: List[int] = [3, 5, 7]  # 允许的卷积核大小
CHANNEL_CHOICES: List[int] = [16, 32, 64, 128]  # 允许的输出通道数
DEPTHS: List[int] = [1, 2, 3, 4]  # 允许的网络深度

# NAS 实验参数
NUM_SAMPLES: int = 20  # 要评估的随机架构数量
NAS_EPOCHS: int = 3  # 每个架构的快速代理训练 epoch 数
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.01

# 数据相关参数
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # CIFAR-10 数据集均值
CIFAR10_STD = (0.2470, 0.2435, 0.2616)  # CIFAR-10 数据集标准差
TRAIN_SUBSET: int = 5000  # 使用 CIFAR-10 的子集以加快搜索速度
VAL_SUBSET: int = 2000  # 固定验证子集，保证评估一致性

# 可复现性种子
SEED: int = 42

# 输出配置
DEVICE = torch.device("cpu")
OUTPUT_PLOT: str = "nas_accuracy_vs_macs.png"  # 结果图保存路径


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


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
        macs: Conv2d 总 MACs（乘加操作数）。
        train_time_s: 训练实际耗时（秒）。
    """

    arch: ArchSpec
    accuracy: float
    macs: int
    train_time_s: float


# ---------------------------------------------------------------------------
# 搜索空间：随机采样器
# ---------------------------------------------------------------------------


def random_sample_architecture(
    kernel_choices: Sequence[int] = KERNEL_SIZES,
    channel_choices: Sequence[int] = CHANNEL_CHOICES,
    depth_choices: Sequence[int] = DEPTHS,
    rng: random.Random | None = None,
) -> ArchSpec:
    """从搜索空间中随机采样一个架构。

    参数:
        kernel_choices:  允许的卷积核大小（默认 [3, 5, 7]）。
        channel_choices: 允许的输出通道数（默认 [16, 32, 64, 128]）。
        depth_choices:   允许的网络深度（默认 [1, 2, 3, 4]）。
        rng:             可选带种子的 random.Random 实例，用于保证可复现性。

    返回:
        一个具有随机选择的深度、卷积核大小和通道数的 ArchSpec。
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


# ---------------------------------------------------------------------------
# CNN 构建器
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 通过前向钩子估算 MACs
# ---------------------------------------------------------------------------


def estimate_macs_conv2d(
    in_c: int,
    out_c: int,
    k: int,
    h: int,
    w: int,
    stride: int = 1,
    padding: int = 0,
) -> int:
    """估算单个 Conv2d 层的 MACs（乘加操作数）。

    参数:
        in_c:    输入通道数。
        out_c:   输出通道数。
        k:       方形卷积核大小。
        h:       输入高度。
        w:       输入宽度。
        stride:  步长。
        padding: 填充。

    返回:
        单次前向传播的 MACs 数量（单个样本）。
    """
    h_out = (h + 2 * padding - k) // stride + 1  # 输出高度
    w_out = (w + 2 * padding - k) // stride + 1  # 输出宽度
    # MACs = 输出通道数 * 输出高度 * 输出宽度 * 输入通道数 * 卷积核高 * 卷积核宽
    return out_c * h_out * w_out * in_c * k * k


def count_macs(model: nn.Module, input_shape: Tuple[int, int, int]) -> int:
    """通过前向钩子追踪一次前向传播，统计所有 Conv2d 层的总 MACs。

    参数:
        model:       PyTorch nn.Module 模型。
        input_shape: 输入张量的形状 (C, H, W)，不含 batch 维度。

    返回:
        所有 Conv2d 层的总 MACs。
    """
    model.eval()
    total_macs: int = 0
    dummy = torch.randn(1, *input_shape)  # 创建一个虚拟输入用于追踪

    def _hook(
        module: nn.Module,
        inp: Tuple[torch.Tensor, ...],
        _out: torch.Tensor,
    ) -> None:
        """前向钩子：当数据流经 Conv2d 层时累加该层的 MACs。"""
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            x = inp[0]  # 获取该层的输入张量
            total_macs += estimate_macs_conv2d(
                in_c=x.shape[1],
                out_c=module.out_channels,
                k=module.kernel_size[0],
                h=x.shape[2],
                w=x.shape[3],
                stride=module.stride[0],
                padding=module.padding[0],
            )

    handles = []  # 存储钩子句柄以便后续清理
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(_hook))  # 为每个 Conv2d 注册前向钩子

    # 使用虚拟输入执行一次前向传播以触发钩子
    with torch.no_grad():
        _ = model(dummy)

    # 清理：移除所有注册的钩子
    for h in handles:
        h.remove()

    return total_macs


# ---------------------------------------------------------------------------
# CIFAR-10 数据加载
# ---------------------------------------------------------------------------


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
        num_val:   验证样本数量（所有架构固定不变）。
        seed:      随机种子，用于保证子集选择的可复现性。

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
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    val_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_val,
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


# ---------------------------------------------------------------------------
# 训练与评估
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------


def plot_accuracy_vs_macs(
    results: List[EvalResult],
    save_path: str = OUTPUT_PLOT,
) -> None:
    """绘制所有采样架构的验证准确率 vs MACs 散点图。

    每个点用紧凑的架构标签标注，显示深度、最大通道数和最小卷积核大小。

    参数:
        results:   来自 NAS 搜索的 EvalResult 列表。
        save_path: 图像保存路径 (PNG)。
    """
    macs_vals = [r.macs for r in results]
    acc_vals = [r.accuracy * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制散点图，颜色用 viridis 渐变色映射准确率
    scatter = ax.scatter(
        macs_vals,
        acc_vals,
        c=acc_vals,
        cmap="viridis",
        s=80,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
    )

    # 为每个点添加紧凑标签
    for r in results:
        label = (
            f"D{r.arch.depth}_C{max(r.arch.out_channels)}_K{min(r.arch.kernel_sizes)}"
        )
        ax.annotate(
            label,
            (r.macs, r.accuracy * 100),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.7,
        )

    ax.set_xlabel("MACs (Multiply-Accumulate Operations)", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title(
        "NAS Random Search: Accuracy vs MACs Trade-off on CIFAR-10",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Accuracy (%)", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nAccuracy vs MACs plot saved to: {save_path}")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def format_macs(macs: int) -> str:
    """将 MACs 计数格式化为带人类可读后缀的字符串。

    参数:
        macs: 原始 MACs 整数。

    返回:
        类似 "12.34M" 的字符串。
    """
    if macs >= 1e9:
        return f"{macs / 1e9:.2f}G"
    if macs >= 1e6:
        return f"{macs / 1e6:.2f}M"
    if macs >= 1e3:
        return f"{macs / 1e3:.2f}K"
    return str(macs)


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


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def main() -> None:
    """运行完整的 NAS 随机搜索流程。"""
    # 设置随机种子以保证可复现性
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    print("=" * 72)
    print("  LECTURE 07: Simplified NAS Random Search on CIFAR-10")
    print("=" * 72)

    # ---- 1. 打印搜索空间 ----------------------------------------------------
    print(f"\n[1] Search space definition:")
    print(f"  Kernel sizes  : {KERNEL_SIZES}")
    print(f"  Channels      : {CHANNEL_CHOICES}")
    print(f"  Depths        : {DEPTHS}")
    # 计算搜索空间中所有可能的架构组合总数
    total_configs = sum(
        len(CHANNEL_CHOICES) ** d * len(KERNEL_SIZES) ** d for d in DEPTHS
    )
    print(f"  Total possible architectures: {total_configs}")
    print(f"  Random samples to evaluate : {NUM_SAMPLES}")
    print(f"  Training epochs per arch    : {NAS_EPOCHS}")

    # ---- 2. 加载数据 --------------------------------------------------------
    print(
        f"\n[2] Loading CIFAR-10 (train subset={TRAIN_SUBSET}, val subset={VAL_SUBSET}) ..."
    )
    train_loader, val_loader = get_cifar10_subset()
    print(f"  Train batches: {len(train_loader)},  Val batches: {len(val_loader)}")

    # ---- 3. 随机搜索 -------------------------------------------------------
    print(f"\n[3] Running random search ({NUM_SAMPLES} architectures) ...")
    print(
        f"     {'#':<4} {'Architecture':<35} {'Accuracy':>8} {'MACs':>10} {'Time':>8}"
    )
    print(f"     {'---':<4} {'---':<35} {'---':>8} {'---':>10} {'---':>8}")

    results: List[EvalResult] = []
    total_search_time = 0.0

    for i in range(NUM_SAMPLES):
        # 随机采样一个架构
        spec = random_sample_architecture(rng=rng)
        # 训练并评估该架构
        acc, train_time = train_and_evaluate(
            spec, train_loader, val_loader, epochs=NAS_EPOCHS
        )

        # 构建一个全新的模型用于 MACs 计数（避免任何副作用）
        macs_model = NasCNN(spec).to(DEVICE)
        macs = count_macs(macs_model, (3, 32, 32))

        # 记录评估结果
        result = EvalResult(arch=spec, accuracy=acc, macs=macs, train_time_s=train_time)
        results.append(result)
        total_search_time += train_time

        print(
            f"     {i + 1:>3d}  {arch_summary(spec):<35} "
            f"{acc * 100:>7.2f}% {format_macs(macs):>9}  {train_time:>6.1f}s"
        )

    print(
        f"  Total search time: {total_search_time:.1f}s ({total_search_time / 60:.1f} min)"
    )

    # ---- 4. 结果汇总 -------------------------------------------------------
    print(f"\n[4] Results summary ({len(results)} architectures):")
    accs = [r.accuracy * 100 for r in results]
    macs_list = [r.macs for r in results]
    print(
        f"  Accuracy: min={min(accs):.2f}%,  max={max(accs):.2f}%,  mean={np.mean(accs):.2f}%"
    )
    print(
        f"  MACs:     min={format_macs(min(macs_list))},  "
        f"max={format_macs(max(macs_list))},  "
        f"mean={format_macs(int(np.mean(macs_list)))}"
    )

    # 找出最高准确率和最低 MACs 的架构
    best_acc = max(results, key=lambda r: r.accuracy)
    print(
        f"\n  Best accuracy:   {arch_summary(best_acc.arch)} -> {best_acc.accuracy * 100:.2f}%"
    )
    lowest_macs = min(results, key=lambda r: r.macs)
    print(
        f"  Lowest MACs:     {arch_summary(lowest_macs.arch)} -> {format_macs(lowest_macs.macs)}"
    )

    # 简单帕累托前沿识别（非被支配架构）
    # 一个架构 A 支配 B，当且仅当 A 的准确率 >= B 且 MACs <= B，至少一个是严格不等
    pareto: List[EvalResult] = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            # 如果 other 在准确率上不差于 r 且在 MACs 上不多于 r
            if other.accuracy >= r.accuracy and other.macs <= r.macs:
                # 至少有一个严格占优
                if other.accuracy > r.accuracy or other.macs < r.macs:
                    dominated = True
                    break
        if not dominated:
            pareto.append(r)
    print(f"\n  Pareto-frontier architectures: {len(pareto)}")

    # ---- 5. 绘制准确率 vs MACs 图 ------------------------------------------
    print(f"\n[5] Plotting accuracy vs MACs trade-off ...")
    plot_accuracy_vs_macs(results, save_path=OUTPUT_PLOT)

    # ---- 6. 结束 -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(
        f"  Search space:    kernel={KERNEL_SIZES}, ch={CHANNEL_CHOICES}, depth={DEPTHS}"
    )
    print(f"  Total configs:   {total_configs}")
    print(f"  Sampled:         {NUM_SAMPLES}")
    print(
        f"  Training:        {NAS_EPOCHS} epochs CIFAR-10 subset ({TRAIN_SUBSET} samples)"
    )
    print(f"  Best accuracy:   {best_acc.accuracy * 100:.2f}%")
    print(f"  Lowest MACs:     {format_macs(lowest_macs.macs)}")
    print(f"  Pareto frontier: {len(pareto)} architectures")
    print(f"  Plot saved to:   {OUTPUT_PLOT}")
    print("=" * 72)

    print("\nLecture 07 complete.")


if __name__ == "__main__":
    main()
