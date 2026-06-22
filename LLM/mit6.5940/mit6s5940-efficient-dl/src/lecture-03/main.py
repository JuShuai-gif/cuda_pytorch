"""
Fine-Grained Magnitude-Based Pruning (Lecture 03)
==================================================
细粒度基于幅度的剪枝（第 03 讲）

Implements magnitude pruning and layer-wise sensitivity analysis for
understanding which layers can tolerate the most sparsity.
实现基于幅度的剪枝和逐层敏感性分析，以理解哪些层可以容忍最高的稀疏度。

Key concepts:
核心概念：
  - magnitude_prune: zeros out the smallest-magnitude weights
    magnitude_prune: 将幅度最小的权重置零
  - apply_pruning_to_model: prunes all Conv2d/Linear layers uniformly
    apply_pruning_to_model: 均匀地剪枝所有 Conv2d/Linear 层
  - sensitivity_scan: prunes each layer independently at multiple sparsity
    levels and measures the accuracy impact
    sensitivity_scan: 在多个稀疏度水平上独立剪枝每一层并测量准确率影响
  - plot_sensitivity: visualises the sensitivity curves with matplotlib
    plot_sensitivity: 使用 matplotlib 可视化敏感性曲线

All computations run on CPU; no GPU required.
所有计算在 CPU 上运行；不需要 GPU。
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Use a non-interactive backend so plots can be saved without a display
# 使用非交互式后端，以便在没有显示器的情况下保存图表
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# 常量定义
# Constants
# ---------------------------------------------------------------------------

SPARSITY_LEVELS: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]  # 要测试的稀疏度水平列表
NUM_CLASSES: int = 10  # 分类类别数
INPUT_CHANNELS: int = 3  # 输入通道数（RGB）
IMAGE_SIZE: int = 32  # 图像尺寸（32×32）
BATCH_SIZE: int = 64  # 训练批次大小
NUM_SAMPLES: int = 2000  # synthetic training set size：合成训练集大小
NUM_TEST: int = 500  # synthetic test set size：合成测试集大小
SEED: int = 42  # 随机种子，保证可复现性


# ===========================================================================
# 基于幅度的剪枝
# Magnitude Pruning
# ===========================================================================


def magnitude_prune(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Zero out the smallest-magnitude weights to achieve the target sparsity.
    将幅度最小的权重置零以达到目标稀疏度。

    The function computes a magnitude threshold at the given sparsity
    percentile and zeros out all weights whose absolute value falls
    below that threshold.
    该函数在给定稀疏度百分位数处计算幅度阈值，并将所有绝对值低于该阈值的权重置零。

    Args:
        weight:   A 2-D (Linear) or 4-D (Conv2d) weight tensor.
                  weight: 2-D（Linear）或 4-D（Conv2d）权重张量。
        sparsity: Target sparsity ratio in (0, 1).  0.5 means 50% of
                  weights are set to zero.
                  sparsity: 目标稀疏度比例，范围 (0, 1)。0.5 表示 50% 的权重被置零。

    Returns:
        A new tensor with the same shape as `weight`, where the smallest
        `sparsity * weight.numel()` values by absolute magnitude are
        replaced with 0.
        返回与 weight 形状相同的新张量，其中按绝对幅度最小的
        sparsity × weight.numel() 个值被替换为 0。

    Raises:
        ValueError: If sparsity is not in [0, 1].
        ValueError: 如果 sparsity 不在 [0, 1] 范围内。
    """
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError(f"sparsity must be in [0, 1]; got {sparsity}")

    # 若稀疏度为 0，直接返回原张量的克隆
    if sparsity == 0.0:
        return weight.clone()

    # 将权重展平并取绝对值
    flat = weight.abs().flatten()
    # 计算需要置零的元素个数 k
    k = max(1, int(sparsity * flat.numel()))

    # k-th smallest absolute value = the magnitude below which we prune
    # 第 k 小的绝对值 = 小于此幅度的权重将被剪枝
    # 使用 kthvalue 找到第 k 小的值作为阈值
    threshold = flat.kthvalue(k).values.item()

    # 创建掩码：保留绝对值 >= 阈值的权重
    mask = weight.abs() >= threshold
    # 应用掩码：将低于阈值的权重置零
    return weight * mask.float()


def _get_prunable_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return (name, module) pairs for all Conv2d and Linear layers.
    返回所有 Conv2d 和 Linear 层的 (名称, 模块) 对。

    Args:
        model: A PyTorch nn.Module.
               model: PyTorch nn.Module 实例。

    Returns:
        List of (name, module) tuples for prunable layers.
        返回可剪枝层的 (名称, 模块) 元组列表。
    """
    prunable: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        # 只收集 Conv2d 和 Linear 层（这些层有可剪枝的权重）
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable.append((name, module))
    return prunable


def apply_pruning_to_model(model: nn.Module, sparsity: float) -> nn.Module:
    """Apply uniform magnitude pruning to all Conv2d and Linear layers.
    对所有 Conv2d 和 Linear 层应用均匀的幅度剪枝。

    Each layer's weight tensor is replaced with a pruned version in-place.
    This is a **global-uniform** pruning strategy: every prunable layer
    receives the same sparsity level.
    每个层的权重张量被就地替换为剪枝后的版本。
    这是一种**全局均匀**剪枝策略：每个可剪枝层都获得相同的稀疏度水平。

    Args:
        model:    A PyTorch nn.Module.
                  model: PyTorch nn.Module 实例。
        sparsity: Target sparsity ratio in (0, 1).
                  sparsity: 目标稀疏度比例，范围 (0, 1)。

    Returns:
        The same model instance (modified in-place).
        返回修改后的同一模型实例（就地修改）。
    """
    for _name, module in _get_prunable_modules(model):
        # 对每个可剪枝层的权重进行剪枝
        pruned = magnitude_prune(module.weight.data, sparsity)
        # 将剪枝后的权重就地复制回原层
        module.weight.data.copy_(pruned)

    return model


def count_sparsity(model: nn.Module) -> Tuple[int, int, float]:
    """Count zero weights across all prunable layers.
    统计所有可剪枝层中零权重的数量。

    Args:
        model: A PyTorch nn.Module.
               model: PyTorch nn.Module 实例。

    Returns:
        A tuple of (total_params, zero_params, sparsity_ratio).
        返回 (总参数量, 零参数个数, 稀疏度比例) 的元组。
    """
    total = 0
    zeros = 0
    for _name, module in _get_prunable_modules(model):
        w = module.weight.data
        total += w.numel()
        zeros += (w == 0).sum().item()
    # 计算实际达到的稀疏度比例
    sparsity = zeros / total if total > 0 else 0.0
    return total, zeros, sparsity


# ===========================================================================
# 用于演示的简单 CNN
# Simple CNN for Demonstration
# ===========================================================================


class SimpleCNN(nn.Module):
    """A compact 4-conv-layer CNN suitable for quick pruning experiments.
    一个紧凑的 4 层卷积 CNN，适合快速剪枝实验。

    Architecture:
    架构：
        Conv2d(3, 16, 3, padding=1) -> BN -> ReLU
        Conv2d(16, 32, 3, stride=2, padding=1) -> BN -> ReLU
        Conv2d(32, 64, 3, padding=1) -> BN -> ReLU
        Conv2d(64, 128, 3, stride=2, padding=1) -> BN -> ReLU
        AdaptiveAvgPool2d(1) -> Flatten -> Linear(128, 10)
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        # 第一卷积块：3→16，保持尺寸
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二卷积块：16→32，尺寸减半（stride=2）
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        # 第三卷积块：32→64，保持尺寸
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        # 第四卷积块：64→128，尺寸减半（stride=2）
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)

        # 全局平均池化 + 分类器
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播：卷积块 → 池化 → 展平 → 分类器
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ===========================================================================
# 数据工具
# Data Utilities
# ===========================================================================


def _create_synthetic_dataset(
    num_samples: int,
    num_classes: int,
    channels: int,
    size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic images and random labels.
    生成合成图像和随机标签。

    Images are drawn from a normal distribution and labels are uniform
    random across num_classes.  This avoids external downloads and runs
    quickly on CPU.
    图像从正态分布中采样，标签在 num_classes 上均匀随机分布。
    这避免了外部下载，并且可以在 CPU 上快速运行。

    Args:
        num_samples: Number of samples to generate.
                     num_samples: 要生成的样本数。
        num_classes: Number of label classes.
                     num_classes: 标签类别数。
        channels:    Number of image channels.
                     channels: 图像通道数。
        size:        Spatial size (square).
                     size: 空间尺寸（方形）。

    Returns:
        Tuple of (images, labels).
        返回 (图像, 标签) 的元组。
    """
    # 从标准正态分布生成图像数据
    images = torch.randn(num_samples, channels, size, size)
    # 生成均匀随机的标签
    labels = torch.randint(0, num_classes, (num_samples,))
    return images, labels


# ===========================================================================
# 训练与评估
# Training & Evaluation
# ===========================================================================


def train_one_epoch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    lr: float = 0.01,
) -> float:
    """Train the model for one epoch on the given data.
    在给定数据上训练模型一个 epoch。

    Args:
        model:      A PyTorch nn.Module.
                    model: PyTorch nn.Module 实例。
        images:     Training images tensor (N, C, H, W).
                    images: 训练图像张量 (N, C, H, W)。
        labels:     Training labels tensor (N,).
                    labels: 训练标签张量 (N,)。
        batch_size: Batch size.
                    batch_size: 批次大小。
        lr:         Learning rate.
                    lr: 学习率。

    Returns:
        Average training loss over the epoch.
        返回该 epoch 的平均训练损失。
    """
    model.train()
    # 使用带动量的 SGD 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    n = images.size(0)
    # 随机打乱数据顺序
    perm = torch.randperm(n)
    total_loss = 0.0
    num_batches = 0

    # 按批次迭代训练
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        xb, yb = images[idx], labels[idx]

        # 标准训练步骤：清零梯度 → 前向 → 计算损失 → 反向 → 更新
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> float:
    """Evaluate top-1 accuracy on the given dataset.
    在给定数据集上评估 top-1 准确率。

    Args:
        model:      A PyTorch nn.Module.
                    model: PyTorch nn.Module 实例。
        images:     Image tensor (N, C, H, W).
                    images: 图像张量 (N, C, H, W)。
        labels:     Label tensor (N,).
                    labels: 标签张量 (N,)。
        batch_size: Batch size for evaluation.
                    batch_size: 评估时的批次大小。

    Returns:
        Accuracy as a float between 0.0 and 1.0.
        返回 0.0 到 1.0 之间的浮点数准确率。
    """
    model.eval()
    n = images.size(0)
    correct = 0

    # 按批次评估
    for i in range(0, n, batch_size):
        xb = images[i : i + batch_size]
        yb = labels[i : i + batch_size]
        logits = model(xb)
        preds = logits.argmax(dim=1)  # 取 logits 最大的类别作为预测
        correct += (preds == yb).sum().item()

    return correct / n


# ===========================================================================
# 敏感性分析
# Sensitivity Analysis
# ===========================================================================


def sensitivity_scan(
    model: nn.Module,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    sparsity_levels: List[float] | None = None,
) -> Dict[str, Dict[float, float]]:
    """Run layer-wise sensitivity analysis via iterative pruning.
    通过迭代剪枝运行逐层敏感性分析。

    For **each** prunable layer and **each** sparsity level, the function:
    对于**每个**可剪枝层和**每个**稀疏度水平，该函数：
      1. Saves a copy of the original weights.
         保存原始权重的副本。
      2. Prunes *only that layer* at the target sparsity.
         在目标稀疏度下剪枝*仅该层*。
      3. Evaluates accuracy on the test set.
         在测试集上评估准确率。
      4. Restores the original weights.
         恢复原始权重。

    This reveals which layers are most sensitive to pruning: layers whose
    accuracy drops sharply even at low sparsity are "bottleneck" layers.
    这揭示了哪些层对剪枝最敏感：即使在低稀疏度下准确率也急剧下降的层
    是"瓶颈"层。

    Args:
        model:           A trained PyTorch nn.Module.
                         model: 已训练的 PyTorch nn.Module 实例。
        test_images:     Test images tensor (N, C, H, W).
                         test_images: 测试图像张量 (N, C, H, W)。
        test_labels:     Test labels tensor (N,).
                         test_labels: 测试标签张量 (N,)。
        sparsity_levels: List of sparsity ratios to try.
                         Defaults to [0.1, 0.3, 0.5, 0.7, 0.9].
                         sparsity_levels: 要尝试的稀疏度比例列表。
                         默认为 [0.1, 0.3, 0.5, 0.7, 0.9]。

    Returns:
        Nested dict: {layer_name: {sparsity: accuracy, ...}, ...}
        返回嵌套字典：{层名称: {稀疏度: 准确率, ...}, ...}
    """
    if sparsity_levels is None:
        sparsity_levels = SPARSITY_LEVELS

    model.eval()

    # Baseline accuracy (unpruned)：未剪枝时的基线准确率
    baseline_acc = evaluate_accuracy(model, test_images, test_labels)
    print(f"\n  Baseline accuracy: {baseline_acc:.4f}")

    results: Dict[str, Dict[float, float]] = {}

    prunable = _get_prunable_modules(model)
    print(
        f"  Found {len(prunable)} prunable layers ({SPARSITY_LEVELS} sparsity levels each)"
    )
    print(f"  Total evaluations: {len(prunable) * len(SPARSITY_LEVELS)}\n")

    for layer_name, module in prunable:
        results[layer_name] = {}
        # 保存该层的原始权重，以便每次剪枝后恢复
        original_weight = module.weight.data.clone()

        for sp in sparsity_levels:
            # Prune only this layer：仅剪枝当前层
            pruned_w = magnitude_prune(original_weight, sp)
            module.weight.data.copy_(pruned_w)

            # 评估剪枝当前层后的准确率
            acc = evaluate_accuracy(model, test_images, test_labels)
            results[layer_name][sp] = acc

            print(f"  {layer_name:<30s}  sp={sp:.1f}  acc={acc:.4f}")

            # Restore original weight for the next iteration
            # 恢复原始权重，准备下一轮迭代
            module.weight.data.copy_(original_weight)

    return results


# ===========================================================================
# 绘图
# Plotting
# ===========================================================================


def plot_sensitivity(
    results: Dict[str, Dict[float, float]],
    baseline_acc: float,
    save_path: str = "sensitivity_curves.png",
) -> None:
    """Plot sensitivity curves: accuracy vs sparsity for each layer.
    绘制敏感性曲线：每层的准确率随稀疏度变化的曲线。

    Each curve shows how pruning a single layer at increasing sparsity
    levels affects overall model accuracy.  Layers whose curves drop
    steeply are the most sensitive to pruning.
    每条曲线展示了在递增的稀疏度水平下仅剪枝单个层如何影响整体模型准确率。
    曲线陡降的层对剪枝最敏感。

    Args:
        results:      Nested dict from sensitivity_scan().
                      results: 来自 sensitivity_scan() 的嵌套字典。
        baseline_acc: Accuracy of the unpruned model.
                      baseline_acc: 未剪枝模型的准确率。
        save_path:    File path to save the plot (PNG).
                      save_path: 保存图表的文件路径（PNG）。
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 为每一层绘制一条准确率 vs 稀疏度的曲线
    for layer_name, acc_dict in results.items():
        sparsities = sorted(acc_dict.keys())
        accuracies = [acc_dict[sp] for sp in sparsities]
        ax.plot(
            [s * 100 for s in sparsities],  # 稀疏度从比例转换为百分比
            [a * 100 for a in accuracies],  # 准确率从比例转换为百分比
            marker="o",
            linewidth=2,
            markersize=6,
            label=layer_name,
        )

    # Baseline horizontal line：基线准确率水平线
    ax.axhline(
        y=baseline_acc * 100,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"baseline ({baseline_acc * 100:.1f}%)",
    )

    ax.set_xlabel("Sparsity (%)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Layer-wise Sensitivity to Magnitude Pruning", fontsize=14)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)  # 关闭图形以释放内存
    print(f"\nSensitivity curves saved to: {save_path}")


# ===========================================================================
# 主函数
# Main
# ===========================================================================


def main() -> None:
    """Run the full pruning pipeline: train, prune, sensitivity scan, plot.
    运行完整的剪枝流程：训练、剪枝、敏感性扫描、绘图。"""
    torch.manual_seed(SEED)  # 设置随机种子以保证可复现性

    print("=" * 70)
    print("  LECTURE 03: Fine-Grained Magnitude-Based Pruning")
    print("=" * 70)

    # ---- 1. Create synthetic data ------------------------------------------
    # ---- 1. 创建合成数据 ---------------------------------------------------
    print("\n[1] Generating synthetic dataset ...")
    train_images, train_labels = _create_synthetic_dataset(
        NUM_SAMPLES, NUM_CLASSES, INPUT_CHANNELS, IMAGE_SIZE
    )
    test_images, test_labels = _create_synthetic_dataset(
        NUM_TEST, NUM_CLASSES, INPUT_CHANNELS, IMAGE_SIZE
    )
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")

    # ---- 2. Build and train model ------------------------------------------
    # ---- 2. 构建并训练模型 -------------------------------------------------
    print("\n[2] Building SimpleCNN and training ...")
    model = SimpleCNN(num_classes=NUM_CLASSES)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Quick training on synthetic data：在合成数据上快速训练
    for epoch in range(1, 11):
        loss = train_one_epoch(model, train_images, train_labels, BATCH_SIZE)
        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2d}  loss={loss:.4f}")

    baseline_acc = evaluate_accuracy(model, test_images, test_labels)
    print(f"\n  Baseline test accuracy: {baseline_acc:.4f}")

    # ---- 3. Sanity check: magnitude_prune on a single tensor ---------------
    # ---- 3. 健全性检查：在单个张量上测试 magnitude_prune -------------------
    print("\n[3] Testing magnitude_prune on a sample tensor ...")
    sample_w = torch.tensor([0.5, -0.1, 0.8, -0.3, 0.02, -0.9, 0.0, 0.15])
    pruned_w = magnitude_prune(sample_w, sparsity=0.5)
    num_zeros = (pruned_w == 0).sum().item()
    print(f"  Original: {sample_w.tolist()}")
    print(f"  Pruned (50%): {pruned_w.tolist()}")
    print(f"  Zeros: {num_zeros} / {sample_w.numel()}")

    # ---- 4. Apply global uniform pruning -----------------------------------
    # ---- 4. 应用全局均匀剪枝 -----------------------------------------------
    print("\n[4] Applying uniform pruning (sparsity=0.5) to entire model ...")
    model.eval()
    apply_pruning_to_model(model, sparsity=0.5)
    total_p, zero_p, achieved_sp = count_sparsity(model)
    pruned_acc = evaluate_accuracy(model, test_images, test_labels)
    print(
        f"  Prunable weights: {total_p:,}  |  zeros: {zero_p:,}  "
        f"|  achieved sparsity: {achieved_sp:.4f}"
    )
    print(f"  Accuracy after 50% uniform pruning: {pruned_acc:.4f}")

    # ---- 5. Sensitivity scan (requires fresh model) ------------------------
    # ---- 5. 敏感性扫描（需要全新的模型）-------------------------------------
    print("\n[5] Sensitivity scan: pruning each layer independently ...")
    model2 = SimpleCNN(num_classes=NUM_CLASSES)
    model2.eval()
    # Train the fresh model：训练新模型
    for epoch in range(1, 11):
        train_one_epoch(model2, train_images, train_labels, BATCH_SIZE)
    baseline_acc2 = evaluate_accuracy(model2, test_images, test_labels)
    print(f"  Fresh model baseline accuracy: {baseline_acc2:.4f}")

    sensitivity_results = sensitivity_scan(
        model2, test_images, test_labels, SPARSITY_LEVELS
    )

    # ---- 6. Plot sensitivity curves ----------------------------------------
    # ---- 6. 绘制敏感性曲线 -------------------------------------------------
    print("\n[6] Plotting sensitivity curves ...")
    plot_sensitivity(
        sensitivity_results,
        baseline_acc2,
        save_path="sensitivity_curves.png",
    )

    # ---- 7. Summary ---------------------------------------------------------
    # ---- 7. 总结 -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Model: SimpleCNN ({total_params:,} parameters)")
    print(f"  Synthetic data: {NUM_SAMPLES} train / {NUM_TEST} test samples")
    print(f"  Baseline accuracy: {baseline_acc2:.4f}")
    print(f"  Uniform pruning (50%): accuracy = {pruned_acc:.4f}")
    print(
        f"  Sensitivity scan: {len(prunable_layers := _get_prunable_modules(model2))} layers "
        f"x {len(SPARSITY_LEVELS)} levels"
    )
    print("=" * 70)

    print("\nLecture 03 complete.")


if __name__ == "__main__":
    main()
