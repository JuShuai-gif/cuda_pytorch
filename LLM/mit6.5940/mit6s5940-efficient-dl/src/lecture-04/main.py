"""
基于 Frobenius 范数的通道剪枝（第 04 讲）
============================================

通过对 Conv2d 层的输出通道按 Frobenius（L2）范数进行重要性排序，
移除重要性最低的通道，实现结构化通道剪枝。

核心概念：
  - frobenius_importance: 通过 ||W_i||_F 对 Conv2d 输出通道进行排序
  - channel_prune: 仅保留 top-k 通道，构建更小的模型
  - fine_tune: 对剪枝后的模型进行 5 个 epoch 的微调以恢复精度
  - compare_metrics: 对比剪枝前后的准确率、参数量、MACs 和延迟

所有计算均在 CPU 上运行；无需 GPU。
"""

from __future__ import annotations

import time
from typing import List, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 10  # 分类类别数
INPUT_CHANNELS: int = 3  # 输入图像通道数
IMAGE_SIZE: int = 32  # 输入图像的空间尺寸
BATCH_SIZE: int = 64  # 小批量大小
NUM_TRAIN: int = 2000  # 合成训练样本数
NUM_TEST: int = 500  # 合成测试样本数
PRUNE_RATIO: float = 0.3  # 每层需要剪枝的通道比例
FINE_TUNE_EPOCHS: int = 5  # 微调 epoch 数
INITIAL_EPOCHS: int = 10  # 初始训练 epoch 数
LR: float = 0.01  # 学习率
WARMUP_RUNS: int = 10  # 延迟测量的预热迭代次数
TIMED_RUNS: int = 100  # 延迟测量的计时迭代次数
SEED: int = 42  # 随机种子，保证可复现性
SAVE_PATH: str = "pruned_model.pth"  # 剪枝模型保存路径


# ===========================================================================
# 模型定义
# ===========================================================================


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU 基础构建模块。

    参数:
        in_c:  输入通道数。
        out_c: 输出通道数。
        stride: Conv2d 步长（默认为 1）。
    """

    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DemoCNN(nn.Module):
    """用于通道剪枝实验的小型 CNN，包含 4 个 ConvBlock 层。

    网络结构:
        ConvBlock(3,   64,  stride=1)
        ConvBlock(64,  128, stride=2)
        ConvBlock(128, 256, stride=1)
        ConvBlock(256, 256, stride=2)
        AdaptiveAvgPool2d(1) -> Flatten -> Linear(256, 10)

    参数:
        num_classes: 输出类别数（默认为 10）。
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.block1 = ConvBlock(3, 64, stride=1)
        self.block2 = ConvBlock(64, 128, stride=2)
        self.block3 = ConvBlock(128, 256, stride=1)
        self.block4 = ConvBlock(256, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @property
    def blocks(self) -> List[ConvBlock]:
        """按前向顺序返回所有 ConvBlock 模块。"""
        return [self.block1, self.block2, self.block3, self.block4]


class PrunedCNN(nn.Module):
    """根据剪枝后的通道配置列表构建的 CNN。

    该类接收预先计算好的通道数，以便用精确的架构实例化剪枝后的模型。

    参数:
        channels:    每个 ConvBlock 的 (in_c, out_c, stride) 元组列表。
        num_classes: 输出类别数。
    """

    def __init__(
        self, channels: List[Tuple[int, int, int]], num_classes: int = NUM_CLASSES
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for in_c, out_c, stride in channels:
            self.blocks.append(ConvBlock(in_c, out_c, stride))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 最后一层的输出通道数作为分类器的输入维度
        final_out = channels[-1][1] if channels else 0
        self.classifier = nn.Linear(final_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ===========================================================================
# 数据工具函数
# ===========================================================================


def _create_synthetic_data(
    n: int, c: int, h: int, w: int, num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成合成图像和随机标签。

    参数:
        n:           样本数量。
        c:           通道数。
        h, w:        空间维度。
        num_classes: 标签类别数。

    返回:
        (images, labels) 元组。
    """
    images = torch.randn(n, c, h, w)
    labels = torch.randint(0, num_classes, (n,))
    return images, labels


# ===========================================================================
# 训练与评估
# ===========================================================================


def train_one_epoch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> float:
    """训练模型一个 epoch。

    参数:
        model:      PyTorch nn.Module 模型。
        images:     训练图像 (N, C, H, W)。
        labels:     训练标签 (N,)。
        batch_size: 小批量大小。
        lr:         学习率。

    返回:
        该 epoch 的平均训练损失。
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    n = images.size(0)
    # 随机打乱样本顺序
    perm = torch.randperm(n)
    total_loss = 0.0
    num_batches = 0

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        xb, yb = images[idx], labels[idx]

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
    """评估 top-1 准确率。

    参数:
        model:      PyTorch nn.Module 模型。
        images:     图像张量 (N, C, H, W)。
        labels:     标签张量 (N,)。
        batch_size: 评估批次大小。

    返回:
        [0.0, 1.0] 范围内的准确率浮点数。
    """
    model.eval()
    n = images.size(0)
    correct = 0

    for i in range(0, n, batch_size):
        xb = images[i : i + batch_size]
        yb = labels[i : i + batch_size]
        logits = model(xb)
        # 取 logits 中最大值对应的索引作为预测类别
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()

    return correct / n


# ===========================================================================
# Frobenius 范数重要性排序
# ===========================================================================


def frobenius_importance(weight: torch.Tensor) -> torch.Tensor:
    """通过 Frobenius 范数计算每个输出通道的重要性得分。

    对于形状为 [C_out, C_in, K, K] 的 Conv2d 权重张量，每个输出通道（滤波器）
    的 Frobenius 范数为：

        ||W[i, :, :, :]||_F = sqrt( sum( W[i, :, :, :] ** 2 ) )

    Frobenius 范数较小的通道对输出的贡献较小，因此是剪枝的候选对象。

    参数:
        weight: 形状为 (C_out, C_in, K, K) 的 Conv2d 权重张量。

    返回:
        形状为 (C_out,) 的一维张量，包含每个输出通道的重要性得分。
    """
    c_out = weight.size(0)
    # 将除第一维外的所有维度展平，然后计算 L2 范数
    return weight.view(c_out, -1).norm(p=2, dim=1)


def select_top_channels(importance: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    """根据重要性得分选择要保留的输出通道。

    参数:
        importance:  形状为 (C_out,) 的重要性得分。
        prune_ratio: 要剪枝的通道比例（0.0 到 1.0）。

    返回:
        要保留的通道索引（已排序），形状为 (C_out * (1-prune_ratio),)。
    """
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError(f"prune_ratio 必须在 [0, 1) 范围内；当前值为 {prune_ratio}")

    num_channels = importance.size(0)
    # 至少保留 1 个通道
    num_keep = max(1, int(num_channels * (1.0 - prune_ratio)))
    # 使用 topk 选取重要性最高的 num_keep 个通道
    _, top_indices = torch.topk(importance, num_keep)
    # 返回排序后的索引，保持通道原有顺序
    return torch.sort(top_indices).values


# ===========================================================================
# 通道剪枝
# ===========================================================================


def channel_prune(original: DemoCNN, prune_ratio: float) -> PrunedCNN:
    """使用 Frobenius 范数排序对每个 ConvBlock 进行通道剪枝。

    算法流程:
      1. 对每个 ConvBlock，计算每个输出通道的 Frobenius 范数。
      2. 按重要性对通道排序，保留 top (1 - prune_ratio) 的通道。
      3. 使用缩减后的通道数构建新的 PrunedCNN。
      4. 将原始模型中保留通道的权重复制到新模型中。

    由于剪枝第 i 层的输出通道会减少第 i+1 层的输入通道数，第 i 层的保留输入
    索引就是第 i-1 层的保留输出索引。第一层始终保留全部 3 个输入通道。

    参数:
        original:    已训练的 DemoCNN 模型。
        prune_ratio: 每个 Block 要移除的输出通道比例。

    返回:
        通道数减少且权重已复制的新 PrunedCNN 实例。
    """
    original.eval()

    # ---- 步骤 1: 确定每个 Block 要保留的输出通道 ----------------------------
    kept_outputs: List[torch.Tensor] = []
    for block in original.blocks:
        imp = frobenius_importance(block.conv.weight.data)
        kept = select_top_channels(imp, prune_ratio)
        kept_outputs.append(kept)

    # ---- 步骤 2: 构建新的通道配置 --------------------------------------------
    new_channels: List[Tuple[int, int, int]] = []
    # block1 保留全部输入通道
    prev_kept_out = torch.arange(INPUT_CHANNELS)

    for i, block in enumerate(original.blocks):
        in_c = prev_kept_out.size(0)  # 当前层的输入通道数
        out_c = kept_outputs[i].size(0)  # 当前层的输出通道数
        stride = block.conv.stride[0]
        new_channels.append((in_c, out_c, stride))
        prev_kept_out = kept_outputs[i]  # 当前层的保留输出作为下一层的输入

    # ---- 步骤 3: 实例化剪枝后的模型 ------------------------------------------
    pruned = PrunedCNN(new_channels, num_classes=original.classifier.out_features)

    # ---- 步骤 4: 复制权重 ---------------------------------------------------
    prev_kept_out = torch.arange(INPUT_CHANNELS)

    for i, (orig_block, kept_out) in enumerate(zip(original.blocks, kept_outputs)):
        new_block = pruned.blocks[i]

        # 复制 Conv2d 权重：选择保留的输出通道和对应的输入通道
        new_block.conv.weight.data.copy_(
            orig_block.conv.weight.data[kept_out][:, prev_kept_out]
        )

        # 为保留的输出通道复制 BatchNorm 参数
        new_block.bn.weight.data.copy_(orig_block.bn.weight.data[kept_out])
        new_block.bn.bias.data.copy_(orig_block.bn.bias.data[kept_out])
        new_block.bn.running_mean.data.copy_(orig_block.bn.running_mean.data[kept_out])
        new_block.bn.running_var.data.copy_(orig_block.bn.running_var.data[kept_out])

        prev_kept_out = kept_out

    # 复制分类器：输入维度对应最后一个 Block 的保留输出通道
    pruned.classifier.weight.data.copy_(
        original.classifier.weight.data[:, prev_kept_out]
    )
    pruned.classifier.bias.data.copy_(original.classifier.bias.data)

    return pruned


# ===========================================================================
# 微调
# ===========================================================================


def fine_tune(
    model: nn.Module,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int = FINE_TUNE_EPOCHS,
    lr: float = LR,
) -> None:
    """对剪枝后的模型进行微调以恢复精度。

    参数:
        model:        要微调的剪枝模型（原地修改）。
        train_images: 训练图像。
        train_labels: 训练标签。
        test_images:  用于跟踪精度的测试图像。
        test_labels:  用于跟踪精度的测试标签。
        epochs:       微调 epoch 数。
        lr:           学习率。
    """
    print(f"\n  正在进行 {epochs} 个 epoch 的微调 (lr={lr}) ...")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_images, train_labels, lr=lr)
        acc = evaluate_accuracy(model, test_images, test_labels)
        print(f"    Epoch {epoch:>2d}/{epochs}  loss={loss:.4f}  acc={acc:.4f}")


# ===========================================================================
# 评估指标
# ===========================================================================


def count_params(model: nn.Module) -> int:
    """统计模型的总参数数量。

    参数:
        model: PyTorch nn.Module 模型。

    返回:
        参数总数（包括可训练和不可训练参数）。
    """
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model: nn.Module, input_shape: Tuple[int, int, int]) -> int:
    """通过注册前向钩子来估算 Conv2d 层的总 MACs（乘加运算次数）。

    仅统计 Conv2d 层；BatchNorm、池化层和全连接层的计算量相对于卷积层
    可忽略不计。

    参数:
        model:       PyTorch nn.Module 模型。
        input_shape: 单个输入样本的 (C, H, W)。

    返回:
        所有 Conv2d 层的估算总 MACs。
    """
    model.eval()
    total_macs = 0
    dummy = torch.randn(1, *input_shape)

    def _hook(
        module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor, /
    ) -> None:
        """前向钩子：在每次卷积前向时累加 MACs。"""
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            x = inp[0]  # (1, C_in, H_in, W_in)
            c_in = x.shape[1]
            h_in = x.shape[2]
            w_in = x.shape[3]
            h_out = out.shape[2]
            w_out = out.shape[3]
            # MACs = C_out * H_out * W_out * C_in * K * K
            macs = (
                module.out_channels
                * h_out
                * w_out
                * c_in
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
            total_macs += macs

    # 为所有 Conv2d 模块注册钩子
    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(_hook))

    # 进行一次虚拟前向传播来触发钩子
    with torch.no_grad():
        _ = model(dummy)

    # 清理钩子
    for h in handles:
        h.remove()

    return total_macs


def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    warmup: int = WARMUP_RUNS,
    repeats: int = TIMED_RUNS,
) -> float:
    """测量在 CPU 上的平均前向传播延迟。

    参数:
        model:       PyTorch nn.Module 模型。
        input_shape: 单个输入样本的 (C, H, W)。
        warmup:      不计时的预热迭代次数。
        repeats:     计时迭代次数。

    返回:
        每次推理的平均延迟，单位为毫秒。
    """
    model.eval()
    dummy = torch.randn(1, *input_shape)

    # 预热：让 CPU 缓存和频率进入稳定状态
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # 正式计时
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy)
    end = time.perf_counter()

    # 返回平均每次推理的毫秒数
    return (end - start) / repeats * 1000.0


def print_comparison(
    label: str,
    accuracy: float,
    params: int,
    macs: int,
    latency_ms: float,
) -> None:
    """打印比较表中的一行数据。

    参数:
        label:      模型描述（例如 "Original" 或 "Pruned + FT"）。
        accuracy:   Top-1 准确率。
        params:     参数数量。
        macs:       Conv2d 总 MACs。
        latency_ms: 推理延迟，单位为毫秒。
    """
    print(
        f"  {label:<20}  acc={accuracy:.4f}  "
        f"params={params:>9,}  MACs={macs:>12,}  latency={latency_ms:.3f} ms"
    )


# ===========================================================================
# 主流程
# ===========================================================================


def main() -> None:
    """运行完整的通道剪枝流水线。"""
    torch.manual_seed(SEED)

    print("=" * 70)
    print("  第 04 讲: 基于 Frobenius 范数的通道剪枝")
    print("=" * 70)

    # ---- 1. 创建合成数据 -----------------------------------------------
    print("\n[1] 正在生成合成数据集 ...")
    train_images, train_labels = _create_synthetic_data(
        NUM_TRAIN, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES
    )
    test_images, test_labels = _create_synthetic_data(
        NUM_TEST, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES
    )
    print(f"  训练集: {train_images.shape}, 测试集: {test_images.shape}")

    # ---- 2. 构建并训练原始模型 -----------------------------------------
    print(f"\n[2] 正在构建 DemoCNN 并训练 {INITIAL_EPOCHS} 个 epoch ...")
    model = DemoCNN(num_classes=NUM_CLASSES)

    for epoch in range(1, INITIAL_EPOCHS + 1):
        loss = train_one_epoch(model, train_images, train_labels)
        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2d}  loss={loss:.4f}")

    original_acc = evaluate_accuracy(model, test_images, test_labels)
    print(f"  原始模型准确率: {original_acc:.4f}")

    # ---- 3. Frobenius 范数正确性检查 -----------------------------------
    print("\n[3] Frobenius 范数重要性排序（正确性检查） ...")
    sample_conv = model.block1.conv.weight.data  # [64, 3, 3, 3]
    importance = frobenius_importance(sample_conv)
    print(f"  block1.conv 权重形状: {tuple(sample_conv.shape)}")
    print(
        f"  重要性得分（前 8 个 / 共 {importance.size(0)} 个）: "
        f"{importance[:8].tolist()}"
    )
    print(f"  Top-5 通道索引: {torch.topk(importance, 5).indices.tolist()}")

    # ---- 4. 通道剪枝 --------------------------------------------------
    print(f"\n[4] 正在对每层剪枝 {PRUNE_RATIO * 100:.0f}% 的通道 ...")
    for i, block in enumerate(model.blocks):
        out_c = block.conv.out_channels
        keep = max(1, int(out_c * (1.0 - PRUNE_RATIO)))
        pruned_c = out_c - keep
        print(f"  block{i + 1}: {out_c} -> {keep} 个输出通道 (剪枝 {pruned_c} 个)")

    pruned_model = channel_prune(model, PRUNE_RATIO)
    pruned_params_before_ft = count_params(pruned_model)
    print(
        f"  剪枝后模型参数: {pruned_params_before_ft:,} "
        f"(原始为 {count_params(model):,})"
    )

    # ---- 5. 微调剪枝后的模型 ------------------------------------------
    print(f"\n[5] 正在微调剪枝模型 ({FINE_TUNE_EPOCHS} 个 epoch) ...")
    fine_tune(pruned_model, train_images, train_labels, test_images, test_labels)

    # ---- 6. 对比指标 ------------------------------------------------
    print("\n[6] 对比: 原始 vs 剪枝（+ 微调）")
    input_shape = (INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    # 原始模型指标
    orig_params = count_params(model)
    orig_macs = estimate_macs(model, input_shape)
    orig_latency = measure_latency(model, input_shape)
    orig_accuracy = original_acc

    # 剪枝模型指标
    pruned_accuracy = evaluate_accuracy(pruned_model, test_images, test_labels)
    pruned_params = count_params(pruned_model)
    pruned_macs = estimate_macs(pruned_model, input_shape)
    pruned_latency = measure_latency(pruned_model, input_shape)

    print(
        f"\n  {'':<20}  {'Accuracy':>8}  {'Params':>10}  {'MACs':>13}  {'Latency':>12}"
    )
    print(f"  {'':->20}  {'':->8}  {'':->10}  {'':->13}  {'':->12}")
    print_comparison("Original", orig_accuracy, orig_params, orig_macs, orig_latency)
    print_comparison(
        "Pruned + FT", pruned_accuracy, pruned_params, pruned_macs, pruned_latency
    )

    # ---- 7. 缩减统计摘要 -----------------------------------------------
    print(f"\n  缩减摘要 (prune_ratio={PRUNE_RATIO}):")
    print(
        f"    准确率:  {orig_accuracy:.4f} -> {pruned_accuracy:.4f}  "
        f"({(pruned_accuracy - orig_accuracy) * 100:+.2f}%)"
    )
    print(
        f"    参数量:  {orig_params:,} -> {pruned_params:,}  "
        f"({(1 - pruned_params / orig_params) * 100:.1f}% 缩减)"
    )
    print(
        f"    MACs:    {orig_macs:,} -> {pruned_macs:,}  "
        f"({(1 - pruned_macs / orig_macs) * 100:.1f}% 缩减)"
    )
    print(
        f"    延迟:    {orig_latency:.3f} -> {pruned_latency:.3f} ms  "
        f"({(1 - pruned_latency / orig_latency) * 100:.1f}% 缩减)"
    )

    # ---- 8. 保存剪枝模型 ------------------------------------------------
    print(f"\n[7] 正在将剪枝模型保存到 '{SAVE_PATH}' ...")
    torch.save(pruned_model.state_dict(), SAVE_PATH)
    import os

    file_size_kb = os.path.getsize(SAVE_PATH) / 1024
    print(f"  已保存: {SAVE_PATH} ({file_size_kb:.1f} KiB)")

    # ---- 9. 完成 -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)
    print(f"  模型: DemoCNN (4 个 ConvBlock)")
    print(f"  合成数据: {NUM_TRAIN} 训练 / {NUM_TEST} 测试")
    print(f"  剪枝比例: {PRUNE_RATIO} (逐层通道剪枝)")
    print(f"  方法: Frobenius 范数重要性排序")
    print(f"  原始参数量: {orig_params:,}  |  剪枝后参数量: {pruned_params:,}")
    print(f"  原始准确率: {orig_accuracy:.4f}  |  剪枝后准确率: {pruned_accuracy:.4f}")
    print(f"  剪枝模型已保存至: {SAVE_PATH}")
    print("=" * 70)

    print("\n第 04 讲完成。")


if __name__ == "__main__":
    main()
