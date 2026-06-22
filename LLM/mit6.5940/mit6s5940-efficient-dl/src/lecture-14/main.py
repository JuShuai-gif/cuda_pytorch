#!/usr/bin/env python3
"""
第 14 讲：参数高效微调（PEFT）—— 从零实现 LoRA
======================================================

本脚本演示了用于高效微调大型预训练模型的低秩适应（LoRA）方法。
内容包括：

  1. 自定义 ``LoRALinear`` 层，用两个低秩矩阵 A 和 B
     包装一个冻结的 ``nn.Linear`` 权重。
  2. 在 MNIST 上预训练一个小型 MLP。
  3. 对选定层应用 LoRA，并在 MNIST 子集上进行微调。
  4. 比较全量微调和不同秩下 LoRA 的可训练参数数量。
  5. 将 LoRA 权重合并回原始权重。

所有计算在 CPU 上运行 —— 无需 CUDA。
"""

import copy
import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# 1. 自定义 LoRA 线性层
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """
    一个用低秩适应（LoRA）增强的线性层。

    原始权重 ``W``（形状：``out_features x in_features``）被冻结
    （``requires_grad = False``）。引入两个可训练的低秩矩阵
    ``A``（``r x in_features``）和 ``B``（``out_features x r``），
    使得有效的前向传播变为：

        h = x @ W^T + (x @ A^T @ B^T) * (alpha / r)

    其中：
        - ``r``      ：低秩分解的秩，
        - ``alpha``  ：缩放因子，控制 LoRA 更新相对于冻结权重的幅度。

    初始化：
        ``A`` 使用 ``kaiming_uniform`` 初始化以保证稳定的梯度流；
        ``B`` 被初始化为**零**，使得 LoRA 分支最初不贡献任何内容
        （``delta_W = 0``）。

    合并 / 取消合并：
        ``merge()`` 将 LoRA 权重折叠到原始权重中：
            ``W_merged = W + (alpha / r) * (B @ A)``
        合并后，可以使用标准的线性前向传播绕过 LoRA 分支。
        ``unmerge()`` 反转该操作。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # --- 冻结的原始权重 ---
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # 立即冻结基础权重和偏置，使其在前向传播中不可训练。
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = alpha / r  # 缓存用于前向传播的缩放因子

        # --- 低秩矩阵 ---
        # A：(r, in_features)  -- 将输入投影到秩 r
        # B：(out_features, r) -- 将秩 r 的表示投影回高维
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))  # 零初始化
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # 跟踪 LoRA 权重是否已合并到原始权重中。
        self._merged = False
        # 备份原始权重用于取消合并。
        self._orig_weight: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 ``x @ W^T + (x @ A^T @ B^T) * (alpha / r)``。

        当已合并时，退回到普通的线性前向传播。
        """
        if self._merged:
            return self.linear(x)

        # 冻结的基础输出：x @ W^T  （batch, out_features）
        base = self.linear(x)

        # LoRA 增量部分：(x @ A^T) @ B^T  （batch, out_features）
        lora = (x @ self.A.T) @ self.B.T

        return base + lora * self.scaling

    # -- 合并 / 取消合并实用方法 ----------------------------------------

    def merge(self) -> None:
        """
        将 LoRA 权重折叠到冻结权重中：

            W := W + (alpha / r) * (B @ A)

        调用此方法后，LoRA 矩阵在推理时不再需要，该层表现为标准的
        ``nn.Linear``。
        """
        if self._merged:
            return  # 已经合并，无需重复操作

        delta = (self.B @ self.A) * self.scaling  # (out_features, in_features)
        self._orig_weight = self.linear.weight.data.clone()  # 备份原始权重
        self.linear.weight.data.add_(delta)  # 将 LoRA 增量加回
        self._merged = True

    def unmerge(self) -> None:
        """恢复原始权重（撤销 ``merge()``）。"""
        if not self._merged or self._orig_weight is None:
            return

        self.linear.weight.data.copy_(self._orig_weight)
        self._orig_weight = None
        self._merged = False

    # -- 便捷属性 -------------------------------------------

    @property
    def trainable_params(self) -> int:
        """A 和 B 中可训练参数的数量。"""
        return self.A.numel() + self.B.numel()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, merged={self._merged}"
        )


# ---------------------------------------------------------------------------
# 2. MLP 模型（简单分类器）
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """
    用于 MNIST 数字分类的三层 MLP。

        第 1 层：784 -> 256  + ReLU
        第 2 层：256 -> 128  + ReLU
        第 3 层：128 -> 10   （logits）
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # 展平为 (batch, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# 3. 数据辅助函数
# ---------------------------------------------------------------------------


def get_mnist_loaders(
    batch_size: int = 64,
    subset_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    返回 MNIST 的训练和测试 DataLoader。

    如果提供了 ``subset_size``，训练数据集将被裁剪到该数量的样本
    （适用于在微型数据集上演示 LoRA 微调）。
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # 转换为张量并缩放到 [0, 1]
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 标准化
        ]
    )

    train_ds = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    if subset_size is not None:
        # 随机采样子集，固定种子以确保可重复性
        indices = np.random.default_rng(42).choice(
            len(train_ds),
            size=min(subset_size, len(train_ds)),
            replace=False,
        )
        train_ds = Subset(train_ds, indices.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 4. 训练 & 评估循环
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练模型一个 epoch。返回平均损失。"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()  # 清零梯度
        loss = F.cross_entropy(model(x), y)  # 交叉熵损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """返回给定数据加载器上的分类准确率。"""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # 取 argmax 获取预测类别，与真实标签比较
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


def count_trainable_params(model: nn.Module) -> int:
    """统计可训练参数的数量（requires_grad=True）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 5. LoRA 应用辅助函数
# ---------------------------------------------------------------------------


def apply_lora_to_model(
    base_model: nn.Module,
    r: int,
    alpha: float,
    target_layers: List[str],
) -> nn.Module:
    """
    将 ``target_layers`` 中命名的线性层替换为共享原始冻结权重的
    ``LoRALinear`` 包装层。

    参数
    ----------
    base_model : nn.Module
        预训练模型（权重将在过程中被冻结）。
    r : int
        LoRA 秩。
    alpha : float
        LoRA 缩放因子。
    target_layers : List[str]
        要增强的 ``nn.Linear`` 层的属性名（例如 ``["fc1", "fc2"]``）。

    返回
    -------
    nn.Module
        已安装 LoRA 层的相同模型。
    """
    for name in target_layers:
        original: nn.Linear = getattr(base_model, name)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"{name} 不是 nn.Linear 类型")

        # 创建包装原始权重的 LoRA 线性层
        lora = LoRALinear(
            in_features=original.in_features,
            out_features=original.out_features,
            r=r,
            alpha=alpha,
            bias=original.bias is not None,
        )

        # 将预训练权重（和偏置）复制到 LoRA 包装层中。
        with torch.no_grad():
            lora.linear.weight.copy_(original.weight)
            if original.bias is not None:
                lora.linear.bias.copy_(original.bias)

        setattr(base_model, name, lora)

    return base_model


# ---------------------------------------------------------------------------
# 6. 主演示
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # 6.0 环境设置
    # ------------------------------------------------------------------
    device = torch.device("cpu")
    print("设备：", device)
    print()

    # ------------------------------------------------------------------
    # 6.1 在完整 MNIST 上预训练一个简单 MLP
    # ------------------------------------------------------------------
    print("=" * 65)
    print("阶段 1：在完整 MNIST 数据集上预训练 MLP")
    print("=" * 65)

    pretrain_loader, full_test_loader = get_mnist_loaders(batch_size=128)

    model = SimpleMLP().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数（总计）：{total_params:,}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pretrain_epochs = 3

    for epoch in range(1, pretrain_epochs + 1):
        t0 = time.perf_counter()
        avg_loss = train_one_epoch(model, pretrain_loader, optimizer, device)
        acc = evaluate(model, full_test_loader, device)
        elapsed = time.perf_counter() - t0
        print(
            f"  Epoch {epoch}/{pretrain_epochs} | loss={avg_loss:.4f} "
            f"| test acc={acc:.4f} | {elapsed:.1f}s"
        )

    pretrain_acc = evaluate(model, full_test_loader, device)
    print(f"\n预训练最终准确率：{pretrain_acc:.4f}")
    print()

    # 保存预训练权重的冻结副本以备后续重用。
    base_state = copy.deepcopy(model.state_dict())

    # ------------------------------------------------------------------
    # 6.2 显示全量微调的可训练参数数量
    # ------------------------------------------------------------------
    full_ft_params = total_params  # 全量微调时每个权重都会被更新
    print(f"全量微调将训练：{full_ft_params:,} 个参数")
    print()

    # ------------------------------------------------------------------
    # 6.3 在不同秩下进行 LoRA 微调
    # ------------------------------------------------------------------
    print("=" * 65)
    print("阶段 2：在小型 MNIST 子集（2048 个样本）上进行 LoRA 微调")
    print("=" * 65)

    ranks = [2, 4, 8, 16]  # 测试不同的 LoRA 秩
    lora_alphas = {2: 2.0, 4: 4.0, 8: 8.0, 16: 16.0}  # alpha 通常设为与 rank 相等
    subset_size = 2048  # 用于微调的子集大小
    lora_epochs = 5
    lora_lr = 5e-4

    results: List[Tuple[int, int, float]] = []  # (秩, 可训练参数数, 准确率)

    for r in ranks:
        print(f"\n--- LoRA 秩 r = {r} ---")

        # 每次重新加载预训练基础模型，从头开始。
        model = SimpleMLP().to(device)
        model.load_state_dict(base_state)

        # 对前两个线性层应用 LoRA。
        alpha = lora_alphas[r]
        apply_lora_to_model(model, r=r, alpha=alpha, target_layers=["fc1", "fc2"])

        # 统计可训练参数（仅 A 和 B 矩阵）。
        trainable = count_trainable_params(model)
        frozen = total_params - trainable
        print(
            f"  可训练参数：{trainable:,} / {total_params:,} "
            f"（{trainable / total_params * 100:.2f}%）"
        )

        # 创建一个子集数据加载器。
        subset_loader, _ = get_mnist_loaders(batch_size=64, subset_size=subset_size)

        # 仅优化需要梯度的参数（A, B）。
        optimizer_lora = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lora_lr,
        )

        for epoch in range(1, lora_epochs + 1):
            avg_loss = train_one_epoch(model, subset_loader, optimizer_lora, device)

        acc = evaluate(model, full_test_loader, device)
        print(f"  LoRA 测试准确率（rank={r}）：{acc:.4f}")

        results.append((r, trainable, acc))

    # ------------------------------------------------------------------
    # 6.4 比较表
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("阶段 3：结果汇总")
    print("=" * 65)

    header = f"{'方法':>22}  {'可训练参数':>17}  {'准确率':>9}  {'% 占全量':>10}"
    print(header)
    print("-" * len(header))

    # 基线：全量微调（所有参数，准确率来自预训练）
    print(
        f"{'全量微调':>22}  {full_ft_params:>17,}  {pretrain_acc:>9.4f}  {'100.00%':>10}"
    )

    # LoRA 行
    for r, tp, acc in results:
        pct = tp / full_ft_params * 100
        print(
            f"{'LoRA (rank=' + str(r) + ')':>22}  {tp:>17,}  {acc:>9.4f}  {pct:>9.2f}%"
        )

    print()

    # ------------------------------------------------------------------
    # 6.5 秩效应分析
    # ------------------------------------------------------------------
    print("=" * 65)
    print("阶段 4：秩效应分析")
    print("=" * 65)
    print(
        f"{'秩':>5}  {'可训练参数':>10}  {'准确率':>9}  {'Δ 准确率（LoRA - 预训练）':>25}"
    )
    print("-" * 65)
    for r, tp, acc in results:
        delta = acc - pretrain_acc  # LoRA 微调后的准确率变化
        print(f"{r:>5}  {tp:>10,}  {acc:>9.4f}  {delta:>+25.4f}")
    print()

    # ------------------------------------------------------------------
    # 6.6 合并演示
    # ------------------------------------------------------------------
    print("=" * 65)
    print("阶段 5：权重合并演示（rank=8）")
    print("=" * 65)

    # 创建一个带有 LoRA（rank=8）的新模型并快速训练。
    model_merge = SimpleMLP().to(device)
    model_merge.load_state_dict(base_state)
    apply_lora_to_model(model_merge, r=8, alpha=8.0, target_layers=["fc1", "fc2"])

    # 快速训练以获得清晰的 LoRA 权重。
    merge_loader, _ = get_mnist_loaders(batch_size=64, subset_size=2048)
    opt_merge = torch.optim.Adam(
        [p for p in model_merge.parameters() if p.requires_grad],
        lr=5e-4,
    )
    for _ in range(3):  # 短期训练
        train_one_epoch(model_merge, merge_loader, opt_merge, device)

    # 比较合并前后的预测（应该完全相同）。
    x_sample, _ = next(iter(full_test_loader))
    x_sample = x_sample[:16].to(device)  # 取前 16 张图片

    model_merge.eval()
    with torch.no_grad():
        pred_before = model_merge(x_sample)  # 合并前的预测

    # 合并并比较。
    for name in ["fc1", "fc2"]:
        layer = getattr(model_merge, name)
        if isinstance(layer, LoRALinear):
            layer.merge()

    with torch.no_grad():
        pred_after = model_merge(x_sample)  # 合并后的预测

    max_diff = (pred_before - pred_after).abs().max().item()
    agreement = (
        (pred_before.argmax(dim=1) == pred_after.argmax(dim=1)).float().mean().item()
    )

    print(f"  合并前后的最大 logit 差异：{max_diff:.2e}")
    print(f"  预测一致性：{agreement:.1%}")
    if max_diff < 1e-5:
        print("  ✓ 合并在数值上是稳定的 -- 输出完全相同。")
    else:
        print("  ⚠ 存在微小的数值差异（在 fp32 下是正常的）。")

    # 取消合并并验证旧行为已恢复。
    for name in ["fc1", "fc2"]:
        layer = getattr(model_merge, name)
        if isinstance(layer, LoRALinear):
            layer.unmerge()

    with torch.no_grad():
        pred_unmerged = model_merge(x_sample)

    max_diff2 = (pred_before - pred_unmerged).abs().max().item()
    print(f"  原始与取消合并后的最大 logit 差异：{max_diff2:.2e}")
    print()

    # ------------------------------------------------------------------
    # 6.7 最终总结
    # ------------------------------------------------------------------
    print("=" * 65)
    print("完成！ 关键要点：")
    print("  - LoRA 显著减少了可训练参数数量（仅占全量微调的 1-5%）。")
    print("  - 更高的秩可以提升准确率，但收益递减。")
    print("  - 将 LoRA 合并回基础权重是无损的（在 fp32 精度下）。")
    print("  - 冻结基础权重 + 仅训练低秩矩阵 = 内存高效微调。")
    print("=" * 65)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
