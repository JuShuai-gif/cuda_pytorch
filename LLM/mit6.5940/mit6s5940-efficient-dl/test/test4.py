"""Lecture 03 pruning mini demos.

这个脚本把 lecture-03 中的剪枝代码拆成多个小 demo：
1. demo_01_tensor_pruning: 单个权重张量的幅度剪枝。
2. demo_02_global_pruning: 对整个模型做全局非结构化剪枝。
3. demo_03_sensitivity_scan: 逐层敏感度扫描，观察哪一层更怕剪。
4. demo_04_channel_importance: 计算 Conv2d 输出通道重要性，这是结构化剪枝的第一步。

运行方式：
    python test/test4.py
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# 公共工具：构造一个很小的 CNN 和 synthetic dataloader，保证 demo 不依赖外部数据集。
# -----------------------------------------------------------------------------
class TinyCNN(nn.Module):
    """用于剪枝演示的小型 CNN。"""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)


def make_synthetic_loader(n: int = 128, batch_size: int = 32) -> DataLoader:
    """生成随机图片和随机标签，用于快速 smoke test。"""
    x = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


# -----------------------------------------------------------------------------
# 公共工具：剪枝、稀疏度、评估、延迟测量。
# -----------------------------------------------------------------------------
def magnitude_prune_tensor(weight: torch.Tensor, sparsity: float):
    """对单个权重张量做幅度剪枝，返回剪枝后的权重和二值 mask。"""
    # 校验稀疏度参数
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")

    # 边界情况：稀疏度为 0，保留所有权重
    if sparsity == 0.0:
        mask = torch.ones_like(weight, dtype=torch.bool)
        return weight.clone(), mask

    # 边界情况：稀疏度为 1，清零所有权重
    if sparsity == 1.0:
        mask = torch.zeros_like(weight, dtype=torch.bool)
        return torch.zeros_like(weight), mask

    # 展平绝对值并找到第 k 小的幅值作为阈值
    flat = weight.detach().abs().flatten()
    k = int(sparsity * flat.numel())                     # 需要置零的权重个数
    threshold = torch.kthvalue(flat, k).values           # 第 k 小的幅值

    # 保留严格大于阈值的权重
    mask = weight.detach().abs() > threshold
    return weight * mask.to(weight.dtype), mask


def prunable_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """遍历模型中可剪枝的 Conv2d/Linear 层。"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            yield name, module


@torch.no_grad()
def global_magnitude_prune(model: nn.Module, sparsity: float) -> dict[str, torch.Tensor]:
    """对整个模型做全局非结构化幅度剪枝，原地修改模型并返回 mask。

    举例：假设有两层 conv1.weight=[0.1, 0.2, 0.5, 0.05], fc1.weight=[0.8, 0.01, 0.3, 0.4]
    sparsity=0.5 时：
        all_scores = [0.1, 0.2, 0.5, 0.05, 0.8, 0.01, 0.3, 0.4]  （8个数）
        k = int(0.5*8) = 4，排序找第4小 -> 0.2，所以 threshold=0.2
        conv1: [0.1,0.2,0.5,0.05] > 0.2? -> [F,F,T,F] -> 保留 0.5，其余置零
        fc1:   [0.8,0.01,0.3,0.4] > 0.2? -> [T,F,T,T] -> 保留 0.8/0.3/0.4
        全局恰好 50% 为零，但 conv1 被剪了 75%，fc1 只剪了 25%。
    """
    # 获取所有可剪枝层
    named_modules = list(prunable_modules(model))
    if not named_modules:
        return {}

    # 将所有层的所有权重绝对值拼成一个长向量，统一排序
    all_scores = torch.cat([m.weight.detach().abs().flatten() for _, m in named_modules])
    k = int(sparsity * all_scores.numel())                    # 全局需要置零的权重个数
    # 找到第 k 小的幅值作为全局阈值；k 超出总个数时设为无穷大（全部剪掉）
    threshold = torch.inf if k >= all_scores.numel() else torch.kthvalue(all_scores, max(k, 1)).values

    # 逐层应用阈值：保留大于 threshold 的权重，其余置零
    masks: dict[str, torch.Tensor] = {}
    for name, module in named_modules:
        mask = module.weight.detach().abs() > threshold
        module.weight.mul_(mask.to(module.weight.dtype))
        masks[name] = mask
    return masks


@torch.no_grad()
def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """微调后重新应用 mask，防止被剪权重恢复成非零。"""
    module_dict = dict(model.named_modules())
    for name, mask in masks.items():
        module = module_dict[name]
        module.weight.mul_(mask.to(module.weight.device, module.weight.dtype))


@torch.no_grad()
def sparsity_of_prunable_weights(model: nn.Module) -> float:
    """统计 Conv2d/Linear 权重中的实际稀疏度。"""
    total = 0
    zeros = 0
    for _, module in prunable_modules(model):
        w = module.weight.detach()
        total += w.numel()
        zeros += int((w == 0).sum())
    return zeros / max(total, 1)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """在 synthetic loader 上计算 top-1 accuracy。随机数据只用于验证代码流程。"""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def benchmark_latency_ms(model: nn.Module, example_input: torch.Tensor, warmup: int = 5, runs: int = 20):
    """CPU 简易延迟测试，返回 mean/p50/p95。"""
    model.eval()
    for _ in range(warmup):
        model(example_input)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(example_input)
        times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return {
        "mean_ms": float(t.mean()),
        "p50_ms": float(t.quantile(0.50)),
        "p95_ms": float(t.quantile(0.95)),
    }


# -----------------------------------------------------------------------------
# Demo 1：单个 tensor 的幅度剪枝。
# 作用：理解 magnitude pruning 的最小单元：小权重被置零，大权重保留。
# -----------------------------------------------------------------------------
def demo_01_tensor_pruning() -> None:
    w = torch.randn(64, 64)
    w_pruned, mask = magnitude_prune_tensor(w, sparsity=0.5)
    print("\n[Demo 1] 单 tensor 幅度剪枝")
    print(f"target sparsity=50%, actual sparsity={(w_pruned == 0).float().mean():.2%}")
    print(f"mask dtype={mask.dtype}, kept weights={int(mask.sum())}/{mask.numel()}")


# -----------------------------------------------------------------------------
# Demo 2：全局非结构化剪枝。
# 作用：理解“全局剪 50%”不是“每层各剪 50%”，而是所有权重一起排序。
# -----------------------------------------------------------------------------
def demo_02_global_pruning() -> None:
    model = TinyCNN()
    before = sparsity_of_prunable_weights(model)
    masks = global_magnitude_prune(model, sparsity=0.5)
    after = sparsity_of_prunable_weights(model)
    print("\n[Demo 2] 全局非结构化剪枝")
    print(f"sparsity before={before:.2%}, after={after:.2%}")
    print("mask layers:", list(masks.keys()))


# -----------------------------------------------------------------------------
# Demo 3：逐层敏感度扫描。
# 作用：每次只剪一层，观察 accuracy drop，用于决定不同层剪多少。
# -----------------------------------------------------------------------------
@dataclass
class SensitivityPoint:
    layer: str
    sparsity: float
    accuracy: float


def layerwise_sensitivity_scan(
    model: nn.Module,
    val_loader: DataLoader,
    sparsities: tuple[float, ...] = (0.3, 0.6),
    device: str = "cpu",
) -> list[SensitivityPoint]:
    results: list[SensitivityPoint] = []
    baseline_acc = evaluate_accuracy(model, val_loader, device=device)
    for layer_name, _ in prunable_modules(model):
        for sparsity in sparsities:
            trial = copy.deepcopy(model).to(device)
            module = dict(trial.named_modules())[layer_name]
            with torch.no_grad():
                pruned_w, _ = magnitude_prune_tensor(module.weight, sparsity)
                module.weight.copy_(pruned_w)
            acc = evaluate_accuracy(trial, val_loader, device=device)
            results.append(SensitivityPoint(layer_name, sparsity, acc))
            print(
                f"layer={layer_name:20s} sparsity={sparsity:.1f} "
                f"acc={acc:.4f} drop={baseline_acc - acc:.4f}"
            )
    return results


def demo_03_sensitivity_scan() -> None:
    print("\n[Demo 3] 逐层敏感度扫描")
    model = TinyCNN()
    val_loader = make_synthetic_loader(n=96, batch_size=32)
    layerwise_sensitivity_scan(model, val_loader)


# -----------------------------------------------------------------------------
# Demo 4：通道重要性排序。
# 作用：结构化剪枝前先找出哪些输出通道更重要；这里只排序，不真正改模型结构。
# -----------------------------------------------------------------------------
@torch.no_grad()
def conv_out_channel_importance(conv: nn.Conv2d) -> torch.Tensor:
    return conv.weight.detach().flatten(1).norm(p=2, dim=1)


@torch.no_grad()
def select_conv_out_channels(conv: nn.Conv2d, keep_ratio: float):
    scores = conv_out_channel_importance(conv)
    keep = max(1, int(scores.numel() * keep_ratio))
    keep_idx = torch.topk(scores, keep).indices.sort().values
    prune_idx = torch.tensor([i for i in range(scores.numel()) if i not in set(keep_idx.tolist())])
    return keep_idx, prune_idx, scores


def demo_04_channel_importance() -> None:
    print("\n[Demo 4] Conv2d 输出通道重要性排序")
    conv = nn.Conv2d(3, 16, 3, padding=1)
    keep_idx, prune_idx, scores = select_conv_out_channels(conv, keep_ratio=0.5)
    print("importance scores:", [round(float(x), 4) for x in scores[:5]])
    print("keep channels:", keep_idx.tolist())
    print("prune channels:", prune_idx.tolist())


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_01_tensor_pruning()
    demo_02_global_pruning()
    demo_03_sensitivity_scan()
    demo_04_channel_importance()
