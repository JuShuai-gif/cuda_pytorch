"""Demo: 全局幅度剪枝与稀疏度统计。

这个脚本演示最基础的 global magnitude pruning：
1. 收集模型中所有 Conv2d/Linear 权重；
2. 按绝对值全局排序，找到剪枝阈值；
3. 将小于阈值的权重置零；
4. 统计剪枝后的实际稀疏度。

注意：这里的剪枝只把权重置零，不会改变 Linear/Conv 的 shape。
如果 runtime 没有 sparse kernel，模型不一定会变快。
"""

import torch
import torch.nn as nn


@torch.no_grad()
def global_magnitude_prune(model: nn.Module, sparsity: float):
    """对整个模型做全局非结构化幅度剪枝。

    参数：
        model: 待剪枝的 PyTorch 模型。
        sparsity: 目标稀疏度，例如 0.5 表示剪掉 50% 权重。

    返回：
        原地修改后的 model。
    """
    weights = []
    modules = []

    # 只剪 Conv2d 和 Linear 的 weight；BN/LayerNorm 通常不在这里剪。
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weights.append(m.weight.detach().abs().flatten())
            modules.append(m)

    # 将所有可剪层的权重绝对值拼在一起，做“全局”阈值选择。
    flat = torch.cat(weights)
    k = int(flat.numel() * sparsity)
    if k <= 0:
        return model

    # 第 k 小的绝对值作为剪枝阈值。
    threshold = torch.kthvalue(flat, k).values

    # 小于等于阈值的权重被置零，大于阈值的权重保留。
    for m in modules:
        mask = (m.weight.detach().abs() > threshold).to(m.weight.dtype)
        m.weight.mul_(mask)
    return model


@torch.no_grad()
def prunable_sparsity(model: nn.Module):
    """统计 Conv2d/Linear 权重中的实际稀疏度。"""
    total = zeros = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.detach()
            total += w.numel()
            zeros += int((w == 0).sum())
    return zeros / max(total, 1)


# 构造一个最小 MLP，用于演示剪枝前后稀疏度。
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
global_magnitude_prune(model, sparsity=0.5)
print(f"actual sparsity = {prunable_sparsity(model):.2%}")
