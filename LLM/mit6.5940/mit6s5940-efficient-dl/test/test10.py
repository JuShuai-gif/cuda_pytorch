"""Demo: Conv2d 输出通道重要性排序。

这个脚本演示结构化通道剪枝的第一步：
1. 对每个输出通道计算权重 L2/Frobenius norm；
2. 按重要性排序，选择要保留的通道；
3. 输出 keep_idx，供后续真正改模型结构时使用。

注意：这里只计算保留通道索引，不直接修改 Conv2d。真正的结构化剪枝还需要同步修改
下一层输入通道、BatchNorm 参数以及 residual/concat 分支。
"""

import torch
import torch.nn as nn


@torch.no_grad()
def conv_out_channel_importance(conv: nn.Conv2d):
    """计算 Conv2d 每个输出通道的重要性分数。

    Conv2d 权重形状是 [out_channels, in_channels, kh, kw]。
    flatten(1) 后，每一行就是一个输出通道的所有参数。
    L2 norm 越大，通常说明这个输出通道越重要。
    """
    return conv.weight.detach().flatten(1).norm(p=2, dim=1)


@torch.no_grad()
def select_channels_to_keep(conv: nn.Conv2d, keep_ratio: float):
    """根据输出通道重要性选择需要保留的通道索引。"""
    score = conv_out_channel_importance(conv)
    keep = max(1, int(score.numel() * keep_ratio))
    return torch.topk(score, keep).indices.sort().values


# 构造一个 Conv2d：输入通道 16，输出通道 32。
conv = nn.Conv2d(16, 32, 3, padding=1, bias=False)

# 保留 70% 输出通道，即 32 * 0.7 ≈ 22 个通道。
keep_idx = select_channels_to_keep(conv, keep_ratio=0.7)
print("keep channels", keep_idx.tolist())
