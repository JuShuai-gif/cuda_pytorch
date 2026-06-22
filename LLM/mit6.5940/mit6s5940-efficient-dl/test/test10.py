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
    """计算 Conv2d 每个输出通道(filter)的重要性分数，返回形状 [out_channels]。

    原理（出自 Pruning Filters for Efficient ConvNets, Li et al. 2017）：
    - 权重形状是 [out_channels, in_channels, kh, kw]。第 0 维的每个切片
      weight[o]（形状 [in_channels, kh, kw]）就是一个独立的 filter，
      单独负责生成第 o 个输出特征图(feature map)；out_channels 个通道 = 同样多个 filter。
    - flatten(1) 把后三维 [in, kh, kw] 拉平：[out, in, kh, kw] -> [out, in*kh*kw]，
      于是每一行 = 一个输出通道的全部参数。
    - 对每行求 L2 范数 sqrt(Σ wᵢ²)，衡量该 filter 权重的整体“能量/幅度”：
        范数大  -> 权重普遍较大 -> 输出响应强 -> 重要，保留；
        范数≈0  -> 权重几乎全是小值 -> 输出≈0 -> 冗余，可整条通道剪掉。
    - 这是结构化剪枝的第一步：按分数排序后剪掉范数最小的通道，能直接缩小
      out_channels，得到真正更小更快的稠密卷积层（区别于只置零的非结构化剪枝）。
    - 局限：只看权重大小，未考虑真实数据下的激活分布；更精细的方法会用 BN 的
      缩放系数 γ、激活统计或 Taylor 展开来估计删除通道对 loss 的实际影响。
    """
    # detach() 切断梯度；最终输出形状 [out_channels]，第 o 个值即第 o 个 filter 的重要性。
    return conv.weight.detach().flatten(1).norm(p=2, dim=1)


@torch.no_grad()
def select_channels_to_keep(conv: nn.Conv2d, keep_ratio: float):
    """根据输出通道重要性选择需要保留的通道索引。"""
    score = conv_out_channel_importance(conv)  # 每个输出通道的 L2 重要性
    keep = max(1, int(score.numel() * keep_ratio))  # 要保留的通道数（至少 1 个）
    # topk 取分数最高的 keep 个通道，再按索引从小到大排序，方便阅读和后续切片。
    return torch.topk(score, keep).indices.sort().values


# 构造一个 Conv2d：输入通道 16，输出通道 32。
conv = nn.Conv2d(16, 32, 3, padding=1, bias=False)

# 保留 70% 输出通道，即 32 * 0.7 ≈ 22 个通道。
keep_idx = select_channels_to_keep(conv, keep_ratio=0.7)
print("keep channels", keep_idx.tolist())
