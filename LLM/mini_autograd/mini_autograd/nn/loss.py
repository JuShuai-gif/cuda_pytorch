"""损失函数：MSELoss 与 CrossEntropyLoss。"""

from __future__ import annotations

import numpy as np

from .. import ops
from ..tensor import Tensor
from .module import Module


class MSELoss(Module):
    """预测值与目标之间的均方（或求和）误差。"""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction 必须是 'mean' 或 'sum'，当前为 {reduction!r}")
        self.reduction = reduction

    def forward(self, pred, target):
        diff = pred - target
        squared = diff * diff
        if self.reduction == "sum":
            return squared.sum()
        return squared.mean()


class CrossEntropyLoss(Module):
    """原始 logits 与整数类别索引之间的交叉熵。

    ``logits`` 形状: (N, C)；``targets`` 形状: (N,)，取值在 [0, C) 的整数。

    完全用可微算子构建（对类别维度做 softmax），因此梯度能通过我们已实现
    的计算图流回去。
    """

    def forward(self, logits, targets):
        # 数值稳定的平移：减去每行的最大值（这是常量，不影响梯度）。
        logits_data = logits.data
        row_max = logits_data.max(axis=1, keepdims=True)
        shifted = logits - Tensor(row_max)

        exp_s = ops.exp(shifted)
        logsumexp = ops.log(exp_s.sum(axis=1, keepdims=True))
        log_softmax = shifted - logsumexp  # (N, C)

        # 用 one-hot 掩码取出真实类别的对数概率。
        onehot = np.eye(logits.shape[1], dtype=np.float64)[targets]  # (N, C)
        loss = -(log_softmax * Tensor(onehot)).sum(axis=1).mean()
        return loss
