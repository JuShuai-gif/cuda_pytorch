"""全连接层（对应 torch.nn.Linear）。"""

from __future__ import annotations

import numpy as np

from .. import ops
from .module import Module
from .parameter import Parameter


class Linear(Module):
    """``y = x @ W.T + b``

    参数:
        in_features:  输入特征数
        out_features: 输出特征数
        bias:         是否包含偏置项
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        # 类 He 初始化：std = sqrt(1 / fan_in)。权重形状和 PyTorch 一致：
        # (out_features, in_features)，这样前向就是 x @ W.T。
        k = 1.0 / np.sqrt(in_features)
        self.weight = Parameter(np.random.uniform(-k, k, (out_features, in_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        y = ops.matmul(x, ops.transpose(self.weight))
        if self.bias is not None:
            y = y + self.bias
        return y
