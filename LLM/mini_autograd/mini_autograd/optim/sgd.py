"""随机梯度下降优化器（对应 torch.optim.SGD）。

支持基础 SGD、动量（momentum）和权重衰减（weight decay）。
参数更新在 ``no_grad()`` 中执行，这样更新操作永远不会进入计算图。
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Optional

from ..grad_mode import no_grad
from ..tensor import Tensor


class SGD:
    """基于梯度下降的参数更新。

    参数:
        params:      可迭代的 Parameter（例如 ``model.parameters()``）
        lr:          学习率
        momentum:    动量系数（0 表示不使用动量）
        weight_decay: L2 惩罚，加到梯度上为 ``g + wd * p``
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # 每个参数的速度缓存（仅当 momentum > 0 时使用）。
        self._velocities: List[Optional[np.ndarray]] = [None] * len(self.params)

    def zero_grad(self) -> None:
        """把所有参数的梯度重置为 None。"""
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """用累积的梯度执行一次参数更新。"""
        # 更新绝不能创建计算图，所以放在 no_grad() 里。
        with no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue  # 该参数没有参与本次 backward

                g = p.grad
                if self.weight_decay:
                    # L2 正则化：在梯度上加 wd * param。
                    g = g + self.weight_decay * p.data

                if self.momentum:
                    v = self._velocities[i]
                    if v is None:
                        v = g.copy()
                    else:
                        v = self.momentum * v + g
                    self._velocities[i] = v
                    g = v

                p.data = p.data - self.lr * g
