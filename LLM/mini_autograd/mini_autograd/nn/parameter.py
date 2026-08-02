"""Parameter：默认需要梯度的 Tensor（对应 torch.nn.Parameter）。"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor


class Parameter(Tensor):
    """在训练中会被优化器更新的叶子张量。

    Parameter 默认 ``requires_grad=True``，并且永远不带 ``grad_fn``
    （即使出现在 ``no_grad`` 代码块内，它们也始终是叶子）。
    """

    def __init__(self, data) -> None:
        super().__init__(np.asarray(data, dtype=np.float64), requires_grad=True)
        self.is_leaf = True

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, data={self.data})"
