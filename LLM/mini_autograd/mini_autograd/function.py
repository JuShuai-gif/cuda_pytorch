"""所有自动微分算子的基类（对应 torch.autograd.Function）。

凡是参与计算图的算子都是 :class:`Function` 的子类。
一个 Function 实例就是计算图中的一个节点，它把输出 Tensor 和它的输入
Tensor 连接起来。

Function 的职责：

- ``forward``  : 计算算子的原始 numpy 结果
- ``backward`` : 接收上游梯度，并为每个输入 Tensor 计算梯度
                 （局部梯度 × 上游梯度）

Function 保存对输入 Tensor 的引用（``self.inputs``）以及反向时需要的
中间值（``self.saved``）。正是这些引用让反向传播时可以遍历整张计算图。
"""

from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np


class Function:
    """所有自动微分算子的基类（一个实例 == 一个图节点）。

    属性:
        inputs: 本算子作用的输入 Tensor（指向父节点的边）。
        output: 本算子产生的输出 Tensor。
        saved:  反向时需要的任意信息（输入形状、中间数据等）。
    """

    def __init__(self, **config: Any) -> None:
        self.inputs: Tuple["Tensor", ...] = ()
        self.output: "Tensor | None" = None
        self.saved: Dict[str, Any] = dict(config)

    # -- 辅助方法 --------------------------------------------------------------
    def save_for_backward(self, **tensors: Any) -> None:
        """保存 ``backward`` 需要的任意值（数组/形状等）。"""
        self.saved.update(tensors)

    # -- 子类必须实现这两个方法 -------------------------------------------------
    def forward(self, *args: np.ndarray) -> np.ndarray:
        """根据原始的 numpy 输入计算前向结果。"""
        raise NotImplementedError(f"{type(self).__name__} 必须实现 forward()")

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """给定上游梯度，返回对每个输入的梯度。"""
        raise NotImplementedError(f"{type(self).__name__} 必须实现 backward()")

    # -- 其他 ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"
