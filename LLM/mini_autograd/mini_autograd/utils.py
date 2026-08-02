"""自动微分引擎共享的工具函数。

其中最重要的是 ``unbroadcast``：它在反向传播时"撤销"numpy 的广播。
前向时，一个小张量被拉伸成更大的形状；反向时，上游梯度必须沿着被拉伸
的维度求和回去，梯度才能与原始（较小的）形状匹配。
"""

import numpy as np

from typing import Sequence


def unbroadcast(grad: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """把广播算子的梯度还原回 ``target_shape``。

    NumPy 广播会默默地把一个形状拉伸成更大的形状，例如
    ``(3,) + (4, 3) -> (4, 3)``。反向时梯度 ``(4, 3)`` 必须被还原成
    ``(3,)``，否则无法把它加回到小张量上。

    规则（广播的逆运算）：

    - 多余的前导维度直接求和去掉：``(1, 4, 3) -> (4, 3)``
    - 目标尺寸为 1 的尾部维度求和去掉：``(4, 3) -> (3,)``

    这正是 PyTorch 在 ``torch.autograd`` 内部做的事情。
    """
    grad = np.asarray(grad)

    # 1) 去掉多余的前导维度（它们只因为广播才存在）。
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # 2) 折叠目标尺寸为 1、但梯度尺寸大于 1 的尾部维度。
    for axis, (g_size, t_size) in enumerate(zip(grad.shape, target_shape)):
        if t_size == 1 and g_size != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    # 规整最终形状（如果已经正确，这一步无副作用）。
    return grad.reshape(target_shape)


def as_float_array(data) -> np.ndarray:
    """把 ``data``（数字 / 列表 / ndarray）转换成 float64 ndarray。

    自动微分需要浮点数值，这样每个算子才可微。
    """
    return np.asarray(data, dtype=np.float64)
