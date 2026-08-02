"""自动微分算子（Function）以及它们的顶层入口函数。

每个算子都遵循同样的模式：

    class <Op>(Function):
        def forward(self, x, y, ...):
            self.save_for_backward(...)   # 记下 backward 需要的数据
            return <numpy 结果>

        def backward(self, grad_output):
            # grad_output 是上游梯度（dc/dz，其中 z = 前向结果）
            # 返回对每个前向输入的梯度
            return <对 x 的梯度>, <对 y 的梯度>, ...

    def <op>(a, b, ...):
        return _from_function(<Op>(), a, b, ...)

这里应用的链式法则：对于算子 ``z = f(x, y)``，
``dc/dx = dc/dz * dz/dx = grad_output * 局部梯度(x)``。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .function import Function
from .tensor import Tensor, _from_function
from .utils import unbroadcast

# --------------------------------------------------------------------------- #
#  add / sub / mul / div / neg / pow（逐元素，支持广播）                       #
# --------------------------------------------------------------------------- #


class Add(Function):
    """z = a + b    dz/da = 1,  dz/db = 1"""

    def forward(self, a, b):
        self.save_for_backward(a_shape=a.shape, b_shape=b.shape)
        return a + b

    def backward(self, grad_output):
        # 上游梯度对两个输入相同，但若发生了广播，必须把梯度还原到
        # 每个输入各自的原始形状。
        return (
            unbroadcast(grad_output, self.saved["a_shape"]),
            unbroadcast(grad_output, self.saved["b_shape"]),
        )


class Sub(Function):
    """z = a - b    dz/da = 1,  dz/db = -1"""

    def forward(self, a, b):
        self.save_for_backward(a_shape=a.shape, b_shape=b.shape)
        return a - b

    def backward(self, grad_output):
        return (
            unbroadcast(grad_output, self.saved["a_shape"]),
            unbroadcast(-grad_output, self.saved["b_shape"]),
        )


class Mul(Function):
    """z = a * b    dz/da = b,  dz/db = a"""

    def forward(self, a, b):
        self.save_for_backward(a=a, b=b)
        return a * b

    def backward(self, grad_output):
        return (
            unbroadcast(grad_output * self.saved["b"], self.saved["a"].shape),
            unbroadcast(grad_output * self.saved["a"], self.saved["b"].shape),
        )


class Div(Function):
    """z = a / b    dz/da = 1/b,  dz/db = -a/b^2"""

    def forward(self, a, b):
        self.save_for_backward(a=a, b=b)
        return a / b

    def backward(self, grad_output):
        a, b = self.saved["a"], self.saved["b"]
        return (
            unbroadcast(grad_output / b, a.shape),
            unbroadcast(grad_output * (-a / (b * b)), b.shape),
        )


class Neg(Function):
    """z = -x    dz/dx = -1"""

    def forward(self, x):
        return -x

    def backward(self, grad_output):
        return (-grad_output,)


class Pow(Function):
    """z = x ** p    dz/dx = p * x^(p-1)   （p 是常量，不是 Tensor）"""

    def forward(self, x):
        p = self.saved["exponent"]
        self.save_for_backward(x=x)
        return x**p

    def backward(self, grad_output):
        x = self.saved["x"]
        p = self.saved["exponent"]
        return (grad_output * p * np.power(x, p - 1.0),)


# --------------------------------------------------------------------------- #
#  matmul                                                                     #
# --------------------------------------------------------------------------- #


class MatMul(Function):
    """z = a @ b    dz/da = z @ b.T,  dz/db = a.T @ z

    支持 numpy 批量矩阵乘法对 batch 维度的广播语义。
    """

    def forward(self, a, b):
        self.save_for_backward(a=a, b=b)
        return np.matmul(a, b)

    def backward(self, grad_output):
        a, b = self.saved["a"], self.saved["b"]
        ga, gb = _matmul_backward(grad_output, a, b)
        return (ga, gb)


def _matmul_backward(grad_out, a, b):
    """numpy matmul 的梯度，处理 1-D 向量和 batch 广播。

    1-D 输入会被临时升成 2-D，套用标准梯度公式后，再把结果压缩回原始形状。
    """
    sq_a, sq_b = a.ndim == 1, b.ndim == 1
    a2 = a.reshape(1, -1) if sq_a else a
    b2 = b.reshape(-1, 1) if sq_b else b

    g2 = np.asarray(grad_out)
    if sq_a and sq_b:
        g2 = g2.reshape(1, 1)
    elif sq_a:
        g2 = g2[None, ...]  # (N,) -> (1, N)
    elif sq_b:
        g2 = g2[..., None]  # (M,) -> (M, 1)

    ga2 = np.matmul(g2, np.swapaxes(b2, -1, -2))
    gb2 = np.matmul(np.swapaxes(a2, -1, -2), g2)
    ga = unbroadcast(ga2, a2.shape).reshape(a.shape)
    gb = unbroadcast(gb2, b2.shape).reshape(b.shape)
    return ga, gb


# --------------------------------------------------------------------------- #
#  归约算子：sum / mean                                                        #
# --------------------------------------------------------------------------- #


def _reduce_backward(grad_output, input_shape, axis, keepdims, divisor=1.0):
    """sum/mean 共用的反向逻辑：把归约后的梯度广播回原始形状，然后
    （对 mean）除以折叠进每个输出元素的元素个数。"""
    grad = grad_output
    if axis is None:
        grad = np.broadcast_to(grad, input_shape)
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        ndim = len(input_shape)
        axes = tuple(sorted(a % ndim for a in axes))
        if not keepdims:
            # 把被归约的维度重新插成大小为 1，然后再广播。
            shape = list(grad.shape)
            for ax in axes:
                shape.insert(ax, 1)
            grad = grad.reshape(shape)
        grad = np.broadcast_to(grad, input_shape)
    return grad / divisor


class Sum(Function):
    """z = sum(x, axis, keepdims)    dz/dx = 把 1 广播回去"""

    def forward(self, x):
        axis, keepdims = self.saved["axis"], self.saved["keepdims"]
        self.save_for_backward(input_shape=x.shape)
        return x.sum(axis=axis, keepdims=keepdims)

    def backward(self, grad_output):
        s = self.saved
        return (
            _reduce_backward(grad_output, s["input_shape"], s["axis"], s["keepdims"]),
        )


class Mean(Function):
    """z = mean(x, axis, keepdims)    dz/dx = 把 1/count 广播回去"""

    def forward(self, x):
        axis, keepdims = self.saved["axis"], self.saved["keepdims"]
        self.save_for_backward(input_shape=x.shape)
        out = x.mean(axis=axis, keepdims=keepdims)
        # 折叠进单个输出元素的元素个数。
        if axis is None:
            count = x.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = int(np.prod([x.shape[a % x.ndim] for a in axes]))
        self.save_for_backward(count=count)
        return out

    def backward(self, grad_output):
        s = self.saved
        return (
            _reduce_backward(
                grad_output,
                s["input_shape"],
                s["axis"],
                s["keepdims"],
                divisor=s["count"],
            ),
        )


# --------------------------------------------------------------------------- #
#  形状算子：reshape / transpose                                               #
# --------------------------------------------------------------------------- #


class Reshape(Function):
    """z = reshape(x, shape)    dz/dx = 把梯度 reshape 回去"""

    def forward(self, x):
        self.save_for_backward(input_shape=x.shape)
        return x.reshape(self.saved["shape"])

    def backward(self, grad_output):
        return (grad_output.reshape(self.saved["input_shape"]),)


class Transpose(Function):
    """z = transpose(x, axes)    dz/dx = 用逆置换转置梯度"""

    def forward(self, x):
        self.save_for_backward(axes=self.saved["axes"])
        return x.transpose(self.saved["axes"])

    def backward(self, grad_output):
        axes = self.saved["axes"]
        inverse = np.argsort(axes)
        return (grad_output.transpose(inverse),)


# --------------------------------------------------------------------------- #
#  逐元素超越函数：exp / log / relu / sigmoid / tanh                           #
# --------------------------------------------------------------------------- #


class Exp(Function):
    """z = exp(x)    dz/dx = exp(x)"""

    def forward(self, x):
        self.save_for_backward(exp_x=np.exp(x))
        return self.saved["exp_x"]

    def backward(self, grad_output):
        return (grad_output * self.saved["exp_x"],)


class Log(Function):
    """z = log(x)    dz/dx = 1 / x"""

    def forward(self, x):
        self.save_for_backward(x=x)
        return np.log(x)

    def backward(self, grad_output):
        return (grad_output / self.saved["x"],)


class ReLU(Function):
    """z = max(0, x)    dz/dx = x>0 时为 1，否则为 0"""

    def forward(self, x):
        self.save_for_backward(x=x)
        return np.maximum(x, 0.0)

    def backward(self, grad_output):
        return (grad_output * (self.saved["x"] > 0.0),)


class Sigmoid(Function):
    """z = 1 / (1 + exp(-x))    dz/dx = z * (1 - z)"""

    def forward(self, x):
        out = 1.0 / (1.0 + np.exp(-x))
        self.save_for_backward(out=out)
        return out

    def backward(self, grad_output):
        out = self.saved["out"]
        return (grad_output * out * (1.0 - out),)


class Tanh(Function):
    """z = tanh(x)    dz/dx = 1 - z^2"""

    def forward(self, x):
        out = np.tanh(x)
        self.save_for_backward(out=out)
        return out

    def backward(self, grad_output):
        out = self.saved["out"]
        return (grad_output * (1.0 - out * out),)


# --------------------------------------------------------------------------- #
#  顶层算子函数（Tensor 和 nn 模块使用的公共 API）                             #
# --------------------------------------------------------------------------- #


def add(a, b) -> Tensor:
    return _from_function(Add(), a, b)


def sub(a, b) -> Tensor:
    return _from_function(Sub(), a, b)


def mul(a, b) -> Tensor:
    return _from_function(Mul(), a, b)


def div(a, b) -> Tensor:
    return _from_function(Div(), a, b)


def neg(x) -> Tensor:
    return _from_function(Neg(), x)


def pow(x, exponent: float) -> Tensor:
    return _from_function(Pow(exponent=exponent), x)


def matmul(a, b) -> Tensor:
    return _from_function(MatMul(), a, b)


def sum(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    return _from_function(Sum(axis=axis, keepdims=keepdims), x)


def mean(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    return _from_function(Mean(axis=axis, keepdims=keepdims), x)


def reshape(x: Tensor, shape: Sequence[int]) -> Tensor:
    return _from_function(Reshape(shape=tuple(shape)), x)


def transpose(x: Tensor, axes: Optional[Tuple[int, int]] = None) -> Tensor:
    if axes is None:
        if x.ndim < 2:
            raise ValueError(f"对 {x.ndim} 维张量转置时必须显式指定 axes")
        axes = tuple(reversed(range(x.ndim)))
    return _from_function(Transpose(axes=axes), x)


def exp(x: Tensor) -> Tensor:
    return _from_function(Exp(), x)


def log(x: Tensor) -> Tensor:
    return _from_function(Log(), x)


def relu(x: Tensor) -> Tensor:
    return _from_function(ReLU(), x)


def sigmoid(x: Tensor) -> Tensor:
    return _from_function(Sigmoid(), x)


def tanh(x: Tensor) -> Tensor:
    return _from_function(Tanh(), x)
