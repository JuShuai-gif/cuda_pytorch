"""核心的 Tensor 类与反向模式（reverse-mode）自动微分引擎。

Tensor 包装了一个 numpy 数组，并可选地记录它是由谁产生的。

计算图在*前向执行时隐式、动态地*构建：

    z = x * y + x

会在前向阶段产生一串由 Tensor 连接起来的 Function 节点
（先 ``Mul`` 再 ``Add``）：

    x --Mul--> t --Add--> z
    y --Mul--> t
    x --------Add--> z

``z.grad_fn`` 指向 ``Add`` 节点；``t.grad_fn`` 指向 ``Mul`` 节点。
反向传播按照反向拓扑顺序遍历这张图，逐节点应用链式法则。
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from . import grad_mode
from .function import Function
from .utils import as_float_array

# 避免循环导入：算子方法内部延迟导入 ops。
# （ops.py 需要导入 tensor.py 来构建 Tensor，所以这里不能直接导入 ops。）
_scalar_types = (int, float, np.number)


def _wrap(data) -> "Tensor":
    """把普通数字/ndarray 转换成（叶子）Tensor。"""
    if isinstance(data, Tensor):
        return data
    return Tensor(data)


class Tensor:
    """基于 numpy 的、带反向自动微分的张量。

    成员（尽量与 torch.Tensor 对应）：

        data          numpy.ndarray，保存数值
        grad          numpy.ndarray，保存累积的梯度（与 data 同形状）
        requires_grad 该张量是否参与计算图
        grad_fn       产生该张量的 Function 节点（叶子节点为 None）
        is_leaf       用户创建时为 True（没有 grad_fn）；requires_grad=False 时也视为叶子
        shape / ndim / dtype
    """

    __array_priority__ = 1000

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        grad_fn: Optional[Function] = None,
    ) -> None:
        # 始终存成浮点数组，这样每个算子都可微。
        self.data: np.ndarray = as_float_array(data)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad: bool = bool(requires_grad)
        self.grad_fn: Optional[Function] = grad_fn
        # 叶子节点是用户创建的张量（没有 grad_fn）。算子通过 _from_function
        # 把输出显式标记为非叶子。
        self.is_leaf: bool = grad_fn is None

    # ------------------------------------------------------------------ #
    #  类工厂方法（zeros / ones / randn / tensor / arange）                 #
    # ------------------------------------------------------------------ #
    @classmethod
    def zeros(cls, *shape: int, requires_grad: bool = False) -> "Tensor":
        return cls(np.zeros(shape, dtype=np.float64), requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape: int, requires_grad: bool = False) -> "Tensor":
        return cls(np.ones(shape, dtype=np.float64), requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape: int, requires_grad: bool = False) -> "Tensor":
        return cls(np.random.randn(*shape), requires_grad=requires_grad)

    # ------------------------------------------------------------------ #
    #  属性                                                                #
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        """矩阵转置（交换最后两个维度）。"""
        return transpose(self)

    def numpy(self) -> np.ndarray:
        """返回底层 numpy 数组（不经过自动微分）。"""
        return self.data

    def item(self) -> float:
        """返回一个 Python float。仅对标量张量有效。"""
        if self.ndim == 0:
            return float(self.data)
        if self.size() == 1:
            return float(self.data.reshape(-1)[0])
        raise ValueError(f"只能转换单元素张量，当前形状为 {self.shape}")

    def size(self) -> int:
        return int(np.prod(self.shape))

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------ #
    #  计算图控制：detach / zero_grad / backward                            #
    # ------------------------------------------------------------------ #
    def detach(self) -> "Tensor":
        """返回一个没有任何梯度历史的新张量。

        新张量的 ``requires_grad=False`` 且没有 ``grad_fn``。
        注意：真实 PyTorch 会共享底层存储；这里我们拷贝数据，
        这样意外发生的就地修改永远不会污染原张量。
        """
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self) -> None:
        """把累积的梯度重置为 None（PyTorch 是清零为 0）。"""
        self.grad = None

    def _accumulate_grad(self, g: np.ndarray) -> None:
        """把梯度 ``g``（与 ``data`` 同形状）累加到 ``self.grad``。

        梯度累加就是为什么一个被多条路径使用的张量，最终梯度是所有路径
        梯度的"和"而不是最后一条路径的梯度。
        """
        if not self.requires_grad:
            return
        g = as_float_array(g)
        if self.grad is None:
            # 拷贝一份，避免后续 numpy 运算意外修改已存好的梯度。
            self.grad = g.copy()
        else:
            self.grad = self.grad + g

    def backward(
        self, gradient: Optional[Union[np.ndarray, float, "Tensor"]] = None
    ) -> None:
        """从该张量开始执行反向模式自动微分。

        参数:
            gradient: 初始上游梯度。仅在输出不是标量时必须显式传入，
                      否则默认为 ``1``。

        引擎按*反向拓扑顺序*遍历计算图：

        1. 如果输出是标量，用 1 作为初始梯度（否则要求显式传入梯度）
        2. 对该张量可达的所有 Function 节点做拓扑排序
        3. 按反向顺序处理节点：读取输出上累积的梯度，调用该节点的
           backward()，把得到的梯度推送到输入上（累加，绝不覆盖）
        """
        # -- (1) 校验输出张量 ------------------------------------------------- #
        if gradient is None:
            if self.ndim != 0:
                raise RuntimeError(
                    "只有标量输出才能隐式创建梯度。"
                    f"当前张量形状为 {self.shape}。"
                    "请通过 y.backward(gradient=...) 显式传入梯度。"
                )
            gradient = np.array(1.0)
        else:
            if isinstance(gradient, Tensor):
                gradient = gradient.data
            gradient = as_float_array(gradient)
            if gradient.shape != self.shape:
                raise RuntimeError(
                    f"梯度形状 {gradient.shape} 与张量形状 {self.shape} 不匹配"
                )

        if self.grad_fn is None:
            if not self.requires_grad:
                raise RuntimeError("在不需要梯度的张量上调用了 backward()")
            self._accumulate_grad(gradient)  # 标量叶子：直接设置它的梯度
            return

        # -- (3) 构建计算图节点的反向拓扑顺序 -------------------------------- #
        order = _reverse_topological_order(self)

        # 先快照并清空每个节点输出的累积梯度（包括根节点，它的 grad_fn 是
        # order[0]）。否则第二次调用 backward() 会复用上一次的脏梯度。
        snapshots = {id(fn.output): fn.output.grad for fn in order}
        for fn in order:
            fn.output.grad = None

        # -- (4) 播种输出梯度 ------------------------------------------------ #
        # 必须在清空之后进行，否则新播种的梯度会被清掉。
        self._accumulate_grad(gradient)

        # -- (5) 按反向拓扑顺序处理节点 -------------------------------------- #
        for fn in order:
            # 上游梯度：该节点输出上已经累积的所有梯度
            # （所有下游消费者都在我们之前执行完了）。
            up_grad = fn.output.grad
            grads = fn.backward(up_grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for inp, g in zip(fn.inputs, grads):
                if isinstance(inp, Tensor) and inp.requires_grad and g is not None:
                    inp._accumulate_grad(g)

        # 恢复之前累积的梯度，这样重复调用 backward() 是累加而不是覆盖。
        for fn in order:
            prev = snapshots[id(fn.output)]
            if prev is not None:
                fn.output._accumulate_grad(prev)

    # ------------------------------------------------------------------ #
    #  类数组的便捷方法                                                     #
    # ------------------------------------------------------------------ #
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        from . import ops

        return ops.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        from . import ops

        return ops.mean(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: int) -> "Tensor":
        from . import ops

        return ops.reshape(self, shape)

    def transpose(self, axes: Optional[Tuple[int, int]] = None) -> "Tensor":
        from . import ops

        return ops.transpose(self, axes)

    # ------------------------------------------------------------------ #
    #  Python 运算符重载                                                  #
    # ------------------------------------------------------------------ #
    def __add__(self, other) -> "Tensor":
        from . import ops

        return ops.add(self, other)

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other) -> "Tensor":
        from . import ops

        return ops.sub(self, other)

    def __rsub__(self, other) -> "Tensor":
        from . import ops

        return ops.sub(other, self)

    def __mul__(self, other) -> "Tensor":
        from . import ops

        return ops.mul(self, other)

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Tensor":
        from . import ops

        return ops.div(self, other)

    def __rtruediv__(self, other) -> "Tensor":
        from . import ops

        return ops.div(other, self)

    def __neg__(self) -> "Tensor":
        from . import ops

        return ops.neg(self)

    def __pow__(self, exponent: float) -> "Tensor":
        from . import ops

        return ops.pow(self, exponent)

    def __matmul__(self, other) -> "Tensor":
        from . import ops

        return ops.matmul(self, other)

    def __rmatmul__(self, other) -> "Tensor":
        from . import ops

        return ops.matmul(other, self)

    # ------------------------------------------------------------------ #
    #  repr                                                               #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        fn = f", grad_fn={self.grad_fn}" if self.grad_fn is not None else ""
        grad = f", grad={self.grad}" if self.grad is not None else ""
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"requires_grad={self.requires_grad}{fn}{grad})"
        )

    def __str__(self) -> str:
        return (
            f"Tensor({self.data},\n requires_grad={self.requires_grad}"
            f", grad_fn={self.grad_fn})"
        )


# --------------------------------------------------------------------------- #
#  模块级工厂函数                                                              #
# --------------------------------------------------------------------------- #
def tensor(data, requires_grad: bool = False) -> Tensor:
    """从数字 / 列表 / ndarray / Tensor 创建一个叶子 Tensor。"""
    if isinstance(data, Tensor):
        return Tensor(data.data, requires_grad=requires_grad or data.requires_grad)
    return Tensor(data, requires_grad=requires_grad)


def as_tensor(data) -> Tensor:
    """把 data 转换成 Tensor 但不追踪梯度；Tensor 输入原样返回。"""
    if isinstance(data, Tensor):
        return data
    return Tensor(data)


def _from_function(fn: Function, *inputs: Union[Tensor, float, np.ndarray]) -> Tensor:
    """``ops.py`` 中每个算子共用的核心管道。

    给定一个 Function 和它的原始输入，它会：

    1. 执行前向计算
    2. 判断输出是否需要梯度（梯度开关打开 且 任一输入需要梯度）
    3. 如果需要，就把 Function 节点挂到输出 Tensor 上（``grad_fn``），
       从而构建计算图的一条边

    必须设置 ``fn.output``，这样反向引擎才能读到上游梯度。
    """
    # 把非 Tensor 输入（数字/ndarray）规整成叶子 Tensor。
    tensors = tuple(_wrap(i) for i in inputs)
    raw_out = fn.forward(*[t.data for t in tensors])

    needs_grad = grad_mode.is_grad_enabled() and any(t.requires_grad for t in tensors)

    out = Tensor(raw_out, requires_grad=needs_grad)
    if needs_grad:
        fn.inputs = tensors
        fn.output = out
        out.grad_fn = fn
        out.is_leaf = False
    return out


def _reverse_topological_order(root: Tensor) -> list:
    """返回计算图中按反向执行顺序排列的 Function 节点列表。

    为什么需要拓扑排序而不是简单的递归？

    考虑一个菱形计算图::

              x
             / \\
            a   b        (a = x*x, b = x*2, y = a + b)
             \\ /
              y

    反向传播必须等到 ``a`` 和 ``b`` 都被求导之后才能到达 ``x``，
    因为 x 的梯度是 ``da/dx + db/dx``。简单的递归可能会先沿某一条路径
    到达 ``x`` 并处理它，而另一条路径的梯度还没有贡献上来。拓扑排序
    保证：一个节点只有当它输出的所有消费者都被处理完之后才会被处理，
    所以当我们对某个节点求导时，它的输出张量上已经积累了*完整*的上游
    梯度。

    这个函数从根节点开始做 DFS，只有当某个节点所有输入对应的节点
    （在前向图中更靠近叶子）都被加入之后，才把该节点加入列表；
    最后把列表反转即可得到我们需要的处理顺序。
    """
    visited: set = set()
    order: list = []

    def dfs(t: Tensor) -> None:
        fn = t.grad_fn
        if fn is None or id(fn) in visited:
            return
        visited.add(id(fn))
        for inp in fn.inputs:
            dfs(inp)
        order.append(fn)

    dfs(root)
    order.reverse()  # 输出侧的节点排在前面，叶子排在最后
    return order
