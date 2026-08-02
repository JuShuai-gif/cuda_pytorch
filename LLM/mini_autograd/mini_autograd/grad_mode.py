"""梯度开关模块，torch.no_grad / set_grad_enabled 的简化版。

本模块控制运算是否构建计算图。
当梯度模式被关闭时：

- 算子返回不带 grad_fn 的普通 Tensor（没有图的边）
- 即使输入需要梯度，输出 requires_grad 也为 False
"""


class _GradMode:
    """全局梯度模式标志，整个库共享同一个实例。"""

    def __init__(self) -> None:
        self.enabled = True

    def is_enabled(self) -> bool:
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled


_global_grad_mode = _GradMode()


def is_grad_enabled() -> bool:
    """返回当前是否开启了梯度计算。"""
    return _global_grad_mode.is_enabled()


def set_grad_enabled(enabled: bool) -> None:
    """全局开启/关闭梯度计算。"""
    _global_grad_mode.set_enabled(enabled)


class no_grad:
    """上下文管理器，在其代码块内关闭梯度计算。

    用法:
        with no_grad():
            y = model(x)          # 不构建计算图
            p.data -= lr * g      # 参数更新永远不会进入计算图
    """

    def __init__(self) -> None:
        self._prev = True

    def __enter__(self) -> "no_grad":
        # 记住进入前的状态，这样嵌套使用时能正确恢复。
        self._prev = is_grad_enabled()
        set_grad_enabled(False)
        return self

    def __exit__(self, *exc) -> None:
        # 离开代码块时恢复之前的梯度开关状态。
        set_grad_enabled(self._prev)


class enable_grad:
    """上下文管理器，在其代码块内重新开启梯度计算。"""

    def __init__(self) -> None:
        self._prev = False

    def __enter__(self) -> "enable_grad":
        self._prev = is_grad_enabled()
        set_grad_enabled(True)
        return self

    def __exit__(self, *exc) -> None:
        set_grad_enabled(self._prev)
