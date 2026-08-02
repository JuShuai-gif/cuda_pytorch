"""所有神经网络模块的基类（对应 torch.nn.Module）。"""

from __future__ import annotations

from typing import Iterator, Tuple

from .parameter import Parameter


class Module:
    """所有 nn 模块的基类。

    一个 Module 追踪两类子对象：

    - 参数      （``Parameter`` 实例），通过 ``self._parameters``
    - 子模块    （``Module`` 实例），通过 ``self._modules``

    ``__setattr__`` 会拦截属性赋值，因此 ``self.weight = Parameter(...)``
    和 ``self.fc1 = Linear(...)`` 都会被自动注册。
    """

    def __init__(self) -> None:
        self._parameters: dict = {}
        self._modules: dict = {}
        self.training: bool = True

    def __setattr__(self, name: str, value) -> None:
        # 注册 Parameter 和子 Module，这样 parameters() 才能找到它们。
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif name in ("_parameters", "_modules"):
            object.__setattr__(self, name, value)
            return
        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[Parameter]:
        """产出本模块以及所有子模块的全部 Parameter。"""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        """产出 (name, Parameter) 对，名字是 模块.路径 的样式。"""
        for name, param in self._parameters.items():
            yield name, param
        for mname, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{mname}.{name}", param

    def zero_grad(self) -> None:
        """把所有参数的梯度清零（只清梯度，不动数据）。"""
        for param in self.parameters():
            param.zero_grad()

    def train(self) -> None:
        """把本模块（以及所有子模块）切换到训练模式。"""
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self) -> None:
        """把本模块（以及所有子模块）切换到评估模式。"""
        self.training = False
        for module in self._modules.values():
            module.eval()

    def __call__(self, *args, **kwargs):
        # ``module(x)`` 会转发到 ``forward(x)``。
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("子类必须实现 forward()")

    def __repr__(self) -> str:
        child = [f"  ({k}): {v}" for k, v in self._modules.items()]
        body = "\n".join(child)
        return f"{type(self).__name__}(\n{body}\n)" if body else type(self).__name__
