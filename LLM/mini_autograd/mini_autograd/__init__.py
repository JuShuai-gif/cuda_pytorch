"""mini_autograd - 一个从零实现的微型自动微分引擎，灵感来自 PyTorch。

用法:

    from mini_autograd import Tensor, no_grad, tensor

    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x ** 2 + x
    y.backward()
    print(x.grad)   # [3.0, 5.0]
"""

from .tensor import Tensor, tensor, as_tensor
from . import ops
from .grad_mode import no_grad, enable_grad, is_grad_enabled, set_grad_enabled
from . import nn
from . import optim

__all__ = [
    "Tensor",
    "tensor",
    "as_tensor",
    "ops",
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "set_grad_enabled",
    "nn",
    "optim",
]

__version__ = "0.1.0"
