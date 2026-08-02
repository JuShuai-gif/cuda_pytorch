"""以 Module 形式实现的常用激活函数。"""

from __future__ import annotations

from .. import ops
from .module import Module


class ReLU(Module):
    def forward(self, x):
        return ops.relu(x)


class Sigmoid(Module):
    def forward(self, x):
        return ops.sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        return ops.tanh(x)
