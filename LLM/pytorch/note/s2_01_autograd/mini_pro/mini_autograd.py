"""
Mini Autograd Engine — 从零实现自动微分
===========================================
对标 PyTorch autograd 的核心概念：
  Value  = Variable (持有 data + grad)
  backward() = 拓扑排序 + 链式法则
  grad_fn  = 记录创建该值的操作

运行: python 04_mini_autograd.py
"""

from __future__ import annotations
import math
from typing import List, Set, Tuple


# ============================================================
# 1. 核心数据结构: Value (对标 torch.Tensor 的 autograd 部分)
# ============================================================
class Value:
    """一个标量值, 持有 data、grad 和 backward 函数。

    对标 PyTorch:
      Value.data    <-> Tensor.data
      Value.grad    <-> Tensor.grad
      Value._prev   <-> grad_fn.next_functions (拓扑前驱)
      Value._backward <-> grad_fn.apply()
    """

    def __init__(self, data: float, _children: Tuple[Value, ...] = (), _op: str = ""):
        self.data = data
        self.grad = 0.0       # 累积梯度 (对标 Tensor.grad)
        self._prev = set(_children)  # 计算图中的父节点 (拓扑排序用)
        self._op = _op         # 操作名 (对标 grad_fn.name())
        self._backward = lambda: None  # 局部梯度计算 (对标 autograd.Function.backward)

    # ---- 前向运算 (每个都注册 backward) ----

    def __add__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(out)/d(self) = other.data, d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other: float) -> Value:
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    # ---- 反向传播 (对标 Tensor.backward()) ----

    def backward(self):
        """拓扑排序所有节点, 从输出向输入传播梯度。

        对标 PyTorch:
          torch.autograd.backward() -> Engine::execute()
          1. 拓扑排序 (topological_sort)
          2. 反向遍历调用 grad_fn.apply()
        """
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0  # d(self)/d(self) = 1 (对标 loss.backward())
        for v in reversed(topo):  # 输出→输入 (对标 Engine 的依赖计数)
            v._backward()

    # ---- 辅助 ----

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other) -> Value:
        return self + other

    def __sub__(self, other) -> Value:
        return self + (-other)

    def __rsub__(self, other) -> Value:
        return (-self) + other

    def __rmul__(self, other) -> Value:
        return self * other

    def __truediv__(self, other) -> Value:
        return self * other**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}, op='{self._op}')"


# ============================================================
# 2. 与 PyTorch 对标演示
# ============================================================
def demo_add_mul():
    """对标 PyTorch 的:  x=2; a=x+1; b=x+2; y=a*b; y.backward()"""
    print("=" * 60)
    print("Demo 1: 对标 PyTorch 01_backward.py")
    print("=" * 60)

    # --- 我们的引擎 ---
    x = Value(2.0)
    a = x + 1
    b = x + 2
    y = a * b
    y.backward()

    print("  [Mini]  x.grad =", x.grad, "  (期望: d((x+1)*(x+2))/dx |x=2 = 2*2 + 3 = 7)")

    # --- PyTorch ---
    import torch
    xt = torch.tensor(2.0, requires_grad=True)
    at = xt + 1
    bt = xt + 2
    yt = at * bt
    yt.backward()

    print("  [Torch] x.grad =", xt.grad.item())
    print("  Match:", abs(x.grad - xt.grad.item()) < 1e-6)


def demo_mlp():
    """小 MLP: 2 输入 → 2 隐藏 → 1 输出"""
    print("\n" + "=" * 60)
    print("Demo 2: 小 MLP (2→2→1)")
    print("=" * 60)

    # 权重初始化
    w1_11, w1_12, b1_1 = Value(0.5), Value(-0.3), Value(0.1)
    w1_21, w1_22, b1_2 = Value(0.2), Value(0.8), Value(-0.2)
    w2_1, w2_2, b2 = Value(0.4), Value(-0.5), Value(0.0)

    # 输入
    x1, x2 = Value(1.0), Value(2.0)

    # 前向
    h1 = (w1_11 * x1 + w1_12 * x2 + b1_1).relu()
    h2 = (w1_21 * x1 + w1_22 * x2 + b1_2).relu()
    y = w2_1 * h1 + w2_2 * h2 + b2
    loss = y * y  # MSE

    loss.backward()

    print(f"  Loss: {loss.data:.4f}")
    print(f"  Gradients:")
    for name, p in [("w1_11", w1_11), ("w1_12", w1_12), ("w2_1", w2_1), ("w2_2", w2_2)]:
        print(f"    {name}: {p.grad:.4f}")


def demo_topo():
    """展示拓扑排序"""
    print("\n" + "=" * 60)
    print("Demo 3: 拓扑排序验证")
    print("=" * 60)

    a = Value(3.0)
    b = a + a       # b ← a
    c = b * a       # c ← b,a
    d = c.relu()    # d ← c
    e = d + b       # e ← d,b

    # 拓扑排序: 所有依赖必须在前
    topo = []
    visited = set()

    def dfs(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                dfs(child)
            topo.append(v)

    dfs(e)
    print("  Topo order (leaves first, output last):")
    for i, v in enumerate(topo):
        print(f"    {i}: data={v.data:.1f}, op='{v._op}', deps={len(v._prev)}")

    # PyTorch 对标: Engine 内部也是拓扑排序后反向执行
    print("\n  PyTorch Engine 流程:")
    print("    1. 从 loss 节点 BFS/DFS 收集所有 Node")
    print("    2. 拓扑排序 (依赖计数)")
    print("    3. release_backward() 遍历执行")


if __name__ == "__main__":
    demo_add_mul()
    demo_mlp()
    demo_topo()
    print("\n[Mini Autograd Engine] DONE")
