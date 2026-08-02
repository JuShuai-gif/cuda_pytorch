# 01 Autograd 总览

## 什么是 Autograd？

Autograd（自动微分）解决一个问题：**给定一个由若干算子组成的复合函数，自动求出它对输入/参数的偏导数。**

在 PyTorch 里，你只需要：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)   # 2*2 + 3 = 7
```

`backward()` 内部完成了两件事：

1. **前向阶段**：逐算子计算结果，同时把"计算图"记录下来。
2. **反向阶段**：从输出往输入走，用**链式法则**把梯度逐层传回去。

本项目 `mini_autograd` 用纯 NumPy 重新实现这一整套机制。

## 核心概念一览

| 概念 | 含义 | 在本项目中的实现 |
| --- | --- | --- |
| Tensor | 数据 + 梯度 + 图信息 | `mini_autograd/tensor.py` |
| 计算图 | 前向执行时隐式构建的有向无环图 | Function 节点互相引用 |
| grad_fn | "这个 Tensor 是谁算出来的" | Tensor 上的 `grad_fn` 属性 |
| 叶子节点 | 用户创建、没有父算子的 Tensor | `is_leaf = True` |
| 链式法则 | 梯度 = 上游梯度 × 局部梯度 | 每个 Function 的 `backward()` |
| 梯度累加 | 一个节点被多条路径使用时梯度相加 | `_accumulate_grad` |
| 反向拓扑排序 | 保证每个节点的梯度先汇总完再向前传 | `_reverse_topological_order` |
| no_grad | 关闭梯度追踪，不建图 | `grad_mode.py` |

## 一个完整的最小例子

```python
from mini_autograd import Tensor

a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)
c = a * b + a
c.backward()

print(a.grad)   # 4.0
print(b.grad)   # 2.0
```

前向时发生了什么：

```
a ──┐
    ├─(*b)──> t ──(+a)──> c
b ──┘                   │
                 c.grad_fn = <Add>
                 t.grad_fn = <Mul>
```

`backward()` 从 `c` 出发，按 `Add -> Mul` 的顺序处理：

1. `c.grad = 1`（标量输出的初始梯度）。
2. `Add.backward`：`dt/dc=1, da/dc=1`，所以 `t.grad += 1`，`a.grad += 1`。
3. `Mul.backward`：`da/d(b)=a=2`，`da/d(a)=b=3`，所以 `b.grad += 2`，`a.grad += 3`。
4. 最终 `a.grad = 1 + 3 = 4`，`b.grad = 2`。

## 为什么不直接用 PyTorch？

PyTorch 的 autograd 用 C++ 实现、考虑了性能、内存、并发等大量工程问题，源码很难读。
本项目剥掉这些，只保留**原理骨架**，让你看清：

- 每个算子如何定义自己的局部梯度
- 反向传播到底按什么顺序跑
- 广播、复用、非标量输出这些"坑"是怎么处理的

读完本项目，再去看 PyTorch 源码或者官方 `extending-autograd` 文档会轻松很多。
