# Mini Autograd — 从零实现 PyTorch 自动微分引擎

> 对照 PyTorch 源码 `torch/csrc/autograd/engine.cpp`，用纯 Python 实现一个最小可运行的自动微分系统。

## 项目结构

```
mini_pro/
├── mini_autograd.py   # 核心引擎: Value + backward() + 拓扑排序   (run me first)
├── graph_trace.py     # 追踪 PyTorch autograd 图 (grad_fn, next_functions)
├── engine_sim.py      # 模拟 Engine::execute() 依赖计数
└── higher_order.py    # 高阶导数 & 实战 (WGAN-GP, Hessian-vector product)
```

## 快速开始

```bash
cd 12_autograd/mini_pro

# 1. 运行 mini 引擎 (对标 PyTorch, 自动验证)
python mini_autograd.py

# 2. 追踪 PyTorch 计算图
python graph_trace.py

# 3. 模拟 Engine 依赖计数
python engine_sim.py

# 4. 高阶导数实战
python higher_order.py
```

## 核心概念对照

| Mini Autograd | PyTorch | 源码位置 |
|---------------|---------|----------|
| `Value.data` | `Tensor.data` | `c10/core/TensorImpl.h` |
| `Value.grad` | `Tensor.grad` | `torch/csrc/autograd/variable.h` |
| `Value._prev` | `Node.next_edges()` | `torch/csrc/autograd/edge.h` |
| `Value._backward` | `Node.apply()` | `torch/csrc/autograd/functions/` |
| `Value.backward()` | `Tensor.backward()` | `torch/_tensor.py` → `engine.cpp` |
| 拓扑排序 | `Engine::compute_dependencies()` | `torch/csrc/autograd/engine.cpp:530` |
| 梯度累加 | `variable.grad() += new_grad` | `torch/csrc/autograd/engine.cpp:680` |

## 学习路径

1. **`mini_autograd.py`** — 先读这个。从 `Value` 类开始，理解 `__add__` / `__mul__` 如何注册 `_backward`，然后看 `backward()` 如何拓扑排序 + 链式法则。运行后自动与 PyTorch 比对验证。

2. **`graph_trace.py`** — 在真实 PyTorch 中追踪 `grad_fn` 链和 `next_functions`，观察 Node 之间的拓扑关系。

3. **`engine_sim.py`** — 模拟 PyTorch Engine 的核心算法：从输出节点反向 BFS，计算每个节点的 `dependency_count`，就绪后执行 `grad_fn`。对照源码 `engine.cpp:compute_dependencies()`。

4. **`higher_order.py`** — 理解 `create_graph=True` 如何保留梯度的计算图以支持高阶导数，实战 WGAN-GP 和 Hessian-vector product。

## 为什么叫 "Mini"？

这个实现仅支持 **标量**（对标 PyTorch 中你 `sum()` 之后对 loss 做 backward 的场景），省略了：
- 批量 tensor 的 stride/broadcast 处理（PyTorch 用 `TensorIterator` 处理）
- CUDA 支持（只做 CPU 标量）
- 多线程 Engine（PyTorch 用 `ReadyQueue` + worker threads）
- `retain_graph` / `create_graph` 等高级特性

但所有这些省略恰好让你聚焦核心：**计算图的拓扑排序 + 链式法则传播**。
