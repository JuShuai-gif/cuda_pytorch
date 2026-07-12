"""
Autograd Graph Visualization & Node Trace
==========================================
从 Python 侧追踪 PyTorch autograd 计算图的结构：
  node.next_functions -> 查看拓扑关系
  node.metadata       -> 查看节点类型
  retain_grad         -> 保留非叶子梯度

运行: python 05_graph_trace.py
"""

import sys
import torch


def trace_graph():
    """追踪 autograd graph 的 Node 结构"""
    print("=" * 60)
    print("1. 追踪计算图: Node / Edge / grad_fn")
    print("=" * 60)

    x = torch.tensor(2.0, requires_grad=True)
    a = x + 1          # AddBackward0
    b = x + 2          # AddBackward0
    y = a * b          # MulBackward0

    print("  grad_fn 链:")
    print(f"    y.grad_fn  = {y.grad_fn}")                     # MulBackward0
    print(f"    a.grad_fn  = {a.grad_fn}")                     # AddBackward0
    print(f"    b.grad_fn  = {b.grad_fn}")                     # AddBackward0
    print(f"    x.grad_fn  = {x.grad_fn}  (leaf)")

    print("\n  next_functions (拓扑边):")
    for i, fn in enumerate(y.grad_fn.next_functions):
        node, idx = fn
        print(f"    MulBackward0.next[{i}]: {node}, output_idx={idx}")

    # 查看 AddBackward0 的前驱
    fn_a = a.grad_fn
    for i, fn in enumerate(fn_a.next_functions):
        node, idx = fn
        print(f"    AddBackward0(a).next[{i}]: {node}, output_idx={idx}")

    print("\n  is_leaf vs grad_fn:")
    for name, t in [("x", x), ("a", a), ("b", b), ("y", y)]:
        print(f"    {name}: is_leaf={t.is_leaf}, requires_grad={t.requires_grad}, grad_fn={'None' if t.grad_fn is None else type(t.grad_fn).__name__}")


def graph_breakdown():
    """复杂的 autograd graph"""
    print("\n" + "=" * 60)
    print("2. 复杂图的 grad_fn 树")
    print("=" * 60)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w = torch.tensor([[0.5, -0.5], [0.2, 0.8]], requires_grad=True)

    h = x @ w             # MmBackward0
    h = h.relu()          # ReluBackward0
    y = h.sum()           # SumBackward0

    y.backward()

    print("  操作链:")
    current = y.grad_fn
    depth = 0
    while current is not None:
        indent = "    " * depth
        print(f"{indent}{type(current).__name__}")
        if hasattr(current, 'next_functions'):
            for fn, _ in current.next_functions:
                if fn is not None:
                    print(f"{indent}  -> {type(fn).__name__}")
        depth += 1
        # 取第一个子节点继续追踪
        if hasattr(current, 'next_functions') and current.next_functions:
            current = current.next_functions[0][0]
        else:
            break

    print(f"\n  x.grad:\n{x.grad}")
    print(f"  w.grad:\n{w.grad}")


def retain_grad_demo():
    """演示 retain_grad 和梯度生命周期"""
    print("\n" + "=" * 60)
    print("3. retain_grad: 非叶子节点梯度保留")
    print("=" * 60)

    x = torch.tensor(3.0, requires_grad=True)
    a = x * 2          # 非叶子
    b = a + 1          # 非叶子
    c = b * b          # 非叶子

    # 默认: 非叶子节点反向传播后梯度被释放
    c.backward()
    print(f"  默认: a.grad={a.grad}, b.grad={b.grad}  (None = 已释放)")

    # 显式保留
    x2 = torch.tensor(3.0, requires_grad=True)
    a2 = x2 * 2
    b2 = a2 + 1
    a2.retain_grad()
    b2.retain_grad()
    (b2 * b2).backward()

    print(f"  保留: a2.grad={a2.grad.item()}, b2.grad={b2.grad.item()}  (保留成功)")

    # 对标: PyTorch Engine 默认仅保留叶子节点梯度 (节省内存)
    # Node::release_variables() 释放中间梯度


EXPERIMENTS = {
    "trace": trace_graph,
    "complex": graph_breakdown,
    "retain": retain_grad_demo,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}'")
            continue
        EXPERIMENTS[name]()
    print("[autograd graph trace] DONE")


if __name__ == "__main__":
    main()
