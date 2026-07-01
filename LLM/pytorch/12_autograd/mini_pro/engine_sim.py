"""
Autograd Engine Internals — Dependency Counting & Topological Execute
=====================================================================
模拟 PyTorch Engine::execute() 的核心逻辑:
  1. 依赖计数 (dependency_count)
  2. 拓扑排序执行
  3. 梯度累积

对标源码: torch/csrc/autograd/engine.cpp

运行: python 06_engine_sim.py
"""

import sys
from collections import defaultdict


class FakeNode:
    """模拟 Autograd Node (对标 torch::autograd::Node)"""

    def __init__(self, name: str, inputs: list = None, grad_fn=None):
        self.name = name
        self.inputs = inputs or []       # 前驱节点
        self.outputs = []                # 后继节点 (拓扑反向)
        self.dependency_count = 0        # 未完成的依赖数
        self.grad = 0.0                  # 累积的梯度
        self.grad_fn = grad_fn or (lambda: None)  # 局部梯度函数

    def add_input(self, node: "FakeNode"):
        self.inputs.append(node)
        node.outputs.append(self)

    def __repr__(self):
        return f"Node({self.name}, deps={self.dependency_count}, grad={self.grad:.2f})"


def simulate_engine():
    """模拟 PyTorch Engine::execute()"""
    print("=" * 60)
    print("1. 模拟 Engine::execute(): 依赖计数 + 拓扑执行")
    print("=" * 60)

    # 构建图: x -> a -> b -> y,  x -> c -> y
    x = FakeNode("x")
    a = FakeNode("a", inputs=[x], grad_fn=lambda: print("  a.grad_fn: d(a)/d(x)=2"))
    b = FakeNode("b", inputs=[a], grad_fn=lambda: print("  b.grad_fn: d(b)/d(a)=1"))
    c = FakeNode("c", inputs=[x], grad_fn=lambda: print("  c.grad_fn: d(c)/d(x)=3"))
    y = FakeNode("y", inputs=[b, c], grad_fn=lambda: print("  y.grad_fn: dy/db=1, dy/dc=1"))

    # Engine 初始化: 对每个输出节点, 计算 dependency_count
    all_nodes = [y]  # output node

    # Phase 1: 从输出向输入 BFS, 构建依赖计数
    queue = [y]
    visited = {y}
    node_list = []
    while queue:
        node = queue.pop(0)
        node_list.append(node)
        for inp in node.inputs:
            inp.dependency_count += 1  # 每个 input 增加一个未完成的依赖
            if inp not in visited:
                visited.add(inp)
                queue.append(inp)

    print("  初始化依赖计数:")
    for n in node_list:
        print(f"    {n.name}: dependency_count = {n.dependency_count}")

    # Phase 2: 反向执行 (从输出到输入)
    print("\n  反向执行 (Engine::evaluate_function):")
    y.grad = 1.0  # 输出梯度初始化为 1

    ready = [y]
    while ready:
        node = ready.pop()
        print(f"    -> 执行 {node.name} (grad={node.grad:.2f})")
        node.grad_fn()

        for inp in node.inputs:
            inp.dependency_count -= 1
            if inp.dependency_count == 0:
                ready.append(inp)

    print("\n  对标 PyTorch 源码:")
    print("    torch/csrc/autograd/engine.cpp: Engine::execute()")
    print("    1. compute_dependencies() -> 计算每个 Node 的前驱数")
    print("    2. ReadyQueue -> 就绪节点队列")
    print("    3. evaluate_function() -> 执行 grad_fn")


def real_autograd_check():
    """验证 PyTorch 的依赖计数行为"""
    print("\n" + "=" * 60)
    print("2. PyTorch autograd 中的依赖计数验证")
    print("=" * 60)

    import torch
    x = torch.tensor(2.0, requires_grad=True)
    a = x * 2      # grad_fn = MulBackward0, 依赖数 = 1 (x)
    b = a + 1      # grad_fn = AddBackward0, 依赖数 = 1 (a)
    c = x * 3      # grad_fn = MulBackward0, 依赖数 = 1 (x)
    y = b + c      # grad_fn = AddBackward0, 依赖数 = 2 (b, c)

    # x 被 a 和 c 使用 -> dependency_count(x) = 2
    y.backward()

    print(f"  a.grad_fn.next_functions: {a.grad_fn.next_functions}")
    print(f"    -> x 是 a 的唯一输入")
    print(f"  y.grad_fn.next_functions: {y.grad_fn.next_functions}")
    print(f"    -> b 和 c 是 y 的输入, x 被两个分支引用")
    print(f"  x.grad = {x.grad.item()}  (x 被使用 2 次, 梯度累加)")


def demonstrate_accumulate():
    """演示梯度累加"""
    print("\n" + "=" * 60)
    print("3. 梯度累加: 多次 backward 不自动清零")
    print("=" * 60)

    import torch
    x = torch.tensor(3.0, requires_grad=True)

    for i in range(3):
        y = x * (i + 1)
        y.backward()
        print(f"  backward #{i+1}: x.grad = {x.grad.item()}  (累加)")

    # 手动清零
    x.grad.zero_()
    print(f"  手动 zero_() 后: x.grad = {x.grad.item()}")

    print("\n  对标: PyTorch 默认累加梯度，需手动 zero_grad()")


EXPERIMENTS = {
    "sim": simulate_engine,
    "real": real_autograd_check,
    "accum": demonstrate_accumulate,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}'")
            continue
        EXPERIMENTS[name]()
    print("[autograd engine sim] DONE")


if __name__ == "__main__":
    main()
