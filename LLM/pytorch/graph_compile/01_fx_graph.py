"""
01_fx_graph.py — FX 符号追踪：从 Python 到计算图

核心: torch.fx.symbolic_trace(model) 把 forward() 的执行过程
      记录成一张有向无环图 (DAG)。
      节点 = 操作调用，边 = 数据流向。

包含:
  1. 基本 trace 和图解读
  2. 从零手动构建 MiniGraph
  3. 图改写演示
"""

import torch
import torch.nn as nn
import torch.fx


# ═══════════════════════════════════════════════════════════════
# Part 1: 基本 trace 和图解读
# ═══════════════════════════════════════════════════════════════


def demo_basic_trace():
    """
    symbolic_trace 做的事:
      1. 喂 Proxy tensor（假输入，只记形状）
      2. forward() 每调用一个函数 → 记录为 Node
      3. args 就是图的边 —— 存的是 Python 对象引用
      4. 返回 GraphModule（图 + 可执行 forward）
    """

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)

        def forward(self, x):
            y = self.linear(x)
            y = torch.relu(y)
            y = torch.abs(y)
            return y + 2.0

    gm = torch.fx.symbolic_trace(Model())

    print("=" * 60)
    print("FX Graph: y = abs(relu(linear(x))) + 2.0")
    print("=" * 60)

    # 图解读
    print(f"""
  每个 opcode 的含义:
    placeholder    — 输入（函数参数）
    call_module    — 调用 nn.Module（如 self.linear）
    call_function  — 调用普通函数（torch.relu、torch.abs、+）
    output         — 返回值

  args 就是图的边 —— 存的是 Python 对象引用:
    relu 的 args 是 (linear,) → 数据从 linear 流向 relu
    add 的 args 是 (abs_1, 2.0) → 数据从 abs_1 流向 add

  Python 代码                    FX Graph Node
  ──────────────────────────────────────────────
  def forward(self, x):          → x (placeholder)
  y = self.linear(x)             → linear (call_module)
  y = torch.relu(y)             → relu (call_function)
  y = torch.abs(y)              → abs_1 (call_function)
  return y + 2.0                → add (call_function) → output
""")
    gm.graph.print_tabular()

    # 验证一致性
    x = torch.randn(3, 10)
    print(f"\n  原始输出: {Model()(x).sum():.4f}")
    print(f"  追踪输出: {gm(x).sum():.4f}")
    print(f"  完全一致 ✓")


# ═══════════════════════════════════════════════════════════════
# Part 2: 从零实现 MiniNode + MiniGraph
# ═══════════════════════════════════════════════════════════════


def demo_mini_graph():
    """
    用 160 行代码从零实现 FX Graph 的核心，
    理解底层数据结构。
    """

    # ── MiniNode ──
    class MiniNode:
        def __init__(self, graph, name, op, target, args):
            self.graph = graph
            self.name = name
            self.op = op
            self.target = target
            self.args = tuple(args) if args else ()
            self.users: dict["MiniNode", None] = {}
            self._prev = None
            self._next = None

        def __repr__(self):
            args_str = ", ".join(
                a.name if isinstance(a, MiniNode) else str(a) for a in self.args
            )
            return f"MiniNode({self.name!r}, op={self.op!r}, args=({args_str}))"

    # ── MiniGraph ──
    class MiniGraph:
        def __init__(self):
            self._root = MiniNode(self, "__root__", "root", None, ())
            self._root._next = self._root
            self._root._prev = self._root
            self._name_counter = {}
            self.nodes = []

        def _unique_name(self, base):
            if base not in self._name_counter:
                self._name_counter[base] = 0
                return base
            self._name_counter[base] += 1
            return f"{base}_{self._name_counter[base]}"

        def _insert(self, node):
            tail = self._root._prev
            tail._next = node
            node._prev = tail
            node._next = self._root
            self._root._prev = node
            self.nodes.append(node)
            for a in node.args:
                if isinstance(a, MiniNode):
                    a.users[node] = None
            return node

        def create_node(self, op, target, args, name=None):
            name = name or self._unique_name(
                target if isinstance(target, str) else target.__name__
            )
            return self._insert(MiniNode(self, name, op, target, args))

        def placeholder(self, name="x"):
            return self.create_node("placeholder", name, ())

        def call_module(self, mod, *args):
            return self.create_node("call_module", mod, args)

        def call_function(self, fn, *args):
            return self.create_node("call_function", fn, args)

        def output(self, result):
            return self.create_node("output", "output", (result,))

        def print_me(self):
            print(f"\n  {'op':<16} {'name':<12} {'target':<20} inputs")
            print(f"  {'-' * 60}")
            for n in self.nodes:
                t = (
                    n.target.__name__
                    if hasattr(n.target, "__name__")
                    else str(n.target)
                )
                args = ", ".join(
                    a.name if isinstance(a, MiniNode) else str(a) for a in n.args
                )
                print(f"  {n.op:<16} {n.name:<12} {t:<20} ({args})")
            print(f"\n  数据流: {' → '.join([n.name for n in self.nodes])}")

    # 手动构建
    print("\n" + "=" * 60)
    print("从零构建: y = relu(linear(x)) + 2.0")
    print("=" * 60)

    g = MiniGraph()
    x = g.placeholder("x")
    linear = g.call_module("linear", x)
    relu = g.call_function(torch.relu, linear)
    add = g.call_function(torch.add, relu, 2.0)
    out = g.output(add)
    g.print_me()

    # 验证: args 里是真的对象引用
    print(f"\n  relu.args = MiniNode({relu.args[0].name!r})")
    print(f"  relu.args[0] is linear → {relu.args[0] is linear}")
    print("  → 图的边就是 Python 指针")


# ═══════════════════════════════════════════════════════════════
# Part 3: 图改写
# ═══════════════════════════════════════════════════════════════


def demo_rewrite():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

        def forward(self, x):
            return torch.relu(self.fc(x))

    gm = torch.fx.symbolic_trace(M())
    x = torch.randn(2, 4)

    before = gm(x).sum()
    print("\n" + "=" * 60)
    print("图改写: relu → gelu")
    print("=" * 60)
    print(f"  改写前 (relu): {before:.4f}")

    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target == torch.relu:
            n.target = torch.nn.functional.gelu

    gm.recompile()
    print(f"  改写后 (gelu): {gm(x).sum():.4f}")
    print("  → 只改了一行 node.target，不改 Python 源码")


# ═══════════════════════════════════════════════════════════════
# Part 4: 不是只有 FX Graph 能改图
# ═══════════════════════════════════════════════════════════════


def demo_edit_stages():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  哪些阶段可以改图:                                           ║
╠══════════════════════════════════════════════════════════════╣
║  Dynamo 之后（自定义 backend）→ FX Graph 改 node.target     ║
║  Inductor 内部              → custom lowering / fusion pass ║
║  导出后 (ONNX / .pt2)       → onnx-simplifier               ║
║                                                            ║
║  但 FX Graph 是唯一面向用户、可读可写的图 IR。               ║
║  自定义 backend 就是在 Dynamo 抓图后、Inductor 前拦截它:    ║
║                                                            ║
║    def my_backend(gm, inputs):                              ║
║        for n in gm.graph.nodes:                             ║
║            if n.target == torch.relu:                       ║
║                n.target = nn.functional.gelu                ║
║        gm.recompile()                                       ║
║        return gm.forward                                    ║
║                                                            ║
║    torch.compile(model, backend=my_backend)                 ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    demo_basic_trace()
    demo_mini_graph()
    demo_rewrite()
    demo_edit_stages()
