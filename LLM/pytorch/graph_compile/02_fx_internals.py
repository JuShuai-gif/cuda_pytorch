"""
01_fx_trace.py 背后的数据结构和代码实现

当你看到这张表输出时：

  placeholder  x     x       ()       {}
  call_module  linear linear  (x,)     {}
  call_function relu  <relu>  (linear,)  {}

背后是三个核心 Python 类在协作：

  Node        — 图中的一个操作节点
  Graph       — 由 Node 组成的 DAG（双向链表 + 字典索引）
  GraphModule — 图 + nn.Module 参数的组合体，可像普通模型一样调用

本文逐层剥开看它们的内存结构和实现方式。
"""

import torch
import torch.fx
import torch.nn as nn


# ═══════════════════════════════════════════════════
# 第 1 层：Node —— 图中的单个节点
# ═══════════════════════════════════════════════════


def layer1_node():
    """
    Node 的简化版实现大概是这样的：

    class Node:
        def __init__(self, graph, name, op, target, args, kwargs):
            self.graph = graph        # 所属的 Graph
            self.name = name          # 唯一标识，如 'linear', 'relu'
            self.op = op              # placeholder | call_function | call_module | output | get_attr
            self.target = target      # 被调用的目标（函数、模块名、字符串）
            self.args = args          # 位置参数（Node 或值的 tuple）
            self.kwargs = kwargs      # 关键字参数
            self._input_nodes = {}    # {node: uses} — 上游依赖（谁喂数据给我）
            self.users = {}           # {node: uses} — 下游依赖（谁要吃我的输出）
            self._prev = None         # 双向链表：前驱
            self._next = None         # 双向链表：后继

        @property
        def all_input_nodes(self):
            # 返回所有作为参数传入的 Node 对象
            return ...

    Node 在内存中存储什么（实际值 vs 文本显示）:
    """

    # 构建一张真实的图并拆开看
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 8)

        def forward(self, x):
            return torch.relu(self.linear(x)) + 2.0

    gm = torch.fx.symbolic_trace(M())

    print("=== Node 内部字段 ===")
    for node in gm.graph.nodes:
        print(f"\n  Node name={node.name!r}")
        print(f"    .op         = {node.op!r}")
        print(f"    .target     = {node.target!r}")
        print(f"    .args       = {node.args!r}")
        print(f"    .kwargs     = {node.kwargs!r}")
        print(f"    .type       = {type(node)}")
        print(f"    ._input_nodes = {dict(node._input_nodes)}")
        print(f"    .users        = {[u.name for u in node.users.keys()]}")

        # args 里存的是真实的 Node 对象
        for i, a in enumerate(node.args):
            if isinstance(a, torch.fx.Node):
                print(
                    f"    .args[{i}]      = Node({a.name}) ← 这是另一个 Node 对象的引用"
                )
            else:
                print(f"    .args[{i}]      = {a} ({type(a).__name__}) ← 常量")


# ═══════════════════════════════════════════════════
# 第 2 层：Graph —— 节点容器
# ═══════════════════════════════════════════════════


def layer2_graph():
    """
    Graph 的简化实现：

    class Graph:
        def __init__(self):
            self._root = Node(...)         # 哨兵节点，链表的头
            self._len = 0                  # 节点数
            self._nodes_by_name = {}       # {name: Node} 快速查找
            self._used_names = {}          # 名字去重计数
            self._owns_graphmodule = None  # 反向引用到 GraphModule

        def create_node(self, op, target, args, kwargs, name):
            # 1. 生成唯一名字（如 'relu', 'relu_1', 'relu_2'）
            # 2. 创建 Node 对象
            # 3. 插入双向链表
            # 4. 注册到 _nodes_by_name
            # 5. 更新输入节点的 users
            ...

    所以 Graph 本质上是一个：
      - 双向链表（Node._prev / Node._next）
      - 字典索引（按名字查 Node）
      - 每个 Node 维护自己的 users / _input_nodes
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 8)

        def forward(self, x):
            return torch.relu(self.linear(x))

    gm = torch.fx.symbolic_trace(M())

    print("\n=== Graph 内部字段 ===")
    g = gm.graph
    print(f"  type(graph)           = {type(g)}")
    print(f"  len(graph.nodes)      = {len(g.nodes)} nodes")
    print(f"  graph._used_names     = {g._used_names}")
    print(f"  graph._graph_namespace = {g._graph_namespace}")

    # 内部字典字段名可能随 PyTorch 版本变化，直接用公共 API
    node_names = [n.name for n in g.nodes]
    print(f"  nodes (by iteration)  = {node_names}")
    print(f"  graph._owning_module  = {type(g._owning_module).__name__}")

    # 看双向链表
    print(f"\n  双向链表遍历:")
    cur = g._root._next
    while cur and cur.op != "root":
        print(f"    {cur.name} → ", end="")
        cur = cur._next
    print("(end)")

    # 反向遍历
    print(f"  反向遍历:")
    cur = g._root._prev
    while cur and cur.op != "root":
        print(f"    {cur.name} → ", end="")
        cur = cur._prev
    print("(end)")


# ═══════════════════════════════════════════════════
# 第 3 层：args 里存的是真的 Node 对象
# ═══════════════════════════════════════════════════


def layer3_args_are_nodes():
    """
    这在 print_tabular 里被简化成字符串了，但实际上 args 存的是
    真正的 Node 对象引用。也就是说图的「边」就是 Python 指针。
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

        def forward(self, x):
            h = self.fc(x)
            h = torch.relu(h)
            return torch.cat([h, h * 2])  # cat 有两个输入 → 图分叉了

    gm = torch.fx.symbolic_trace(M())

    print("\n=== args 里的 Node 对象（从表里看不到的真实内存结构） ===")
    for node in gm.graph.nodes:
        print(f"\n  {node.name} ({node.op})")
        for i, a in enumerate(node.args):
            if isinstance(a, torch.fx.Node):
                print(f"    args[{i}] = Node(name={a.name!r}, op={a.op!r}, id={id(a)})")
            elif isinstance(a, (tuple, list)):
                print(f"    args[{i}] = {type(a).__name__}:")
                for j, item in enumerate(a):
                    if isinstance(item, torch.fx.Node):
                        print(f"      [{j}] = Node(name={item.name!r})")
                    else:
                        print(f"      [{j}] = {item} ({type(item).__name__})")
            else:
                print(f"    args[{i}] = {a} ({type(a).__name__})")


# ═══════════════════════════════════════════════════
# 第 4 层：GraphModule —— 让图可执行
# ═══════════════════════════════════════════════════


def layer4_graphmodule():
    """
    GraphModule 把 Graph 和 nn.Module 的 state_dict 绑在一起。

    简化实现：
    class GraphModule(nn.Module):
        def __init__(self, root, graph):
            super().__init__()
            self.graph = graph
            # 把原模型的参数拷贝过来
            self.__dict__.update(root.__dict__)
            # 生成可执行的 Python 代码（compile graph → code → python function）
            self.recompile()

        def recompile(self):
            # 核心：把 Graph 节点列表翻译成 Python 源码
            code = self.graph.python_code(root_module='self')
            # 然后用 exec 执行这段代码，生成 forward 方法
            exec(code, global_dict)
            self.forward = global_dict['forward']

    翻译过程大概是这样：

      Graph:  x → linear → relu → output
      ↓ 翻译成 Python 代码
      def forward(self, x):
          linear = self.linear(x)     # call_module
          relu = torch.relu(linear)   # call_function
          return relu                 # output
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)

        def forward(self, x):
            return torch.relu(self.fc1(x)) + 1.0

    gm = torch.fx.symbolic_trace(M())

    print("\n=== GraphModule 内部 ===")
    print(f"  type(gm)          = {type(gm).__name__}")
    print(f"  gm.graph          = {type(gm.graph).__name__}")
    print(f"  gm.code           = 生成的 Python 源码 (见下方):")

    # print(gm.code) 是编译后的 Python 源码
    print(f"\n  ── gm.code (graph → python 代码) ──")
    print(gm.code)

    print(f"\n  ── gm 包含的参数 ──")
    for name, p in gm.named_parameters():
        print(f"    {name}: {tuple(p.shape)}")

    # 调用方式和普通模型完全一样
    x = torch.randn(2, 4)
    print(f"\n  gm(x)   = {gm(x).sum():.4f}    ← 可以像普通模型一样调用")
    print(f"  M()(x)   = {M()(x).sum():.4f}    ← 和原始模型结果一致")


# ═══════════════════════════════════════════════════
# 第 5 层：Proxy —— 图的「木马」构建器
# ═══════════════════════════════════════════════════


def layer5_proxy():
    """
    Proxy 是 symbolic_trace 用来「骗」forward 的假 tensor。

    它伪装成 Tensor，重载了所有运算符：
      __add__, __matmul__, __mul__, ...

    但实际不做计算，而是在 Graph 上创建新 Node。

    简化实现：
    class Proxy:
        def __init__(self, node, tracer):
            self.node = node     # 对应 Graph 中的 Node
            self.tracer = tracer

        def __add__(self, other):
            # 在 Graph 中创建一个 call_function(torch.add) 的 Node
            return self.tracer.create_proxy(
                "call_function", torch.add, (self, other), {}
            )

        @property
        def shape(self):
            # 从图推导形状，不实际计算
            ...

    于是 forward 里的代码：
      y = self.linear(x)      # x 是 Proxy → linear 返回新 Proxy
      z = y + 2.0             # y 是 Proxy → + 被 Proxy.__add__ 拦截
                              #   → 在 Graph 中创建 add Node
                              #   → 返回新 Proxy 指向这个 add Node
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

        def forward(self, x):
            y = self.fc(x)
            z = y + 2.0
            return z

    gm = torch.fx.symbolic_trace(M())

    print("=== Proxy 的工作过程 ===")
    print()
    print("forward 中的 Python 代码:")
    print("  y = self.fc(x)       # x 是 Proxy → fc 内部调用被拦截")
    print("                       #   → Graph 中创建 call_module Node")
    print("                       #   → 返回新 Proxy 指向这个 Node")
    print()
    print("  z = y + 2.0          # y 是 Proxy → Proxy.__add__ 被拦截")
    print("                       #   → Graph 中创建 call_function(add) Node")
    print("                       #   → 返回新 Proxy 指向这个 add Node")
    print()
    print("  return z             # z 是 Proxy → 记录为 output Node")
    print()
    print("最终 Graph 里的节点:")

    for node in gm.graph.nodes:
        if node.op != "output":
            print(f"  {node.name} ({node.op}): {node.target}")


# ═══════════════════════════════════════════════════
# 第 6 层：完整的对象关系图
# ═══════════════════════════════════════════════════


def layer6_summary():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                 FX Graph 完整对象关系图                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  GraphModule (nn.Module 子类)                               ║
║   ├── graph: Graph                                          ║
║   │    ├── _root: Node (哨兵，链表头尾)                      ║
║   │    ├── _len: int                                        ║
║   │    ├── _nodes_by_name: {str → Node}                     ║
║   │    └── _owning_module → 指回 GraphModule               ║
║   │                                                         ║
║   │    双向链表: _root ↔ x ↔ linear ↔ relu ↔ output ↔ _root ║
║   │                                                         ║
║   │    每个 Node:                                            ║
║   │      .name      = "relu"                                ║
║   │      .op        = "call_function"                        ║
║   │      .target    = <built-in function relu>              ║
║   │      .args      = (Node("linear"),)   ← 真的 Node 对象  ║
║   │      ._input_nodes = {Node("linear"): 1}                ║
║   │      .users        = {Node("abs_1"): 1}                 ║
║   │                                                         ║
║   ├── parameters (来自原始模型):                             ║
║   │    fc1.weight: (8, 4)                                   ║
║   │    fc1.bias:   (8,)                                     ║
║   │                                                         ║
║   └── code: str  ← graph.python_code() 生成的 Python 源码   ║
║        def forward(self, x):                                 ║
║            linear = self.linear(x)                           ║
║            relu = torch.relu(linear)                         ║
║            return relu                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ═══════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("FX Graph 底层：数据结构和代码实现")
    print("=" * 60)

    layer1_node()
    print("\n\n")

    layer2_graph()
    print("\n\n")

    layer3_args_are_nodes()
    print("\n\n")

    layer4_graphmodule()
    print("\n\n")

    layer5_proxy()
    print("\n\n")

    layer6_summary()
