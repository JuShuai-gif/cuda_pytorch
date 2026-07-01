"""FX Graphs 源码分析: Proxy 追踪, Interpreter, node 内部结构。

使用工具: torch.fx.Proxy / Interpreter / Tracer / node.meta

运行:
  python test3.py                  # 全链路分析
  python test3.py proxy_trace      # Proxy 追踪原理
  python test3.py interpreter      # Interpreter 节点遍历
  python test3.py node_meta        # Node 元数据 (val, stack_trace)

参考源码:
  torch/fx/proxy.py                — Proxy 类 (惰性追踪)
  torch/fx/interpreter.py          — Interpreter (图执行器)
  torch/fx/node.py                 — Node 数据结构
  torch/fx/_symbolic_trace.py      — Tracer
"""

import sys
import torch
import torch.nn as nn
import torch.fx as fx


# ============ 1. Proxy 追踪原理 ============
def exp_proxy_trace():
    """理解 torch.fx.Proxy 如何惰性记录操作。"""
    print("=" * 60)
    print("1. Proxy: 惰性追踪 — 每个 op 不计算, 只记录")
    print("=" * 60)

    class TracerDemo:
        """最小复现 Tracer 如何工作。"""

        def __init__(self):
            self.graph = fx.Graph()
            self._node_counter = 0

        def create_arg(self, value):
            if isinstance(value, fx.Proxy):
                return value.node
            else:
                return value

        def trace(self, fn, *args):
            self.graph = fx.Graph()
            # 创建 placeholder 节点
            placeholder = self.graph.placeholder("x")
            proxy = fx.Proxy(placeholder, self)
            # 执行 fn — 每个操作都被 Proxy 拦截
            result = fn(proxy)
            # 添加 output 节点
            self.graph.output(result.node)
            return self.graph

    def my_fn(x):
        a = x * 2
        b = a + 1
        c = b.relu()
        return c

    tracer = TracerDemo()
    g = tracer.trace(my_fn, torch.randn(1))

    print(f"  Traced graph nodes: {len(g.nodes)}")
    for node in g.nodes:
        print(
            f"    {node.op:>15s} {node.name:10s} "
            f"target={str(node.target)[:30]} "
            f"args={[a.name if hasattr(a, 'name') else str(a)[:20] for a in node.args]}"
        )

    print(f"\n  Proxy 原理:")
    print(f"  1. __call__ → create_node('call_function', target, args)")
    print(f"  2. __getattr__ → create_node('call_method', 'relu', [self])")
    print(f"  3. 每次返回新的 Proxy → 链式构建图")
    print(f"  4. 最终 graph.output(last_proxy) → 图完成")
    print(f"  5. 图的执行由 Interpreter 或 GraphModule.__call__ 完成")
    print()


# ============ 2. Interpreter 节点遍历 ============
def exp_interpreter():
    """使用 Interpreter 自定义图执行逻辑。"""
    print("=" * 60)
    print("2. Interpreter: 自定义图遍历/执行")
    print("=" * 60)

    class Model(nn.Module):
        def forward(self, x):
            return x * 2 + 1

    model = Model()
    gm = fx.symbolic_trace(model)

    # 自定义 Interpreter: 打印每个节点的执行
    class TracingInterpreter(fx.Interpreter):
        def run_node(self, n):
            val = super().run_node(n)
            print(f"    [{n.op}] {n.name} = {val}")
            return val

    print(f"  Standard execution:")
    x = torch.tensor([3.0])
    y = gm(x)
    print(f"    result = {y}")

    print(f"\n  Interpreter execution (tracing each node):")
    interpreter = TracingInterpreter(gm)
    y2 = interpreter.run(x)
    print(f"    result = {y2}")

    # Interpreter 可以替换节点实现
    class DoublingInterpreter(fx.Interpreter):
        def call_function(self, target, args, kwargs):
            if target == torch.add:
                return super().call_function(torch.mul, args, kwargs)  # add → mul
            return super().call_function(target, args, kwargs)

    print(f"\n  Modified Interpreter (add → mul):")
    x = torch.tensor([3.0])
    y3 = DoublingInterpreter(gm).run(x)
    print(f"    original: (3*2)+1 = 7")
    print(f"    modified: (3*2)*1 = {y3.item():.0f}")
    print()


# ============ 3. Node 元数据深度分析 ============
def exp_node_meta():
    """分析 FX Node 携带的元数据 (stack_trace, val, meta)。"""
    print("=" * 60)
    print("3. Node Meta: 每个节点记录的元数据")
    print("=" * 60)

    class MetaModel(nn.Module):
        def forward(self, x):
            return x * 2 + 1

    model = MetaModel()
    gm = fx.symbolic_trace(model)

    # Propagate fake values for metadata
    from torch.fx.passes.shape_prop import ShapeProp

    ShapeProp(gm).propagate(torch.randn(4))

    print(f"  Node metadata:")
    for node in gm.graph.nodes:
        print(f"  [{node.op:>15s}] {node.name}")
        print(f"    meta keys: {list(node.meta.keys())}")
        if "stack_trace" in node.meta:
            trace = node.meta["stack_trace"]
            print(f"    stack_trace:")
            for line in trace.split("\n")[:3]:
                print(f"      {line.strip()[:80]}")
        if "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "shape"):
                print(f"    fake_val: shape={list(val.shape)} dtype={val.dtype}")
        if "tensor_meta" in node.meta:
            meta = node.meta["tensor_meta"]
            print(
                f"    tensor_meta: shape={list(meta.shape)} dtype={meta.dtype} "
                f"requires_grad={meta.requires_grad}"
            )
        print()

    print(f"  meta 用途:")
    print(f"  stack_trace: 调试 — 知道图中每个 op 来自源代码的哪一行")
    print(f"  val:         形状推断 — FakeTensor 的结果")
    print(f"  tensor_meta: 编译 — codegen 时用于分配内存/选 kernel")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_proxy_trace()
        exp_interpreter()
        exp_node_meta()

    print("[FX Graphs source analysis] DONE")


if __name__ == "__main__":
    main()
