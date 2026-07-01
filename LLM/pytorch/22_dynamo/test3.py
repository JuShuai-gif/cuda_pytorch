"""Dynamo 源码分析: VariableTracker, symbolic_convert, graph capture 内部。

使用工具: torch._dynamo.eval_frame / TORCH_LOGS='graph_code' /
         symbolic_trace 对比 / graph break 细粒度分析

运行:
  python test3.py                    # 全链路分析
  python test3.py variable_tracker   # VariableTracker 类型探究
  python test3.py graph_compare      # Dynamo graph vs symbolic_trace
  python test3.py graph_code         # 查看生成的 FX 图 Python 代码

参考源码:
  torch/_dynamo/symbolic_convert.py  — InstructionTranslator (符号执行)
  torch/_dynamo/variables/           — VariableTracker 子类
  torch/_dynamo/guards.py             — Guard 构建
"""

import sys
import torch
import torch.fx as fx


# ============ 1. VariableTracker 类型探究 ============
def exp_variable_tracker():
    """查看 Dynamo 如何追踪各种 Python 对象。"""
    print("=" * 60)
    print("1. VariableTracker: Dynamo 如何抽象 Python 对象")
    print("=" * 60)

    # VariableTracker 的类型层次 (从源码 torch/_dynamo/variables/):
    tracker_types = {
        "TensorVariable": "包装 torch.Tensor",
        "SymNodeVariable": "符号维度变量",
        "ConstantVariable": "Python 常量 (int, float, str, None)",
        "ListVariable": "Python list",
        "DictVariable": "Python dict",
        "UserDefinedObjectVariable": "任意 Python 对象 (slow path)",
        "ModuleVariable": "nn.Module 实例",
        "BuiltinVariable": "Python 内建函数 (print, range, len)",
        "NNModuleVariable": "nn.functional 等",
    }

    print("  VariableTracker 类型层次:")
    for name, desc in tracker_types.items():
        print(f"    {name:35s}: {desc}")

    # 展示 Dynamo 如何处理不同类型
    @torch.compile(fullgraph=True)
    def fn(x):
        a = x * 2  # TensorVariable
        b = 3.14  # ConstantVariable
        c = [a, a + 1]  # ListVariable
        d = a.sum()  # TensorVariable with reduction
        return a + d

    x = torch.randn(4)
    try:
        y = fn(x)
        print(f"\n  Compiled successfully: output={y.item():.2f}")
        print(f"  Dynamo 把 Python code 转换为:")
        print(f"    x → TensorVariable")
        print(f"    3.14 → ConstantVariable")
        print(f"    [a, a+1] → ListVariable (延迟求值)")
        print(f"    a.sum() → TensorVariable (reduction)")
    except Exception as e:
        print(f"\n  Compile failed: {e}")

    print()


# ============ 2. Dynamo vs symbolic_trace 对比 ============
def exp_graph_compare():
    """对比 Dynamo 图和 torch.fx.symbolic_trace 图。"""
    print("=" * 60)
    print("2. Dynamo Graph vs symbolic_trace: 两种追踪方式对比")
    print("=" * 60)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

        def forward(self, x):
            a = self.linear(x)
            b = torch.relu(a)
            return b

    model = Model()

    # Method 1: torch.fx.symbolic_trace
    gm_sym = fx.symbolic_trace(model)
    print(f"  symbolic_trace:")
    print(f"    nodes: {len(gm_sym.graph.nodes)}")
    for n in gm_sym.graph.nodes:
        print(f"      {n.op:>15s} {n.name:10s} target={str(n.target)[:40]}")

    # Method 2: Dynamo (torch.compile → capture graph)
    captured_graph = None

    def backend_capture(gm, example_inputs):
        nonlocal captured_graph
        captured_graph = gm
        return gm.forward  # 返回原始 forward (不编译)

    @torch.compile(backend=backend_capture)
    def dynamo_fn(x):
        return model(x)

    x = torch.randn(2, 4)
    dynamo_fn(x)

    if captured_graph:
        print(f"\n  Dynamo captured graph:")
        print(f"    nodes: {len(captured_graph.graph.nodes)}")
        for n in captured_graph.graph.nodes:
            print(f"      {n.op:>15s} {n.name:10s} target={str(n.target)[:40]}")

        print(f"\n  Difference: symbolic_trace traces at Python level")
        print(f"  Dynamo traces at bytecode level → can handle more patterns")
    print()


# ============ 3. Graph code generation ============
def exp_graph_code():
    """查看图生成的 Python 代码。"""
    print("=" * 60)
    print("3. Graph Code: 查看 FX 图生成的 Python forward")
    print("=" * 60)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, bias=False)
            self.bn = nn.BatchNorm2d(8)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = torch.relu(x)
            return x

    model = Net()
    gm = fx.symbolic_trace(model)

    print(f"  Generated Python code:")
    print(f"  --- code start ---")
    code_lines = gm.code.split("\n")
    for line in code_lines[:25]:
        print(f"  {line}")
    if len(code_lines) > 25:
        print(f"  ... ({len(code_lines)} lines total)")
    print(f"  --- code end ---")

    # 展示 codegen 如何工作
    print(f"\n  Graph Code Generation (torch/fx/graph.py):")
    print(f"  1. 遍历 graph.nodes (拓扑序)")
    print(f"  2. 对每个 node 生成一行 Python 赋值语句")
    print(f"     placeholder  → 不做生成 (在函数签名中)")
    print(f"     call_module  → self_{node.name} = self.{node.target}(...);")
    print(f"     call_function→ {node.name} = torch.{fn}(...);")
    print(f"     call_method  → {node.name} = {obj}.{method}(...);")
    print(f"     output       → return ({args});")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_variable_tracker()
        exp_graph_compare()
        exp_graph_code()

    print("[Dynamo source analysis] DONE")


if __name__ == "__main__":
    main()
